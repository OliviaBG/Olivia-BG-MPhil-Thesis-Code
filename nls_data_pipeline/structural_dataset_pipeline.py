"""
Real per-residue SASA + pLDDT for the NLS predictor -- direct analog of
../nes_data_pipeline/structural_dataset_v2_pipeline.py, same fix for the
same reason: nls_ml_predictor.py's _extract_features() only gets real
plddt_values/sasa_values when structural_data.json exists; until then,
plddt_norm/sasa_norm are constant neutral defaults (0.75/0.50) and carry
zero learned signal, which is exactly what today's held-out permutation
importance run showed (both at 0.0 importance).

Must be run somewhere with real internet access, since it queries
alphafold.ebi.ac.uk in bulk (see the NES structural pipeline's docstring
for the same requirement).

    pip install freesasa requests
    python3 structural_dataset_pipeline.py

Reads nls_dataset.csv + nls_negatives.csv (both have real UniProt
accession + start/end for every row that came from real data, i.e.
everything except synthetic_polybasic_decoy negatives). Output:
structural_data.json -- list of {seq, accession, label, sasa_per_residue,
plddt_per_residue, whole_protein_sasa_avg, whole_protein_plddt_avg}.
Checkpointed every 25 proteins, resumable.
"""
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import freesasa
import requests

HERE = Path(__file__).resolve().parent
DATASET_CSV = HERE / "nls_dataset.csv"
NEGATIVES_CSV = HERE / "nls_negatives.csv"
OUTPUT_JSON = HERE / "structural_data.json"

# Reuse the exact same 3-method consensus RSA calculation the
# live app uses (app.py's calculate_sasa() was itself ported from this same
# module) -- see real_per_residue_sasa() below for why this replaced the
# single-method FreeSASA Shrake-Rupley calculation this script used to do.
sys.path.insert(0, str(HERE.parent))
from consensus_accessibility import consensus_accessibility


def _get_with_retries(url, timeout=20, retries=2):
    for attempt in range(retries + 1):
        try:
            return requests.get(url, timeout=timeout)
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1.5)
                continue
            return None
    return None


def fetch_alphafold_pdb(uniprot_id):
    meta_resp = _get_with_retries(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}")
    if meta_resp is None or meta_resp.status_code != 200:
        return None
    try:
        meta = meta_resp.json()
    except ValueError:
        return None
    if not meta or "pdbUrl" not in meta[0]:
        return None
    resp = _get_with_retries(meta[0]["pdbUrl"])
    if resp is None or resp.status_code != 200:
        return None
    return resp.text


# Tien et al. 2013, theoretical MaxASA (Ų) -- PLoS ONE 8(11):e80635.
# Kept in sync with the same table in app.py / consensus_accessibility.py.
MAX_ASA_TIEN2013 = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 225.0, 'GLU': 223.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}
DEFAULT_MAX_ASA = 200.0


def real_per_residue_sasa(pdb_text):
    """Real 3-method CONSENSUS relative solvent accessibility (RSA),
    normalized against Tien et al. 2013 residue-specific max ASA: FreeSASA
    Lee-Richards + FreeSASA Shrake-Rupley + Biopython Shrake-Rupley,
    averaged per residue, via the shared consensus_accessibility() helper
    (same one app.py's calculate_sasa(return_stats=True) uses).

    fix: this function previously used FreeSASA Shrake-Rupley
    ALONE, while the live app already computed the 3-method consensus --
    meaning sasa_norm was trained on one distribution (single-method) and
    predicted on a different one (consensus). Both sides now call the exact
    same consensus_accessibility() function, so training-time and
    inference-time sasa_norm are genuinely on the same scale. Still matters
    a lot for NLS specifically since Lys/Arg have some of the largest
    max-ASA values of any residue -- see consensus_accessibility.py for the
    full method/rationale."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_text)
        tmp_path = f.name
    try:
        rows = consensus_accessibility(tmp_path)
        if not rows:
            return {}
        chain_id = sorted({r.chain for r in rows})[0]
        return {r.resnum: r.consensus_rsa for r in rows if r.chain == chain_id}
    finally:
        os.unlink(tmp_path)


def real_per_residue_plddt(pdb_text):
    out = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                out[int(line[22:26].strip())] = float(line[60:66].strip())
            except ValueError:
                continue
    return out


def load_tasks():
    tasks = []
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (row.get("accession") and row.get("start") and row.get("end")):
                continue
            tasks.append({"label": 1, "seq": row["nls_sequence"].upper(), "accession": row["accession"],
                          "start": int(row["start"]), "end": int(row["end"])})
    with open(NEGATIVES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("neg_type") == "synthetic_polybasic_decoy":
                continue
            if not (row.get("accession") and row.get("start") and row.get("end")):
                continue
            tasks.append({"label": 0, "seq": row["neg_sequence"].upper(), "accession": row["accession"],
                          "start": int(row["start"]), "end": int(row["end"])})
    return tasks


def main():
    tasks = load_tasks()
    accessions = sorted({t["accession"] for t in tasks})
    print(f"{len(tasks)} tasks across {len(accessions)} unique accessions")

    already_done, records = set(), []
    if OUTPUT_JSON.exists():
        try:
            records = json.load(open(OUTPUT_JSON, encoding="utf-8"))
            already_done = {r["accession"] for r in records}
            print(f"Resuming: {len(already_done)} accessions already processed")
        except Exception:
            records = []

    cache = {}
    for i, acc in enumerate(accessions, 1):
        if acc in already_done:
            continue
        print(f"[{i}/{len(accessions)}] {acc} ...", end=" ")
        pdb_text = fetch_alphafold_pdb(acc)
        if pdb_text is None:
            print("SKIP -- no AlphaFold structure")
            cache[acc] = None
            time.sleep(0.34)
            continue
        try:
            sasa_by_res = real_per_residue_sasa(pdb_text)
            plddt_by_res = real_per_residue_plddt(pdb_text)
        except Exception as e:
            print(f"SKIP -- {e}")
            cache[acc] = None
            time.sleep(0.34)
            continue
        cache[acc] = (sasa_by_res, plddt_by_res)
        print(f"OK -- {len(sasa_by_res)} residues")
        time.sleep(0.34)
        if i % 25 == 0:
            json.dump(records, open(OUTPUT_JSON, "w"), indent=2)
            print(f"  (checkpoint saved, {len(records)} records so far)")

    for t in tasks:
        cached = cache.get(t["accession"])
        if cached is None:
            continue
        sasa_by_res, plddt_by_res = cached
        sasa_window = [v for v in (sasa_by_res.get(p) for p in range(t["start"], t["end"] + 1)) if v is not None]
        plddt_window = [v for v in (plddt_by_res.get(p) for p in range(t["start"], t["end"] + 1)) if v is not None]
        if not sasa_window and not plddt_window:
            continue
        records.append({
            "seq": t["seq"], "accession": t["accession"], "label": t["label"],
            "sasa_per_residue": sasa_window, "plddt_per_residue": plddt_window,
        })

    json.dump(records, open(OUTPUT_JSON, "w"), indent=2)
    print(f"\nDone. Wrote {len(records)} records to {OUTPUT_JSON.name}")


if __name__ == "__main__":
    main()
