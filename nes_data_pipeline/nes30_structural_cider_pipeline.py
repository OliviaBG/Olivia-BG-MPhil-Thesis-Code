"""
Real structural + CIDER dataset builder for 30 (31) validated NES-containing
proteins, drawn from nes_dataset.csv (real UniProt accessions extracted from
the NESbase db_reference field).

For each protein:
  1. Fetches the real AlphaFold structure via its UniProt accession.
  2. Runs freesasa with the Shrake-Rupley ("rolling ball") algorithm to get
     real per-residue SASA -- not the neutral-fallback placeholder.
  3. Runs localCIDER on the full sequence to get linear hydropathy, linear
     NCPR (net charge per residue), and linear FCR (fraction of charged
     residues) -- one value per residue.
  4. Aligns all of this to the annotated NES region (and a flanking window)
     so you can directly compare "inside the NES" vs "rest of the protein"
     for real exposure and real linear charge/hydropathy patterning.

Must be run somewhere with real internet access and both packages
installed, since it queries alphafold.ebi.ac.uk directly:

    pip install freesasa localcider requests biopython
    python3 nes30_structural_cider_pipeline.py

Output: nes30_structural_cider_data.json (one record per protein) and
nes30_structural_cider_summary.csv (one row per protein, NES-region vs
whole-protein averages, easy to eyeball in Excel/pandas for patterns).
"""
import csv
import json
import time
import sys

import requests
import freesasa
from localcider.sequenceParameters import SequenceParameters

CANDIDATES_FILE = 'nes30_candidates.json'
FLANK = 20  # residues of context on each side of the annotated NES, for comparison
BLOB_LEN = 5  # localCIDER sliding window size


def fetch_alphafold_pdb(uniprot_id):
    """
    Fetch the real, CURRENT AlphaFold model for a UniProt accession.

    Do NOT hardcode a version suffix (e.g. '..._v4.pdb') -- AlphaFold DB has
    moved on to v6 (confirmed live: P04637's API metadata reports
    latestVersion=6, file storage has also moved behind a GCS bucket, so the
    old alphafold.ebi.ac.uk/files/AF-<id>-F1-model_v4.pdb URL now 404s with a
    'NoSuchKey' GCS error for every accession regardless of whether the
    structure exists). Always ask the metadata API for the real current
    pdbUrl instead.
    """
    api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}'
    try:
        meta_resp = requests.get(api_url, timeout=30)
    except requests.RequestException as e:
        print(f"metadata API request failed: {e}", end=' ')
        return None
    if meta_resp.status_code != 200:
        return None
    try:
        meta = meta_resp.json()
    except ValueError:
        return None
    if not meta or 'pdbUrl' not in meta[0]:
        return None

    pdb_url = meta[0]['pdbUrl']
    resp = requests.get(pdb_url, timeout=30)
    if resp.status_code != 200:
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


def real_sasa_per_residue(pdb_text):
    """Real per-residue RELATIVE solvent accessibility (RSA) via freesasa's
    Shrake-Rupley (rolling ball) algorithm, normalized against Tien et al.
    2013 residue-specific max ASA (NOT raw Ų -- must match app.py's
    calculate_sasa() scale so this training data lines up with live
    inference features). Returns a list of (residue_number, rsa) in file
    order, restricted to a single chain (AlphaFold single-protein models are
    always one chain, but this is filtered explicitly rather than assumed --
    tested this function against a real multi-chain complex (crm1.pdb) and
    confirmed that without this filter, residue numbers from different
    chains collide and silently corrupt the alignment)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_text)
        tmp_path = f.name
    try:
        structure = freesasa.Structure(tmp_path)
        params = freesasa.Parameters({'algorithm': freesasa.ShrakeRupley})
        result = freesasa.calc(structure, params)
        residue_areas = result.residueAreas()
        if len(residue_areas) > 1:
            print(f"    (note: {len(residue_areas)} chains in structure, "
                  f"using chain '{sorted(residue_areas.keys())[0]}' only)")
        chain_id = sorted(residue_areas.keys())[0]
        per_residue = [
            (int(res_num_str), min(area.total / MAX_ASA_TIEN2013.get(area.residueType, DEFAULT_MAX_ASA), 1.5))
            for res_num_str, area in residue_areas[chain_id].items()
        ]
        per_residue.sort(key=lambda x: x[0])
        return per_residue
    finally:
        os.unlink(tmp_path)


def linear_cider(sequence):
    sp = SequenceParameters(sequence)
    _, hydro = sp.get_linear_hydropathy(blobLen=BLOB_LEN)
    _, ncpr = sp.get_linear_NCPR(blobLen=BLOB_LEN)
    _, fcr = sp.get_linear_FCR(blobLen=BLOB_LEN)
    return [float(v) for v in hydro], [float(v) for v in ncpr], [float(v) for v in fcr]


def main():
    with open(CANDIDATES_FILE) as f:
        candidates = json.load(f)

    print(f"Processing {len(candidates)} proteins...")
    records = []
    summary_rows = []

    for i, c in enumerate(candidates, 1):
        uid = c['uniprot']
        name = c['protein_name']
        full_seq = c['full_sequence']
        nes_start, nes_end = c['nes_start'], c['nes_end']  # 1-based inclusive

        print(f"[{i}/{len(candidates)}] {uid} ({name[:40]}) ...", end=' ')

        pdb_text = fetch_alphafold_pdb(uid)
        if pdb_text is None:
            print("SKIP -- no AlphaFold structure found")
            continue

        try:
            sasa_pairs = real_sasa_per_residue(pdb_text)
        except Exception as e:
            print(f"SKIP -- freesasa failed: {e}")
            continue

        sasa_by_pos = dict(sasa_pairs)
        # Confirm the structure's residue numbering roughly matches the
        # dataset's full_sequence length before trusting alignment
        if abs(len(sasa_by_pos) - len(full_seq)) > 5:
            print(f"WARNING -- structure has {len(sasa_by_pos)} residues, "
                  f"dataset sequence has {len(full_seq)}; using structure's own numbering")

        try:
            hydro, ncpr, fcr = linear_cider(full_seq)
        except Exception as e:
            print(f"SKIP -- localCIDER failed: {e}")
            continue

        nes_flank_start = max(1, nes_start - FLANK)
        nes_flank_end = min(len(full_seq), nes_end + FLANK)

        def region_avg(values_by_index_1based, start, end):
            vals = [values_by_index_1based.get(p) for p in range(start, end + 1)]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        sasa_nes_avg = region_avg(sasa_by_pos, nes_start, nes_end)
        sasa_flank_avg = region_avg(sasa_by_pos, nes_flank_start, nes_flank_end)
        sasa_whole_avg = sum(sasa_by_pos.values()) / len(sasa_by_pos) if sasa_by_pos else None

        hydro_by_pos = {p + 1: v for p, v in enumerate(hydro)}
        ncpr_by_pos = {p + 1: v for p, v in enumerate(ncpr)}
        fcr_by_pos = {p + 1: v for p, v in enumerate(fcr)}

        hydro_nes_avg = region_avg(hydro_by_pos, nes_start, nes_end)
        ncpr_nes_avg = region_avg(ncpr_by_pos, nes_start, nes_end)
        fcr_nes_avg = region_avg(fcr_by_pos, nes_start, nes_end)
        hydro_whole_avg = sum(hydro) / len(hydro) if hydro else None
        ncpr_whole_avg = sum(ncpr) / len(ncpr) if ncpr else None
        fcr_whole_avg = sum(fcr) / len(fcr) if fcr else None

        record = {
            'uniprot': uid,
            'protein_name': name,
            'nes_start': nes_start,
            'nes_end': nes_end,
            'full_sequence': full_seq,
            'nes_sequence': full_seq[nes_start - 1:nes_end],
            'sasa_per_residue': [sasa_by_pos.get(p) for p in range(1, len(full_seq) + 1)],
            'linear_hydropathy': hydro,
            'linear_ncpr': ncpr,
            'linear_fcr': fcr,
            'sasa_nes_avg': sasa_nes_avg,
            'sasa_flank_avg': sasa_flank_avg,
            'sasa_whole_protein_avg': sasa_whole_avg,
            'hydro_nes_avg': hydro_nes_avg,
            'ncpr_nes_avg': ncpr_nes_avg,
            'fcr_nes_avg': fcr_nes_avg,
            'hydro_whole_protein_avg': hydro_whole_avg,
            'ncpr_whole_protein_avg': ncpr_whole_avg,
            'fcr_whole_protein_avg': fcr_whole_avg,
        }
        records.append(record)
        summary_rows.append({
            'uniprot': uid, 'protein_name': name, 'nes_start': nes_start, 'nes_end': nes_end,
            'sasa_nes_avg': round(sasa_nes_avg, 2) if sasa_nes_avg else '',
            'sasa_flank_avg': round(sasa_flank_avg, 2) if sasa_flank_avg else '',
            'sasa_whole_protein_avg': round(sasa_whole_avg, 2) if sasa_whole_avg else '',
            'hydro_nes_avg': round(hydro_nes_avg, 3) if hydro_nes_avg else '',
            'hydro_whole_protein_avg': round(hydro_whole_avg, 3) if hydro_whole_avg else '',
            'ncpr_nes_avg': round(ncpr_nes_avg, 3) if ncpr_nes_avg else '',
            'ncpr_whole_protein_avg': round(ncpr_whole_avg, 3) if ncpr_whole_avg else '',
            'fcr_nes_avg': round(fcr_nes_avg, 3) if fcr_nes_avg else '',
            'fcr_whole_protein_avg': round(fcr_whole_avg, 3) if fcr_whole_avg else '',
        })
        print(f"OK -- SASA(NES)={sasa_nes_avg:.1f} SASA(whole)={sasa_whole_avg:.1f}" if sasa_nes_avg else "OK")
        time.sleep(0.5)  # be polite to the AlphaFold DB server

    with open('nes30_structural_cider_data.json', 'w') as f:
        json.dump(records, f, indent=2)

    if summary_rows:
        with open('nes30_structural_cider_summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"\nDone. {len(records)}/{len(candidates)} proteins processed successfully.")
    print("Wrote nes30_structural_cider_data.json (full per-residue arrays)")
    print("Wrote nes30_structural_cider_summary.csv (NES-region vs whole-protein averages)")


if __name__ == '__main__':
    main()
