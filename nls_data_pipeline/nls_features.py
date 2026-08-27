"""
Standalone NLS feature-extraction script -- runs directly on the raw data
(nls_dataset.csv / nls_negatives.csv), with NO trained model/classifier
involved. This is the NLS-side analog of nes_data_pipeline/nes_features.py:
a plain, dependency-light module you can import or run from the command
line to turn raw sequence windows into a flat numeric feature table,
independent of nls_ml_predictor.py's NLSPredictor class (which wraps this
same logic plus the trained sklearn model on top).

Every feature family used by the NLS model is computed here, each in its
own clearly-separated function so it can be audited/re-run individually:
    - PSSM (built fresh from the raw positives every run -- see build_pssm)
    - net charge / basic & acidic fraction (charge)
    - Kyte-Doolittle hydrophobicity (hydrophobicity)
    - flanking-region disorder + charge (flanking regions)
    - per-residue disorder propensity (disorder)
    - real localCIDER linear charge/hydropathy/complexity profiles (CIDER)
    - SASA/pLDDT pass-through if structural_data.json exists (SASA) --
      otherwise reported as missing rather than silently defaulted, since
      this script is meant to show you the raw data, not paper over gaps
    - monopartite / bipartite consensus pattern flags
    - full 20-aa composition

Usage as a library:
    from nls_features import featurize, build_pssm, pssm_score

    pssm = build_pssm(["PKKKRKV", "KRPAATKKAGQAKKKK", ...])
    featurize("PKKKRKV", pssm=pssm)   # -> dict of ~40 numeric features

Usage as a script (writes a flat, ready-to-analyze/ready-to-train CSV):
    python3 nls_features.py --dataset nls_dataset.csv --negatives nls_negatives.csv --out nls_training_table.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

try:
    from localcider.sequenceParameters import SequenceParameters
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Residue scales (identical to nls_ml_predictor.py -- kept in sync
# deliberately so the standalone feature audit and the shipped model agree)
# ---------------------------------------------------------------------------

KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2, "X": 0.0,
}

DISORDER_PROPENSITY = {
    'P': 1.0, 'R': 0.9, 'E': 0.9, 'K': 0.9, 'S': 0.8,
    'Q': 0.8, 'D': 0.8, 'G': 0.7, 'A': 0.5, 'T': 0.5,
    'N': 0.5, 'H': 0.4, 'M': 0.3, 'C': 0.3, 'L': 0.2,
    'F': 0.2, 'I': 0.2, 'V': 0.2, 'W': 0.1, 'Y': 0.1,
    'X': 0.5,
}

BASIC = set("KR")
ACIDIC = set("DE")
HYDROPHOBIC = set("LIVFM")
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

MONOPARTITE_RE = re.compile(r"[KR][KR].{0,2}[KR]")

PSSM_LEFT, PSSM_RIGHT = 8, 10
PSSM_WIDTH = PSSM_LEFT + PSSM_RIGHT


# ---------------------------------------------------------------------------
# 1. Bipartite detector
# ---------------------------------------------------------------------------

def detect_bipartite(seq):
    n = len(seq)
    for i in range(n - 1):
        if seq[i] in BASIC and seq[i + 1] in BASIC:
            for spacer in range(9, 13):
                j = i + 2 + spacer
                if j + 5 <= n:
                    window = seq[j:j + 5]
                    if sum(1 for c in window if c in BASIC) >= 3:
                        return True, spacer
    return False, None


# ---------------------------------------------------------------------------
# 2. PSSM -- built fresh from whatever sequences you pass in (no pickle,
#    no trained model; this is the raw-data version)
# ---------------------------------------------------------------------------

def _pssm_anchor(seq):
    m = MONOPARTITE_RE.search(seq)
    if m:
        return m.start()
    best_i, best_score = 0, -1
    for i in range(max(1, len(seq) - 3)):
        score = sum(1 for c in seq[i:i + 4] if c in BASIC)
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def build_pssm(sequences):
    """Returns (pssm_matrix, aa_to_idx, n_sequences_used)."""
    aa_to_idx = {a: i for i, a in enumerate(AA_ALPHABET)}
    counts = np.ones((PSSM_WIDTH, 20))
    n_used = 0
    for seq in sequences:
        seq = seq.upper()
        if len(seq) < 4:
            continue
        anchor = _pssm_anchor(seq)
        for c in range(PSSM_WIDTH):
            idx = anchor - (PSSM_LEFT - 1) + c
            if 0 <= idx < len(seq) and seq[idx] in aa_to_idx:
                counts[c, aa_to_idx[seq[idx]]] += 1
        n_used += 1
    freqs = counts / counts.sum(axis=1, keepdims=True)
    pssm = np.log2(freqs / (np.ones(20) / 20.0))
    return pssm, aa_to_idx, n_used


def pssm_score(seq, pssm):
    matrix, aa_to_idx, _ = pssm
    seq = seq.upper()
    anchor = _pssm_anchor(seq)
    score = 0.0
    for c in range(PSSM_WIDTH):
        idx = anchor - (PSSM_LEFT - 1) + c
        if 0 <= idx < len(seq) and seq[idx] in aa_to_idx:
            score += matrix[c, aa_to_idx[seq[idx]]]
    return float(score)


# ---------------------------------------------------------------------------
# 3. Charge
# ---------------------------------------------------------------------------

def charge_features(seq):
    n = max(1, len(seq))
    n_basic = sum(1 for a in seq if a in BASIC)
    n_acidic = sum(1 for a in seq if a in ACIDIC)
    return {
        "net_charge": float(n_basic - n_acidic),
        "frac_basic": n_basic / n,
        "frac_acidic": n_acidic / n,
    }


# ---------------------------------------------------------------------------
# 4. Hydrophobicity
# ---------------------------------------------------------------------------

def hydrophobicity_features(seq):
    n = max(1, len(seq))
    return {
        "mean_kd_hydropathy": float(np.mean([KD_SCALE.get(a, 0.0) for a in seq])),
        "frac_hydrophobic": sum(1 for a in seq if a in HYDROPHOBIC) / n,
    }


# ---------------------------------------------------------------------------
# 5. Flanking regions (disorder + charge of the 10 aa on either side)
# ---------------------------------------------------------------------------

def flanking_region_features(full_sequence, start, end, flank_len=10):
    if not full_sequence or start is None or end is None:
        return {"n_flank_disorder": None, "c_flank_disorder": None,
                "n_flank_net_charge": None, "c_flank_net_charge": None}
    n_flank = full_sequence[max(0, start - flank_len):start]
    c_flank = full_sequence[end:end + flank_len]
    return {
        "n_flank_disorder": float(np.mean([DISORDER_PROPENSITY.get(a, 0.5) for a in n_flank])) if n_flank else None,
        "c_flank_disorder": float(np.mean([DISORDER_PROPENSITY.get(a, 0.5) for a in c_flank])) if c_flank else None,
        "n_flank_net_charge": float(sum(1 for a in n_flank if a in BASIC) - sum(1 for a in n_flank if a in ACIDIC)) if n_flank else None,
        "c_flank_net_charge": float(sum(1 for a in c_flank if a in BASIC) - sum(1 for a in c_flank if a in ACIDIC)) if c_flank else None,
    }


# ---------------------------------------------------------------------------
# 6. Disorder propensity (region itself, not flanks)
# ---------------------------------------------------------------------------

def disorder_features(seq):
    return {"disorder_mean": float(np.mean([DISORDER_PROPENSITY.get(a, 0.5) for a in seq]))}


# ---------------------------------------------------------------------------
# 7. CIDER (real localCIDER linear/sliding-window profiles)
# ---------------------------------------------------------------------------

def cider_features(seq):
    if not CIDER_AVAILABLE or len(seq) < 6:
        return {"cider_ncpr_range": None, "cider_hydropathy_range": None, "cider_complexity_mean": None}
    try:
        sp = SequenceParameters(seq)
        ncpr = sp.get_linear_NCPR()[1]
        hydro = sp.get_linear_hydropathy()[1]
        complexity = sp.get_linear_complexity()[1]
        return {
            "cider_ncpr_range": float(max(ncpr) - min(ncpr)) if len(ncpr) else None,
            "cider_hydropathy_range": float(max(hydro) - min(hydro)) if len(hydro) else None,
            "cider_complexity_mean": float(np.mean(complexity)) if len(complexity) else None,
        }
    except Exception:
        return {"cider_ncpr_range": None, "cider_hydropathy_range": None, "cider_complexity_mean": None}


# ---------------------------------------------------------------------------
# 8. SASA / pLDDT pass-through (only if structural_data.json exists --
#    reports None rather than a fake neutral default, since this script's
#    purpose is showing you the real raw data)
# ---------------------------------------------------------------------------

def load_structural_lookup():
    path = HERE / "structural_data.json"
    if not path.exists():
        return {}
    records = json.load(open(path, encoding="utf-8"))
    return {r["seq"].upper(): r for r in records}


def structural_features(seq, structural_lookup):
    rec = structural_lookup.get(seq.upper())
    if not rec:
        return {"mean_sasa": None, "mean_plddt": None}
    sasa = rec.get("sasa_per_residue") or []
    plddt = rec.get("plddt_per_residue") or []
    return {
        "mean_sasa": float(np.mean(sasa)) if sasa else None,
        "mean_plddt": float(np.mean(plddt)) if plddt else None,
    }


# ---------------------------------------------------------------------------
# 9. Composition (full 20-aa)
# ---------------------------------------------------------------------------

def composition_features(seq):
    n = max(1, len(seq))
    return {f"comp_{aa}": seq.count(aa) / n for aa in AA_ALPHABET}


# ---------------------------------------------------------------------------
# Bring it all together
# ---------------------------------------------------------------------------

def featurize(window_seq, full_sequence=None, start=None, end=None, pssm=None, structural_lookup=None):
    seq = window_seq.upper()
    feats = {"sequence": seq, "length": len(seq)}
    feats["pssm_score"] = pssm_score(seq, pssm) if pssm is not None else None
    feats.update(charge_features(seq))
    feats.update(hydrophobicity_features(seq))
    feats.update(disorder_features(seq))
    feats.update(flanking_region_features(full_sequence, start, end))
    is_bip, spacer = detect_bipartite(seq)
    feats["is_bipartite_pattern"] = int(is_bip)
    feats["bipartite_spacer"] = spacer if spacer is not None else -1
    feats["is_monopartite_pattern"] = int(bool(MONOPARTITE_RE.search(seq)))
    feats.update(cider_features(seq))
    feats.update(structural_features(seq, structural_lookup or {}))
    feats.update(composition_features(seq))
    return feats


# ---------------------------------------------------------------------------
# CLI: run on the raw dataset CSVs, no model involved
# ---------------------------------------------------------------------------

def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_training_table(dataset_csv, negatives_csv):
    pos_rows = _read_csv(dataset_csv)
    neg_rows = _read_csv(negatives_csv)

    pssm = build_pssm([r["nls_sequence"] for r in pos_rows if r.get("nls_sequence")])
    structural_lookup = load_structural_lookup()

    out = []
    for r in pos_rows:
        seq = r["nls_sequence"]
        start = int(r["start"]) - 1 if r.get("start") else None
        end = int(r["end"]) if r.get("end") else None
        feats = featurize(seq, r.get("full_sequence"), start, end, pssm=pssm, structural_lookup=structural_lookup)
        feats.update({"label": 1, "accession": r["accession"], "organism": r.get("organism"),
                      "bipartite_annotation": r.get("bipartite"), "confidence": r.get("confidence")})
        out.append(feats)
    for r in neg_rows:
        seq = r["neg_sequence"]
        start = int(r["start"]) - 1 if r.get("start") else None
        end = int(r["end"]) if r.get("end") else None
        feats = featurize(seq, r.get("full_sequence"), start, end, pssm=pssm, structural_lookup=structural_lookup)
        feats.update({"label": 0, "accession": r["accession"], "organism": r.get("organism"),
                      "neg_type": r.get("neg_type")})
        out.append(feats)
    return out, pssm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "nls_dataset.csv"))
    ap.add_argument("--negatives", default=str(HERE / "nls_negatives.csv"))
    ap.add_argument("--out", default=str(HERE / "nls_training_table.csv"))
    args = ap.parse_args()

    table, pssm = build_training_table(args.dataset, args.negatives)
    if not table:
        print("No rows produced -- check --dataset/--negatives paths.")
        return

    fieldnames = list(table[0].keys())
    for row in table:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(table)

    n_pos = sum(1 for r in table if r["label"] == 1)
    n_neg = sum(1 for r in table if r["label"] == 0)
    print(f"PSSM built from {int(pssm[2])} positive sequences ({PSSM_WIDTH}-column, basic-cluster-anchored)")
    print(f"Wrote {args.out}: {n_pos} positive rows, {n_neg} negative rows, {len(fieldnames)} columns")

    # quick raw-data sanity summary -- no model, just the numbers
    import statistics
    for feat in ["pssm_score", "net_charge", "frac_basic", "mean_kd_hydropathy", "disorder_mean"]:
        pos_vals = [r[feat] for r in table if r["label"] == 1 and r.get(feat) is not None]
        neg_vals = [r[feat] for r in table if r["label"] == 0 and r.get(feat) is not None]
        if pos_vals and neg_vals:
            print(f"  {feat:22s} positives mean={statistics.mean(pos_vals):+.3f}  "
                  f"negatives mean={statistics.mean(neg_vals):+.3f}")


if __name__ == "__main__":
    main()
