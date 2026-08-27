"""
Starting feature set + consensus-motif scanner for building your own leucine-
rich NES (nuclear export signal) predictor from nes_dataset.csv/json
(produced by build_dataset.py).

What's well-established (used with confidence here):
    The classic hydrophobic-spacing consensus for CRM1-dependent NESs,
    Phi-x(2,3)-Phi-x(2,3)-Phi-x-Phi (Phi = L/I/V/F/M), first characterized by
    Bogerd et al. 1996 / la Cour et al. 2004 and reproduced in essentially
    every later study. `CLASS1_CORE_RE` implements exactly this and is a
    solid default.

What's approximate (double-check before relying on it for anything precise):
    Kosugi et al. (2008, J. Biol. Chem. 283:9418; PNAS-associated
    yeast-selection studies) proposed refined subclasses of the core motif
    with different spacer lengths (their "class 1/2/3/4"). The variants below
    (`CLASS2_RE`, `CLASS3_RE`, `CLASS4_RE`) are reconstructed from memory of
    that literature and are included only to widen recall during candidate
    scanning -- verify the exact spacer definitions against the primary
    paper (or NESmapper / LocNES, which re-implement Kosugi's classes) before
    treating class-level distinctions as authoritative.

Usage as a library:
    from nes_features import scan_sequence, featurize, build_training_table

    scan_sequence("MAGRSG...LQLPPLERLTLC...")   # -> list of candidate hits
    featurize("LQLPPLERLTLD")                    # -> dict of numeric features
    build_training_table("nes_dataset.csv")      # -> (X, y, meta) for sklearn

Usage as a script (writes a ready-to-train CSV of positive + negative windows):
    python nes_features.py --dataset nes_dataset.csv --out training_table.csv
"""

import argparse
import csv
import random
import re
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Residue classes / scales
# ---------------------------------------------------------------------------

HYDROPHOBIC = set("LIVFM")  # Phi positions in the classic NES consensus
CHARGED_POS = set("KR")
CHARGED_NEG = set("DE")

# Kyte-Doolittle hydropathy scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# ---------------------------------------------------------------------------
# Consensus motifs. Phi = [LIVFM]. "x" = any residue.
# ---------------------------------------------------------------------------

_PHI = "[LIVFM]"

# The well-established core: Phi-x(2,3)-Phi-x(2,3)-Phi-x-Phi
CLASS1_CORE_RE = re.compile(
    _PHI + r".{2,3}" + _PHI + r".{2,3}" + _PHI + r"." + _PHI
)

# Approximate variants widening the spacer search (verify before trusting
# class-level labels -- see module docstring).
CLASS2_RE = re.compile(_PHI + r".{2,3}" + _PHI + r".{2,3}" + _PHI + r".{2}" + _PHI)
CLASS3_RE = re.compile(_PHI + r".{2,3}" + _PHI + r".{3}" + _PHI + r"." + _PHI)
CLASS4_RE = re.compile(_PHI + r".{3}" + _PHI + r".{2,3}" + _PHI + r"." + _PHI)

CONSENSUS_PATTERNS = {
    "class1_core": CLASS1_CORE_RE,
    "class2_approx": CLASS2_RE,
    "class3_approx": CLASS3_RE,
    "class4_approx": CLASS4_RE,
}


def scan_sequence(sequence: str) -> List[Dict]:
    """Return every (possibly overlapping) regex hit for each consensus
    pattern, as candidate NES windows: [{'start','end','sequence','class'}]
    (0-based half-open start/end within `sequence`)."""
    hits = []
    for cls_name, pattern in CONSENSUS_PATTERNS.items():
        for m in pattern.finditer(sequence):
            hits.append({
                "start": m.start(),
                "end": m.end(),
                "sequence": m.group(0),
                "class": cls_name,
            })
    return hits


# ---------------------------------------------------------------------------
# Featurization of a single candidate window
# ---------------------------------------------------------------------------

def featurize(window_seq: str) -> Dict[str, float]:
    """Turn a short peptide window into a flat numeric feature dict suitable
    for scikit-learn / xgboost / a simple logistic regression baseline."""
    n = len(window_seq)
    if n == 0:
        return {}

    feats = {}
    feats["length"] = n
    feats["frac_hydrophobic"] = sum(1 for a in window_seq if a in HYDROPHOBIC) / n
    feats["frac_charged_pos"] = sum(1 for a in window_seq if a in CHARGED_POS) / n
    feats["frac_charged_neg"] = sum(1 for a in window_seq if a in CHARGED_NEG) / n
    feats["net_charge"] = (sum(1 for a in window_seq if a in CHARGED_POS)
                            - sum(1 for a in window_seq if a in CHARGED_NEG))
    feats["mean_kd_hydropathy"] = sum(KD_SCALE.get(a, 0.0) for a in window_seq) / n

    # amino-acid composition (20-dim)
    for aa in AA_ALPHABET:
        feats[f"comp_{aa}"] = window_seq.count(aa) / n

    # does the window match each consensus class at all (0/1)?
    for cls_name, pattern in CONSENSUS_PATTERNS.items():
        feats[f"matches_{cls_name}"] = 1.0 if pattern.search(window_seq) else 0.0

    # spacing between consecutive hydrophobic (Phi) residues -- the single
    # most informative feature family for this problem
    phi_positions = [i for i, a in enumerate(window_seq) if a in HYDROPHOBIC]
    gaps = [b - a - 1 for a, b in zip(phi_positions, phi_positions[1:])]
    feats["n_phi_residues"] = len(phi_positions)
    feats["mean_phi_gap"] = sum(gaps) / len(gaps) if gaps else -1
    feats["max_phi_gap"] = max(gaps) if gaps else -1
    feats["min_phi_gap"] = min(gaps) if gaps else -1

    return feats


# ---------------------------------------------------------------------------
# Building a labeled training table from nes_dataset.csv (build_dataset.py)
# ---------------------------------------------------------------------------

def _read_dataset_csv(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_training_table(dataset_csv: str, window_size: Optional[int] = None,
                          neg_per_pos: int = 3, seed: int = 0):
    """
    Positives: every row with a known nes_sequence (and, when full_sequence
    is available, its exact start/end).
    Negatives: `neg_per_pos` random windows sampled from the *same* protein's
    full_sequence, outside the annotated NES region, matched in length to the
    positive window (falls back to `window_size` or 15 aa when a protein's
    full_sequence isn't available).

    Returns a list of dict rows: {sequence, label, source, protein_name, ...features}
    ready to be written to CSV / loaded into pandas.
    """
    rng = random.Random(seed)
    rows = _read_dataset_csv(dataset_csv)
    out = []

    for r in rows:
        nes_seq = (r.get("nes_sequence") or "").strip()
        if not nes_seq:
            continue
        full_seq = (r.get("full_sequence") or "").strip()
        start = r.get("nes_start")
        end = r.get("nes_end")

        # positive example
        pos_feats = featurize(nes_seq)
        out.append({
            "sequence": nes_seq, "label": 1, "source": r["source"],
            "protein_name": r["protein_name"], **pos_feats,
        })

        # negative examples, sampled from elsewhere in the same protein
        win_len = len(nes_seq) if not window_size else window_size
        if full_seq and len(full_seq) > win_len + 1:
            try:
                nes_start_i = int(start) - 1 if start not in (None, "",) else None
                nes_end_i = int(end) if end not in (None, "") else None
            except ValueError:
                nes_start_i = nes_end_i = None

            tries = 0
            made = 0
            while made < neg_per_pos and tries < neg_per_pos * 20:
                tries += 1
                i = rng.randint(0, len(full_seq) - win_len)
                j = i + win_len
                if nes_start_i is not None and nes_end_i is not None:
                    if not (j <= nes_start_i or i >= nes_end_i):
                        continue  # overlaps the real NES -- skip
                cand = full_seq[i:j]
                neg_feats = featurize(cand)
                out.append({
                    "sequence": cand, "label": 0, "source": r["source"],
                    "protein_name": r["protein_name"], **neg_feats,
                })
                made += 1

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nes_dataset.csv")
    ap.add_argument("--out", default="training_table.csv")
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = build_training_table(args.dataset, neg_per_pos=args.neg_per_pos, seed=args.seed)
    if not table:
        print("No rows produced -- check that --dataset has a nes_sequence column populated.")
        return

    fieldnames = list(table[0].keys())
    # some rows may be missing keys the first row happened to have (rare)
    for row in table:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in table:
            w.writerow(row)

    n_pos = sum(1 for r in table if r["label"] == 1)
    n_neg = sum(1 for r in table if r["label"] == 0)
    print(f"Wrote {args.out}: {n_pos} positive windows, {n_neg} negative windows.")


if __name__ == "__main__":
    main()
