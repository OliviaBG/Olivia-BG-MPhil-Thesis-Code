#!/usr/bin/env python3
"""
aggregate_subset_replicates.py
============================================================
Combines 2-3 independent replicate runs of rerun_subset_annealing_check.py
(each a full re-run of the same 10-example subset, differing only in
OpenMM's random initial-velocity seed -- no seed is ever fixed anywhere in
md_refinement.py, confirmed by grep, so back-to-back runs of the same
script are genuine independent replicates, not deterministic repeats) into
per-example mean +/- std anchor_occupancy_score, then redoes the
old-vs-new delta/gap analysis on the AVERAGED values instead of a single
noisy 2 ns trajectory per example.

WHY: the first single-replicate subset run showed a slightly NEGATIVE
positive-vs-negative delta gap (-0.041) and a scrambled ranking (2 of the
top 3 scores under the new protocol were hard negatives, not real
positives) -- but with only 1 trajectory per condition, there's no way to
tell how much of that is a genuine annealing effect vs. plain MD
stochasticity (different random velocity seed each run). Averaging
replicates answers that directly.

USAGE:
    python3 aggregate_subset_replicates.py rerun_subset_annealing_check.json \
        rerun_subset_annealing_check_rep2.json rerun_subset_annealing_check_rep3.json
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 aggregate_subset_replicates.py <rep1.json> [rep2.json] [rep3.json] ...")
        return

    paths = [Path(p) for p in sys.argv[1:]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"Missing replicate file(s), skipping: {missing}")
        paths = [p for p in paths if p.exists()]
    if not paths:
        print("No replicate files found.")
        return

    print(f"Aggregating {len(paths)} replicate(s): {[p.name for p in paths]}\n")

    by_example = defaultdict(list)
    old_occ_by_example = {}
    label_by_example = {}
    for p in paths:
        data = json.loads(p.read_text())
        for r in data:
            key = (r['accession'], r['start'], r['end'])
            occ = r.get('anchor_occupancy_score')
            if occ is not None:
                by_example[key].append(occ)
            old_occ_by_example[key] = r['old_anchor_occupancy_score']
            label_by_example[key] = r['label']

    rows = []
    for key, vals in by_example.items():
        accession, start, end = key
        mean_new = float(np.mean(vals))
        std_new = float(np.std(vals)) if len(vals) > 1 else None
        old = old_occ_by_example[key]
        rows.append({
            'accession': accession, 'start': start, 'end': end,
            'label': label_by_example[key],
            'old_anchor_occupancy_score': old,
            'new_mean': mean_new, 'new_std': std_new,
            'n_replicates': len(vals),
            'delta': mean_new - old,
            'raw_values': vals,
        })

    rows.sort(key=lambda r: -r['old_anchor_occupancy_score'])

    print(f"{'accession':10s} {'label':6s} {'old':>7s} {'new_mean':>9s} {'new_std':>8s} {'n':>3s} {'delta':>8s}")
    pos_deltas, neg_deltas = [], []
    for r in rows:
        std_str = f"{r['new_std']:.3f}" if r['new_std'] is not None else "n/a"
        print(f"{r['accession']:10s} {r['label']:<6} {r['old_anchor_occupancy_score']:>7.3f} "
              f"{r['new_mean']:>9.3f} {std_str:>8s} {r['n_replicates']:>3d} {r['delta']:>+8.3f}")
        (pos_deltas if r['label'] == 1 else neg_deltas).append(r['delta'])

    print()
    if pos_deltas:
        print(f"Mean delta, positives: {np.mean(pos_deltas):+.3f}  (n={len(pos_deltas)})")
    if neg_deltas:
        print(f"Mean delta, negatives: {np.mean(neg_deltas):+.3f}  (n={len(neg_deltas)})")
    if pos_deltas and neg_deltas:
        gap = np.mean(pos_deltas) - np.mean(neg_deltas)
        print(f"\nGap (mean positive delta - mean negative delta): {gap:+.3f}")
        print("  Positive gap => annealing (on average, across replicates) helps real positives")
        print("  MORE than hard negatives -- ranking/percentile structure likely holds.")
        print("  Negative/near-zero gap => still no clean class-separating effect even after")
        print("  averaging out MD noise -- the old reference percentiles are not safely reusable")
        print("  for annealed results, and this reflects a real (not just noisy) property of the")
        print("  new protocol.")

    print(f"\n{'='*70}\nWithin-example replicate spread (new_std) -- how much of the single-run")
    print("swings we saw before were just MD stochasticity:")
    stds = [r['new_std'] for r in rows if r['new_std'] is not None]
    if stds:
        print(f"  mean std across examples: {np.mean(stds):.3f}")
        print(f"  (compare to the mean |delta| from a single replicate, ~0.15-0.2, from the first run --")
        print(f"   if replicate std is comparable to that, most of what we saw was noise, not signal)")

    out_path = Path('aggregate_subset_replicates_result.json')
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
