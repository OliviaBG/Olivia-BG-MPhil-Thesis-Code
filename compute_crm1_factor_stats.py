#!/usr/bin/env python3
"""
compute_crm1_factor_stats.py
============================================================
Computes AUC / 95% CI / p-value (vs. random) for each of the 7 raw factors
inside pocket_detector.py's crm1_compatibility_score, from real evaluated
examples in crm1_eval_results.json (produced by
evaluate_crm1_pocket_signal.py). Writes crm1_factor_stats.json, consumed by
plot_crm1_factor_significance.py.

Only examples with a non-null 'subscores' dict are used (i.e. where a real
fpocket-detected pocket actually overlapped the candidate's residue span --
see CRM1_pocket_scoring_evaluation_2026-07-27.md for why a null subscores
entry is a legitimate result, not missing data).

Usage: python3 compute_crm1_factor_stats.py
       python3 compute_crm1_factor_stats.py --results crm1_eval_results.json --out crm1_factor_stats.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

FACTORS = ['volume_A3', 'hydrophobicity', 'shape_similarity', 'druggability',
           'charge_score', 'composition_similarity', 'residue_residue_score']
# Dropped the hardcoded WEIGHTS table that used to be attached
# to each factor's stats -- it was already stale (showing pre-reweight
# values) and would go stale again every time compute_crm1_joint_weights.py
# is re-run. This script is about AUC/significance, not the model's
# current weight; see plot_crm1_factor_significance.py for the same reasoning
# applied to the figure.


def auc_ci(auc, n1, n2):
    """Hanley & McNeil (1982) approximate SE, computed on whichever side of
    0.5 is >=0.5 (the formula is only defined there; SE is symmetric)."""
    a = auc if auc >= 0.5 else 1 - auc
    Q1 = a / (2 - a)
    Q2 = 2 * a ** 2 / (1 + a)
    se = math.sqrt((a * (1 - a) + (n1 - 1) * (Q1 - a ** 2) + (n2 - 1) * (Q2 - (1 - a) ** 2)) / (n1 * n2))
    z = (auc - 0.5) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return se, auc - 1.96 * se, auc + 1.96 * se, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='crm1_eval_results.json')
    ap.add_argument('--out', default='crm1_factor_stats.json')
    args = ap.parse_args()

    d = json.loads(Path(args.results).read_text())
    d = [r for r in d if r.get('subscores') is not None]
    print(f"{len(d)} examples with subscores "
          f"({sum(1 for r in d if r['label']==1)} positive, "
          f"{sum(1 for r in d if r['label']==0)} negative)")

    results = {}
    for factor in FACTORS:
        vals, labs = [], []
        for r in d:
            v = r['subscores'].get(factor)
            if v is not None:
                vals.append(v)
                labs.append(r['label'])
        vals, labs = np.array(vals), np.array(labs)
        n1, n2 = int((labs == 1).sum()), int((labs == 0).sum())
        if n1 < 5 or n2 < 5:
            print(f"  {factor}: only {n1} pos / {n2} neg -- skipping (too few)")
            continue
        auc = roc_auc_score(labs, vals)
        se, lo, hi, p = auc_ci(auc, n1, n2)
        results[factor] = {
            'n': len(vals), 'n_pos': n1, 'n_neg': n2,
            'auc': round(float(auc), 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
            'p': round(p, 4),
            'mean_pos': round(float(vals[labs == 1].mean()), 4),
            'mean_neg': round(float(vals[labs == 0].mean()), 4),
        }
        print(f"  {factor}: AUC={auc:.3f}  p={p:.3f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
