#!/usr/bin/env python3
"""
compute_crm1_joint_weights.py
============================================================
Joint standardized logistic regression across all 7 crm1_compatibility_score
factors, fit against real evaluated examples in crm1_eval_results.json.
Marginal (one-at-a-time) AUC ignores inter-feature correlation; this fits
all 7 together so each coefficient reflects that factor's contribution
once the others are accounted for -- same method used for the fpocket-vs-
burial blend question, reused here for the 7-factor scoring weights in
pocket_detector.py's _filter_for_crm1_compatibility(). (second pass): the first version of this reweighting was fit
on a leucine_zipper-underpowered sample (n=13 leucine_zipper hard
negatives). Growing that sample to n=41+ via expand_leucine_zipper_negatives.py
changed the picture materially (see CRM1_pocket_scoring_evaluation_2026-07-27.md
Part 4) -- charge_score in particular is no longer significant once a
properly-sized leucine_zipper sample is included. This script re-fits on
whatever is currently in crm1_eval_results.json, so it should be re-run
(and pocket_detector.py's weights re-derived from its output) any time the
eval sample composition changes meaningfully, rather than trusting a
one-off snapshot.

Only examples where ALL 7 factors are non-null are used (joint regression
needs a complete feature matrix -- this is a stricter filter than the
per-factor marginal AUCs in compute_crm1_factor_stats.py, which only
require that ONE factor be non-null at a time).

Usage: python3 compute_crm1_joint_weights.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
FACTORS = ['volume_A3', 'hydrophobicity', 'shape_similarity', 'druggability',
           'charge_score', 'composition_similarity', 'residue_residue_score']


def main():
    d = json.loads((THIS_DIR / 'crm1_eval_results.json').read_text())
    d = [r for r in d if r.get('subscores') is not None
         and all(r['subscores'].get(f) is not None for f in FACTORS)]
    labels = np.array([r['label'] for r in d])
    X = np.array([[r['subscores'][f] for f in FACTORS] for r in d])
    print(f"{len(d)} examples with ALL 7 factors present "
          f"({int((labels==1).sum())} positive, {int((labels==0).sum())} negative)")

    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xs, labels)
    coefs = clf.coef_[0]

    print("\nStandardized joint logistic regression coefficients:")
    for f, c in sorted(zip(FACTORS, coefs), key=lambda t: -abs(t[1])):
        print(f"  {f:24s} {c:+.3f}")

    # Convert to a point-budget allocation: only positive-coefficient
    # factors get weight (matches the precedent of zeroing
    # negative/backwards factors rather than penalizing them, given
    # n<300 makes "actively harmful" hard to distinguish from "no
    # effect"). Budget matches the current pocket_detector.py total
    # (1.45) so the existing >=0.3 pass threshold and confidence bands
    # stay meaningful without a separate threshold recalibration.
    BUDGET = 1.45
    positive = {f: c for f, c in zip(FACTORS, coefs) if c > 0}
    total_pos = sum(positive.values())
    weights = {f: (c / total_pos) * BUDGET if c > 0 else 0.0 for f, c in zip(FACTORS, coefs)}

    print(f"\nProposed weights (positive-coefficient factors only, summing to {BUDGET}):")
    for f in FACTORS:
        share = weights[f] / BUDGET * 100 if weights[f] else 0.0
        print(f"  {f:24s} weight={weights[f]:.3f}  ({share:.1f}% of budget)")

    out = {f: round(weights[f], 4) for f in FACTORS}
    out['_coefficients'] = {f: round(float(c), 4) for f, c in zip(FACTORS, coefs)}
    out['_n'] = len(d)
    out['_n_pos'] = int((labels == 1).sum())
    out['_n_neg'] = int((labels == 0).sum())
    (THIS_DIR / 'crm1_joint_weights.json').write_text(json.dumps(out, indent=2))
    print("\nWrote crm1_joint_weights.json")


if __name__ == '__main__':
    main()
