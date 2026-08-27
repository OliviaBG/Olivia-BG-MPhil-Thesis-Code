#!/usr/bin/env python3
"""
sanity_check_reweight.py
============================================================
Compares crm1_compatibility_score under the OLD (first-pass, n=13
leucine_zipper) weights vs. the NEW (second-pass, n=41+ leucine_zipper)
weights applied to pocket_detector.py on , using the real,
already-computed per-factor subscores in crm1_eval_results.json -- no
fresh fpocket/AlphaFold calls needed, since subscores are weight-
independent raw measurements.

This is a pure arithmetic replay of both scoring formulas, so it verifies
the reweight actually moves scores the direction it should (real NES up
or flat, hard negatives -- especially leucine_zipper, the case that
motivated the reweight -- down or flat) BEFORE trusting it in the live app.

Usage: python3 sanity_check_reweight.py
"""
import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def old_score(s):
    score = 0.0
    vol = s.get('volume_A3')
    if vol is not None and 200 <= vol <= 2500:
        score += 0.451 if 800 <= vol <= 1500 else 0.338
    hydro = s.get('hydrophobicity')
    if hydro is not None and hydro >= 0.35:
        score += 0.061
    shape = s.get('shape_similarity')
    if shape is not None:
        score += 0.466 * shape
    drug = s.get('druggability')
    if drug is not None and drug >= 0.25:
        score += 0.028
    charge = s.get('charge_score')
    if charge is not None:
        score += 0.444 * charge
    return score


def new_score(s):
    score = 0.0
    vol = s.get('volume_A3')
    if vol is not None and 200 <= vol <= 2500:
        score += 0.468 if 800 <= vol <= 1500 else 0.351
    # hydrophobicity: zeroed in the new weights, contributes nothing
    shape = s.get('shape_similarity')
    if shape is not None:
        score += 0.545 * shape
    drug = s.get('druggability')
    if drug is not None and drug >= 0.25:
        score += 0.117
    charge = s.get('charge_score')
    if charge is not None:
        score += 0.320 * charge
    return score


def main():
    d = json.loads((THIS_DIR / 'crm1_eval_results.json').read_text())
    d = [r for r in d if r.get('subscores') is not None]

    groups = {}
    for r in d:
        key = 'positive' if r['label'] == 1 else r.get('feature_kind', 'unknown')
        old = old_score(r['subscores'])
        new = new_score(r['subscores'])
        groups.setdefault(key, []).append((old, new, new - old))

    print(f"{'group':16s} {'n':>4s} {'mean_old':>9s} {'mean_new':>9s} {'mean_shift':>11s}  interpretation")
    print("-" * 90)
    for key in ['positive', 'leucine_zipper', 'coiled_coil']:
        rows = groups.get(key, [])
        if not rows:
            continue
        n = len(rows)
        mean_old = sum(r[0] for r in rows) / n
        mean_new = sum(r[1] for r in rows) / n
        mean_shift = sum(r[2] for r in rows) / n
        print(f"{key:16s} {n:4d} {mean_old:9.3f} {mean_new:9.3f} {mean_shift:+11.3f}")

    print()
    pos_shift = sum(r[2] for r in groups.get('positive', [])) / max(1, len(groups.get('positive', [])))
    lz_shift = sum(r[2] for r in groups.get('leucine_zipper', [])) / max(1, len(groups.get('leucine_zipper', [])))
    cc_shift = sum(r[2] for r in groups.get('coiled_coil', [])) / max(1, len(groups.get('coiled_coil', [])))
    print("What we want to see: positives roughly flat/up, leucine_zipper (the case that motivated")
    print("this reweight) down relative to positives, coiled_coil roughly unchanged.")
    print(f"  positive shift:       {pos_shift:+.3f}")
    print(f"  leucine_zipper shift: {lz_shift:+.3f}   (gap vs positive: {lz_shift - pos_shift:+.3f})")
    print(f"  coiled_coil shift:    {cc_shift:+.3f}   (gap vs positive: {cc_shift - pos_shift:+.3f})")

    # Does the reweight actually improve separation, not just shift everything?
    import statistics
    try:
        from sklearn.metrics import roc_auc_score
        for label_name, neg_key in [('positive vs leucine_zipper', 'leucine_zipper'),
                                     ('positive vs coiled_coil', 'coiled_coil')]:
            pos_new = [r[1] for r in groups.get('positive', [])]
            neg_new = [r[1] for r in groups.get(neg_key, [])]
            pos_old = [r[0] for r in groups.get('positive', [])]
            neg_old = [r[0] for r in groups.get(neg_key, [])]
            labels = [1] * len(pos_new) + [0] * len(neg_new)
            auc_old = roc_auc_score(labels, pos_old + neg_old)
            auc_new = roc_auc_score(labels, pos_new + neg_new)
            print(f"\n  {label_name}: AUC old={auc_old:.3f}  AUC new={auc_new:.3f}  "
                  f"({'improved' if auc_new > auc_old else 'worse' if auc_new < auc_old else 'unchanged'})")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
