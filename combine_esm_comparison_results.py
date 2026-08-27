#!/usr/bin/env python3
"""
combine_esm_comparison_results.py
============================================================
esm_full_comparison.py and esm_finetune_kfold.py each checkpoint to their
own per-target JSON (esm_full_comparison_{nes,nls}.json,
esm_finetune_kfold_{nes,nls}.json) -- neither writes a combined CSV on its
own. This stitches all four into ONE tidy long-format CSV: every
(target, feature_set, classifier) combination side by side, hand-engineered
vs frozen ESM vs combined vs combined+PCA{10,30,50,100,200} vs fine-tuned
ESM, for both NES and NLS -- exactly the shape needed for one thesis table/
figure instead of reading four separate JSON files.

Run this AFTER esm_full_comparison.py and esm_finetune_kfold.py have both
finished (or partially finished -- picks up whatever's in each checkpoint
file, so it's safe to run early and re-run later as more results land).

Usage (from the AlphaFold directory):
    python3 combine_esm_comparison_results.py
"""
import json
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
TARGETS = ["nes", "nls"]


def load_full_comparison(target):
    """esm_full_comparison_{target}.json -- flat dict keyed 'fs__clf', each
    value {target, fs, clf, f1_mean, f1_std, auc_mean, auc_std}."""
    path = THIS_DIR / f"esm_full_comparison_{target}.json"
    if not path.exists():
        print(f"  {path.name} not found -- skipping (run esm_full_comparison.py first)")
        return []
    data = json.loads(path.read_text())
    rows = []
    for key, r in data.items():
        rows.append({
            "target": r.get("target", target), "feature_set": r.get("fs"),
            "classifier": r.get("clf"), "f1_mean": r.get("f1_mean"), "f1_std": r.get("f1_std"),
            "auc_mean": r.get("auc_mean"), "auc_std": r.get("auc_std"),
            "esm_mode": "frozen" if r.get("fs") not in (None,) and "esm" in r.get("fs", "") else "n/a",
        })
    return rows


def load_finetune(target):
    """esm_finetune_kfold_{target}.json -- single result (fine-tuning isn't
    swappable across classifiers/feature-sets the way frozen embeddings
    are), so this becomes ONE row with feature_set='esm_finetuned',
    classifier='finetuned_esm2_head' -- same 6 columns as the grid above so
    both concat cleanly."""
    path = THIS_DIR / f"esm_finetune_kfold_{target}.json"
    if not path.exists():
        print(f"  {path.name} not found -- skipping (run esm_finetune_kfold.py first)")
        return []
    data = json.loads(path.read_text())
    if "f1_mean" not in data:
        print(f"  {path.name} exists but has no folds finished yet -- skipping for now")
        return []
    return [{
        "target": data.get("target", target), "feature_set": "esm_finetuned",
        "classifier": "finetuned_esm2_head", "f1_mean": data.get("f1_mean"),
        "f1_std": data.get("f1_std"), "auc_mean": data.get("auc_mean"),
        "auc_std": data.get("auc_std"), "esm_mode": "fine-tuned",
    }]


def main():
    rows = []
    for target in TARGETS:
        rows += load_full_comparison(target)
        rows += load_finetune(target)

    if not rows:
        print("\nNothing found -- run esm_full_comparison.py and/or esm_finetune_kfold.py first.")
        return

    df = pd.DataFrame(rows)
    # tag frozen-ESM-derived feature sets clearly (esm_only, combined,
    # esm_pca*, combined_pca*) vs hand_engineered (esm_mode='n/a') vs
    # fine-tuned (esm_mode='fine-tuned', set explicitly above)
    df.loc[df["feature_set"] == "hand_engineered", "esm_mode"] = "n/a"
    df = df.sort_values(["target", "feature_set", "classifier"]).reset_index(drop=True)

    out_path = THIS_DIR / "esm_comparison_master.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{out_path}  ({len(df)} rows)")
    print(df.to_string(index=False))

    missing = []
    for target in TARGETS:
        if not (THIS_DIR / f"esm_full_comparison_{target}.json").exists():
            missing.append(f"esm_full_comparison_{target}.json")
        ft_path = THIS_DIR / f"esm_finetune_kfold_{target}.json"
        if not ft_path.exists() or "f1_mean" not in json.loads(ft_path.read_text() or "{}"):
            missing.append(f"esm_finetune_kfold_{target}.json (complete)")
    if missing:
        print(f"\nNote: still missing/incomplete -- {missing}. Re-run this script once those finish "
              f"to get the full grid.")


if __name__ == "__main__":
    main()
