#!/usr/bin/env python3
"""
generate_esm_oof_predictions.py
============================================================
esm_comparison_figures.py's ROC/PR and agreement-scatter figures both read
esm_comparison/oof_predictions.npz -- a file esm_full_comparison.py never
writes (it only keeps aggregate F1/AUC mean+std per fold, not per-example
out-of-fold probabilities). The only thing that used to produce this file
was esm_model_comparison.py, an older, narrower script (predates today's
viral-NLS-data retrain, single-target-at-a-time, fewer classifiers) that
esm_full_comparison.py's own docstring says it generalizes/supersedes.
Using that stale file would silently mix pre-retrain predictions into an
otherwise fresh figure set.

This regenerates real out-of-fold probabilities from CURRENT data, using
the exact same data loading, feature-set construction, and classifier
definitions as esm_full_comparison.py (imported directly, not copy-pasted,
so the two scripts can't drift apart) -- just swapping cross_val_score's
aggregate scoring for cross_val_predict's actual per-example probabilities.
Deliberately scoped to the 3 feature sets esm_comparison_figures.py's
ROC/PR/agreement figures actually plot (hand_engineered, esm_only,
combined -- see its FEATURE_SET_ORDER) x every classifier, both targets --
not the full PCA sweep, which those two figures don't use anyway.

REQUIREMENTS: same as esm_full_comparison.py -- esm_embeddings/*.npz must
already exist (run esm_embed_sequences.py first).

Usage (from the AlphaFold directory, AFTER esm_embed_sequences.py):
    python3 generate_esm_oof_predictions.py

Output:
    esm_comparison/oof_predictions.npz -- keys '{target}_y' (true labels)
    and '{target}__{feature_set}__{classifier}' (out-of-fold probabilities,
    same order as y), for target in {nes, nls}.

Run esm_comparison_figures.py again afterward to pick this up.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from esm_full_comparison import _classifiers, _load_target, _load_embeddings  # noqa: E402

FEATURE_SETS_NEEDED = ["hand_engineered", "esm_only", "combined"]


def main():
    out_dir = THIS_DIR / "esm_comparison"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "oof_predictions.npz"

    npz_data = {}
    for target in ("nes", "nls"):
        print(f"\n{'=' * 60}\n{target.upper()}\n{'=' * 60}")
        X_hand, y, seqs = _load_target(target)
        seq_to_vec = _load_embeddings(target)
        missing = sorted(set(s for s in seqs if s not in seq_to_vec))
        if missing:
            raise RuntimeError(
                f"{len(missing)} {target} sequences have no cached embedding -- "
                f"run esm_embed_sequences.py first.")
        X_esm = np.stack([seq_to_vec[s] for s in seqs])
        X_combined = np.concatenate([X_hand, X_esm], axis=1)
        feature_sets = {"hand_engineered": X_hand, "esm_only": X_esm, "combined": X_combined}

        npz_data[f"{target}_y"] = y
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        classifiers = _classifiers()

        for fs_name in FEATURE_SETS_NEEDED:
            X = feature_sets[fs_name]
            for clf_name, clf_factory in classifiers.items():
                key = f"{target}__{fs_name}__{clf_name}"
                t0 = time.time()
                try:
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf_factory())])
                    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
                except Exception as e:
                    print(f"  SKIP {fs_name}/{clf_name}: {e}")
                    continue
                npz_data[key] = proba
                print(f"  {fs_name:16s} {clf_name:22s} done ({time.time() - t0:.1f}s)")
                # checkpoint after every combo -- safe to interrupt/resume
                np.savez(out_path, **npz_data)

    np.savez(out_path, **npz_data)
    print(f"\n{out_path}")


if __name__ == "__main__":
    main()
