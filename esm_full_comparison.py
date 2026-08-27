#!/usr/bin/env python3
"""
esm_full_comparison.py
============================================================
5-fold CV comparison of hand-engineered features vs frozen ESM2 embeddings
(raw and PCA-reduced) vs combined, for NES and/or NLS in one run. Companion
to esm_finetune_kfold.py (same 5-fold protocol, for fine-tuned ESM2), so all
the numbers from both scripts are directly comparable for a single thesis
figure/table.

Generalizes an earlier NLS-only prototype to both targets and every
classifier esm_model_comparison.py originally considered.

Run esm_embed_sequences.py FIRST to (re)generate esm_embeddings/*.npz --
this script needs every current training sequence to already have a cached
embedding and will raise a clear error listing what's missing if not.

xgboost is entirely optional -- skipped automatically if not installed, same
degrade-gracefully pattern nls_ml_predictor.py itself already uses. No need
to pip install it just for this.

Checkpointed per (target, feature_set, classifier) combo so it's safe to
interrupt and rerun -- picks up where it left off.

Requires: pip install joblib scikit-learn numpy

Usage:
    python3 esm_full_comparison.py                        # both nes and nls
    python3 esm_full_comparison.py --target nls
    python3 esm_full_comparison.py --target nes --pca-components 10 30 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("(xgboost not installed -- skipping that candidate classifier, everything else still runs fine)")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def _classifiers():
    c = {
        "svm_linear": lambda: SVC(kernel="linear", C=0.1, probability=True, random_state=42, class_weight="balanced"),
        "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42, class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=42),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42, early_stopping=True),
    }
    if XGBOOST_AVAILABLE:
        c["xgboost"] = lambda: XGBClassifier(random_state=42, eval_metric="logloss")
    return c


def _load_target(target):
    if target == "nls":
        from nls_ml_predictor import NLSPredictor
        p = NLSPredictor()
    else:
        from nes_ml_predictor_improved import ImprovedNESPredictor
        p = ImprovedNESPredictor()
    dataset = p.build_training_dataset()
    X_hand = np.asarray(dataset["X"], dtype=float)
    y = np.asarray(dataset["y"], dtype=int)
    seqs = [d["seq"].upper() for d in dataset["positives"]] + [d["seq"].upper() for d in dataset["negatives"]]
    return X_hand, y, seqs


def _load_embeddings(target):
    path = THIS_DIR / "esm_embeddings" / f"{target}_embeddings.npz"
    npz = np.load(path, allow_pickle=True)
    return {s: v for s, v in zip(npz["sequences"], npz["vectors"])}


def _write_oof_npz(npz_path, oof_all):
    # np.savez needs a plain dict of arrays; write to a tmp file first so an
    # interrupted write can never corrupt the existing checkpoint. The tmp
    # name must itself end in ".npz" -- np.savez silently APPENDS ".npz" to
    # any filename that doesn't already end with it, so a naive ".npz.tmp"
    # suffix actually gets written to "*.npz.tmp.npz", and the rename below
    # then fails with FileNotFoundError (this broke the first version of
    # this fix, caught mid-run on the pod -- only the very first
    # feature_set/classifier combo's OOF predictions got saved before it
    # crashed; re-running is safe since the resume logic below only skips a
    # combo once its OOF array is confirmed present in the loaded npz).
    tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez(tmp_path, **oof_all)
    tmp_path.replace(npz_path)


def _write_comparison_csv(csv_path, rows_by_key):
    import csv as csv_mod
    fieldnames = ["target", "feature_set", "classifier", "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_by_key.values():
            w.writerow(row)


def run_target(target, pca_components, out_dir):
    print(f"\n{'=' * 60}\n{target.upper()}\n{'=' * 60}")
    X_hand, y, seqs = _load_target(target)
    seq_to_vec = _load_embeddings(target)

    missing = sorted(set(s for s in seqs if s not in seq_to_vec))
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(set(seqs))} {target} training sequences have no cached "
            f"embedding -- run esm_embed_sequences.py first. First missing: {missing[0][:40]!r}...")

    X_esm = np.stack([seq_to_vec[s] for s in seqs])
    print(f"n={len(y)} ({int(y.sum())} pos / {int((1 - y).sum())} neg)")

    feature_sets = {
        "hand_engineered": X_hand, "esm_only": X_esm,
        "combined": np.concatenate([X_hand, X_esm], axis=1),
    }
    for k in pca_components:
        feature_sets[f"esm_pca{k}"] = ("pca", X_esm, k)
        feature_sets[f"combined_pca{k}"] = ("pca_combined", X_esm, k)

    classifiers = _classifiers()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ckpt_path = out_dir / f"esm_full_comparison_{target}.json"
    results = json.load(open(ckpt_path)) if ckpt_path.exists() else {}

    # --- esm_comparison/ outputs -----------------------------------------
    # esm_umap_agreement.py and generate_thesis_figures.py (fig_esm_comparison,
    # fig_esm_roc_pr, fig_esm_agreement) don't read the per-target JSON above --
    # they read esm_comparison/esm_comparison_results.csv (aggregate F1/AUC,
    # columns target/feature_set/classifier/f1_mean/f1_std/roc_auc_mean/
    # roc_auc_std) and esm_comparison/oof_predictions.npz (out-of-fold predicted
    # probabilities, keys "{target}_y" and "{target}__{feature_set}__{classifier}").
    # This script used to only write the aggregate JSON above and silently
    # produced neither file, so those three downstream figures always failed.
    # Fixed by accumulating OOF proba during the same CV loop below
    # and persisting both files here, checkpointed every combo same as the JSON.
    comp_dir = out_dir / "esm_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    npz_path = comp_dir / "oof_predictions.npz"
    csv_path = comp_dir / "esm_comparison_results.csv"

    oof_all = dict(np.load(npz_path, allow_pickle=True)) if npz_path.exists() else {}
    oof_all[f"{target}_y"] = y

    csv_rows = {}
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                csv_rows[(row["target"], row["feature_set"], row["classifier"])] = row

    for fs_name, spec in feature_sets.items():
        for clf_name, clf_factory in classifiers.items():
            key = f"{fs_name}__{clf_name}"
            oof_key = f"{target}__{fs_name}__{clf_name}"
            # Only skip recompute if BOTH the aggregate metrics (JSON) and the
            # OOF predictions (npz) are already checkpointed -- otherwise an
            # older run (from before this fix) would leave oof_predictions.npz
            # permanently missing this combo.
            if key in results and oof_key in oof_all:
                continue
            t0 = time.time()
            fold_f1, fold_auc, failed = [], [], False
            oof_proba = np.full(len(y), np.nan)
            for tr_idx, te_idx in cv.split(X_hand, y):
                if isinstance(spec, tuple):
                    mode, X_src, k = spec
                    scaler_pre = StandardScaler().fit(X_src[tr_idx])
                    Xtr_s, Xte_s = scaler_pre.transform(X_src[tr_idx]), scaler_pre.transform(X_src[te_idx])
                    k_eff = min(k, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
                    pca = PCA(n_components=k_eff, random_state=42).fit(Xtr_s)
                    Xtr_p, Xte_p = pca.transform(Xtr_s), pca.transform(Xte_s)
                    if mode == "pca_combined":
                        Xtr = np.concatenate([X_hand[tr_idx], Xtr_p], axis=1)
                        Xte = np.concatenate([X_hand[te_idx], Xte_p], axis=1)
                    else:
                        Xtr, Xte = Xtr_p, Xte_p
                else:
                    Xtr, Xte = spec[tr_idx], spec[te_idx]
                try:
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf_factory())])
                    pipe.fit(Xtr, y[tr_idx])
                    proba = pipe.predict_proba(Xte)[:, 1]
                except Exception as e:
                    print(f"  SKIP {fs_name}/{clf_name}: {e}")
                    failed = True
                    break
                pred = (proba >= 0.5).astype(int)
                fold_f1.append(f1_score(y[te_idx], pred, zero_division=0))
                fold_auc.append(roc_auc_score(y[te_idx], proba))
                oof_proba[te_idx] = proba
            if failed:
                continue
            results[key] = {
                "target": target, "fs": fs_name, "clf": clf_name,
                "f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
                "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
            }
            json.dump(results, open(ckpt_path, "w"), indent=2)
            r = results[key]
            print(f"  {fs_name:20s} {clf_name:22s} F1={r['f1_mean']:.3f}+/-{r['f1_std']:.3f}  "
                  f"AUC={r['auc_mean']:.3f}+/-{r['auc_std']:.3f}  ({time.time() - t0:.1f}s)")

            oof_all[oof_key] = oof_proba
            _write_oof_npz(npz_path, oof_all)
            csv_rows[(target, fs_name, clf_name)] = {
                "target": target, "feature_set": fs_name, "classifier": clf_name,
                "f1_mean": r["f1_mean"], "f1_std": r["f1_std"],
                "roc_auc_mean": r["auc_mean"], "roc_auc_std": r["auc_std"],
            }
            _write_comparison_csv(csv_path, csv_rows)

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["nes", "nls"], default=None, help="omit to run both")
    ap.add_argument("--pca-components", type=int, nargs="+", default=[10, 30, 50, 100, 200])
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out_dir = Path(args.out)
    targets = [args.target] if args.target else ["nls", "nes"]
    for target in targets:
        run_target(target, args.pca_components, out_dir)
    print("\nDone. Results saved to esm_full_comparison_<target>.json "
          "and esm_comparison/esm_comparison_results.csv + esm_comparison/oof_predictions.npz")


if __name__ == "__main__":
    main()
