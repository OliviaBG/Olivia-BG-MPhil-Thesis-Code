#!/usr/bin/env python3
"""
esm_model_comparison.py
============================================================
3-way comparison for both NES and NLS: hand-engineered features only
vs frozen-ESM2-150M embeddings only vs the two combined, across every
classifier your pipeline already considers, using the SAME cross-
validation protocol for all three feature sets so comparisons are fair.

(An earlier version restricted this to just the classifier each
predictor's nested-CV _train_model() selected for shipping; it was
reverted to all candidates, because restricting to one classifier
answers "does this
feature set help the classifier hand-engineered features happened to
pick", not the broader "does this feature set help, period" question this
script is meant for.)

Run this AFTER esm_embed_sequences.py has produced esm_embeddings/
(nes_embeddings.npz, nls_embeddings.npz).

Usage:
    python3 esm_model_comparison.py
    python3 esm_model_comparison.py --embeddings esm_embeddings --out esm_comparison --folds 5

Outputs (written to --out, default 'esm_comparison/'):
    esm_comparison_results.csv      one row per (target, feature_set, classifier)
                                     with CV-mean F1 / ROC-AUC / PR-AUC (+/- std)
    oof_predictions.npz             out-of-fold predicted probabilities for every
                                     (target, feature_set, classifier) cell -- needed
                                     by esm_umap_agreement.py for the Spearman step
    figures/roc_<target>.png        ROC curves, one panel per classifier, 3 lines
                                     each (hand-eng / ESM / combined)
    figures/pr_<target>.png         same, precision-recall
    figures/confusion_<target>_<classifier>_<feature_set>.png
    figures/comparison_bars_<target>.png   grouped bar chart, F1 and ROC-AUC,
                                     classifier x feature_set, with error bars
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve, confusion_matrix)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from thesis_plot_style import apply_style, style_axes, CAM, CATEGORICAL, CAM_SEQUENTIAL_CMAP, save_fig

# XGBoost is an optional extra, same degrade-gracefully pattern as
# nes_ml_predictor_improved.py / nls_ml_predictor.py.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Exact classifier candidates + hyperparameters, copied from each predictor's
# own _train_model() so results are directly comparable to your existing
# hand-engineered-only models -- not a re-tuned/different set.
NES_CLASSIFIERS = {
    'svm_linear': lambda: SVC(kernel='linear', C=0.01, probability=True, random_state=42, class_weight='balanced'),
    'svm_rbf': lambda: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced'),
    'random_forest': lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'extra_trees': lambda: ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'gradient_boosting': lambda: GradientBoostingClassifier(random_state=42),
    'hist_gradient_boosting': lambda: HistGradientBoostingClassifier(random_state=42),
    'mlp': lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42, early_stopping=True),
}
if XGBOOST_AVAILABLE:
    NES_CLASSIFIERS['xgboost'] = lambda: XGBClassifier(random_state=42, eval_metric='logloss')
else:
    print("  (xgboost not installed -- skipping that candidate; pip install xgboost to include it)")

NLS_CLASSIFIERS = {
    'svm_linear': lambda: SVC(kernel='linear', C=0.1, probability=True, random_state=42, class_weight='balanced'),
    'svm_rbf': lambda: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced'),
    'random_forest': lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'extra_trees': lambda: ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'gradient_boosting': lambda: GradientBoostingClassifier(random_state=42),
    'hist_gradient_boosting': lambda: HistGradientBoostingClassifier(random_state=42),
    'mlp': lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42, early_stopping=True),
}
if XGBOOST_AVAILABLE:
    NLS_CLASSIFIERS['xgboost'] = lambda: XGBClassifier(random_state=42, eval_metric='logloss')


def _load_hand_engineered(target, embeddings_dir):
    """Rebuilds X/y/sequences from the predictor's own build_training_dataset(),
    same call esm_embed_sequences.py used -- guarantees row-for-row alignment
    is possible via sequence lookup (order itself isn't relied on)."""
    if target == 'nes':
        from nes_ml_predictor_improved import ImprovedNESPredictor
        predictor = ImprovedNESPredictor()
    else:
        from nls_ml_predictor import NLSPredictor
        predictor = NLSPredictor()

    dataset = predictor.build_training_dataset()
    X_hand = np.asarray(dataset['X'], dtype=float)
    y = np.asarray(dataset['y'], dtype=int)

    seqs = [p['seq'].upper() for p in dataset['positives']] + [n['seq'].upper() for n in dataset['negatives']]
    if len(seqs) != len(y):
        raise RuntimeError(f"{target}: sequence count ({len(seqs)}) != label count ({len(y)}) -- "
                            "build_training_dataset() row order assumption broke, check predictor source.")
    return X_hand, y, seqs


def _load_esm(target, embeddings_dir):
    npz = np.load(Path(embeddings_dir) / f'{target}_embeddings.npz', allow_pickle=True)
    seq_to_vec = {s: v for s, v in zip(npz['sequences'], npz['vectors'])}
    return seq_to_vec


def _build_feature_sets(X_hand, seqs, seq_to_vec):
    missing = [s for s in seqs if s not in seq_to_vec]
    if missing:
        raise RuntimeError(f"{len(missing)} sequences have no ESM embedding cached -- "
                            f"re-run esm_embed_sequences.py (first missing: {missing[0][:30]}...)")
    X_esm = np.stack([seq_to_vec[s] for s in seqs])
    X_combined = np.concatenate([X_hand, X_esm], axis=1)
    return {'hand_engineered': X_hand, 'esm_only': X_esm, 'combined': X_combined}


def _make_classifier_pipeline(clf_factory):
    """StandardScaler in front of every model -- required for the SVMs, and
    harmless (helps, if anything, on the wide/mixed-scale combined feature
    set) for the tree-based models."""
    return Pipeline([('scaler', StandardScaler()), ('clf', clf_factory())])


def run_target(target, classifiers, embeddings_dir, out_dir, n_folds, seed):
    print(f"\n{'=' * 60}\n{target.upper()}\n{'=' * 60}")
    X_hand, y, seqs = _load_hand_engineered(target, embeddings_dir)
    seq_to_vec = _load_esm(target, embeddings_dir)
    feature_sets = _build_feature_sets(X_hand, seqs, seq_to_vec)
    print(f"  n={len(y)} ({int(y.sum())} positive / {int((1 - y).sum())} negative)")
    for name, X in feature_sets.items():
        print(f"  {name}: {X.shape[1]} features")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    rows = []
    oof_store = {}  # (feature_set, classifier) -> oof probability array, aligned to y/seqs

    for fs_name, X in feature_sets.items():
        for clf_name, clf_factory in classifiers.items():
            pipe = _make_classifier_pipeline(clf_factory)
            try:
                oof_proba = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
            except Exception as e:
                print(f"  SKIP {fs_name}/{clf_name}: {e}")
                continue
            oof_pred = (oof_proba >= 0.5).astype(int)

            # Also collect per-fold metrics for a proper mean +/- std (rather
            # than a single pooled-OOF number, which understates variance).
            fold_f1, fold_auc, fold_ap = [], [], []
            for tr_idx, te_idx in cv.split(X, y):
                fold_pipe = _make_classifier_pipeline(clf_factory)
                fold_pipe.fit(X[tr_idx], y[tr_idx])
                proba = fold_pipe.predict_proba(X[te_idx])[:, 1]
                pred = (proba >= 0.5).astype(int)
                fold_f1.append(f1_score(y[te_idx], pred, zero_division=0))
                if len(np.unique(y[te_idx])) > 1:
                    fold_auc.append(roc_auc_score(y[te_idx], proba))
                    fold_ap.append(average_precision_score(y[te_idx], proba))

            row = {
                'target': target, 'feature_set': fs_name, 'classifier': clf_name,
                'n_features': X.shape[1], 'n_samples': len(y),
                'f1_mean': np.mean(fold_f1), 'f1_std': np.std(fold_f1),
                'roc_auc_mean': np.mean(fold_auc) if fold_auc else np.nan,
                'roc_auc_std': np.std(fold_auc) if fold_auc else np.nan,
                'pr_auc_mean': np.mean(fold_ap) if fold_ap else np.nan,
                'pr_auc_std': np.std(fold_ap) if fold_ap else np.nan,
            }
            rows.append(row)
            oof_store[(fs_name, clf_name)] = oof_proba
            print(f"  {fs_name:16s} {clf_name:18s} F1={row['f1_mean']:.3f}+/-{row['f1_std']:.3f}  "
                  f"AUC={row['roc_auc_mean']:.3f}+/-{row['roc_auc_std']:.3f}")

    results_df = pd.DataFrame(rows)

    # -- Figures --------------------------------------------------------
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fs_colors = {'hand_engineered': CATEGORICAL[0], 'esm_only': CATEGORICAL[1], 'combined': CATEGORICAL[3]}
    fs_labels = {'hand_engineered': 'Hand-engineered', 'esm_only': 'ESM2-150M only', 'combined': 'Combined'}

    # ROC curves: one panel per classifier, 3 lines (feature sets)
    n_clf = len(classifiers)
    fig, axes = plt.subplots(1, n_clf, figsize=(4.2 * n_clf, 4), squeeze=False)
    axes = axes[0]
    for ax, clf_name in zip(axes, classifiers.keys()):
        for fs_name in feature_sets.keys():
            if (fs_name, clf_name) not in oof_store:
                continue
            fpr, tpr, _ = roc_curve(y, oof_store[(fs_name, clf_name)])
            auc = results_df.loc[(results_df.feature_set == fs_name) & (results_df.classifier == clf_name), 'roc_auc_mean'].values[0]
            ax.plot(fpr, tpr, color=fs_colors[fs_name], lw=1.6, label=f'{fs_labels[fs_name]} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], color=CAM['slate2'], lw=0.8, ls='--')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(clf_name)
        ax.legend(fontsize=6.5, loc='lower right')
        style_axes(ax)
    save_fig(fig, fig_dir / f'roc_{target}.png')

    # Precision-recall curves
    fig, axes = plt.subplots(1, n_clf, figsize=(4.2 * n_clf, 4), squeeze=False)
    axes = axes[0]
    base_rate = y.mean()
    for ax, clf_name in zip(axes, classifiers.keys()):
        for fs_name in feature_sets.keys():
            if (fs_name, clf_name) not in oof_store:
                continue
            prec, rec, _ = precision_recall_curve(y, oof_store[(fs_name, clf_name)])
            ap = results_df.loc[(results_df.feature_set == fs_name) & (results_df.classifier == clf_name), 'pr_auc_mean'].values[0]
            ax.plot(rec, prec, color=fs_colors[fs_name], lw=1.6, label=f'{fs_labels[fs_name]} (AP={ap:.3f})')
        ax.axhline(base_rate, color=CAM['slate2'], lw=0.8, ls='--')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(clf_name)
        ax.legend(fontsize=6.5, loc='lower left')
        style_axes(ax)
    save_fig(fig, fig_dir / f'pr_{target}.png')

    # Confusion matrices, one file per (classifier, feature_set)
    for (fs_name, clf_name), oof_proba in oof_store.items():
        pred = (oof_proba >= 0.5).astype(int)
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(3.2, 3))
        im = ax.imshow(cm, cmap=CAM_SEQUENTIAL_CMAP)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                txt_color = CAM['white'] if val > cm.max() / 2 else CAM['slate4']
                ax.text(j, i, str(val), ha='center', va='center', color=txt_color, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Neg', 'Pos'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Neg', 'Pos'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{clf_name} / {fs_labels[fs_name]}', fontsize=8)
        style_axes(ax, top_right_spines=False)
        save_fig(fig, fig_dir / f'confusion_{target}_{clf_name}_{fs_name}.png')

    # Grouped bar chart: F1 and ROC-AUC, classifier x feature_set, error bars = std
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.2 * n_clf * 0.55 + 2, 4))
    clf_names = list(classifiers.keys())
    x = np.arange(len(clf_names))
    width = 0.25
    for i, fs_name in enumerate(feature_sets.keys()):
        f1_vals = [results_df.loc[(results_df.feature_set == fs_name) & (results_df.classifier == c), 'f1_mean'].values
                   for c in clf_names]
        f1_err = [results_df.loc[(results_df.feature_set == fs_name) & (results_df.classifier == c), 'f1_std'].values
                  for c in clf_names]
        auc_vals = [results_df.loc[(results_df.feature_set == fs_name) & (results_df.classifier == c), 'roc_auc_mean'].values
                    for c in clf_names]
        auc_err = [results_df.loc[(results_df.feature_set == fs_name) & (results_df.classifier == c), 'roc_auc_std'].values
                   for c in clf_names]
        f1_vals = [v[0] if len(v) else np.nan for v in f1_vals]
        f1_err = [v[0] if len(v) else 0 for v in f1_err]
        auc_vals = [v[0] if len(v) else np.nan for v in auc_vals]
        auc_err = [v[0] if len(v) else 0 for v in auc_err]
        offset = (i - 1) * width
        ax1.bar(x + offset, f1_vals, width, yerr=f1_err, color=fs_colors[fs_name], edgecolor=CAM['slate4'],
                linewidth=0.6, capsize=2.5, label=fs_labels[fs_name])
        ax2.bar(x + offset, auc_vals, width, yerr=auc_err, color=fs_colors[fs_name], edgecolor=CAM['slate4'],
                linewidth=0.6, capsize=2.5, label=fs_labels[fs_name])
    for ax, ylabel, title in [(ax1, 'F1 score', 'F1 (5-fold CV mean +/- std)'), (ax2, 'ROC-AUC', 'ROC-AUC (5-fold CV mean +/- std)')]:
        ax.set_xticks(x); ax.set_xticklabels(clf_names, rotation=20, ha='right', fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 1.05)
        style_axes(ax)
    ax1.legend(fontsize=6.5)
    save_fig(fig, fig_dir / f'comparison_bars_{target}.png')

    return results_df, oof_store, y, seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--embeddings', default='esm_embeddings')
    ap.add_argument('--out', default='esm_comparison')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    all_results = []
    oof_npz_payload = {}

    for target, classifiers in [('nes', NES_CLASSIFIERS), ('nls', NLS_CLASSIFIERS)]:
        results_df, oof_store, y, seqs = run_target(target, classifiers, args.embeddings, out_dir, args.folds, args.seed)
        all_results.append(results_df)
        oof_npz_payload[f'{target}_y'] = y
        oof_npz_payload[f'{target}_seqs'] = np.array(seqs)
        for (fs_name, clf_name), proba in oof_store.items():
            oof_npz_payload[f'{target}__{fs_name}__{clf_name}'] = proba

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(out_dir / 'esm_comparison_results.csv', index=False)
    np.savez_compressed(out_dir / 'oof_predictions.npz', **oof_npz_payload)

    print(f"\nResults: {out_dir}/esm_comparison_results.csv")
    print(f"OOF predictions cached: {out_dir}/oof_predictions.npz (for esm_umap_agreement.py)")
    print(f"Figures: {out_dir}/figures/")


if __name__ == '__main__':
    main()
