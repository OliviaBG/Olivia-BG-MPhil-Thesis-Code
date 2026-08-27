#!/usr/bin/env python3
"""
esm_comparison_figures.py
============================================================
Standalone ESM-comparison figure generator -- pulled out of
generate_thesis_figures.py so it can be run on its own (seconds, not the
full master script's runtime) whenever new ESM comparison data comes back
from the GPU pod. Same Cambridge/Nature theme (thesis_plot_style.py) as
every other thesis figure, same output location.

Data sources (all produced on the GPU pod, copied back locally -- none of
this needs a GPU or network to plot):
  - esm_full_comparison_nes.json, esm_full_comparison_nls.json
        (esm_full_comparison.py) -- hand-engineered / frozen ESM2 / combined
        / combined+PCA(10,30,50,100,200) / esm+PCA, 5-fold CV, every
        classifier (svm x2, random_forest, extra_trees, gradient_boosting,
        hist_gradient_boosting, mlp, xgboost if installed).
  - esm_finetune_kfold_nes.json, esm_finetune_kfold_nls.json
        (esm_finetune_kfold.py) -- fine-tuned ESM2 (last 2 layers + head
        unfrozen), proper 5-fold CV.
  - esm_comparison/esm_comparison_results.csv + oof_predictions.npz
        (older esm_model_comparison.py run) -- only source with per-example
        out-of-fold predictions, needed for the ROC/PR curves and the
        rank-agreement scatter plots. Only covers hand-engineered / frozen /
        combined (no PCA, no fine-tuned) -- esm_full_comparison.py doesn't
        save per-example predictions, only aggregate fold stats, so those
        two plot types can't include the new PCA/fine-tuned variants without
        rerunning esm_model_comparison.py's OOF-saving approach for them too
        (not done here -- flag if you want that as a follow-up).
  - esm_embeddings/nes_embeddings.npz, nls_embeddings.npz -- for the UMAP
        embedding-space plot.

Each data source is optional -- whatever isn't present yet is skipped with
a printed note rather than crashing the rest of the script, same
degrade-gracefully convention as generate_thesis_figures.py.

Usage:
    python3 esm_comparison_figures.py
    python3 esm_comparison_figures.py --out thesis_figures

Outputs -> <out>/esm_comparison/:
    esm_comparison_<target>.png   -- MAIN figure: F1+AUC bars, hand-eng /
                                      frozen / combined / combined+PCA(best)
                                      per classifier, fine-tuned ESM2 shown
                                      as a horizontal reference line
    esm_pca_sweep_<target>.png    -- F1 vs PCA component count, one line per
                                      classifier, ESM-only and Combined panels
    esm_roc_<target>.png, esm_pr_<target>.png   -- hand-eng/frozen/combined only
    esm_agreement_<target>.png    -- hand-eng vs frozen-ESM rank agreement
    esm_umap_<target>.png         -- frozen ESM2 embedding space
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from thesis_plot_style import apply_style, style_axes, CAM, CATEGORICAL, sig_stars, save_fig

CLASSIFIER_LABELS = {
    'svm_linear': 'Support Vector Machine (Linear)',
    'svm_rbf': 'Support Vector Machine (Radial Basis Function)',
    'random_forest': 'Random Forest',
    'extra_trees': 'Extra Trees',
    'gradient_boosting': 'Gradient Boosting',
    'hist_gradient_boosting': 'Histogram-Based Gradient Boosting',
    'mlp': 'Multi-Layer Perceptron',
    'xgboost': 'XGBoost',
}
CLASSIFIER_ORDER = list(CLASSIFIER_LABELS.keys())


def pretty_clf(name):
    return CLASSIFIER_LABELS.get(name, name.replace('_', ' ').title())


FEATURE_SET_LABELS = {
    'hand_engineered': 'Hand-Engineered',
    'esm_only': 'ESM2-150M (Frozen)',
    'combined': 'Combined',
    'combined_pca_best': 'Combined + PCA (best k)',
}
FEATURE_SET_ORDER = ['hand_engineered', 'esm_only', 'combined']
# House order: dark blue, visible light blue, teal, then grey for anything
# past the first three (see thesis_plot_style.py CATEGORICAL).
FEATURE_SET_COLORS = {
    'hand_engineered': CATEGORICAL[0], 'esm_only': CATEGORICAL[1],
    'combined': CATEGORICAL[2], 'combined_pca_best': CATEGORICAL[3],
}
FINETUNE_COLOR = CATEGORICAL[4]
PCA_KS = [10, 30, 50, 100, 200]


def _load_json(path):
    return json.load(open(path)) if path.exists() else None


# -------------------------------------------------------------------------
# 1. Main comparison bars: hand-engineered / frozen / combined /
#    combined+PCA(best), fine-tuned as a horizontal reference line
# -------------------------------------------------------------------------

def fig_master_comparison(out_dir):
    for target in ('nes', 'nls'):
        full_path = THIS_DIR / f'esm_full_comparison_{target}.json'
        full = _load_json(full_path)
        if full is None:
            print(f"\n(skipping {target.upper()} master comparison -- {full_path.name} not found)")
            continue
        ft_path = THIS_DIR / f'esm_finetune_kfold_{target}.json'
        ft = _load_json(ft_path)

        clf_order = [c for c in CLASSIFIER_ORDER if f'hand_engineered__{c}' in full]
        if not clf_order:
            continue
        labels = [pretty_clf(c) for c in clf_order]
        x = np.arange(len(clf_order))
        width = 0.2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5 * len(clf_order) + 2, 4))
        for i, fs in enumerate(FEATURE_SET_ORDER):
            f1_vals, f1_err, auc_vals, auc_err = [], [], [], []
            for c in clf_order:
                row = full.get(f'{fs}__{c}')
                f1_vals.append(row['f1_mean'] if row else np.nan)
                f1_err.append(row['f1_std'] if row else 0)
                auc_vals.append(row['auc_mean'] if row else np.nan)
                auc_err.append(row['auc_std'] if row else 0)
            offset = (i - 1.5) * width
            ax1.bar(x + offset, f1_vals, width, yerr=f1_err, color=FEATURE_SET_COLORS[fs],
                    edgecolor=CAM['slate4'], linewidth=0.6, capsize=2, label=FEATURE_SET_LABELS[fs])
            ax2.bar(x + offset, auc_vals, width, yerr=auc_err, color=FEATURE_SET_COLORS[fs],
                    edgecolor=CAM['slate4'], linewidth=0.6, capsize=2, label=FEATURE_SET_LABELS[fs])

        # Combined + PCA (best k per classifier, by F1)
        f1_vals, f1_err, auc_vals, auc_err = [], [], [], []
        for c in clf_order:
            best = None
            for k in PCA_KS:
                row = full.get(f'combined_pca{k}__{c}')
                if row and (best is None or row['f1_mean'] > best['f1_mean']):
                    best = row
            f1_vals.append(best['f1_mean'] if best else np.nan)
            f1_err.append(best['f1_std'] if best else 0)
            auc_vals.append(best['auc_mean'] if best else np.nan)
            auc_err.append(best['auc_std'] if best else 0)
        offset = 2.5 * width
        ax1.bar(x + offset, f1_vals, width, yerr=f1_err, color=FEATURE_SET_COLORS['combined_pca_best'],
                edgecolor=CAM['slate4'], linewidth=0.6, capsize=2, label=FEATURE_SET_LABELS['combined_pca_best'])
        ax2.bar(x + offset, auc_vals, width, yerr=auc_err, color=FEATURE_SET_COLORS['combined_pca_best'],
                edgecolor=CAM['slate4'], linewidth=0.6, capsize=2, label=FEATURE_SET_LABELS['combined_pca_best'])

        if ft and ft.get('folds'):
            ax1.axhline(ft['f1_mean'], color=FINETUNE_COLOR, lw=1.4, ls='--',
                        label=f"Fine-tuned ESM2 (F1={ft['f1_mean']:.3f}±{ft['f1_std']:.3f})")
            ax2.axhline(ft['auc_mean'], color=FINETUNE_COLOR, lw=1.4, ls='--',
                        label=f"Fine-tuned ESM2 (AUC={ft['auc_mean']:.3f}±{ft['auc_std']:.3f})")

        for ax, ylabel, title in [(ax1, 'F1 score', 'F1 (5-fold CV, mean ± sd)'),
                                   (ax2, 'ROC-AUC', 'ROC-AUC (5-fold CV, mean ± sd)')]:
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
            ax.set_ylabel(ylabel); ax.set_title(title, fontsize=8); ax.set_ylim(0, 1.05)
            style_axes(ax)
        ax1.legend(fontsize=6, loc='lower right')
        fig.suptitle(f'{target.upper()}: hand-engineered vs. ESM2-150M (frozen, +PCA, fine-tuned) vs. combined',
                     y=1.03, fontsize=9)
        save_fig(fig, out_dir / f'esm_comparison_{target}.png')

    print(f"  Saved master ESM comparison figures to {out_dir}/")


# -------------------------------------------------------------------------
# 2. PCA component sweep -- F1 vs k, one line per classifier
# -------------------------------------------------------------------------

def fig_pca_sweep(out_dir):
    for target in ('nes', 'nls'):
        full_path = THIS_DIR / f'esm_full_comparison_{target}.json'
        full = _load_json(full_path)
        if full is None:
            print(f"\n(skipping {target.upper()} PCA sweep -- {full_path.name} not found)")
            continue
        clf_order = [c for c in CLASSIFIER_ORDER if f'hand_engineered__{c}' in full]
        if not clf_order:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
        colors = (CATEGORICAL * 2)[:len(clf_order)]
        for ax, prefix, no_pca_key, ref_label in [
            (ax1, 'esm_pca', 'esm_only', 'ESM-only, no PCA'),
            (ax2, 'combined_pca', 'combined', 'Combined, no PCA'),
        ]:
            for c, color in zip(clf_order, colors):
                ys = [full[f'{prefix}{k}__{c}']['f1_mean'] if f'{prefix}{k}__{c}' in full else np.nan
                      for k in PCA_KS]
                ax.plot(PCA_KS, ys, marker='o', ms=3, lw=1.2, color=color, label=pretty_clf(c))
                ref = full.get(f'{no_pca_key}__{c}')
                if ref:
                    ax.axhline(ref['f1_mean'], color=color, lw=0.6, ls=':', alpha=0.5)
            hand = full.get(f'hand_engineered__{c}') if clf_order else None
            ax.set_xlabel('PCA components (k)')
            ax.set_ylabel('F1 score')
            ax.set_title(f'{ref_label} (dotted = no-PCA baseline per classifier)', fontsize=7.5)
            ax.set_ylim(0, 1.0)
            style_axes(ax)
        ax1.legend(fontsize=5.5, loc='lower right', ncol=2)
        fig.suptitle(f'{target.upper()}: F1 vs. PCA dimensionality (5-fold CV)', y=1.03, fontsize=9)
        save_fig(fig, out_dir / f'esm_pca_sweep_{target}.png')

    print(f"  Saved PCA sweep figures to {out_dir}/")


# -------------------------------------------------------------------------
# 3. ROC / PR curves -- hand-engineered / frozen / combined only (needs
#    per-example OOF predictions, which only the older esm_model_comparison.py
#    run saved; esm_full_comparison.py only saves aggregate fold stats)
# -------------------------------------------------------------------------

def fig_esm_roc_pr(out_dir, npz_rel='esm_comparison/oof_predictions.npz'):
    npz_path = THIS_DIR / npz_rel
    if not npz_path.exists():
        print(f"\n(skipping ESM ROC/PR figures -- {npz_path} not found)")
        return

    data = np.load(npz_path, allow_pickle=True)
    for target in ('nes', 'nls'):
        y_key = f'{target}_y'
        if y_key not in data.files:
            continue
        y = data[y_key]
        classifiers = sorted({k.split('__')[2] for k in data.files if k.startswith(f'{target}__hand_engineered__')})
        clf_order = [c for c in CLASSIFIER_ORDER if c in classifiers] + [c for c in classifiers if c not in CLASSIFIER_ORDER]
        if not clf_order:
            continue
        n_clf = len(clf_order)

        fig, axes = plt.subplots(1, n_clf, figsize=(4.2 * n_clf, 4), squeeze=False)
        axes = axes[0]
        for ax, clf_name in zip(axes, clf_order):
            for fs in FEATURE_SET_ORDER:
                key = f'{target}__{fs}__{clf_name}'
                if key not in data.files:
                    continue
                proba = data[key]
                fpr, tpr, _ = roc_curve(y, proba)
                auc = roc_auc_score(y, proba)
                ax.plot(fpr, tpr, color=FEATURE_SET_COLORS[fs], lw=1.6, label=f'{FEATURE_SET_LABELS[fs]} (AUC={auc:.3f})')
            ax.plot([0, 1], [0, 1], color=CAM['slate2'], lw=0.8, ls='--')
            ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
            ax.set_title(pretty_clf(clf_name), fontsize=8)
            ax.legend(fontsize=6.5, loc='lower right')
            style_axes(ax)
        fig.suptitle(f'{target.upper()}: ROC, hand-engineered vs. ESM2-150M vs. combined', y=1.03, fontsize=9)
        save_fig(fig, out_dir / f'esm_roc_{target}.png')

        fig, axes = plt.subplots(1, n_clf, figsize=(4.2 * n_clf, 4), squeeze=False)
        axes = axes[0]
        base_rate = y.mean()
        for ax, clf_name in zip(axes, clf_order):
            for fs in FEATURE_SET_ORDER:
                key = f'{target}__{fs}__{clf_name}'
                if key not in data.files:
                    continue
                proba = data[key]
                prec, rec, _ = precision_recall_curve(y, proba)
                ap = average_precision_score(y, proba)
                ax.plot(rec, prec, color=FEATURE_SET_COLORS[fs], lw=1.6, label=f'{FEATURE_SET_LABELS[fs]} (AP={ap:.3f})')
            ax.axhline(base_rate, color=CAM['slate2'], lw=0.8, ls='--')
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
            ax.set_title(pretty_clf(clf_name), fontsize=8)
            ax.legend(fontsize=6.5, loc='lower left')
            style_axes(ax)
        fig.suptitle(f'{target.upper()}: precision-recall, hand-engineered vs. ESM2-150M vs. combined', y=1.03, fontsize=9)
        save_fig(fig, out_dir / f'esm_pr_{target}.png')

    print(f"  Saved ESM ROC/PR figures to {out_dir}/ (hand-eng/frozen/combined only -- see docstring)")


# -------------------------------------------------------------------------
# 4. Rank agreement scatter -- hand-engineered vs. frozen ESM, same caveat
# -------------------------------------------------------------------------

def fig_esm_agreement(out_dir, npz_rel='esm_comparison/oof_predictions.npz'):
    npz_path = THIS_DIR / npz_rel
    if not npz_path.exists():
        print(f"\n(skipping ESM agreement figure -- {npz_path} not found)")
        return

    data = np.load(npz_path, allow_pickle=True)
    for target in ('nes', 'nls'):
        y_key = f'{target}_y'
        if y_key not in data.files:
            continue
        y = data[y_key]
        classifiers = sorted({k.split('__')[2] for k in data.files if k.startswith(f'{target}__hand_engineered__')})
        clf_order = [c for c in CLASSIFIER_ORDER if c in classifiers] + [c for c in classifiers if c not in CLASSIFIER_ORDER]
        if not clf_order:
            continue
        n_clf = len(clf_order)

        fig, axes = plt.subplots(1, n_clf, figsize=(4 * n_clf, 4), squeeze=False)
        axes = axes[0]
        for ax, clf_name in zip(axes, clf_order):
            hand_key = f'{target}__hand_engineered__{clf_name}'
            esm_key = f'{target}__esm_only__{clf_name}'
            if hand_key not in data.files or esm_key not in data.files:
                ax.set_visible(False)
                continue
            hand_proba, esm_proba = data[hand_key], data[esm_key]
            rho, p = spearmanr(hand_proba, esm_proba)
            neg_mask, pos_mask = (y == 0), (y == 1)
            ax.scatter(hand_proba[neg_mask], esm_proba[neg_mask], s=10, alpha=0.7, color=CATEGORICAL[1],
                       edgecolors=CAM['slate4'], linewidths=0.2, label=f'Negative (n={neg_mask.sum()})')
            ax.scatter(hand_proba[pos_mask], esm_proba[pos_mask], s=10, alpha=0.7, color=CATEGORICAL[0],
                       edgecolors=CAM['slate4'], linewidths=0.2, label=f'Positive (n={pos_mask.sum()})')
            ax.plot([0, 1], [0, 1], color=CAM['slate3'], lw=0.8, ls='--', label='y = x')
            ax.set_xlabel('Hand-engineered P(positive)'); ax.set_ylabel('ESM-only P(positive)')
            ax.set_title(f'{pretty_clf(clf_name)}\n' + r'Spearman $\rho$=' + f'{rho:.3f} ({sig_stars(p)})', fontsize=8)
            ax.legend(fontsize=6, loc='lower right', frameon=True, facecolor=CAM['white'], edgecolor=CAM['slate2'])
            style_axes(ax)
        fig.suptitle(f'{target.upper()}: rank agreement, hand-engineered vs. ESM-only', y=1.03, fontsize=9)
        save_fig(fig, out_dir / f'esm_agreement_{target}.png')

    print(f"  Saved ESM agreement figures to {out_dir}/ (hand-eng vs. frozen only -- see docstring)")


# -------------------------------------------------------------------------
# 5. UMAP embedding space -- always regenerated fresh from current embeddings
# -------------------------------------------------------------------------

def fig_esm_umap(out_dir, embeddings_dir='esm_embeddings'):
    try:
        import umap
    except ImportError:
        print("\n(skipping ESM UMAP figure -- umap-learn not installed. "
              "Install with: pip install umap-learn --break-system-packages)")
        return

    emb_dir = THIS_DIR / embeddings_dir
    for target in ('nes', 'nls'):
        npz_path = emb_dir / f'{target}_embeddings.npz'
        if not npz_path.exists():
            print(f"\n(skipping {target.upper()} UMAP -- {npz_path} not found)")
            continue

        if target == 'nes':
            from nes_ml_predictor_improved import ImprovedNESPredictor
            predictor = ImprovedNESPredictor()
        else:
            from nls_ml_predictor import NLSPredictor
            predictor = NLSPredictor()
        dataset = predictor.build_training_dataset()
        labels_dict = {}
        for n in dataset['negatives']:
            labels_dict[n['seq'].upper()] = 0
        for p in dataset['positives']:
            labels_dict[p['seq'].upper()] = 1

        npz = np.load(npz_path, allow_pickle=True)
        seqs, vecs = npz['sequences'], npz['vectors']
        labels = np.array([labels_dict.get(s, -1) for s in seqs])
        keep = labels >= 0
        vecs, labels = vecs[keep], labels[keep]
        if len(vecs) < 5:
            print(f"  [{target}] too few labeled sequences ({len(vecs)}) for UMAP -- skipping")
            continue

        print(f"  [{target}] running UMAP on {vecs.shape[0]} sequences ({vecs.shape[1]}-dim)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb2d = reducer.fit_transform(vecs)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        for lab, name, color in [(0, 'Negative', CATEGORICAL[1]), (1, 'Positive', CATEGORICAL[0])]:
            mask = labels == lab
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1], s=12, alpha=0.7, color=color,
                       edgecolors=CAM['slate4'], linewidths=0.2, label=f'{name} (n={mask.sum()})')
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        ax.set_title(f'{target.upper()}: frozen ESM2-150M embedding space', fontsize=9)
        ax.legend(fontsize=7.5, frameon=True, facecolor=CAM['white'], edgecolor=CAM['slate2'])
        style_axes(ax, top_right_spines=True)
        save_fig(fig, out_dir / f'esm_umap_{target}.png')

    print(f"  Saved ESM UMAP figures to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='thesis_figures')
    args = ap.parse_args()

    apply_style()
    out_dir = THIS_DIR / args.out / 'esm_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_master_comparison(out_dir)
    fig_pca_sweep(out_dir)
    fig_esm_roc_pr(out_dir)
    fig_esm_agreement(out_dir)
    fig_esm_umap(out_dir)

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == '__main__':
    main()
