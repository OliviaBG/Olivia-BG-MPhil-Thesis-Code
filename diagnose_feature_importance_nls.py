#!/usr/bin/env python3
"""
NLS analog of diagnose_feature_importance.py -- same four independent
checks (permutation importance, feature correlation matrix, univariate
model-free ROC-AUC, point-biserial correlation), run against the REAL
trained NLSPredictor and REAL training data (not synthetic placeholder data),
so the accessibility (sasa_norm) feature's importance can be assessed now
that it's real Tien et al. 2013-normalized RSA instead of a flat-divided
raw SASA placeholder.

This script does NOT reimplement feature extraction or data loading -- it
imports NLSPredictor from nls_ml_predictor.py and calls its
build_training_dataset() method directly, so it is guaranteed to use the
exact same positives/negatives/features as the actual training run.

IMPORTANT: this is only meaningful once nls_data_pipeline/structural_data.json
has been regenerated with the corrected per-residue RSA (see
nls_data_pipeline/structural_dataset_pipeline.py's real_per_residue_sasa()).
If that file still holds the old flat-divided-Ų-scale data, sasa_norm's
measured importance here reflects the OLD buggy feature, not the fix.

Usage:
    python3 diagnose_feature_importance_nls.py
    python3 diagnose_feature_importance_nls.py --model-dir models_nls --n-repeats 30
    python3 diagnose_feature_importance_nls.py --scoring roc_auc --corr-threshold 0.6

Outputs (written to <model-dir>/feature_diagnosis/):
    permutation_importance.png     ALL features: permutation vs impurity
    correlation_heatmap.png        full feature x feature correlation matrix
    correlation_with_dominant.png  |r| of every feature vs. the dominant one
    univariate_auc.png             ALL features: model-free single-feature AUC
    point_biserial_correlation.png ALL features: model-free feature-vs-label r
    diagnosis_report.json          all numbers, for reference / re-plotting
    diagnosis_report.txt           human-readable summary (also printed)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nls_ml_predictor import NLSPredictor

THIS_DIR = Path(__file__).resolve().parent

# ============================== pastel theme ==============================
# Same palette as diagnose_feature_importance.py (NES side) so the two
# reports read as a matched pair in a report/thesis.
PASTEL_BLUE = '#A9C9E8'
PASTEL_PINK = '#F4B6C2'
PASTEL_MINT = '#AEE0C8'
PASTEL_PEACH = '#FBD7A8'
PASTEL_ROSE = '#F0A8B4'
PASTEL_LAVENDER = '#D3C1EC'
TEXT_GRAY = '#555555'
GRID_GRAY = '#E7E7EC'
SPINE_GRAY = '#C9C9D2'
from matplotlib.colors import LinearSegmentedColormap
PASTEL_DIVERGING = LinearSegmentedColormap.from_list(
    'pastel_diverging', [PASTEL_PINK, '#FFFFFF', PASTEL_BLUE])
PASTEL_DIVERGING.set_bad('#EDEDED')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'text.color': TEXT_GRAY,
    'axes.edgecolor': SPINE_GRAY,
    'axes.labelcolor': TEXT_GRAY,
    'xtick.color': TEXT_GRAY,
    'ytick.color': TEXT_GRAY,
    'axes.titleweight': 'bold',
    'axes.titlesize': 13,
    'axes.labelsize': 10.5,
})

_ABBREV = {
    'pssm': 'PSSM', 'hpr': 'HPR', 'nc': 'NC', 'ncpr': 'NCPR',
    'cider': 'CIDER', 'sasa': 'SASA', 'plddt': 'pLDDT', 'kd': 'KD',
}


def _pretty_label(name):
    words = []
    for part in name.split('_'):
        low = part.lower()
        words.append(_ABBREV.get(low, part.capitalize()))
    return ' '.join(words)


def _style_axes(ax, xgrid=True):
    ax.set_facecolor('#FFFFFF')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE_GRAY)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=TEXT_GRAY, labelsize=9, length=3)
    if xgrid:
        ax.grid(axis='x', color=GRID_GRAY, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', default=str(THIS_DIR / 'models_nls'),
                     help='Folder containing the trained nls_classifier.pkl etc. '
                          '(default: ./models_nls next to this script)')
    ap.add_argument('--n-repeats', type=int, default=30,
                     help='Permutation repeats per feature (default 30)')
    ap.add_argument('--scoring', default='f1',
                     help='sklearn scoring metric for permutation importance '
                          '(default f1; try roc_auc for a second opinion)')
    ap.add_argument('--corr-threshold', type=float, default=0.5,
                     help='|correlation| threshold for flagging a feature as '
                          'related to the dominant one (default 0.5)')
    args = ap.parse_args()

    out_dir = Path(args.model_dir) / 'feature_diagnosis'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_dir} ...")
    predictor = NLSPredictor(model_dir=args.model_dir)
    if predictor.model is None:
        raise SystemExit("No trained model found/loaded -- run training first "
                          "(python nls_ml_predictor.py train).")

    feature_names = predictor._feature_names()

    print("\nRebuilding the real training dataset (same pipeline as training) ...")
    dataset = predictor.build_training_dataset()
    X, y = dataset['X'], dataset['y']
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"  {X.shape[0]} examples ({n_pos} positive / {n_neg} negative), "
          f"{X.shape[1]} features")

    if n_pos < 10 or n_neg < 10:
        raise SystemExit("Not enough data for a held-out split (need >=10 "
                          "per class) -- can't run this diagnosis yet.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = clone(predictor.model)
    model.fit(X_train_s, y_train)
    print(f"  Re-fit a fresh '{predictor.model_name}' on the train split only "
          f"({len(y_train)} examples) for honest held-out evaluation.")

    impurity_importance = {}
    if hasattr(model, 'feature_importances_'):
        impurity_importance = dict(zip(feature_names, np.asarray(model.feature_importances_).ravel()))
    elif hasattr(model, 'coef_'):
        impurity_importance = dict(zip(feature_names, np.abs(np.asarray(model.coef_).ravel())))

    print(f"\nComputing permutation importance ({args.n_repeats} repeats, "
          f"scoring={args.scoring}) ...")
    perm = permutation_importance(
        model, X_test_s, y_test, n_repeats=args.n_repeats,
        random_state=42, scoring=args.scoring, n_jobs=-1,
    )
    perm_importance = dict(zip(feature_names, perm.importances_mean))
    perm_std = dict(zip(feature_names, perm.importances_std))

    print("Computing feature correlation matrix ...")
    X_all_s = StandardScaler().fit_transform(X)
    corr = np.corrcoef(X_all_s, rowvar=False)

    print("Computing univariate (model-free) feature separation ...")
    univariate_auc = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if np.all(col == col[0]):
            univariate_auc[name] = 0.5
            continue
        auc = roc_auc_score(y, col)
        univariate_auc[name] = max(auc, 1.0 - auc)

    print("Computing point-biserial correlation (feature vs. label) ...")
    point_biserial = {}
    point_biserial_p = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if np.all(col == col[0]):
            point_biserial[name] = 0.0
            point_biserial_p[name] = 1.0
            continue
        r, p = pointbiserialr(y, col)
        point_biserial[name] = float(r)
        point_biserial_p[name] = float(p)

    dominant_feature = max(impurity_importance, key=impurity_importance.get) if impurity_importance else None
    dominant_idx = feature_names.index(dominant_feature) if dominant_feature else None

    correlated_with_dominant = {}
    if dominant_idx is not None:
        for i, name in enumerate(feature_names):
            if name == dominant_feature:
                continue
            c = corr[dominant_idx, i]
            if abs(c) >= args.corr_threshold:
                correlated_with_dominant[name] = float(c)

    _plot_importance_comparison(feature_names, impurity_importance, perm_importance, perm_std, out_dir)
    _plot_correlation_heatmap(feature_names, corr, out_dir)
    if dominant_feature:
        _plot_correlation_with_dominant(dominant_feature, correlated_with_dominant, out_dir)
    _plot_univariate_auc(univariate_auc, out_dir)
    _plot_point_biserial(point_biserial, point_biserial_p, out_dir)

    report = {
        'n_examples': int(X.shape[0]),
        'n_positive': n_pos,
        'n_negative': n_neg,
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'model_type': predictor.model_name,
        'scoring': args.scoring,
        'n_repeats': args.n_repeats,
        'impurity_importance': {k: float(v) for k, v in impurity_importance.items()},
        'permutation_importance_mean': {k: float(v) for k, v in perm_importance.items()},
        'permutation_importance_std': {k: float(v) for k, v in perm_std.items()},
        'dominant_feature_by_impurity': dominant_feature,
        'dominant_feature_by_permutation': (
            max(perm_importance, key=perm_importance.get) if perm_importance else None),
        'univariate_auc': {k: float(v) for k, v in univariate_auc.items()},
        'dominant_feature_by_univariate_auc': (
            max(univariate_auc, key=univariate_auc.get) if univariate_auc else None),
        'point_biserial_correlation': dict(point_biserial),
        'point_biserial_p_value': dict(point_biserial_p),
        'dominant_feature_by_point_biserial': (
            max(point_biserial, key=lambda n: abs(point_biserial[n])) if point_biserial else None),
        'features_correlated_with_dominant': correlated_with_dominant,
        'corr_threshold': args.corr_threshold,
    }
    with open(out_dir / 'diagnosis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    _write_text_report(report, out_dir)

    print(f"\nDiagnosis complete. Outputs in {out_dir}/")


def _plot_importance_comparison(feature_names, impurity, perm_mean, perm_std, out_dir, top_n=None):
    order = sorted(feature_names, key=lambda n: perm_mean.get(n, 0), reverse=True)[:top_n]
    y_pos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(order))))
    perm_vals = [perm_mean.get(n, 0) for n in order]
    perm_errs = [perm_std.get(n, 0) for n in order]
    imp_vals = [impurity.get(n, 0) for n in order]
    imp_sum = sum(imp_vals) or 1.0
    perm_pos_sum = sum(v for v in perm_vals if v > 0) or 1.0
    imp_vals_norm = [v / imp_sum * perm_pos_sum for v in imp_vals]

    labels = [_pretty_label(n) for n in order]
    bar_h = 0.38
    ax.barh(y_pos + bar_h / 2, perm_vals, height=bar_h, xerr=perm_errs,
            color=PASTEL_BLUE, ecolor=SPINE_GRAY, label='Permutation importance (held-out)',
            capsize=2, edgecolor='white', linewidth=0.4)
    ax.barh(y_pos - bar_h / 2, imp_vals_norm, height=bar_h,
            color=PASTEL_PINK, label='Impurity importance (rescaled for comparison)',
            edgecolor='white', linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'NLS: Permutation vs. Impurity-Based Feature Importance\n(all {len(order)} features)')
    ax.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white', edgecolor=GRID_GRAY)
    ax.axvline(0, color=SPINE_GRAY, linewidth=0.8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'permutation_importance.png', dpi=150, facecolor='white')
    plt.close(fig)


def _plot_correlation_heatmap(feature_names, corr, out_dir):
    n = len(feature_names)
    labels = [_pretty_label(name) for name in feature_names]
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n), max(7, 0.35 * n)))
    im = ax.imshow(corr, cmap=PASTEL_DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6, color=TEXT_GRAY)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6, color=TEXT_GRAY)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, label='Pearson correlation', shrink=0.8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_GRAY, labelsize=8)
    cbar.set_label('Pearson correlation', color=TEXT_GRAY, fontsize=9.5)
    ax.set_title('NLS: Feature Correlation Matrix', pad=14)
    fig.tight_layout()
    fig.savefig(out_dir / 'correlation_heatmap.png', dpi=150, facecolor='white')
    plt.close(fig)


def _plot_correlation_with_dominant(dominant_feature, correlated, out_dir):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * max(1, len(correlated)))))
    if not correlated:
        ax.text(0.5, 0.5, f'No feature correlates with\n"{_pretty_label(dominant_feature)}" above threshold',
                ha='center', va='center', transform=ax.transAxes, color=TEXT_GRAY, fontsize=11)
        ax.axis('off')
    else:
        items = sorted(correlated.items(), key=lambda kv: abs(kv[1]), reverse=True)
        names = [_pretty_label(k) for k, _ in items]
        vals = [v for _, v in items]
        colors = [PASTEL_BLUE if v > 0 else PASTEL_PINK for v in vals]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, color=colors, edgecolor='white', linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.axvline(0, color=SPINE_GRAY, linewidth=0.8)
        ax.set_xlabel('Pearson correlation')
        ax.set_title(f'NLS: Features Correlated with "{_pretty_label(dominant_feature)}"')
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'correlation_with_dominant.png', dpi=150, facecolor='white')
    plt.close(fig)


def _plot_univariate_auc(univariate_auc, out_dir, top_n=None):
    order = sorted(univariate_auc, key=univariate_auc.get, reverse=True)[:top_n]
    vals = [univariate_auc[n] for n in order]
    y_pos = np.arange(len(order))
    labels = [_pretty_label(n) for n in order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
    colors = [PASTEL_MINT if v >= 0.7 else (PASTEL_PEACH if v >= 0.6 else PASTEL_ROSE) for v in vals]
    ax.barh(y_pos, vals, color=colors, edgecolor='white', linewidth=0.4)
    ax.axvline(0.5, color=PASTEL_LAVENDER, linewidth=1.4, linestyle='--', label='0.5 = no separation')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('Univariate ROC-AUC (feature alone vs. label, no model fit)')
    ax.set_title(f'NLS: Model-Free Single-Feature Separation\n(all {len(order)} features)')
    ax.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white', edgecolor=GRID_GRAY)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'univariate_auc.png', dpi=150, facecolor='white')
    plt.close(fig)


def _plot_point_biserial(point_biserial, point_biserial_p, out_dir, top_n=None):
    order = sorted(point_biserial, key=lambda n: abs(point_biserial[n]), reverse=True)[:top_n]
    vals = [point_biserial[n] for n in order]
    pvals = [point_biserial_p.get(n, 1.0) for n in order]
    labels = [_pretty_label(n) for n in order]
    y_pos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
    colors = [PASTEL_BLUE if v >= 0 else PASTEL_PINK for v in vals]
    for yi, (v, p, c) in enumerate(zip(vals, pvals, colors)):
        ax.barh(yi, v, color=c, edgecolor='white', linewidth=0.4,
                hatch='///' if p >= 0.05 else None)
    ax.axvline(0, color=SPINE_GRAY, linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel('Point-biserial correlation (feature vs. label, no model fit)')
    ax.set_title(f'NLS: Model-Free Feature-Label Correlation\n(all {len(order)} features)')
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PASTEL_BLUE, edgecolor='white', label='Positive correlation'),
        Patch(facecolor=PASTEL_PINK, edgecolor='white', label='Negative correlation'),
        Patch(facecolor='white', edgecolor=SPINE_GRAY, hatch='///', label='p ≥ 0.05 (not significant)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8, frameon=True,
              facecolor='white', edgecolor=GRID_GRAY)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'point_biserial_correlation.png', dpi=150, facecolor='white')
    plt.close(fig)


def _write_text_report(report, out_dir):
    lines = []
    lines.append("NLS feature importance diagnosis")
    lines.append("=" * 60)
    lines.append(f"Model: {report['model_type']}")
    lines.append(f"Data: {report['n_examples']} examples ({report['n_positive']} positive / "
                 f"{report['n_negative']} negative)")
    lines.append(f"Held-out test set: {report['n_test']} examples "
                 f"(a fresh model was trained on the other {report['n_train']} "
                 f"just for this diagnosis)")
    lines.append("")
    lines.append(f"Dominant feature by impurity/coefficient: {report['dominant_feature_by_impurity']}")
    lines.append(f"Dominant feature by permutation importance: {report['dominant_feature_by_permutation']}")
    lines.append("")

    corr = report['features_correlated_with_dominant']
    if corr:
        lines.append(f"Features correlated with '{report['dominant_feature_by_impurity']}' "
                     f"(|r| >= {report['corr_threshold']}):")
        for name, val in sorted(corr.items(), key=lambda kv: abs(kv[1]), reverse=True):
            lines.append(f"  {name}: r = {val:+.3f}")
        lines.append("")

    lines.append("All features by permutation importance:")
    top = sorted(report['permutation_importance_mean'].items(), key=lambda kv: kv[1], reverse=True)
    for name, val in top:
        std = report['permutation_importance_std'].get(name, 0)
        lines.append(f"  {name}: {val:+.4f} (+/- {std:.4f})")
    lines.append("")

    lines.append("-" * 60)
    lines.append("Model-free check: does this hold up without fitting any model?")
    lines.append("-" * 60)
    lines.append("All features by univariate ROC-AUC:")
    top_uni = sorted(report['univariate_auc'].items(), key=lambda kv: kv[1], reverse=True)
    for name, val in top_uni:
        lines.append(f"  {name}: AUC = {val:.3f}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("Cross-check: point-biserial correlation (linear, not just monotonic)")
    lines.append("-" * 60)
    lines.append("All features by point-biserial correlation (with significance):")
    top_pb = sorted(report['point_biserial_correlation'].items(),
                     key=lambda kv: abs(kv[1]), reverse=True)
    for name, val in top_pb:
        p = report['point_biserial_p_value'].get(name, 1.0)
        sig = '*' if p < 0.05 else ' '
        lines.append(f"  {name}: r = {val:+.3f}{sig} (p = {p:.2e})")
    lines.append("  (* p < 0.05)")

    with open(out_dir / 'diagnosis_report.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
