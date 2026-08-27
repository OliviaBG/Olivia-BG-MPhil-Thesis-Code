"""
Figure generation for the NES ML predictor training/validation pipeline.

Called automatically from nes_ml_predictor_improved.py's _train_model() at
the end of every training run, so figures are always regenerated alongside
the model/metrics whenever you retrain. Every figure is written to disk as a
PNG (never just shown) so it can be dropped straight into a thesis or paper.

All functions degrade gracefully (print a warning and return None) rather
than raising, so a plotting failure never breaks model training.

Figures produced (saved into <model_dir>/figures/):
  01_feature_distributions.png   - violin plots, positives vs negatives,
                                    for a panel of the most discriminative
                                    features (mirrors a typical "feature
                                    characterization" figure in an ML paper)
  02_cv_model_comparison.png     - cross-validated F1 across the candidate
                                    models considered (SVM linear/RBF, GB)
  03_roc_curve.png               - ROC curve + AUC on the held-out test split
  04_precision_recall_curve.png  - precision-recall curve on the same split
  05_confusion_matrix.png        - confusion matrix on the held-out test split
  06_feature_importance.png      - top-N feature importances/coefficients
  07_probability_distribution.png- predicted-probability histograms by class
"""

import warnings

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # headless -- never try to open a GUI window
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, confusion_matrix
    )
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False


# Features shown in the distribution panel, in display order. Picked for
# interpretability (mirrors the kind of "structural/biophysical
# characterization" panel common in NES/NLS predictor papers) rather than
# picking purely by importance rank, though several of these also rank
# highly. Falls back gracefully to whichever of these are actually present
# in feature_names if the feature set changes.
DEFAULT_DISTRIBUTION_FEATURES = [
    'pssm_score', 'nes_disorder_mean', 'c_flank_disorder',
    'n_flank_disorder', 'spacer_hydrophobicity', 'mean_hydro',
    'ncpr_local', 'frac_phi_total',
]

_STYLE = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

POS_COLOR = '#2166AC'   # blue - positives (real/curated NES)
NEG_COLOR = '#B2182B'   # red  - negatives


def _ensure_out_dir(model_dir):
    from pathlib import Path
    out_dir = Path(model_dir) / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_feature_distributions(X, y, feature_names, model_dir,
                                features=None, filename='01_feature_distributions.png'):
    """Violin plots comparing each chosen feature's distribution between
    positive (real NES) and negative examples, with a Mann-Whitney U test
    p-value annotated per panel (nonparametric, doesn't assume normality --
    appropriate here since several features are bounded/skewed)."""
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available -- skipping feature distribution plot")
        return None
    try:
        from scipy.stats import mannwhitneyu
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False

    features = features or DEFAULT_DISTRIBUTION_FEATURES
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    features = [f for f in features if f in name_to_idx]
    if not features:
        print("  Warning: none of the requested features are present -- skipping distribution plot")
        return None

    y = np.asarray(y)
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, feat in zip(axes, features):
            col = X[:, name_to_idx[feat]]
            pos_vals = col[y == 1]
            neg_vals = col[y == 0]

            parts = ax.violinplot([neg_vals, pos_vals], showmeans=True, showextrema=True)
            for body, color in zip(parts['bodies'], [NEG_COLOR, POS_COLOR]):
                body.set_facecolor(color)
                body.set_alpha(0.55)
            for key in ('cbars', 'cmeans', 'cmaxes', 'cmins'):
                if key in parts:
                    parts[key].set_color('#333333')

            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_title(feat, fontsize=10)

            if SCIPY_AVAILABLE and len(pos_vals) > 0 and len(neg_vals) > 0:
                try:
                    _, p = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
                    label = 'p < 0.0001' if p < 1e-4 else f'p = {p:.3g}'
                    ax.text(0.5, -0.18, label, transform=ax.transAxes,
                            ha='center', fontsize=8, color='#555555')
                except Exception:
                    pass

        for ax in axes[len(features):]:
            ax.axis('off')

        fig.suptitle('Feature distributions: positive vs. negative training examples', y=1.02)
        fig.tight_layout()
        out_path = _ensure_out_dir(model_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")
        return out_path


def plot_cv_model_comparison(cv_report, model_dir, filename='02_cv_model_comparison.png'):
    """Bar chart of cross-validated F1 (mean +/- std) for every candidate
    model considered during model selection."""
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available -- skipping CV model comparison plot")
        return None

    names, means, stds = [], [], []
    for name, res in cv_report.items():
        if 'mean_f1' not in res:
            continue
        names.append(name)
        means.append(res['mean_f1'])
        stds.append(res.get('std_f1', 0.0))
    if not names:
        print("  Warning: no valid CV results -- skipping CV model comparison plot")
        return None

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color='#4393C3', edgecolor='#222222')
        best_idx = int(np.argmax(means))
        bars[best_idx].set_color('#2166AC')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel('Cross-validated F1 score')
        ax.set_ylim(0, 1.05)
        ax.set_title('Model selection: cross-validated F1 by candidate model')
        for xi, m in zip(x, means):
            ax.text(xi, m + 0.02, f'{m:.3f}', ha='center', fontsize=9)
        fig.tight_layout()
        out_path = _ensure_out_dir(model_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")
        return out_path


def plot_roc_curve(y_test, y_proba, model_dir, filename='03_roc_curve.png'):
    """ROC curve + AUC on the held-out test split."""
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_METRICS_AVAILABLE:
        print("  Warning: matplotlib/sklearn not available -- skipping ROC curve")
        return None
    if len(set(y_test)) != 2:
        print("  Warning: held-out test set is not binary -- skipping ROC curve")
        return None

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(fpr, tpr, color=POS_COLOR, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#999999', lw=1, linestyle='--', label='Chance')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve (held-out test set)')
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        out_path = _ensure_out_dir(model_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")
        return out_path


def plot_precision_recall_curve(y_test, y_proba, model_dir, filename='04_precision_recall_curve.png'):
    """Precision-recall curve on the held-out test split -- more informative
    than ROC alone when classes are imbalanced (here negatives outnumber
    positives roughly 3:1)."""
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_METRICS_AVAILABLE:
        print("  Warning: matplotlib/sklearn not available -- skipping PR curve")
        return None
    if len(set(y_test)) != 2:
        print("  Warning: held-out test set is not binary -- skipping PR curve")
        return None

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    baseline = float(np.mean(y_test))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(recall, precision, color=POS_COLOR, lw=2, label='Model')
        ax.axhline(baseline, color='#999999', lw=1, linestyle='--',
                   label=f'Baseline (prevalence = {baseline:.2f})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-recall curve (held-out test set)')
        ax.legend(loc='lower left', fontsize=9)
        fig.tight_layout()
        out_path = _ensure_out_dir(model_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")
        return out_path


def plot_confusion_matrix(y_test, y_pred, model_dir, filename='05_confusion_matrix.png',
                           class_names=('Negative', 'Positive')):
    """Confusion matrix heatmap on the held-out test split."""
    if not MATPLOTLIB_AVAILABLE or not SKLEARN_METRICS_AVAILABLE:
        print("  Warning: matplotlib/sklearn not available -- skipping confusion matrix")
        return None

    cm = confusion_matrix(y_test, y_pred)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion matrix (held-out test set)')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = _ensure_out_dir(model_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")
        return out_path


def plot_feature_importance(importance_dict, model_dir, top_n=15,
                             filename='06_feature_importance.png', method='auto'):
    """Horizontal bar chart of the top-N features by importance.

    method controls the axis label / title only (it does not affect the
    values -- pass whichever method the importance_dict actually came from,
    i.e. ImprovedNESPredictor.get_feature_importance()'s own `method` arg):
      'auto'/'permutation' -- labelled as permutation importance (held-out
        F1 drop when a feature is shuffled) -- the default and more
        trustworthy measure for correlated features; see
        diagnose_feature_importance.py.
      'impurity' -- labelled as tree-impurity/coefficient importance (what
        this chart showed before permutation importance was added).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available -- skipping feature importance plot")
        return None
    if not importance_dict:
        print("  Warning: no feature importances available for this model type -- skipping")
        return None

    items = sorted(importance_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    items = items[::-1]  # largest at top when plotted horizontally
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in values]

    if method in ('auto', 'permutation'):
        xlabel = 'Permutation importance (held-out F1 drop when shuffled)'
        title_suffix = '(permutation importance)'
    else:
        xlabel = 'Impurity importance / coefficient'
        title_suffix = '(impurity importance)'

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.2))
        ax.barh(names, values, color=colors)
        ax.set_xlabel(xlabel)
        ax.set_title(f'Top {len(names)} features {title_suffix}')
        fig.tight_layout()
        out_path = _ensure_out_dir(model_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")
        return out_path


def plot_probability_distribution(y_test, y_proba, model_dir,
                                   filename='07_probability_distribution.png'):
    """Histogram of predicted probabilities split by true class -- shows how
    cleanly the model separates the two classes and where the decision
    threshold sits relative to that separation."""
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available -- skipping probability distribution plot")
        return None

    y_test = np.asarray(y_test)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0, 1, 21)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(y_proba[y_test == 0], bins=bins, alpha=0.6, color=NEG_COLOR, label='Negative')
        ax.hist(y_proba[y_test == 1], bins=bins, alpha=0.6, color=POS_COLOR, label='Positive')
        ax.axvline(0.5, color='#333333', linestyle='--', lw=1, label='Decision threshold (0.5)')
        ax.set_xlabel('Predicted probability of being a real NES')
        ax.set_ylabel('Count (held-out test set)')
        ax.set_title('Predicted probability distribution by true class')
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_path = _ensure_out_dir(model_dir) / filename
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")
        return out_path


def generate_all_figures(model_dir, feature_names, X, y,
                          cv_report=None, importance_dict=None, importance_method='auto',
                          y_test=None, y_pred=None, y_proba=None):
    """
    Orchestrator: generate every figure this module knows how to make from
    whatever data is available. Called automatically at the end of
    ImprovedNESPredictor._train_model(). Never raises -- each individual
    plot function already catches its own missing-dependency/missing-data
    cases, and this wraps the whole batch in a try/except as a final safety
    net so a plotting bug can never break model training.
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("matplotlib is not installed -- no figures were generated. "
                       "Install with: pip install matplotlib --break-system-packages")
        return []

    print("\n" + "-" * 70)
    print("Generating training/validation figures...")
    print("-" * 70)

    saved = []
    try:
        saved.append(plot_feature_distributions(X, y, feature_names, model_dir))
        if cv_report:
            saved.append(plot_cv_model_comparison(cv_report, model_dir))
        if importance_dict:
            saved.append(plot_feature_importance(importance_dict, model_dir, method=importance_method))
        if y_test is not None and y_proba is not None:
            saved.append(plot_roc_curve(y_test, y_proba, model_dir))
            saved.append(plot_precision_recall_curve(y_test, y_proba, model_dir))
            saved.append(plot_probability_distribution(y_test, y_proba, model_dir))
        if y_test is not None and y_pred is not None:
            saved.append(plot_confusion_matrix(y_test, y_pred, model_dir))
    except Exception as e:
        print(f"  Warning: Figure generation encountered an error (training/model unaffected): {e}")

    saved = [p for p in saved if p is not None]
    print(f"{len(saved)} figure(s) saved to {model_dir}/figures/")
    print("-" * 70 + "\n")
    return saved
