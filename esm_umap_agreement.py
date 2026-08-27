#!/usr/bin/env python3
"""
esm_umap_agreement.py
============================================================
Two analyses on top of esm_embed_sequences.py + esm_model_comparison.py:

1. UMAP of the frozen ESM2-150M embedding space, coloured by positive/
   negative label -- shows whether the pretrained language model already
   separates real NES/NLS motifs from background sequence *before* any
   classifier is trained on top of it. This is a legitimate, standard
   sanity check for embedding-based features (as seen in the ESM/protein-
   LM literature) and is worth including in your thesis for that reason.

2. Spearman rank agreement between the hand-engineered-feature classifier
   and the ESM-only classifier's predicted probabilities on the same
   examples (out-of-fold, from esm_model_comparison.py's cross-validation)
   -- NOTE this is deliberately NOT the same thing as the Spearman-vs-
   experimental-value correlation you may have seen in DMS/mutational-
   scanning papers (those compare a model's score against a *measured*
   continuous label, which you don't have here since NES/NLS is a binary
   classification task). What this instead answers is a different, still
   useful question: "do the two feature representations rank candidate
   sites the same way, or are they picking up genuinely different
   signal?" A high correlation (~1.0) would suggest ESM is mostly
   rediscovering the same information your hand-engineered features
   already capture; a low/moderate one suggests it's contributing
   something the hand-engineered features miss.

Run this AFTER esm_model_comparison.py (needs its oof_predictions.npz).

Usage:
    python3 esm_umap_agreement.py
    python3 esm_umap_agreement.py --embeddings esm_embeddings --comparison esm_comparison --out esm_umap

Requires: pip install umap-learn scipy

Outputs (written to --out, default 'esm_umap/'):
    umap_<target>.png                  2D UMAP, coloured by label
    agreement_<target>.png             scatter of hand-eng vs ESM-only OOF
                                        probability, one panel per classifier,
                                        annotated with Spearman rho + p
    rank_agreement_summary.csv         one row per (target, classifier):
                                        Spearman rho, p-value, n
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from thesis_plot_style import apply_style, style_axes, CAM, CATEGORICAL, sig_stars, save_fig


def _labels_for_sequences(target):
    """Re-derives a seq -> label dict from the same build_training_dataset()
    call used everywhere else, so UMAP colouring is guaranteed consistent
    with what was actually trained on. If a sequence appears as both a
    positive and a negative window (possible with overlapping negative
    sampling) it's counted as positive and flagged."""
    if target == 'nes':
        from nes_ml_predictor_improved import ImprovedNESPredictor
        predictor = ImprovedNESPredictor()
    else:
        from nls_ml_predictor import NLSPredictor
        predictor = NLSPredictor()
    dataset = predictor.build_training_dataset()
    labels = {}
    ambiguous = 0
    for n in dataset['negatives']:
        labels[n['seq'].upper()] = 0
    for p in dataset['positives']:
        s = p['seq'].upper()
        if s in labels and labels[s] == 0:
            ambiguous += 1
        labels[s] = 1
    if ambiguous:
        print(f"  [{target}] note: {ambiguous} sequences appear in both positive and negative "
              f"sets -- labelled positive for UMAP colouring, flagging for awareness only.")
    return labels


def make_umap(target, embeddings_dir, out_dir):
    try:
        import umap
    except ImportError:
        print(f"  [{target}] umap-learn not installed (pip install umap-learn) -- skipping UMAP.")
        return

    npz = np.load(Path(embeddings_dir) / f'{target}_embeddings.npz', allow_pickle=True)
    seqs, vecs = npz['sequences'], npz['vectors']
    labels_dict = _labels_for_sequences(target)
    labels = np.array([labels_dict.get(s, -1) for s in seqs])
    keep = labels >= 0
    seqs, vecs, labels = seqs[keep], vecs[keep], labels[keep]

    print(f"  [{target}] running UMAP on {vecs.shape[0]} sequences ({vecs.shape[1]}-dim)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb2d = reducer.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for lab, name, color in [(0, 'Negative', CATEGORICAL[1]), (1, 'Positive', CATEGORICAL[0])]:
        mask = labels == lab
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], s=10, alpha=0.65, color=color,
                   edgecolors=CAM['slate4'], linewidths=0.2, label=f'{name} (n={mask.sum()})')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_title(f'{target.upper()}: frozen ESM2-150M embedding space')
    ax.legend(fontsize=7)
    style_axes(ax)
    save_fig(fig, out_dir / f'umap_{target}.png')
    print(f"  [{target}] saved umap_{target}.png")


def make_agreement(target, comparison_dir, out_dir, summary_rows):
    npz_path = Path(comparison_dir) / 'oof_predictions.npz'
    if not npz_path.exists():
        print(f"  [{target}] {npz_path} not found -- run esm_model_comparison.py first. Skipping agreement analysis.")
        return
    data = np.load(npz_path, allow_pickle=True)
    y = data[f'{target}_y']

    classifiers = sorted({k.split('__')[2] for k in data.files
                           if k.startswith(f'{target}__hand_engineered__')})
    if not classifiers:
        print(f"  [{target}] no cached OOF predictions found in {npz_path}.")
        return

    n_clf = len(classifiers)
    fig, axes = plt.subplots(1, n_clf, figsize=(4 * n_clf, 4), squeeze=False)
    axes = axes[0]

    for ax, clf_name in zip(axes, classifiers):
        hand_key = f'{target}__hand_engineered__{clf_name}'
        esm_key = f'{target}__esm_only__{clf_name}'
        if hand_key not in data.files or esm_key not in data.files:
            ax.set_visible(False)
            continue
        hand_proba = data[hand_key]
        esm_proba = data[esm_key]
        rho, p = spearmanr(hand_proba, esm_proba)
        summary_rows.append({'target': target, 'classifier': clf_name, 'spearman_rho': rho,
                              'p_value': p, 'n': len(hand_proba), 'significance': sig_stars(p)})

        colors = [CATEGORICAL[0] if lab == 1 else CATEGORICAL[1] for lab in y]
        ax.scatter(hand_proba, esm_proba, s=10, alpha=0.6, c=colors, edgecolors=CAM['slate4'], linewidths=0.2)
        ax.plot([0, 1], [0, 1], color=CAM['slate3'], lw=0.8, ls='--')
        ax.set_xlabel('Hand-engineered P(positive)')
        ax.set_ylabel('ESM-only P(positive)')
        ax.set_title(f'{clf_name}\n' + r'Spearman $\rho$=' + f'{rho:.3f} ({sig_stars(p)})', fontsize=8)
        style_axes(ax)

    save_fig(fig, out_dir / f'agreement_{target}.png')
    print(f"  [{target}] saved agreement_{target}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--embeddings', default='esm_embeddings')
    ap.add_argument('--comparison', default='esm_comparison')
    ap.add_argument('--out', default='esm_umap')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    summary_rows = []
    for target in ('nes', 'nls'):
        print(f"\n{target.upper()}")
        make_umap(target, args.embeddings, out_dir)
        make_agreement(target, args.comparison, out_dir, summary_rows)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / 'rank_agreement_summary.csv', index=False)
        print(f"\n{out_dir}/rank_agreement_summary.csv")
        print(summary_df.to_string(index=False))

    print(f"\nAll figures in {out_dir}/")


if __name__ == '__main__':
    main()
