#!/usr/bin/env python3
"""
Figure C -- receptor specificity.

Does anything other than importin-alpha read the ACK1 basic cluster as a SEQUENCE,
rather than just as a lump of positive charge?

Ten receptor surfaces, one protocol, one peptide set: KPNA1-4 major and minor sites
(409 runs) plus importin-beta1 and transportin-1 (204 runs).

ONE VISUAL GRAMMAR, BOTH PANELS, NO EXCEPTIONS
    open dot     the control it is measured against
    filled dot   the test
    the line     the result -- long line = this receptor tells the two apart
    grey bar     +/- 2 x that receptor's own seed noise, drawn around the control

  (a) negatives -> positives      does the assay work on this receptor at all?
  (b) scramble  -> ACK1 71-73     does it tell the real sequence from a shuffle of
                                  the same residues?

Panel (a) is what makes (b) interpretable. Without it, a short line in (b) could just
mean the calculation failed on that receptor.

The grey bar in (b) replaces what used to be a third panel of margin/noise ratios. Same
information, shown in place and in the same units as the data, so no rescaled axis has
to be held in the head: if the filled dot clears the grey bar, the receptor is
separating sequence from composition by more than its own run-to-run scatter.

Cluster 64-67 is deliberately NOT here. Which of ACK1's basic clusters binds better is a
different question, answered by the importin-alpha panel; mixing it in forced the open
circle to mean two different things.

Usage:  python3 figures_ack1_receptors.py [results_dir] [outdir]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_plot_style import apply_style, style_axes, CAM, GRID_GREY

import analyse_beta as A

apply_style()

RESULTS = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'

ALPHA_C = CAM['dark_blue_strong']
BETA_C = CAM['warm_blue']
NOISE_C = CAM['slate1']
AX_C = CAM['slate3']

PRETTY = {('IMPB1', 'groove'): 'importin-β1',
          ('TNPO1', 'groove'): 'transportin-1'}

MS = 4.2          # marker size
LW = 1.5          # connector width


def collect():
    A.RESULTS = RESULTS
    A.QC_DROPPED.clear()
    beta = A.load(['beta_results*.tsv'])
    alpha = A.load(['pod_results.tsv', 'screen_results*.tsv'])

    reports = []
    for rec in ('KPNA1', 'KPNA2', 'KPNA3', 'KPNA4'):
        for site in ('major', 'minor'):
            rp = A.receptor_report(alpha, rec, site)
            if rp:
                reports.append(rp)
    for rec, site in sorted({(r['rec'], r['site']) for r in beta}):
        rp = A.receptor_report(beta, rec, site)
        if rp:
            reports.append(rp)

    tests = [(rp, n) for n in ('71-73', '64-67') for rp in reports
             if n in rp['pairs'] and rp['pairs'][n]['p_best'] == rp['pairs'][n]['p_best']]
    order = sorted(range(len(tests)),
                   key=lambda k: tests[k][0]['pairs'][tests[k][1]]['p_best'])
    m, run = len(tests), 0.0
    for rank, k in enumerate(order):
        rp, n = tests[k]
        run = max(run, min(1.0, rp['pairs'][n]['p_best'] * (m - rank)))
        rp['pairs'][n]['p_holm'] = run
    return reports


def label(rp):
    return PRETTY.get((rp['rec'], rp['site']), f"{rp['rec']}  {rp['site']}")


def stars(p):
    if p != p:
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def dumbbell(ax, ypos, lo, hi, cols):
    """open dot at `lo` (control), filled dot at `hi` (test), joined by a line."""
    for y, a, b, c in zip(ypos, lo, hi, cols):
        ax.plot([a, b], [y, y], color=c, lw=LW, solid_capstyle='round', zorder=1)
        ax.plot([a], [y], 'o', ms=MS, mfc='white', mec=c, mew=1.2, zorder=2)
        ax.plot([b], [y], 'o', ms=MS, color=c, zorder=3)


def main():
    reports = collect()
    alpha = [r for r in reports if r['rec'].startswith('KPNA')]
    beta = [r for r in reports if not r['rec'].startswith('KPNA')]
    rows = alpha + [None] + beta
    ypos, ylab, kinds, y = [], [], [], 0.0
    for r in rows:
        if r is None:
            y += 0.85
            continue
        ypos.append(y); ylab.append(label(r))
        kinds.append('a' if r['rec'].startswith('KPNA') else 'b')
        y += 1.0
    ypos = np.array(ypos)
    items = [r for r in rows if r is not None]
    cols = [ALPHA_C if k == 'a' else BETA_C for k in kinds]
    sep = ypos[len(alpha) - 1] + 0.5 + 0.42

    TS, LS = 7.6, 7.0
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), sharey=True,
                             gridspec_kw=dict(wspace=0.10))

    # ------------------------------------------- (a) does the assay work at all?
    ax = axes[0]
    dumbbell(ax, ypos, [r['neg'] for r in items], [r['pos'] for r in items], cols)
    ax.set_xlabel('MM-GBSA score (kcal mol$^{-1}$)', fontsize=LS)
    ax.set_title('a   every receptor separates real NLSs from junk',
                 loc='left', fontsize=TS, pad=7)
    ax.set_ylim(ypos[-1] + 0.75, ypos[0] - 0.75)      # headroom, and inverts
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=LS)
    style_axes(ax)

    # ------------------------------------------- (b) does it read the sequence?
    ax = axes[1]
    lo = [r['pairs']['71-73']['scr'] for r in items]
    hi = [r['pairs']['71-73']['best'] for r in items]
    # +/- 2 x seed noise around the control, drawn first so the dumbbell sits on top
    for yy, r, a in zip(ypos, items, lo):
        sd = r['seed_sd']
        ax.plot([a - 2 * sd, a + 2 * sd], [yy, yy], lw=5.5, color=NOISE_C,
                solid_capstyle='butt', zorder=0)
    dumbbell(ax, ypos, lo, hi, cols)
    left = min(min(hi), min(a - 2 * r['seed_sd'] for r, a in zip(items, lo)))
    right = max(a + 2 * r['seed_sd'] for r, a in zip(items, lo))
    span = right - left
    ax.set_xlim(left - span * 0.15, right + span * 0.03)
    # significance in a single aligned column, not floating at each dumbbell
    for yy, r in zip(ypos, items):
        ax.text(left - span * 0.045, yy,
                stars(r['pairs']['71-73'].get('p_holm', float('nan'))),
                ha='center', va='center', fontsize=6.2, color=AX_C)
    ax.set_xlabel('MM-GBSA score (kcal mol$^{-1}$)', fontsize=LS)
    ax.set_title('b   only importin-$\\alpha$ separates ACK1 from its own scramble',
                 loc='left', fontsize=TS, pad=7)
    style_axes(ax)

    for a_ in axes:
        a_.tick_params(labelsize=LS)
        a_.axhline(sep, color=GRID_GREY, lw=0.6, ls=(0, (3, 3)), zorder=0)

    handles = [
        Line2D([], [], color=ALPHA_C, lw=LW, marker='o', ms=MS,
               label='importin-$\\alpha$ (KPNA1-4)'),
        Line2D([], [], color=BETA_C, lw=LW, marker='o', ms=MS,
               label='$\\beta$-family receptor'),
        Line2D([], [], color=AX_C, lw=0, marker='o', ms=MS, mfc='white',
               mec=AX_C, mew=1.2,
               label='open dot = control (negatives in a, scramble in b)'),
        Line2D([], [], color=NOISE_C, lw=5.5, solid_capstyle='butt',
               label='$\\pm$2 $\\times$ seed noise'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=LS, bbox_to_anchor=(0.5, -0.01), handletextpad=0.6,
               columnspacing=2.2)

    fig.subplots_adjust(left=0.17, right=0.99, top=0.90, bottom=0.28)
    fig.savefig(f'{OUT}/fig_ack1_receptor_specificity.png', dpi=300,
                facecolor='white')
    plt.close(fig)
    print(f'wrote {OUT}/fig_ack1_receptor_specificity.png')

    print('\nvalues plotted (kcal/mol)')
    print(f'{"receptor":<22}{"neg":>8}{"pos":>8}{"scr":>8}{"ACK1":>8}'
          f'{"margin":>8}{"p_holm":>9}{"d":>7}')
    for r in items:
        p1 = r['pairs']['71-73']
        print(f'{label(r):<22}{r["neg"]:>8.1f}{r["pos"]:>8.1f}{p1["scr"]:>8.1f}'
              f'{p1["best"]:>8.1f}{p1["d_best"]:>8.1f}'
              f'{p1.get("p_holm", float("nan")):>9.4f}'
              f'{p1["d_best"] / r["seed_sd"]:>7.1f}')


if __name__ == '__main__':
    main()
