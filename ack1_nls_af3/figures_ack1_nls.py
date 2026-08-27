#!/usr/bin/env python3
"""
Thesis figures for the ACK1 NLS computational work, in the house style.

  Figure A  the 409-run MM-GBSA panel: calibration, the matched-pair result,
            the charge bias, and the paralogue comparison
  Figure B  the mechanism: accessibility versus alpha5 unwinding, and the total
            cost of presenting each basic cluster in monomer versus dimer

Usage:  python3 figures_ack1_nls.py [results.tsv] [outdir]
"""
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_plot_style import (apply_style, style_axes, CAM, save_fig)

apply_style()

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'pod_results.tsv'
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'
KCAL = 1 / 4.184

# house palette, mapped onto the four peptide classes
CLASS_COLOUR = {'ack1': CAM['dark_blue_strong'],
                'mutant': CAM['warm_blue'],
                'pos': CAM['light_blue_strong'],
                'neg': CAM['slate2']}
CLASS_LABEL = {'ack1': 'ACK1 register', 'mutant': 'ACK1 mutant',
               'pos': 'positive control', 'neg': 'negative control'}
PARALOGUES = ('KPNA1', 'KPNA2', 'KPNA3', 'KPNA4')

PRETTY = {
    'ACK1_64-67_K64atP2': 'ACK1 64-67 (K64 P2)',
    'ACK1_64-67_R65atP2': 'ACK1 64-67 (R65 P2)',
    'ACK1_71-73_K71atP2': 'ACK1 71-73 (K71 P2)',
    'ACK1_71-73_R72atP2': 'ACK1 71-73 (R72 P2)',
    'ACK1_71-73_K73atP2': 'ACK1 71-73 (K73 P2)',
    'ACK1_R57R58_R58atP2': 'ACK1 57-58 (R58 P2)',
    'MUT_71KRK73QQQ': '71KRK73>QQQ',
    'MUT_64QQQQ67': '64QQQQ67',
    'MUT_64EEEE67': '64EEEE67',
    'POS_SV40': 'SV40',
    'POS_nucleoplasmin_major': 'nucleoplasmin',
    'POS_cMyc': 'c-Myc',
    'NEG_ACK1_668-674': 'ACK1 668-674',
    'NEG_scramble_of_71reg': 'scramble (71-73)',
    'NEG_scramble_of_64reg': 'scramble (64-67)',
    'NEG_polyAla': 'poly-Ala',
    'NEG_polyGln': 'poly-Gln',
}
PAIRS = [('ACK1_71-73_K71atP2', 'NEG_scramble_of_71reg', '71-73'),
         ('ACK1_64-67_K64atP2', 'NEG_scramble_of_64reg', '64-67')]


def load(path):
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip('\n').split('\t')
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) != len(hdr):
                continue
            d = dict(zip(hdr, f))
            d['dG'] = float(d['dG_total']) * KCAL
            d['q'] = int(d['pep_charge'])
            rows.append(d)
    return rows


ROWS = load(RESULTS)
BY = defaultdict(list)
for r in ROWS:
    BY[(r['paralogue'], r['site'], r['peptide'])].append(r['dG'])
CLS = {r['peptide']: r['class'] for r in ROWS}
CHG = {r['peptide']: r['q'] for r in ROWS}


def mean_over_paralogues(site, pep):
    v = [np.mean(BY[(p, site, pep)]) for p in PARALOGUES if BY[(p, site, pep)]]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


# ============================================================== FIGURE A
def figure_a():
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.34,
                          height_ratios=[1.7, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- (a) ranking, both sites, averaged over the four paralogues
    peps = sorted(CLS, key=lambda p: mean_over_paralogues('major', p)[0])
    y = np.arange(len(peps))
    h = 0.38
    for k, (site, hatch, alpha) in enumerate((('major', None, 1.0),
                                              ('minor', '///', 0.55))):
        means = [mean_over_paralogues(site, p)[0] for p in peps]
        errs = [mean_over_paralogues(site, p)[1] for p in peps]
        cols = [CLASS_COLOUR[CLS[p]] for p in peps]
        ax_a.barh(y + (h / 2 if k == 0 else -h / 2), means, height=h,
                  color=cols, alpha=alpha, hatch=hatch,
                  edgecolor='white', linewidth=0.4,
                  xerr=errs, error_kw=dict(ecolor=CAM['slate3'], lw=0.7,
                                           capsize=1.5))
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([PRETTY.get(p, p) for p in peps], fontsize=7)
    ax_a.set_xlabel(r'MM-GBSA $\Delta G$ (kcal mol$^{-1}$, relative)')
    ax_a.axvline(0, color=CAM['slate3'], lw=0.6)
    ax_a.set_title('a   Peptide ranking, mean of KPNA1-4', loc='left',
                   fontsize=9, fontweight='bold')
    style_axes(ax_a)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOUR[c])
               for c in ('ack1', 'mutant', 'pos', 'neg')]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=CAM['slate3'], alpha=1.0),
                plt.Rectangle((0, 0), 1, 1, fc=CAM['slate3'], alpha=0.55,
                              hatch='///')]
    labels = [CLASS_LABEL[c] for c in ('ack1', 'mutant', 'pos', 'neg')]
    labels += ['major site', 'minor site']
    ax_a.legend(handles, labels, fontsize=6.4, ncol=6, handlelength=1.1,
                frameon=False, loc='upper center',
                bbox_to_anchor=(0.5, -0.115), columnspacing=1.1)

    # ---- (b) matched pairs
    xs, off = [], {'major': -0.11, 'minor': 0.11}
    for j, (ack, scr, tag) in enumerate(PAIRS):
        for site in ('major', 'minor'):
            for i, pa in enumerate(PARALOGUES):
                a = np.mean(BY[(pa, site, ack)])
                b = np.mean(BY[(pa, site, scr)])
                x = j + off[site]
                ax_b.plot([x, x], [b, a], color=CAM['slate2'], lw=0.7,
                          zorder=1)
                ax_b.scatter([x], [a], s=16, color=CAM['dark_blue_strong'],
                             zorder=3, edgecolor='white', linewidth=0.4)
                ax_b.scatter([x], [b], s=16, color=CAM['slate2'], zorder=3,
                             edgecolor='white', linewidth=0.4)
    ticks, ticklabels = [], []
    for j, (_, _, tag) in enumerate(PAIRS):
        for site in ('major', 'minor'):
            ticks.append(j + off[site]); ticklabels.append(site)
    ax_b.set_xticks(ticks)
    ax_b.set_xticklabels(ticklabels, fontsize=6.5)
    for j, (ack, scr, tag) in enumerate(PAIRS):
        gaps = [np.mean(BY[(pa, st, ack)]) - np.mean(BY[(pa, st, scr)])
                for st in ('major', 'minor') for pa in PARALOGUES]
        ax_b.annotate(tag, xy=(j, 0.965), xycoords=('data', 'axes fraction'),
                      ha='center', fontsize=7.5, color=CAM['slate3'])
        ax_b.annotate(f'mean {np.mean(gaps):+.1f}', xy=(j, -0.16),
                      xycoords=('data', 'axes fraction'), ha='center',
                      fontsize=6.3, color=CAM['dark_blue_strong'])
    ax_b.set_ylabel(r'$\Delta G$ (kcal mol$^{-1}$)')
    ax_b.set_title('b   Composition-matched pairs', loc='left', fontsize=9,
                   fontweight='bold', pad=6)
    ax_b.legend([Line2D([], [], marker='o', ls='', color=CAM['dark_blue_strong']),
                 Line2D([], [], marker='o', ls='', color=CAM['slate2'])],
                ['ACK1', 'scramble'], fontsize=6.5, frameon=True,
                loc='lower right')
    style_axes(ax_b)

    # ---- (c) charge regression
    pts = defaultdict(lambda: ([], []))
    for p in CLS:
        for site in ('major', 'minor'):
            m, _ = mean_over_paralogues(site, p)
            if np.isfinite(m):
                pts[CLS[p]][0].append(CHG[p])
                pts[CLS[p]][1].append(m)
    allq, allg = [], []
    for c, (q, g) in pts.items():
        ax_c.scatter(q, g, s=14, color=CLASS_COLOUR[c], alpha=0.85,
                     edgecolor='white', linewidth=0.3, label=CLASS_LABEL[c])
        allq += q; allg += g
    allq, allg = np.array(allq, float), np.array(allg, float)
    A = np.vstack([allq, np.ones_like(allq)]).T
    sol = np.linalg.lstsq(A, allg, rcond=None)[0]
    r2 = 1 - np.sum((allg - A @ sol) ** 2) / np.sum((allg - allg.mean()) ** 2)
    xr = np.linspace(allq.min() - 0.4, allq.max() + 0.4, 20)
    ax_c.plot(xr, sol[0] * xr + sol[1], color=CAM['dark_crest'], lw=1.1,
              zorder=1)
    ax_c.text(0.04, 0.06, f'$R^2$ = {r2:.2f}', transform=ax_c.transAxes,
              fontsize=7.5, color=CAM['dark_crest'])
    ax_c.set_xlabel('net peptide charge')
    ax_c.set_ylabel(r'$\Delta G$ (kcal mol$^{-1}$)')
    ax_c.set_title('c   Charge dependence', loc='left', fontsize=9,
                   fontweight='bold')
    style_axes(ax_c)

    save_fig(fig, f'{OUT}/fig_ack1_mmgbsa_panel.png')
    print('wrote fig_ack1_mmgbsa_panel.png')


# ============================================================== FIGURE B
UNWIND = {1: 1.18, 2: 2.58, 3: 5.68, 4: 10.35, 5: 13.59, 6: 16.11}
TETHER = {
    ('minor', 'K71-R72-K73'): {1: (0.1125, 0.1110), 2: (0.2065, 0.1923),
                               3: (0.2755, 0.2162), 4: (0.3287, 0.2642),
                               5: (0.2390, 0.2028), 6: (0.2893, 0.1872)},
    ('major', 'K71-R72-K73'): {1: (0.0003, 0.0000), 2: (0.0075, 0.0063),
                               3: (0.0375, 0.0138), 4: (0.0470, 0.0220),
                               5: (0.0290, 0.0165), 6: (0.0535, 0.0205)},
    ('minor', 'R57-R58'):     {1: (0.0693, 0.0000), 2: (0.1507, 0.0003),
                               3: (0.3108, 0.0025), 4: (0.2630, 0.0070),
                               5: (0.3405, 0.0683), 6: (0.2717, 0.1123)},
    ('major', 'K64-K67'):     {3: (0.0090, 0.0000), 4: (0.0302, 0.0013),
                               5: (0.0127, 0.0003), 6: (0.0220, 0.0003)},
}
RT = 0.593
TOTALS = {'K71-R72-K73\nminor site': (2.48, 2.49),
          'K71-R72-K73\nmajor site': (5.48, 5.59),
          'R57-R58\nminor site': (2.77, 7.39),
          'K64-K67\nmajor site': (8.47, 14.29)}


def figure_b():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # ---- (a) accessibility versus unwinding
    styles = [(('minor', 'K71-R72-K73'), CAM['dark_blue_strong'], '-', 'o'),
              (('major', 'K71-R72-K73'), CAM['light_blue_strong'], '-', 's'),
              (('minor', 'R57-R58'), CAM['warm_blue'], '--', '^'),
              (('major', 'K64-K67'), CAM['slate3'], '--', 'v')]
    for key, col, ls, mk in styles:
        us = sorted(TETHER[key])
        ax1.plot(us, [TETHER[key][u][0] * 100 for u in us], ls=ls, marker=mk,
                 ms=3.4, lw=1.2, color=col,
                 label=f'{key[1]}, {key[0]}')
        ax1.plot(us, [TETHER[key][u][1] * 100 for u in us], ls=':', marker=mk,
                 ms=2.4, lw=0.9, color=col, alpha=0.55)
    ax1.set_xlabel(r'residues of SAM $\alpha$5 unwound ($u$)')
    ax1.set_ylabel('clash-free conformations (%)')
    ax1.set_title(r'a   Accessibility vs $\alpha$5 fraying', loc='left',
                  fontsize=9, fontweight='bold')
    leg1 = ax1.legend(fontsize=5.8, frameon=True, loc='upper left',
                      title='solid = monomer,  dotted = dimer',
                      title_fontsize=5.8)
    leg1.get_title().set_color(CAM['slate3'])
    ax1.set_ylim(-1.5, 40)
    style_axes(ax1)

    # ---- (b) total presentation cost, monomer vs dimer
    labels = list(TOTALS)
    x = np.arange(len(labels))
    w = 0.36
    mono = [TOTALS[k][0] for k in labels]
    dim = [TOTALS[k][1] for k in labels]
    ax2.bar(x - w / 2, mono, w, color=CAM['light_blue_strong'],
            edgecolor='white', linewidth=0.5, label='monomer')
    ax2.bar(x + w / 2, dim, w, color=CAM['dark_blue_strong'],
            edgecolor='white', linewidth=0.5, label='SAM dimer')
    for xi, (m, d) in enumerate(zip(mono, dim)):
        if d - m <= 0.5:
            ax2.text(xi, max(m, d) + 0.45, 'no change', fontsize=5.8,
                     ha='center', color=CAM['slate3'])
        if d - m > 1.0:
            ax2.annotate('', xy=(xi + w / 2, d), xytext=(xi + w / 2, m),
                         arrowprops=dict(arrowstyle='->', color=CAM['dark_crest'],
                                         lw=0.9))
            ax2.text(xi + w / 2 + 0.06, (m + d) / 2, f'+{d - m:.1f}',
                     fontsize=6, color=CAM['dark_crest'], va='center')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.2)
    ax2.set_ylabel(r'cost of presentation (kcal mol$^{-1}$)')
    ax2.set_title('b   Dimerisation suppresses the competitors', loc='left',
                  fontsize=9, fontweight='bold')
    ax2.legend(fontsize=6.5, frameon=True, loc='upper left')
    style_axes(ax2)

    save_fig(fig, f'{OUT}/fig_ack1_mechanism_panel.png')
    print('wrote fig_ack1_mechanism_panel.png')


# ============================================================== FIGURE C
def figure_c():
    """Paralogue comparison - the negative result, on its own axes."""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    peps = [('ACK1_71-73_K71atP2', CAM['dark_blue_strong'], 'o', '71-73'),
            ('ACK1_64-67_K64atP2', CAM['light_blue_strong'], 's', '64-67'),
            ('POS_SV40', CAM['slate3'], '^', 'SV40')]
    x = np.arange(len(PARALOGUES))
    for pep, col, mk, lab in peps:
        for site, ls, alpha in (('major', '-', 1.0), ('minor', '--', 0.6)):
            m = [np.mean(BY[(p, site, pep)]) for p in PARALOGUES]
            e = [np.std(BY[(p, site, pep)]) for p in PARALOGUES]
            ax.errorbar(x, m, yerr=e, ls=ls, marker=mk, ms=3.6, lw=1.1,
                        color=col, alpha=alpha, capsize=2, elinewidth=0.7,
                        label=f'{lab} ({site})')
    ax.set_xticks(x)
    ax.set_xticklabels(PARALOGUES, fontsize=7.5)
    ax.set_ylabel(r'$\Delta G$ (kcal mol$^{-1}$)')
    ax.set_title('Paralogue comparison', loc='left', fontsize=9,
                 fontweight='bold')
    ax.legend(fontsize=5.6, frameon=True, ncol=2, loc='center right')
    style_axes(ax)
    save_fig(fig, f'{OUT}/fig_ack1_paralogues.png')
    print('wrote fig_ack1_paralogues.png')


if __name__ == '__main__':
    figure_a()
    figure_b()
    figure_c()
