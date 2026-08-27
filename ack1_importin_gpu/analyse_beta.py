#!/usr/bin/env python3
"""
Analysis of the beta-family MM-GBSA screen, alongside the importin-alpha panel.

The rule this script enforces:

    Absolute dG_total is NOT comparable between receptors. Different templates,
    different interface areas, different burial, different numbers of first-shell
    contacts. A receptor with a bigger groove gives every peptide a bigger number.

    What IS comparable across receptors is each receptor's OWN internal contrast:
        (a) matched-pair margin  = ACK1 sequence  minus  its composition-matched
            scramble, within that receptor. Identical composition, identical net
            charge, only the order differs -- so the difference cannot be a
            charge artefact.
        (b) discrimination       = known positives minus true negatives, i.e.
            whether the receptor's groove can tell any NLS from any junk.
        (c) seed dispersion      = SD across independent MD seeds. A groove that
            actually restrains the peptide gives reproducible numbers; one that
            does not gives scatter.

Every cross-receptor statement below is built on (a)-(c), never on raw ranking.
"""
import glob
import math
import os
import sys

import numpy as np

RESULTS = os.environ.get('ACK1_RESULTS', '.')

TRUE_NEG = {'NEG_polyAla', 'NEG_polyGln', 'NEG_ACK1_668-674'}
SCRAMBLE = {'NEG_scramble_of_71reg', 'NEG_scramble_of_64reg'}
POS = {'POS_SV40', 'POS_cMyc', 'POS_nucleoplasmin_major', 'POS_nucleoplasmin_minor'}


QC_DROPPED = []


def qc(r):
    """Reject physically impossible runs.

    dE_vdw is the Lennard-Jones interaction energy between receptor and peptide.
    For two bodies in contact but not interpenetrating it MUST be negative. A
    large positive value means the minimiser failed to resolve a steric overlap
    for that seed and the trajectory is sampling atoms on top of each other, so
    dG_total for that run is meaningless -- typically thousands of kcal/mol.

    This is a physics criterion, not an outlier filter tuned to the data: no
    threshold is fitted, the cut is at zero. Every rejection is printed.
    """
    if r['vdw'] > 0:
        QC_DROPPED.append(r)
        return False
    return True


def load(patterns, key_col=0):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(RESULTS, pat))):
            with open(path, errors='replace') as fh:
                head = None
                for line in fh:
                    if '\x00' in line:
                        continue
                    parts = line.rstrip('\n').split('\t')
                    if head is None:
                        head = parts
                        continue
                    if parts and parts[0] in (head[0],):
                        continue
                    if len(parts) != len(head):
                        continue
                    rows.append(dict(zip(head, parts)))
    out, seen = [], set()
    for r in rows:
        rec = r.get('receptor') or r.get('paralogue')
        try:
            g = float(r['dG_total'])
        except (KeyError, ValueError):
            continue
        # the finishing pod's beta_results.tsv sits alongside the six shards and the
        # globs overlap; a run must never be counted twice.
        uid = (rec, r['site'], r['peptide'], r['seed'])
        if uid in seen:
            continue
        seen.add(uid)
        rec_d = dict(rec=rec, site=r['site'], pep=r['peptide'], seq=r['seq'],
                     cls=r['class'], seed=int(r['seed']), dG=g,
                     vdw=float(r.get('dE_vdw', -1)), q=float(r['pep_charge']))
        if qc(rec_d):
            out.append(rec_d)
    return out


def by_pep(rows):
    d = {}
    for r in rows:
        d.setdefault((r['rec'], r['site'], r['pep']), []).append(r)
    return d


def mean_sem(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return math.nan, math.nan, 0
    if len(v) == 1:
        return float(v[0]), math.nan, 1
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))), len(v)


def welch(a, b):
    """Return (diff, se, t, approx two-sided p) for mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float(a.mean() - b.mean()), math.nan, math.nan, math.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    diff = float(a.mean() - b.mean())
    if se == 0:
        return diff, 0.0, math.inf, 0.0
    t = diff / se
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    # normal approximation to the t tail; df here is >= 4 in practice
    p = math.erfc(abs(t) / math.sqrt(2))
    return diff, se, t, p


CLUSTERS = {
    '71-73': ('ACK1_71-73', 'NEG_scramble_of_71reg'),
    '64-67': ('ACK1_64-67', 'NEG_scramble_of_64reg'),
}


def receptor_report(rows, rec, site):
    sel = [r for r in rows if r['rec'] == rec and r['site'] == site]
    if not sel:
        return None
    peps = by_pep(sel)
    rep = dict(rec=rec, site=site, n=len(sel))

    # ---- seed dispersion: median SD across peptides with >=3 seeds
    sds = [np.std([x['dG'] for x in v], ddof=1)
           for v in peps.values() if len(v) >= 3]
    rep['seed_sd'] = float(np.median(sds)) if sds else math.nan
    rep['n_multi'] = len(sds)

    # ---- discrimination: positives vs true negatives
    pos = [x['dG'] for k, v in peps.items() if k[2] in POS for x in v]
    neg = [x['dG'] for k, v in peps.items() if k[2] in TRUE_NEG for x in v]
    if pos and neg:
        d, se, t, p = welch(pos, neg)
        rep['pos'] = np.mean(pos)
        rep['neg'] = np.mean(neg)
        rep['discrim'] = d
        rep['discrim_se'] = se
        rep['discrim_p'] = p
        rep['n_pos'], rep['n_neg'] = len(pos), len(neg)

    # ---- matched pairs
    rep['pairs'] = {}
    for name, (ack_prefix, scr) in CLUSTERS.items():
        regs = {k[2]: [x['dG'] for x in v]
                for k, v in peps.items() if k[2].startswith(ack_prefix)}
        scr_v = [x['dG'] for k, v in peps.items() if k[2] == scr for x in v]
        if not regs or len(scr_v) < 2:
            continue
        # best register = most negative mean, as in the alpha panel
        best = min(regs.items(), key=lambda kv: np.mean(kv[1]))
        pooled = [g for v in regs.values() for g in v]
        d_best, se_b, t_b, p_b = welch(best[1], scr_v)
        d_pool, se_p, t_p, p_p = welch(pooled, scr_v)
        rep['pairs'][name] = dict(
            reg=best[0].replace(ack_prefix + '_', ''),
            best=np.mean(best[1]), scr=np.mean(scr_v),
            d_best=d_best, se_best=se_b, p_best=p_b, n_best=len(best[1]),
            d_pool=d_pool, se_pool=se_p, p_pool=p_p,
            n_scr=len(scr_v), n_reg=len(regs))

    # ---- mutant contrast (NOT charge-matched; report but flag)
    for wt_prefix, mut, tag in (('ACK1_71-73', 'MUT_71KRK73QQQ', '71-73'),
                                ('ACK1_64-67', 'MUT_64QQQQ67', '64-67')):
        wt = [x['dG'] for k, v in peps.items()
              if k[2].startswith(wt_prefix) for x in v]
        mv = [x['dG'] for k, v in peps.items() if k[2] == mut for x in v]
        if wt and len(mv) >= 2:
            d, se, t, p = welch(wt, mv)
            rep.setdefault('mut', {})[tag] = dict(d=d, se=se, p=p)

    # ---- how much of the spread is just net charge?
    qs = np.array([x['q'] for x in sel], float)
    gs = np.array([x['dG'] for x in sel], float)
    if len(set(qs.tolist())) > 2:
        r = np.corrcoef(qs, gs)[0, 1]
        rep['charge_r2'] = float(r ** 2)
        rep['charge_slope'] = float(np.polyfit(qs, gs, 1)[0])
    return rep


def fmt_p(p):
    if p != p:
        return '   n/a'
    if p < 1e-4:
        return '<1e-4'
    return f'{p:6.4f}'


def main():
    beta = load(['beta_results*.tsv'])
    alpha = load(['pod_results.tsv', 'screen_results*.tsv'])

    print('=' * 100)
    print('BETA-FAMILY SCREEN -- STATUS')
    print('=' * 100)
    if not beta:
        print('no beta rows found in', os.path.abspath(RESULTS))
        return
    combos = sorted({(r['rec'], r['site']) for r in beta})
    for rec, site in combos:
        n = len([r for r in beta if r['rec'] == rec and r['site'] == site])
        peps = len({r['pep'] for r in beta
                    if r['rec'] == rec and r['site'] == site})
        print(f'  {rec:<8} {site:<8} {n:>4} runs   {peps:>3} distinct peptides')
    print(f'  TOTAL {len(beta)} runs passing QC')

    if QC_DROPPED:
        print()
        print('  QC REJECTIONS (dE_vdw > 0 -- unresolved steric overlap, run is')
        print('  physically meaningless and is excluded from every statistic below)')
        for d in QC_DROPPED:
            print(f'    {d["rec"]:<8} {d["pep"]:<24} seed{d["seed"]}  '
                  f'dE_vdw = {d["vdw"]:+9.1f}   dG_total = {d["dG"]:+9.1f}')

    reports = []
    for rec, site in combos:
        rp = receptor_report(beta, rec, site)
        if rp:
            reports.append(rp)
    for rec in ('KPNA1', 'KPNA2', 'KPNA3', 'KPNA4'):
        for site in ('major', 'minor'):
            rp = receptor_report(alpha, rec, site)
            if rp:
                reports.append(rp)

    print()
    print('=' * 100)
    print('(c) SEED DISPERSION -- does the groove actually hold the peptide?')
    print('=' * 100)
    print('median SD of dG_total across independent MD seeds, per peptide')
    print(f'{"receptor":<16}{"site":<8}{"median seed SD":>16}{"peptides":>10}')
    print('-' * 52)
    for rp in reports:
        print(f'{rp["rec"]:<16}{rp["site"]:<8}{rp["seed_sd"]:>14.1f}  '
              f'{rp["n_multi"]:>9}')

    print()
    print('=' * 100)
    print('(b) DISCRIMINATION -- known cNLS positives vs true negatives')
    print('=' * 100)
    print(f'{"receptor":<16}{"site":<8}{"pos":>9}{"neg":>9}{"gap":>9}'
          f'{"+/-":>8}{"p":>8}')
    print('-' * 67)
    for rp in reports:
        if 'discrim' not in rp:
            continue
        print(f'{rp["rec"]:<16}{rp["site"]:<8}{rp["pos"]:>9.1f}{rp["neg"]:>9.1f}'
              f'{rp["discrim"]:>9.1f}{rp["discrim_se"]:>8.1f}'
              f'{fmt_p(rp["discrim_p"]):>8}')
    print('\nnegatives = polyAla, polyGln, ACK1 668-674 (an uncharged kinase-domain 7-mer)')
    print('positives = SV40, cMyc, nucleoplasmin')

    print()
    print('=' * 100)
    print('(a) MATCHED-PAIR MARGIN -- ACK1 minus its OWN composition-matched scramble')
    print('=' * 100)
    print('identical residues, identical net charge, order shuffled.')
    print('negative margin = the real sequence beats the scramble = order matters.\n')
    # Holm-Bonferroni across EVERY matched-pair test in the panel. Twenty tests are
    # run; at a nominal 0.05 one false positive is expected by chance alone, so an
    # uncorrected p near 0.05 cannot carry a claim.
    tests = [(rp, name) for name in ('71-73', '64-67') for rp in reports
             if name in rp['pairs'] and rp['pairs'][name]['p_best'] == rp['pairs'][name]['p_best']]
    order = sorted(range(len(tests)), key=lambda k: tests[k][0]['pairs'][tests[k][1]]['p_best'])
    m = len(tests)
    running = 0.0
    for rank, k in enumerate(order):
        rp, name = tests[k]
        p = rp['pairs'][name]['p_best']
        running = max(running, min(1.0, p * (m - rank)))
        rp['pairs'][name]['p_holm'] = running

    for name in ('71-73', '64-67'):
        print(f'--- cluster {name} ---')
        print(f'{"receptor":<16}{"site":<8}{"register":<12}{"ACK1":>9}'
              f'{"scramble":>10}{"margin":>9}{"+/-":>8}{"p":>8}{"p_holm":>8}'
              f'{"d":>7}  verdict')
        print('-' * 103)
        for rp in reports:
            pr = rp['pairs'].get(name)
            if not pr:
                continue
            # effect size: margin in units of this receptor's own seed noise, so
            # receptors with different interface sizes can be compared directly
            d = abs(pr['d_best']) / rp['seed_sd'] if rp['seed_sd'] else float('nan')
            ph = pr.get('p_holm', float('nan'))
            real = (pr['d_best'] < 0 and ph < 0.05)
            v = ('REAL' if real else
                 ('REVERSED' if pr['d_best'] > 0 else
                  ('ns after correction' if pr['p_best'] < 0.05 else 'ns')))
            print(f'{rp["rec"]:<16}{rp["site"]:<8}{pr["reg"]:<12}'
                  f'{pr["best"]:>9.1f}{pr["scr"]:>10.1f}{pr["d_best"]:>9.1f}'
                  f'{pr["se_best"]:>8.1f}{fmt_p(pr["p_best"]):>8}'
                  f'{fmt_p(ph):>8}{d:>7.1f}  {v}')
        print()
    print('p_holm = Holm-Bonferroni over all', m, 'matched-pair tests in the panel')
    print('d      = |margin| divided by that receptor\'s own median seed SD.')
    print('         Dimensionless, so it compares fairly across receptors whose')
    print('         interfaces differ in size and whose raw dG scales differ.\n')

    print('=' * 100)
    print('CHARGE ARTEFACT CHECK')
    print('=' * 100)
    print(f'{"receptor":<16}{"site":<8}{"R2(dG~q)":>10}{"kcal per +1":>14}')
    print('-' * 48)
    for rp in reports:
        if 'charge_r2' in rp:
            print(f'{rp["rec"]:<16}{rp["site"]:<8}{rp["charge_r2"]:>10.2f}'
                  f'{rp["charge_slope"]:>14.1f}')

    print()
    print('=' * 100)
    print('MUTANT CONTRAST (ACK1 wild-type minus Gln mutant) -- NOT charge-matched')
    print('=' * 100)
    print(f'{"receptor":<16}{"site":<8}{"71-73":>10}{"64-67":>10}')
    print('-' * 44)
    for rp in reports:
        m = rp.get('mut', {})
        a = f'{m["71-73"]["d"]:>10.1f}' if '71-73' in m else f'{"-":>10}'
        b = f'{m["64-67"]["d"]:>10.1f}' if '64-67' in m else f'{"-":>10}'
        print(f'{rp["rec"]:<16}{rp["site"]:<8}{a}{b}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        RESULTS = sys.argv[1]
    main()
