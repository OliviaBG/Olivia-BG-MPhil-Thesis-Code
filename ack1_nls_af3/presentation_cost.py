#!/usr/bin/env python3
"""
Total free-energy cost of presenting each ACK1 basic cluster to importin-alpha,
in the monomer and in the SAM dimer.

Three terms, computed independently and summed because they are sequential steps:

  1. dG_helix(u)     intrinsic helix-coil cost of fraying u residues of SAM alpha5
                     (Lifson-Roig, Pace-Scholtz propensities)
  2. dG_packing(u)   cost of stripping those residues off the SAM core
                     (buried surface x 0.025 kcal/mol/A^2)
  3. -RT ln f(u)     conformational cost of then placing the folded remainder somewhere
                     the receptor is not (tethered-ensemble sampling)

Minimising the sum over u gives the cheapest route to a bound state; the u that
achieves it predicts how far alpha5 actually has to fray.

This is an order-of-magnitude free energy, not a measured affinity. The packing term
dominates and rests on a single burial coefficient; the ensemble term is backbone-only.
Treat differences between clusters as meaningful and absolute values as indicative.
"""
import numpy as np
import helix_cost as H

RT = 0.593
SIGMA = 0.025

TETHER = {
    ('minor site', 'K71-R72-K73'): {1: (0.1125, 0.1110), 2: (0.2065, 0.1923),
                                    3: (0.2755, 0.2162), 4: (0.3287, 0.2642),
                                    5: (0.2390, 0.2028), 6: (0.2893, 0.1872)},
    ('major site', 'K71-R72-K73'): {1: (0.0003, 0.0000), 2: (0.0075, 0.0063),
                                    3: (0.0375, 0.0138), 4: (0.0470, 0.0220),
                                    5: (0.0290, 0.0165), 6: (0.0535, 0.0205)},
    ('minor site', 'R57-R58'):     {1: (0.0693, 0.0000), 2: (0.1507, 0.0003),
                                    3: (0.3108, 0.0025), 4: (0.2630, 0.0070),
                                    5: (0.3405, 0.0683), 6: (0.2717, 0.1123)},
    ('major site', 'K64-K67'):     {3: (0.0090, 0.0000), 4: (0.0302, 0.0013),
                                    5: (0.0127, 0.0003), 6: (0.0220, 0.0003)},
}


def main():
    bur = H.tertiary_cost()
    pack, run = {}, 0.0
    for u, i in enumerate(range(H.A5_END, H.A5_END - 9, -1), start=1):
        if i in bur:
            run += bur[i][1] * SIGMA
            pack[u] = run

    print('=' * 92)
    print('COST OF PRESENTING EACH BASIC CLUSTER TO IMPORTIN-ALPHA (kcal/mol)')
    print('=' * 92)
    print('u = residues of SAM alpha5 that must unwind\n')
    print(f'{"u":>3}{"helix":>8}{"packing":>10}{"unwind total":>14}')
    print('-' * 36)
    for u in sorted(pack):
        print(f'{u:>3}{H.dG_unwind(u):>8.2f}{pack[u]:>10.2f}'
              f'{H.dG_unwind(u) + pack[u]:>14.2f}')

    print('\n' + '=' * 92)
    print(f'{"cluster / site":<28}{"u*":>4}{"unwind":>9}{"place":>8}{"TOTAL":>9}'
          f'{"  |":>4}{"u*":>4}{"unwind":>9}{"place":>8}{"TOTAL":>9}')
    print(f'{"":<28}{"--------- MONOMER ---------":>30}'
          f'{"  |":>4}{"---------- DIMER ----------":>30}')
    print('-' * 92)
    best = {}
    for (site, cluster), fs in TETHER.items():
        row = {}
        for tag in (0, 1):
            cands = []
            for u, fpair in sorted(fs.items()):
                f = fpair[tag]
                if f <= 0 or u not in pack:
                    continue
                unwind = H.dG_unwind(u) + pack[u]
                place = -RT * np.log(f)
                cands.append((unwind + place, u, unwind, place))
            row[tag] = min(cands) if cands else None
        best[(site, cluster)] = row
        label = f'{cluster}  {site}'
        m, d = row[0], row[1]
        ms = (f'{m[1]:>4}{m[2]:>9.2f}{m[3]:>8.2f}{m[0]:>9.2f}' if m
              else f'{"-":>4}{"-":>9}{"-":>8}{"blocked":>9}')
        ds = (f'{d[1]:>4}{d[2]:>9.2f}{d[3]:>8.2f}{d[0]:>9.2f}' if d
              else f'{"-":>4}{"-":>9}{"-":>8}{"blocked":>9}')
        print(f'{label:<28}{ms}{"  |":>4}{ds}')

    print('\n' + '=' * 92)
    print('WHAT THIS SAYS')
    print('=' * 92)
    k71m = best[('minor site', 'K71-R72-K73')]
    k71M = best[('major site', 'K71-R72-K73')]
    r58 = best[('minor site', 'R57-R58')]
    k64 = best[('major site', 'K64-K67')]

    def fold(a, b):
        return np.exp((b - a) / RT)

    print(f'\n1. K71-R72-K73 prefers the MINOR site by '
          f'{k71M[0][0] - k71m[0][0]:.1f} kcal/mol '
          f'({fold(k71m[0][0], k71M[0][0]):.0f}-fold), and needs only '
          f'u={k71m[0][1]} residue(s) of alpha5 to fray.')
    print(f'   Most monopartite cNLSs prefer the major site. ACK1 cannot, because a '
          f'folded domain\n   sits two residues away.')

    print(f'\n2. In the DIMER, K71-R72-K73 costs {k71m[1][0]:.2f} kcal/mol - '
          f'essentially unchanged from\n   the monomer ({k71m[0][0]:.2f}). '
          f'Dimerisation does not impede it.')

    print(f'\n3. The competing clusters are suppressed by dimerisation:')
    print(f'   R57-R58   {r58[0][0]:>6.2f} -> {r58[1][0]:>6.2f} kcal/mol   '
          f'({fold(r58[0][0], r58[1][0]):.0f}-fold worse)')
    print(f'   K64-K67   {k64[0][0]:>6.2f} -> {k64[1][0]:>6.2f} kcal/mol   '
          f'({fold(k64[0][0], k64[1][0]):.0f}-fold worse)')
    print(f'\n   So in the dimer K71-R72-K73 is favoured over R57-R58 by '
          f'{r58[1][0] - k71m[1][0]:.1f} kcal/mol\n   and over K64-K67 by '
          f'{k64[1][0] - k71m[1][0]:.1f} kcal/mol. That is the structural reason '
          f'57RRQQ58 and\n   64QQQQ67 are silent while 71KRK73QQQ is not.')
    print()


if __name__ == '__main__':
    main()
