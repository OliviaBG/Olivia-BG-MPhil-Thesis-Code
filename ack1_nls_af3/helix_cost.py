#!/usr/bin/env python3
"""
What does it cost, energetically, to unwind the C-terminal turns of ACK1 SAM alpha5?

The tether calculation gives the CONFORMATIONAL cost of presenting K71-R72-K73 to
importin-alpha: given that u residues of alpha5 have unwound, what fraction of linker
conformations place the folded SAM domain somewhere the receptor is not. It says
nothing about how hard it is to unwind those u residues in the first place, so the
totals quoted there are a lower bound.

This closes that gap with a Lifson-Roig helix-coil calculation on the actual alpha5
sequence, giving dG_unwind(u). The two terms are then additive, because they are
sequential and independent steps:

    dG_present(u) = dG_unwind(u)  +  -RT ln f(u)
                    \\___________/     \\___________/
                    break u turns     find a pose for what is left

Minimising over u gives the cheapest route to a bound state, and the u that achieves
it is a structural prediction: how far alpha5 has to fray.

METHOD
Lifson-Roig with residue-specific propagation weights. w_i = w_Ala * exp(-ddG_i / RT)
with ddG from Pace & Scholtz (1998) helix propensities (kcal/mol relative to Ala) and
w_Ala = 1.60; nucleation v = 0.036, both standard. Transfer matrices give the partition
function; constraining the C-terminal u residues to coil and taking the ratio gives the
unwinding free energy. Proline is treated as helix-breaking (w = 0.01).

CAVEATS
Lifson-Roig describes an isolated peptide helix. Alpha5 is packed against the SAM
core, and tertiary contacts stabilise it beyond its intrinsic propensity, so the real
cost is HIGHER than this. The numbers below are therefore still a lower bound, but a
much tighter one than assuming the cost is zero.
"""
import numpy as np

RT = 0.593                      # kcal/mol at 298 K
W_ALA = 1.60                    # Lifson-Roig propagation weight for alanine
V_NUC = 0.036                   # nucleation weight

# Pace & Scholtz (1998), kcal/mol relative to Ala; larger = weaker helix former
DDG = {'A': 0.00, 'L': 0.21, 'R': 0.21, 'M': 0.24, 'K': 0.26, 'Q': 0.39, 'E': 0.40,
       'I': 0.41, 'W': 0.49, 'S': 0.50, 'Y': 0.53, 'F': 0.54, 'H': 0.61, 'V': 0.61,
       'N': 0.65, 'T': 0.66, 'C': 0.68, 'D': 0.69, 'G': 1.00, 'P': 3.16}

# Human ACK1 Q07912. Alpha5 of the SAM domain runs to residue 69; the NLS follows.
ACK1 = ('MQPEEGTGWLLELLSEVQLQQYFLRLRDDLNVTRLSHFEYVKNEDLEKIGMGRPGQRRLW'
        'EAVKRRKALCKRKSWMSKVFSGKR')
A5_START, A5_END = 48, 69        # the helix whose C-terminus has to fray


def weights(seq):
    w = []
    for c in seq:
        if c == 'P':
            w.append(0.01)
        else:
            w.append(W_ALA * np.exp(-DDG.get(c, 0.5) / RT))
    return np.array(w)


def partition(seq, forced_coil_tail=0):
    """
    Lifson-Roig partition function. `forced_coil_tail` residues at the C-terminus are
    constrained to the coil state, which is what "unwound by u" means here.
    """
    w = weights(seq)
    n = len(w)
    # states: 0 = helical (h), 1 = coil preceded by helix (c after h), 2 = coil (c)
    vec = np.array([0.0, 0.0, 1.0])
    for i in range(n):
        forced = i >= n - forced_coil_tail
        M = np.zeros((3, 3))
        if not forced:
            M[0, 0] = w[i]          # h -> h
            M[2, 0] = V_NUC         # c -> h (nucleation)
            M[1, 0] = V_NUC
        M[0, 1] = V_NUC             # h -> c
        M[1, 2] = 1.0               # c -> c
        M[2, 2] = 1.0
        vec = vec @ M
        s = vec.sum()
        if s > 0:
            vec = vec / s
            M_scale = s
        else:
            M_scale = 1.0
        if i == 0:
            logZ = np.log(M_scale)
        else:
            logZ += np.log(M_scale)
    return logZ + np.log(vec.sum())


def dG_unwind(u):
    seq = ACK1[A5_START - 1:A5_END]
    return -RT * (partition(seq, forced_coil_tail=u) - partition(seq, 0))


# tether results: f(u) for the registers that matter, monomer and dimer
TETHER = {
    ('minor', 'K71 at P2'): {1: (0.1125, 0.1110), 2: (0.2065, 0.1923),
                             3: (0.2755, 0.2162), 4: (0.3287, 0.2642),
                             5: (0.2390, 0.2028), 6: (0.2893, 0.1872)},
    ('major', 'K71 at P2'): {1: (0.0003, 0.0000), 2: (0.0075, 0.0063),
                             3: (0.0375, 0.0138), 4: (0.0470, 0.0220),
                             5: (0.0290, 0.0165), 6: (0.0535, 0.0205)},
    ('minor', 'R58 at P2'): {1: (0.0693, 0.0000), 2: (0.1507, 0.0003),
                             3: (0.3108, 0.0025), 4: (0.2630, 0.0070),
                             5: (0.3405, 0.0683), 6: (0.2717, 0.1123)},
    ('major', 'K64 at P2'): {3: (0.0090, 0.0000), 4: (0.0302, 0.0013),
                             5: (0.0127, 0.0003), 6: (0.0220, 0.0003)},
}

if __name__ == '__main__':
    print('=' * 78)
    print('COST OF UNWINDING ACK1 SAM alpha5 (residues %d-%d)' % (A5_START, A5_END))
    print('=' * 78)
    print(f'sequence: {ACK1[A5_START-1:A5_END]}')
    print(f'\n{"u":>3}{"residues freed":>18}{"dG_unwind":>12}')
    print('-' * 40)
    for u in range(0, 9):
        freed = ACK1[A5_END - u:A5_END] if u else '-'
        print(f'{u:>3}{freed:>18}{dG_unwind(u):>12.2f}')

    print('\n' + '=' * 78)
    print('TOTAL COST OF PRESENTING THE NLS   dG = dG_unwind(u) - RT ln f(u)')
    print('=' * 78)
    for (site, reg), fs in TETHER.items():
        print(f'\n--- {site} site, {reg} ---')
        print(f'{"u":>3}{"unwind":>9}{"-RTlnf mono":>13}{"TOTAL mono":>12}'
              f'{"-RTlnf dim":>12}{"TOTAL dim":>11}')
        best_m = best_d = (1e9, None)
        for u, (fm, fd) in sorted(fs.items()):
            du = dG_unwind(u)
            cm = -RT * np.log(fm) if fm > 0 else np.inf
            cd = -RT * np.log(fd) if fd > 0 else np.inf
            tm, td = du + cm, du + cd
            if tm < best_m[0]:
                best_m = (tm, u)
            if td < best_d[0]:
                best_d = (td, u)
            print(f'{u:>3}{du:>9.2f}{cm:>13.2f}{tm:>12.2f}'
                  f'{cd:>12.2f}{td:>11.2f}' if np.isfinite(cd) else
                  f'{u:>3}{du:>9.2f}{cm:>13.2f}{tm:>12.2f}{"inf":>12}{"inf":>11}')
        print(f'    cheapest: monomer {best_m[0]:.2f} kcal/mol at u={best_m[1]}; '
              f'dimer {best_d[0]:.2f} at u={best_d[1]}'
              if np.isfinite(best_d[0]) else
              f'    cheapest: monomer {best_m[0]:.2f} kcal/mol at u={best_m[1]}; '
              f'dimer inaccessible')
    print()


# ===================================================================== tertiary term
# The Lifson-Roig number above is small because this sequence has poor intrinsic helix
# propensity -- as an isolated peptide alpha5 would be mostly coil. It is helical in the
# folded protein because it packs against the SAM core. So the real cost of fraying is
# the cost of stripping those tertiary contacts, which is estimated here from the
# surface each residue buries against the rest of the domain.
MAXASA = {'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167, 'Q': 225, 'E': 223,
          'G': 104, 'H': 224, 'I': 197, 'L': 201, 'K': 236, 'M': 224, 'F': 240,
          'P': 159, 'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174}
SIGMA = 0.025      # kcal/mol/A^2, standard burial coefficient


def tertiary_cost():
    import warnings; warnings.filterwarnings('ignore')
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.PDB.Polypeptide import protein_letters_3to1 as t31
    st = PDBParser(QUIET=True).get_structure('s', 'sam_dimer_fixed.pdb')
    model = st[0]
    chain = sorted(model, key=lambda c: c.id)[0]
    for a in list(model.get_atoms()):
        if a.element == 'H':
            a.get_parent().detach_child(a.get_id())
    # keep one protomer only: alpha5 packs against its own core
    for c in list(model):
        if c.id != chain.id:
            model.detach_child(c.id)
    ShrakeRupley().compute(model, level='R')
    out = {}
    for r in chain:
        if r.id[0] != ' ':
            continue
        aa = t31.get(r.get_resname(), 'X')
        mx = MAXASA.get(aa)
        if mx:
            out[r.id[1]] = (aa, max(0.0, mx - r.sasa))
    return out


if __name__ == '__main__':
    print('\n' + '=' * 78)
    print('TERTIARY PACKING: surface each C-terminal alpha5 residue buries against')
    print('the SAM core, and what breaking that contact costs')
    print('=' * 78)
    bur = tertiary_cost()
    print(f'{"res":>6}{"aa":>4}{"buried A^2":>13}{"dG (kcal/mol)":>16}')
    print('-' * 42)
    run = 0.0
    cum = {}
    for i in range(A5_END, A5_END - 9, -1):
        if i not in bur:
            continue
        aa, b = bur[i]
        run += b * SIGMA
        cum[A5_END - i + 1] = run
        print(f'{i:>6}{aa:>4}{b:>13.0f}{b * SIGMA:>16.2f}')
    print(f'\n{"u":>3}{"intrinsic":>12}{"tertiary":>11}{"total unwind":>15}')
    print('-' * 42)
    for u in range(1, 9):
        if u in cum:
            print(f'{u:>3}{dG_unwind(u):>12.2f}{cum[u]:>11.2f}'
                  f'{dG_unwind(u) + cum[u]:>15.2f}')
