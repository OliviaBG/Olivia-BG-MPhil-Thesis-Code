#!/usr/bin/env python3
"""
Does SAM-mediated dimerisation sterically block importin-alpha from engaging the
ACK1 NLS -- and can one dimer load two receptors at once?

Method: take the crystallographic importin-alpha:NLS complex (1EJL), superpose its
bound SV40 peptide onto ACK1's K71-R72-K73 in the composite SAM-dimer model, carry
importin-alpha along with it, then count steric clashes against the ACK1 dimer.

Two registers are tested per site (which ACK1 residue sits in the P2 pocket), and
both protomers, giving a small ensemble rather than one arbitrary placement.

WORST-CASE FRAMING: the grafted linker is in the AlphaFold alpha-helical
conformation, which holds the NLS as CLOSE to the SAM domain as it can be. Binding
requires the linker to extend, which moves the NLS further away. So clash counts
here are an upper bound -- if it does not clash helical, it cannot clash extended.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from Bio.PDB import PDBParser, Superimposer, PDBIO, Select

P = PDBParser(QUIET=True)
BB = ('N', 'CA', 'C')

ejl = P.get_structure('1ejl', '1EJL.pdb')[0]
IMP = [r for r in ejl['I'] if r.id[0] == ' ']            # importin-alpha 72-497
PEP_MAJOR = {r.id[1]: r for r in ejl['B'] if r.id[0] == ' '}   # SV40 in major site
PEP_MINOR = {r.id[1]: r for r in ejl['A'] if r.id[0] == ' '}   # SV40 in minor site

comp = P.get_structure('c', 'ack1_sam_dimer_with_NLS.pdb')[0]
ACK = {c.id: {r.id[1]: r for r in c if r.id[0] == ' '} for c in comp}

imp_atoms = [a for r in IMP for a in r if a.element != 'H']
imp_coords = np.array([a.get_coord() for a in imp_atoms])


def ack_coords(chain, lo, hi):
    return np.array([a.get_coord() for n, r in ACK[chain].items()
                     if lo <= n <= hi for a in r if a.element != 'H'])


def place(pep, tmpl_start, ack_chain, ack_start):
    """superpose 3 template peptide residues onto 3 ACK1 residues; return moved importin"""
    fixed, moving = [], []
    for k in range(3):
        ra = ACK[ack_chain].get(ack_start + k)
        rt = pep.get(tmpl_start + k)
        if ra is None or rt is None:
            return None, None
        for at in BB:
            if at in ra and at in rt:
                fixed.append(ra[at]); moving.append(rt[at])
    if len(fixed) < 6:
        return None, None
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    rot, tran = sup.rotran
    return imp_coords @ rot + tran, sup.rms


def clashes(moved, target, hard=3.0, soft=4.0):
    if moved is None or len(target) == 0:
        return None
    d = np.linalg.norm(moved[:, None, :] - target[None, :, :], axis=2)
    mn = d.min(axis=1)
    return int((mn < hard).sum()), int((mn < soft).sum()), float(mn.min())


print('=' * 86)
print('IMPORTIN-ALPHA PLACED ON THE ACK1 NLS IN THE SAM DIMER')
print('=' * 86)
print(f'{"site":<7}{"protomer":<10}{"P2 res":<8}{"fit RMSD":>9} | '
      f'{"own SAM 2-69":>14} {"partner 2-95":>14}   nearest')
print('-' * 86)

results = {}
for site, pep, tstart in (('major', PEP_MAJOR, 128), ('minor', PEP_MINOR, 128)):
    for ch, other in (('A', 'B'), ('B', 'A')):
        for p2 in (71, 72):
            moved, rms = place(pep, tstart, ch, p2)
            if moved is None:
                continue
            own = clashes(moved, ack_coords(ch, 2, 69))
            par = clashes(moved, ack_coords(other, 2, 95))
            results[(site, ch, p2)] = (moved, own, par)
            print(f'{site:<7}{ch:<10}{p2:<8}{rms:>9.2f} | '
                  f'{own[0]:>5} hard {own[1]:>4} soft {par[0]:>5} hard {par[1]:>4} soft'
                  f'   {min(own[2], par[2]):>5.1f} A')

print('\n  hard = importin heavy atom within 3.0 A of ACK1 heavy atom (real overlap)')
print('  soft = within 4.0 A (contact distance)')

# ---------------------------------------------------------------- two receptors
print('\n' + '=' * 86)
print('CAN ONE DIMER LOAD TWO RECEPTORS? (importin-importin clash)')
print('=' * 86)
for site in ('major', 'minor'):
    for p2 in (71, 72):
        a = results.get((site, 'A', p2))
        b = results.get((site, 'B', p2))
        if not a or not b:
            continue
        d = np.linalg.norm(a[0][:, None, :] - b[0][None, :, :], axis=2)
        mn = d.min()
        hard = int((d.min(axis=1) < 3.0).sum())
        soft = int((d.min(axis=1) < 4.0).sum())
        com_sep = np.linalg.norm(a[0].mean(axis=0) - b[0].mean(axis=0))
        print(f'  {site} site, P2={p2}: centroid separation {com_sep:6.1f} A   '
              f'{hard:>5} hard, {soft:>5} soft clashes   nearest {mn:.1f} A')

# ---------------------------------------------------------------- write a model
best = results.get(('major', 'A', 71))
if best is not None:
    io = PDBIO()
    with open('ack1_dimer_plus_importin.pdb', 'w') as fh:
        fh.write('HEADER    ACK1 SAM DIMER + IMPORTIN-ALPHA DOCKED ON PROTOMER A NLS\n')
        fh.write('REMARK    importin-alpha from 1EJL chain I, superposed via its\n')
        fh.write('REMARK    bound SV40 peptide onto ACK1 K71-R72-K73 (major site register)\n')
        n = 1
        for ch in comp:
            for r in ch:
                if r.id[0] != ' ':
                    continue
                for a in r:
                    if a.element == 'H':
                        continue
                    c = a.get_coord()
                    fh.write(f'ATOM  {n:>5} {a.get_id():<4}{r.get_resname():>3} {ch.id}'
                             f'{r.id[1]:>4}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}'
                             f'  1.00  0.00\n')
                    n += 1
            fh.write('TER\n')
        for a, c in zip(imp_atoms, best[0]):
            r = a.get_parent()
            fh.write(f'ATOM  {n:>5} {a.get_id():<4}{r.get_resname():>3} I'
                     f'{r.id[1]:>4}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}'
                     f'  1.00  0.00\n')
            n += 1
        fh.write('TER\nEND\n')
    print('\n  wrote ack1_dimer_plus_importin.pdb (protomer A, major-site register)')
print()
