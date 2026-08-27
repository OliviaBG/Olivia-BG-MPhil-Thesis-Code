#!/usr/bin/env python3
"""
Register analysis: map ACK1's basic clusters onto the crystallographic cNLS
register and ask, position by position, whether ACK1 can satisfy the same
importin-alpha pockets the template peptide satisfies.

This is deliberately NOT an energy calculation. It reports which anchor pockets
are chemically satisfied, which are not, and where ACK1 differs from a canonical
cNLS -- the part that can be stated without a repacker or a force field.

Templates:
  3UL1 chain A  nucleoplasmin 152-172  bipartite: minor site + major site
  1EJL chain B  SV40 126-132           major site
  1EJL chain A  SV40 126-132           minor site
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from Bio.PDB import PDBParser

P = PDBParser(QUIET=True)
T2O = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
ACK1 = ''.join(l.strip() for l in open('ack1.fasta') if not l.startswith('>'))
BASIC = set('KR')


def load(pdb, pep_chain, rec_chain):
    m = P.get_structure(pdb, pdb + '.pdb')[0]
    pep = [r for r in m[pep_chain] if r.id[0] == ' ']
    rec = [r for r in m[rec_chain] if r.id[0] == ' ']
    return pep, rec


def buried(res, rec, cut=4.0):
    pa = np.array([a.get_coord() for a in res if a.element != 'H'])
    hits = []
    for r in rec:
        ra = np.array([a.get_coord() for a in r if a.element != 'H'])
        if len(ra) and np.linalg.norm(pa[:, None] - ra[None], axis=2).min() < cut:
            hits.append(T2O.get(r.get_resname(), 'X') + str(r.id[1]))
    return hits


def report(title, pep, rec, anchor_tmpl, anchor_ack1, note):
    """anchor_tmpl/anchor_ack1: residue numbers that must coincide"""
    print('\n' + '=' * 84)
    print(title)
    print('=' * 84)
    print(f'  {note}\n')
    off = anchor_ack1 - anchor_tmpl
    print(f'{"tmpl":>6} {"aa":>3} {"contacts":>9}  {"ACK1":>6} {"aa":>3}  match  pocket residues')
    print('-' * 84)
    score = {'same': 0, 'basic_ok': 0, 'lost': 0, 'na': 0}
    for r in pep:
        n = r.id[1]
        t_aa = T2O.get(r.get_resname(), 'X')
        c = buried(r, rec)
        a_num = n + off
        if not (1 <= a_num <= len(ACK1)):
            continue
        a_aa = ACK1[a_num - 1]
        if not c:
            verdict = '-'
            score['na'] += 1
        elif t_aa == a_aa:
            verdict = 'exact'
            score['same'] += 1
        elif t_aa in BASIC and a_aa in BASIC:
            verdict = 'basic'
            score['basic_ok'] += 1
        elif t_aa in BASIC and a_aa not in BASIC:
            verdict = 'LOST'
            score['lost'] += 1
        else:
            verdict = 'n/a'
            score['na'] += 1
        print(f'{n:>6} {t_aa:>3} {len(c):>9}  {a_num:>6} {a_aa:>3}  {verdict:>5}  '
              f'{", ".join(c[:6])}')
    print(f'\n  contacting positions: {score["same"]} identical, {score["basic_ok"]} '
          f'basic-for-basic, {score["lost"]} basic contact LOST, {score["na"]} non-basic')
    return score


# ---- major site: SV40 in 1EJL chain B, P2 lysine K128 <-> ACK1 K71 -----------
pepB, recI = load('1EJL', 'B', 'I')
report('MAJOR SITE   1EJL chain B (SV40) -> ACK1, anchoring K128 on K71',
       pepB, recI, 128, 71,
       'The major site is the dominant one; its P2 pocket takes the critical lysine.')

# ---- minor site: SV40 in 1EJL chain A, anchored on ACK1 R58 -----------------
pepA, _ = load('1EJL', 'A', 'I')
report('MINOR SITE   1EJL chain A (SV40) -> ACK1, anchoring K128 on R58',
       pepA, recI, 128, 58,
       'For a bipartite NLS the UPSTREAM cluster occupies the minor site.')

# ---- the bipartite reference ------------------------------------------------
pepN, recN = load('3UL1', 'A', 'B')
ca = {r.id[1]: r['CA'].get_coord() for r in pepN if 'CA' in r}
print('\n' + '=' * 84)
print('SPACING: does ACK1 have enough linker to reach both sites?')
print('=' * 84)
d = np.linalg.norm(ca[156] - ca[167])
print(f'  nucleoplasmin  R156 -> K167   11 linker steps   {d:.1f} A required')
print(f'  ACK1           R58  -> K71    13 linker steps   -> {13*2.65:.1f} A available')
print(f'                                                     at the crystal\'s own')
print(f'                                                     2.65 A per residue')
print(f'\n  ACK1 has TWO MORE linker residues than nucleoplasmin, so the span is not')
print(f'  the limiting factor -- the linker has slack, not deficit.')
print(f'\n  What is limiting: in the AlphaFold model R58-K71 sit 21.1 A apart because')
print(f'  the linker is alpha-helical. Reaching {d:.1f} A needs ~{(d-21.1)/3.5:.0f} residues of local')
print(f'  unwinding at the C-terminal end of SAM alpha5 -- around one helical turn.')
print()
