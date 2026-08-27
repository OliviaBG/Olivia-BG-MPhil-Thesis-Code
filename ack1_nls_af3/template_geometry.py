#!/usr/bin/env python3
"""
Measure, from crystal structures rather than from an assumed 3.4 A/residue rise,
exactly what geometry importin-alpha demands of a cNLS.

3UL1 = mouse importin-alpha dIBB (chain B, 72-497) + nucleoplasmin BIPARTITE cNLS
       (chain A, 152-172).  The reference for the two-cluster mode.
1EJL = mouse importin-alpha dIBB (chain I, 72-497) + two copies of the SV40
       MONOPARTITE NLS (chains A and B) -- one in each site.

Question: ACK1 has R57-R58, a 12-residue spacer, then K71-R72-K73. Nucleoplasmin
has K155-R156, a 10-residue spacer, then K167-K168-K169-K170. Is ACK1's spacing
compatible, and how far apart must the two clusters actually sit?
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from Bio.PDB import PDBParser

P = PDBParser(QUIET=True)
T2O = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}


def sec(t):
    print('\n' + '=' * 82); print(t); print('=' * 82)


def contacts(pep_res, rec_chain, cut=4.0):
    """receptor residues with any heavy atom within cut of this peptide residue"""
    out = []
    pa = np.array([a.get_coord() for a in pep_res if a.element != 'H'])
    for r in rec_chain:
        if r.id[0] != ' ':
            continue
        ra = np.array([a.get_coord() for a in r if a.element != 'H'])
        if len(ra) == 0:
            continue
        d = np.linalg.norm(pa[:, None, :] - ra[None, :, :], axis=2).min()
        if d < cut:
            out.append((T2O.get(r.get_resname(), 'X') + str(r.id[1]), round(float(d), 2)))
    return out


# ---------------------------------------------------------------- 3UL1
sec('3UL1  nucleoplasmin BIPARTITE cNLS bound to importin-alpha')
s = P.get_structure('3ul1', '3UL1.pdb')[0]
pep, rec = s['A'], s['B']
pres = [r for r in pep if r.id[0] == ' ']
print('peptide: ' + ''.join(T2O.get(r.get_resname(), 'X') for r in pres) +
      f'   (residues {pres[0].id[1]}-{pres[-1].id[1]})')

print(f'\n{"res":>6}  {"n_contacts":>10}  receptor residues within 4.0 A')
print('-' * 82)
site = {}
for r in pres:
    c = contacts(r, rec)
    site[r.id[1]] = [x[0] for x in c]
    lab = T2O.get(r.get_resname(), 'X') + str(r.id[1])
    print(f'{lab:>6}  {len(c):>10}  {", ".join(x[0] for x in c[:9])}')

# geometry between the two basic clusters
ca = {r.id[1]: r['CA'].get_coord() for r in pres if 'CA' in r}
print('\nCa-Ca distances between the two basic clusters (crystal, bound state):')
for a, b in [(155, 167), (156, 167), (156, 168), (155, 170), (156, 170)]:
    if a in ca and b in ca:
        d = np.linalg.norm(ca[a] - ca[b])
        print(f'  {a}-{b}  ({b-a:>2} residues apart):  {d:6.1f} A   '
              f'=> {d/(b-a):4.2f} A per residue')

# rise per residue along the whole bound peptide
rises = [np.linalg.norm(ca[i+1] - ca[i]) for i in range(pres[0].id[1], pres[-1].id[1])
         if i in ca and i+1 in ca]
print(f'\nmean Ca-Ca step along the bound peptide: {np.mean(rises):.2f} A '
      f'(n={len(rises)})  -- fully extended reference is ~3.5 A')

# ---------------------------------------------------------------- 1EJL
sec('1EJL  SV40 MONOPARTITE NLS, two copies -- major and minor sites')
s2 = P.get_structure('1ejl', '1EJL.pdb')[0]
rec2 = s2['I']
for cid in ('A', 'B'):
    p = [r for r in s2[cid] if r.id[0] == ' ']
    seq = ''.join(T2O.get(r.get_resname(), 'X') for r in p)
    allc = set()
    for r in p:
        allc.update(x[0] for x in contacts(r, rec2))
    nums = sorted(int(x[1:]) for x in allc)
    print(f'\n  chain {cid}  {seq}  ({p[0].id[1]}-{p[-1].id[1]})')
    print(f'    contacts ARM repeats spanning receptor residues {min(nums)}-{max(nums)}')
    print(f'    {len(allc)} contact residues: {", ".join(sorted(allc, key=lambda x:int(x[1:])))}')

# which copy is major vs minor: the major site is the one nearer ARM2-4 (res ~150-235)
sec('ACK1 comparison')
ACK1 = ''.join(l.strip() for l in open('ack1.fasta') if not l.startswith('>'))
print(f'  nucleoplasmin:  K155 R156  ... 10-residue spacer ...  K167 K168 K169 K170')
print(f'  ACK1        :  R57  R58   ... 12-residue spacer ...  K71  R72  K73')
print(f'  ACK1 57-73  :  {ACK1[56:73]}')
d_np = np.linalg.norm(ca[156] - ca[167])
print(f'\n  Crystal requires the two anchor residues to sit {d_np:.1f} A apart.')
print(f'  In the AlphaFold ACK1 model, R58 Ca to K71 Ca measured 21.1 A (helical).')
print(f'  Shortfall: {d_np - 21.1:.1f} A, i.e. the helix must unwind by roughly')
print(f'  {(d_np - 21.1)/3.5:.0f} residues worth of extension for both clusters to engage.')
print()
