#!/usr/bin/env python3
"""
ACK1 NLS vs importin-alpha -- structural modelling and physics scoring, GPU version.
No AlphaFold anywhere in this pipeline.

WHAT IT DOES
------------
stage `pockets`  Map the NLS-binding pockets of importin-alpha from the crystal template
                 and work out which first-shell residues differ between human KPNA1,
                 KPNA2, KPNA3 and KPNA4.

stage `screen`   For every (paralogue x site x peptide): thread the test peptide onto the
                 crystallographic NLS in that site, mutate the receptor first shell to the
                 paralogue, rebuild side chains, run restrained MD in Amber14/GBn2
                 implicit solvent, and score with ensemble MM-GBSA over the trajectory
                 frames. Components (vdW, Coulomb, polar solvation, nonpolar) are kept
                 separate so the electrostatic bias is visible.

stage `tether`   Ask whether the folded SAM domain can be accommodated at all while the
                 NLS sits in the groove. The peptide is held in the bound conformation and
                 the chain is grown backwards from it, sampling phi/psi; the folded SAM
                 domain is docked onto the grown stretch and tested for clashes against
                 the receptor, in the monomer and in the SAM dimer. Reported as the
                 fraction of conformations that work, versus how many residues of SAM
                 alpha5 are allowed to unwind. CPU only, takes a couple of minutes.

stage `analyse`  Ranking, control calibration, charge-bias regression, matched-pair test.

REQUIREMENTS
------------
    conda create -n ack1 -c conda-forge python=3.11 openmm cudatoolkit pdbfixer \
          biopython numpy
    conda activate ack1
Check the GPU is found:
    python -m openmm.testInstallation

INPUT FILES (same directory, or pass --datadir)
    1EJL.pdb              importin-alpha dIBB + SV40 NLS in BOTH sites
    sam_dimer_fixed.pdb   the ACK SAM dimer structure   (only needed for `tether`)
    kpna_seqs.fasta       KPNA1-4 human + Kpna2 mouse   (only needed for `pockets`)

USAGE
    python ack1_importin_gpu.py                        # everything, default settings
    python ack1_importin_gpu.py --stage screen --md 400 --seeds 4
    python ack1_importin_gpu.py --stage screen --quick # short run to check it works
    python ack1_importin_gpu.py --stage analyse

The screen writes results incrementally to screen_results.tsv and skips work already in
that file, so it is safe to interrupt and restart.

A NOTE ON WHAT THE NUMBERS MEAN
-------------------------------
MM-GBSA single-trajectory scores are RELATIVE. They are not affinities, and for highly
charged ligands binding an acidic groove -- exactly this case -- they systematically
over-reward net positive charge. That is why the panel carries composition-matched
scrambles: the only comparisons that are free of the charge artefact are between
peptides of identical amino-acid composition. `analyse` reports the charge regression
explicitly so you can see how much of the spread is just charge.
"""
import argparse, gc, json, math, os, sys, time
from collections import defaultdict

# Thread caps must be set before numpy / OpenMM are imported, otherwise the BLAS
# backend and the OpenMM CPU platform have already sized their pools.
_THREADS = os.environ.get('ACK1_THREADS', '13')
for _v in ('OPENMM_CPU_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, _THREADS)

import numpy as np

# ----------------------------------------------------------------- configuration
TMPL_PEP = 'PKKKRKV'
PEP0 = 126
PEPCHAIN = {'major': 'B', 'minor': 'A'}
RECCHAIN = 'I'
RECRANGE = (72, 497)
GAMMA = 0.00542 * 4.184          # kJ/mol/A^2, MM-GBSA nonpolar surface term
KCAL = 1 / 4.184

try:
    from scipy.spatial import cKDTree as _KDT
except Exception:
    _KDT = None


def _obstacle_tree(points):
    return _KDT(points) if _KDT is not None else points


def _clashfree(pts, obstacle, cutoff=3.0):
    """True if no atom in `pts` lies within `cutoff` A of the obstacle."""
    if _KDT is not None:
        return not (obstacle.query_ball_point(pts, cutoff,
                                              return_length=True) > 0).any()
    for i in range(0, len(pts), 4000):
        if np.linalg.norm(pts[i:i + 4000, None] - obstacle[None], axis=2).min() < cutoff:
            return False
    return True

RT = 0.593                       # kcal/mol at 298 K

ONE2THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
    'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

PEPTIDES = [
    # id                          seq        class     note
    ('ACK1_64-67_K64atP2',        'AVKRRKA', 'ack1',   'ACK1 62-68, K64 in the P2 pocket'),
    ('ACK1_64-67_R65atP2',        'VKRRKAL', 'ack1',   'ACK1 63-69, R65 in P2'),
    ('ACK1_71-73_K71atP2',        'LCKRKSW', 'ack1',   'ACK1 69-75, K71 in P2'),
    ('ACK1_71-73_R72atP2',        'CKRKSWM', 'ack1',   'ACK1 70-76, R72 in P2'),
    ('ACK1_71-73_K73atP2',        'KRKSWMS', 'ack1',   'ACK1 71-77, K73 in P2'),
    ('ACK1_R57R58_R58atP2',       'QRRLWEA', 'ack1',   'ACK1 56-62, R58 in P2'),
    ('MUT_71KRK73QQQ',            'LCQQQSW', 'mutant', 'your 71-73 QQQ construct'),
    ('MUT_64QQQQ67',              'AVQQQQA', 'mutant', 'your 64-67 QQQQ construct'),
    ('MUT_64EEEE67',              'AVEEEEA', 'mutant', 'your 64-67 EEEE construct'),
    ('POS_SV40',                  'PKKKRKV', 'pos',    'template sequence, strong cNLS'),
    ('POS_nucleoplasmin_major',   'AKKKKLD', 'pos',    'nucleoplasmin 166-172'),
    ('POS_cMyc',                  'AAKRVKL', 'pos',    'c-Myc 321-327'),
    ('NEG_ACK1_668-674',          'TNYAFVP', 'neg',    'ACK1 non-basic negative'),
    ('NEG_scramble_of_71reg',     'WKSLRCK', 'neg',    'composition-matched to LCKRKSW'),
    ('NEG_scramble_of_64reg',     'RAKAVKR', 'neg',    'composition-matched to AVKRRKA'),
    ('NEG_polyAla',               'AAAAAAA', 'neg',    'apolar null'),
    ('NEG_polyGln',               'QQQQQQQ', 'neg',    'polar null'),
]

MATCHED_PAIRS = [('ACK1_71-73_K71atP2', 'NEG_scramble_of_71reg'),
                 ('ACK1_64-67_K64atP2', 'NEG_scramble_of_64reg')]

OUT_TSV = 'screen_results.tsv'
HDR = ('paralogue\tsite\tpeptide\tseq\tclass\tseed\tdG_total\tdE_vdw\tdE_coul\tdG_polar'
       '\tdG_nonpolar\tdSASA\tn_frames\tn_saltbridge\tn_hbond\tpep_charge\truntime_s')


# ============================================================ stage: pocket mapping
def read_fasta(p):
    out, n, b = [], None, []
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        if l[0] == '>':
            if n:
                out.append((n, ''.join(b)))
            n, b = l[1:], []
        else:
            b.append(l)
    if n:
        out.append((n, ''.join(b)))
    return out


def stage_pockets(args):
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1 as t31
    from Bio.Align import PairwiseAligner, substitution_matrices

    d = args.datadir
    seqs = dict(read_fasta(os.path.join(d, 'kpna_seqs.fasta')))
    TEMPL = [k for k in seqs if k.startswith('Kpna2_MOUSE')][0]
    tmpl = seqs[TEMPL]

    m = PDBParser(QUIET=True).get_structure('x', os.path.join(d, '1EJL.pdb'))[0]
    rec = {r.id[1]: r for r in m[RECCHAIN] if r.id[0] == ' '}
    recseq = {i: t31.get(r.get_resname(), 'X') for i, r in rec.items()}
    bad = [i for i, s in recseq.items() if tmpl[i - 1] != s]
    print(f'template numbering check: {len(bad)} mismatches of {len(recseq)}')

    contacts = {}
    for site, ch in PEPCHAIN.items():
        pep = [r for r in m[ch] if r.id[0] == ' ']
        pc = np.array([a.get_coord() for r in pep for a in r if a.element != 'H'])
        contacts[site] = sorted(
            i for i, r in rec.items()
            if np.linalg.norm(np.array([a.get_coord() for a in r if a.element != 'H'])
                              [:, None, :] - pc[None, :, :], axis=2).min() < 5.0)

    al = PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load('BLOSUM62')
    al.open_gap_score, al.extend_gap_score, al.mode = -11, -1, 'global'
    proj = {}
    for name, s in seqs.items():
        if name == TEMPL:
            proj[name] = {i: tmpl[i - 1] for i in range(1, len(tmpl) + 1)}
            continue
        a = al.align(tmpl, s)[0]
        ti, dd = 0, {}
        for ca, cb in zip(a[0], a[1]):
            if ca != '-':
                ti += 1
                dd[ti] = cb
        proj[name] = dd

    muts = {}
    for name in seqs:
        if name == TEMPL:
            continue
        short = name.split('_')[0]
        muts[short] = {}
        for site in contacts:
            lst = [(i, recseq[i], proj[name][i]) for i in contacts[site]
                   if proj[name].get(i, '-') not in (recseq[i], '-')]
            muts[short][site] = lst
    muts.setdefault('KPNA2', {'major': [], 'minor': []})
    json.dump({'contacts': contacts, 'mutations': muts},
              open('pocket_map.json', 'w'), indent=1)

    print('\nfirst-shell substitutions relative to the template:')
    for p in sorted(muts):
        for site in ('major', 'minor'):
            lst = muts[p][site]
            txt = ', '.join(f'{a}{i}{b}' for i, a, b in lst) or 'none'
            print(f'  {p:<7} {site:<6} ({len(lst):>2}) {txt}')
    print('\nwrote pocket_map.json')


# ================================================================== stage: screen
def _sane_pdb(path, min_atoms=200):
    """A derived file on a network volume can come back NUL-padded after a hard kill.
    Reject it rather than parsing garbage: a corrupt carve surfaces later as an
    IndexError or `int('\\x00\\x00')` deep inside PDBFixer, which is very hard to
    trace back to its cause."""
    try:
        with open(path, 'rb') as fh:
            data = fh.read()
    except OSError:
        return False
    if b'\x00' in data:
        return False
    return data.count(b'\nATOM  ') + data.count(b'\nHETATM') >= min_atoms


def _carve(datadir, site, path, lo, hi):
    """write receptor residues lo-hi + the peptide of `site` (relabelled chain P)"""
    from Bio.PDB import PDBParser, PDBIO, Select

    class Keep(Select):
        def accept_chain(self, c):
            return c.id in (RECCHAIN, PEPCHAIN[site])

        def accept_residue(self, r):
            if r.id[0] != ' ':
                return 0
            if r.get_parent().id == RECCHAIN:
                return 1 if lo <= r.id[1] <= hi else 0
            return 1 if PEP0 <= r.id[1] <= PEP0 + 6 else 0

        def accept_atom(self, a):
            return 0 if a.element == 'H' else 1

    st = PDBParser(QUIET=True).get_structure('t', os.path.join(datadir, '1EJL.pdb'))
    stage = f'{path}.{os.getpid()}.tmp'
    io_ = PDBIO(); io_.set_structure(st); io_.save(stage, Keep())
    out = []
    for l in open(stage):
        if l.startswith(('ATOM', 'HETATM')) and l[21] == PEPCHAIN[site]:
            l = l[:21] + 'P' + l[22:]
        out.append(l)
    with open(stage, 'w') as fh:
        fh.writelines(out)
    os.replace(stage, path)     # atomic: a reader sees either the old file or the new


def _platform(prefer=None):
    import openmm as mm
    order = ([prefer] if prefer else []) + ['CUDA', 'HIP', 'OpenCL', 'CPU']
    for name in order:
        try:
            p = mm.Platform.getPlatformByName(name)
        except Exception:
            continue
        if name in ('CUDA', 'HIP', 'OpenCL'):
            props = {'Precision': 'mixed'}
            if name == 'CUDA':
                gpu = os.environ.get('ACK1_GPU')
                if gpu:                 # only pin a device when asked;
                    props['DeviceIndex'] = gpu   # otherwise let CUDA choose
        else:
            props = {'Threads': os.environ.get('OPENMM_CPU_THREADS', '13')}
        return p, props, name
    raise RuntimeError('no usable OpenMM platform')


def _make_system(topology, gb=True, zero_charges=False):
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    xmls = ['amber14-all.xml'] + (['implicit/gbn2.xml'] if gb else [])
    s = app.ForceField(*xmls).createSystem(
        topology, nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.8 * unit.nanometer, constraints=app.HBonds)
    if zero_charges:
        for f in s.getForces():
            if isinstance(f, mm.NonbondedForce):
                for i in range(f.getNumParticles()):
                    q, sig, eps = f.getParticleParameters(i)
                    f.setParticleParameters(i, 0.0, sig, eps)
                for i in range(f.getNumExceptions()):
                    a, b, qq, sig, eps = f.getExceptionParameters(i)
                    f.setExceptionParameters(i, a, b, 0.0, sig, eps)
            elif 'GB' in f.__class__.__name__:
                try:
                    for i in range(f.getNumParticles()):
                        pr = list(f.getParticleParameters(i))
                        pr[0] = 0.0
                        f.setParticleParameters(i, *pr)
                except Exception:
                    pass
    return s


class EnergyBank:
    """
    Reusable single-point energy contexts for MM-GBSA.

    The decomposition needs nine energies per frame: three species (complex, receptor,
    peptide) x three Hamiltonians (with GB, vacuum, vacuum with charges removed).
    Building a fresh Context for each of those on every frame creates ~180 CUDA contexts
    per run; they are not released fast enough and the GPU eventually refuses to hand
    out another one ("The requested CUDA device could not be loaded"). The topologies
    are identical across frames, so the contexts are built once here and only the
    positions change.

    If the contexts cannot be created on the requested platform, the whole bank falls
    back to the CPU platform. These are single-point evaluations, so that is a slowdown
    rather than a failure.
    """

    KEYS = (('gb', True, False), ('mm', False, False), ('vw', False, True))

    def __init__(self, plat, props, threads=None):
        self.plat, self.props = plat, props
        self.threads = threads or os.environ.get('OPENMM_CPU_THREADS', '13')
        self.ctx = {}
        self.fellback = False

    def _build(self, species, topology):
        import openmm as mm
        from openmm import unit
        made = {}
        try:
            for tag, gb, zc in self.KEYS:
                s = _make_system(topology, gb=gb, zero_charges=zc)
                made[tag] = mm.Context(
                    s, mm.VerletIntegrator(0.001 * unit.picoseconds),
                    self.plat, self.props or {})
        except Exception:
            for c in made.values():
                del c
            made = {}
            cpu = mm.Platform.getPlatformByName('CPU')
            for tag, gb, zc in self.KEYS:
                s = _make_system(topology, gb=gb, zero_charges=zc)
                made[tag] = mm.Context(
                    s, mm.VerletIntegrator(0.001 * unit.picoseconds),
                    cpu, {'Threads': str(self.threads)})
            if not self.fellback:
                print('    note: single-point energies fell back to the CPU platform',
                      flush=True)
                self.fellback = True
        self.ctx[species] = made

    def terms(self, species, topology, positions):
        from openmm import unit
        if species not in self.ctx:
            self._build(species, topology)
        out = {}
        for tag, _, _ in self.KEYS:
            c = self.ctx[species][tag]
            c.setPositions(positions)
            out[tag] = c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)
        return out

    def close(self):
        import gc
        for d in self.ctx.values():
            for k in list(d):
                del d[k]
        self.ctx.clear()
        gc.collect()


def _split(topology, positions, keep_peptide):
    import openmm.app as app
    mdl = app.Modeller(topology, positions)
    mdl.delete([r for r in mdl.topology.residues()
                if (r.chain.id == 'P') != keep_peptide])
    return mdl.topology, mdl.positions


def _sasa(topology, positions):
    # The scratch file MUST be unique per process. Two runs sharing one working
    # directory -- which happens whenever an orphaned process survives a relaunch --
    # otherwise write the same path concurrently and read back each other's
    # half-written bytes. That surfaces as PDBConstructionException, a NUL-byte
    # ValueError, or an IndexError depending on where the damage lands.
    import openmm.app as app
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    tmp = f'_sasa_tmp_{os.getpid()}.pdb'
    with open(tmp, 'w') as fh:
        app.PDBFile.writeFile(topology, positions, fh, keepIds=True)
    st = PDBParser(QUIET=True).get_structure('s', tmp)
    for a in list(st.get_atoms()):
        if a.element == 'H':
            a.get_parent().detach_child(a.get_id())
    ShrakeRupley().compute(st[0], level='M')
    return float(st[0].sasa)


def _contacts(topology, positions):
    from openmm import unit
    pos = np.array(positions.value_in_unit(unit.angstrom))
    pep, rec = [], []
    for a in topology.atoms():
        if a.element is not None and a.element.symbol == 'H':
            continue
        (pep if a.residue.chain.id == 'P' else rec).append((a.index, a.name))
    cat = [i for i, n in pep if n in ('NZ', 'NH1', 'NH2', 'NE')]
    ani = [i for i, n in rec if n in ('OD1', 'OD2', 'OE1', 'OE2')]
    nsb = int((np.linalg.norm(pos[cat][:, None] - pos[ani][None], axis=2) < 4.0).sum()) \
        if cat and ani else 0
    pol = {'N', 'O', 'OG', 'OG1', 'OH', 'NZ', 'NE', 'NH1', 'NH2', 'ND2', 'NE2',
           'OD1', 'OD2', 'OE1', 'OE2', 'SG'}
    pd = [i for i, n in pep if n in pol]
    rd = [i for i, n in rec if n in pol]
    nhb = int((np.linalg.norm(pos[pd][:, None] - pos[rd][None], axis=2) < 3.4).sum()) \
        if pd and rd else 0
    return nsb, nhb


def run_one(args, paralogue, site, pid, seq, cls, seed, pocket, plat, props):
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    from pdbfixer import PDBFixer

    t0 = time.time()
    lo, hi = getattr(args, 'trunc', RECRANGE)
    src = f'_site_{site}_{lo}-{hi}.pdb'      # range in the name: a changed --trunc
    if os.path.exists(src) and not _sane_pdb(src):   # must not silently reuse an old
        print(f'    note: {src} is corrupt, regenerating', flush=True)
        os.remove(src)                               # or a damaged carve
    if not os.path.exists(src):
        _carve(args.datadir, site, src, lo, hi)
        if not _sane_pdb(src):
            raise RuntimeError(f'{src} is unusable after regeneration - check that '
                               f'{os.path.join(args.datadir, "1EJL.pdb")} is intact')

    fx = PDBFixer(filename=src)
    present = {int(r.id) for c in fx.topology.chains() if c.id == RECCHAIN
               for r in c.residues()}
    if paralogue != 'KPNA2':
        wanted = pocket['mutations'][paralogue][site]
        lst = [f'{ONE2THREE[a]}-{i}-{ONE2THREE[b]}'
               for i, a, b in wanted if i in present]
        missing = [i for i, a, b in wanted if i not in present]
        if missing:
            print(f'    note: {paralogue} {site} substitutions at {missing} lie outside '
                  f'the carved receptor range and were skipped', flush=True)
        if lst:
            fx.applyMutations(lst, RECCHAIN)
    mp = [f'{ONE2THREE[TMPL_PEP[k]]}-{PEP0 + k}-{ONE2THREE[seq[k]]}'
          for k in range(7) if seq[k] != TMPL_PEP[k]]
    if mp:
        fx.applyMutations(mp, 'P')
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.4)
    top, pos = fx.topology, fx.positions

    system = _make_system(top, gb=True)
    free = set(pocket['contacts'][site])
    rf = mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    rf.addGlobalParameter('k', args.restraint * unit.kilojoule_per_mole /
                          unit.nanometer**2)
    for p in ('x0', 'y0', 'z0'):
        rf.addPerParticleParameter(p)
    for atom in top.atoms():
        if atom.residue.chain.id == 'P' or atom.name not in ('CA', 'N', 'C'):
            continue
        if int(atom.residue.id) in free and atom.name == 'CA':
            continue
        rf.addParticle(atom.index, pos[atom.index].value_in_unit(unit.nanometer))
    system.addForce(rf)

    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 2 / unit.picosecond,
                                        0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(seed + 1)
    sim = app.Simulation(top, system, integ, plat, props)
    sim.context.setPositions(pos)
    sim.minimizeEnergy(maxIterations=args.minimise)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed + 1)
    sim.step(int(args.equil / 0.002))                      # equilibration

    nframes = max(1, args.frames)
    stride = max(1, int((args.md / 0.002) / nframes))
    acc = defaultdict(list)
    ebplat = plat if getattr(args, 'energy_platform', 'auto') == 'auto' else \
        mm.Platform.getPlatformByName(args.energy_platform)
    ebprops = props if ebplat is plat else \
        {'Threads': os.environ.get('OPENMM_CPU_THREADS', '13')}
    bank = EnergyBank(ebplat, ebprops)
    try:
        for _ in range(nframes):
            sim.step(stride)
            P = sim.context.getState(getPositions=True).getPositions()
            rtop, rpos = _split(top, P, False)
            ptop, ppos = _split(top, P, True)
            E = {}
            for tag, (t, p) in (('c', (top, P)), ('r', (rtop, rpos)),
                                ('p', (ptop, ppos))):
                for k, v in bank.terms(tag, t, p).items():
                    E[tag + k] = v
            d_vdw = E['cvw'] - E['rvw'] - E['pvw']
            d_mm = E['cmm'] - E['rmm'] - E['pmm']
            d_coul = d_mm - d_vdw
            d_pol = (E['cgb'] - E['rgb'] - E['pgb']) - d_mm
            ds = _sasa(top, P) - _sasa(rtop, rpos) - _sasa(ptop, ppos)
            nsb, nhb = _contacts(top, P)
            acc['vdw'].append(d_vdw); acc['coul'].append(d_coul)
            acc['pol'].append(d_pol); acc['sasa'].append(ds)
            acc['np'].append(GAMMA * ds); acc['sb'].append(nsb); acc['hb'].append(nhb)
    finally:
        # Release everything this run allocated on the device before the next one
        # starts, otherwise memory accumulates until CUDA refuses a new context.
        bank.close()
        try:
            del sim.context
        except Exception:
            pass
        del sim, system, integ
        gc.collect()

    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    q = sum({'K': 1, 'R': 1, 'D': -1, 'E': -1}.get(c, 0) for c in seq)
    tot = mean['vdw'] + mean['coul'] + mean['pol'] + mean['np']
    return dict(dG_total=tot, dE_vdw=mean['vdw'], dE_coul=mean['coul'],
                dG_polar=mean['pol'], dG_nonpolar=mean['np'], dSASA=mean['sasa'],
                n_frames=nframes, n_saltbridge=int(mean['sb']),
                n_hbond=int(mean['hb']), pep_charge=q, runtime_s=time.time() - t0)


def stage_screen(args):
    global OUT_TSV
    if args.results:
        OUT_TSV = args.results
    elif args.shard:
        OUT_TSV = 'screen_results_shard{}of{}.tsv'.format(*args.shard.split('/'))
    plat, props, pname = _platform(getattr(args, 'platform', None))
    print(f'OpenMM platform: {pname}   properties: {props}')
    print(f'CPU thread cap: {os.environ.get("OPENMM_CPU_THREADS")}')
    if not os.path.exists('pocket_map.json'):
        stage_pockets(args)
    pocket = json.load(open('pocket_map.json'))
    pocket['contacts'] = {k: [int(x) for x in v]
                          for k, v in pocket['contacts'].items()}

    # A run counts as done if ANY results file in this directory contains it -- this
    # pod's own shard file, a file left over from an earlier unsharded run, or a
    # `screen_results_known.tsv` pushed out by the multi-pod driver carrying what the
    # other pods have already finished. Without this, pods duplicate each other's work.
    import glob as _glob
    done = set()
    seen_files = []
    for path in sorted(set(_glob.glob('screen_results*.tsv'))):
        try:
            with open(path) as fh:
                n0 = len(done)
                for l in fh:
                    f = l.rstrip('\n').split('\t')
                    if len(f) > 5 and f[0] != 'paralogue':
                        done.add((f[0], f[1], f[2], f[5]))
                if len(done) > n0:
                    seen_files.append(f'{path} (+{len(done) - n0})')
        except OSError:
            pass
    if seen_files:
        print('already complete, from: ' + ', '.join(seen_files))
    if not os.path.exists(OUT_TSV):
        open(OUT_TSV, 'w').write(HDR + '\n')

    todo = [(p, s, pid, seq, cls, sd)
            for p in args.paralogues for s in ('major', 'minor')
            for pid, seq, cls, _ in PEPTIDES for sd in range(args.seeds)
            if (p, s, pid, str(sd)) not in done]

    # Sharding: every worker builds the SAME ordered list and keeps every Nth entry.
    # Interleaving rather than splitting into blocks matters -- it spreads paralogues,
    # sites and peptide classes evenly, so a partial result set from any single shard
    # is still a usable cross-section rather than one corner of the panel.
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split('/'))
        except Exception:
            raise SystemExit('--shard must look like 2/4')
        if not 1 <= i <= n:
            raise SystemExit(f'--shard {args.shard} is out of range')
        todo = [t for k, t in enumerate(todo) if k % n == i - 1]
        print(f'shard {i} of {n}')

    # --priority moves the named peptides to the front of this shard's queue. Used when
    # topping a panel up to more seeds: the matched-pair peptides decide the statistics,
    # everything else only completes the table, so the decisive runs should land first.
    if getattr(args, 'priority', None):
        hot = [k for k, t in enumerate(todo) if any(x in t[2] for x in args.priority)]
        hotset = set(hot)
        todo = ([todo[k] for k in hot] +
                [t for k, t in enumerate(todo) if k not in hotset])
        print(f'priority: {len(hot)} of {len(todo)} runs moved to the front '
              f'({", ".join(args.priority)})')
    print(f'{len(todo)} runs to do '
          f'({args.md} ps production, {args.frames} frames, {args.seeds} seeds)')

    consecutive = 0
    for k, job in enumerate(todo, 1):
        p, s, pid, seq, cls, sd = job
        try:
            r = run_one(args, p, s, pid, seq, cls, sd, pocket, plat, props)
            consecutive = 0
        except Exception as e:
            consecutive += 1
            print(f'  FAIL {p} {s} {pid} seed{sd}: {type(e).__name__}: {e}', flush=True)
            if consecutive == 1:            # full traceback for the first one only
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
            if consecutive >= args.max_fails:
                print(f'\n{consecutive} consecutive failures - stopping rather than '
                      f'grinding through the rest of the panel.', flush=True)
                if 'CUDA' in str(e):
                    print('The GPU refused a context. Almost always one of:', flush=True)
                    print('  * another python process still holds the device'
                          ' -- check `nvidia-smi` and kill it', flush=True)
                    print('  * not enough free VRAM -- lower --frames, or run with'
                          ' --energy-platform CPU', flush=True)
                    print('  * driver/toolkit mismatch -- try --platform CPU to'
                          ' confirm the science path works', flush=True)
                raise SystemExit(1)
            continue
        row = [p, s, pid, seq, cls, str(sd)] + [
            (f'{r[c]:.2f}' if isinstance(r[c], float) else str(r[c]))
            for c in ('dG_total', 'dE_vdw', 'dE_coul', 'dG_polar', 'dG_nonpolar',
                      'dSASA', 'n_frames', 'n_saltbridge', 'n_hbond', 'pep_charge',
                      'runtime_s')]
        open(OUT_TSV, 'a').write('\t'.join(row) + '\n')
        print(f'[{k}/{len(todo)}] {p} {s:<5} {pid:<26} seed{sd} '
              f'dG={r["dG_total"]*KCAL:8.1f} kcal/mol  ({r["runtime_s"]:.0f}s)',
              flush=True)


# ================================================================== stage: tether
B_N_CA, B_CA_C, B_C_N = 1.458, 1.525, 1.329
A_N_CA_C, A_CA_C_N, A_C_N_CA = np.deg2rad(111.2), np.deg2rad(116.2), np.deg2rad(121.7)
B_CA_CB, A_N_CA_CB = 1.53, np.deg2rad(110.5)
BASINS = [(-63.0, -43.0, 0.42, 12.0), (-120.0, 130.0, 0.44, 25.0),
          (-90.0, 0.0, 0.10, 20.0), (57.0, 40.0, 0.04, 12.0)]
_BW = np.array([b[2] for b in BASINS]); _BW = _BW / _BW.sum()


def _place(a, b, c, bond, angle, torsion):
    bc = c - b; bc = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc); nn = np.linalg.norm(n)
    n = np.array([0.0, 0.0, 1.0]) if nn < 1e-8 else n / nn
    m = np.cross(n, bc)
    d = np.array([-bond * np.cos(angle), bond * np.sin(angle) * np.cos(torsion),
                  bond * np.sin(angle) * np.sin(torsion)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def _kabsch(fixed, moving):
    cf, cm = fixed.mean(0), moving.mean(0)
    U, S, Vt = np.linalg.svd((fixed - cf).T @ (moving - cm))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = (Vt.T @ D @ U.T).T
    return R, cm - cf @ R


def _minsep(a, b, block=4000):
    best = np.inf
    for i in range(0, len(a), block):
        best = min(best, np.linalg.norm(a[i:i + block, None] - b[None], axis=2).min())
        if best < 1.0:
            break
    return best


def stage_tether(args):
    from Bio.PDB import PDBParser
    rng = np.random.default_rng(args.tether_seed)
    P = PDBParser(QUIET=True)
    d = args.datadir
    ejl = P.get_structure('e', os.path.join(d, '1EJL.pdb'))[0]
    rec = np.array([a.get_coord() for r in ejl[RECCHAIN] if r.id[0] == ' '
                    for a in r if a.element != 'H'])
    peps = {s: {r.id[1]: r for r in ejl[c] if r.id[0] == ' '}
            for s, c in PEPCHAIN.items()}
    dim = P.get_structure('d', os.path.join(d, 'sam_dimer_fixed.pdb'))[0]
    chains = sorted(c.id for c in dim)
    SAMA = {r.id[1]: r for r in dim[chains[0]] if r.id[0] == ' '}
    SAMB = ({r.id[1]: r for r in dim[chains[1]] if r.id[0] == ' '}
            if len(chains) > 1 else None)

    def atoms(cr, lo, hi):
        return np.array([a.get_coord() for n, r in cr.items() if lo <= n <= hi
                         for a in r if a.element != 'H'])

    def bb(cr, nums):
        o = []
        for n in nums:
            r = cr.get(n)
            if r is None or not all(x in r for x in ('N', 'CA', 'C')):
                return None
            o.append([r['N'].get_coord(), r['CA'].get_coord(), r['C'].get_coord()])
        return np.array(o)

    def sample(n):
        i = rng.choice(len(BASINS), size=n, p=_BW)
        ph = np.array([BASINS[j][0] for j in i]) + rng.normal(0, 1, n) * \
            np.array([BASINS[j][3] for j in i])
        ps = np.array([BASINS[j][1] for j in i]) + rng.normal(0, 1, n) * \
            np.array([BASINS[j][3] for j in i])
        return np.deg2rad(ph), np.deg2rad(ps)

    REG = [('major', 'K71 at P2 (bound 69-75)', 69),
           ('major', 'R72 at P2 (bound 70-76)', 70),
           ('major', 'K73 at P2 (bound 71-77)', 71),
           ('major', 'K64 at P2 (bound 62-68)', 62),
           ('minor', 'K71 at P2 (bound 69-75)', 69),
           ('minor', 'R58 at P2 (bound 56-62)', 56)]

    print('=' * 96)
    print('TETHERED-ENSEMBLE TEST')
    print('  u = C-terminal SAM residues allowed to unwind')
    print('  f = fraction of sampled linker conformations placing the folded SAM domain')
    print('      without clashing (>= 3.0 A) while the NLS stays bound')
    print('  cost = -RT ln f at 298 K, kcal/mol  (conformational availability, not affinity)')
    print('=' * 96)
    for site, label, first in REG:
        pep = peps[site]
        anc = pep[PEP0]
        N0, CA0, C0 = (anc['N'].get_coord(), anc['CA'].get_coord(),
                       anc['C'].get_coord())
        obstacle = _obstacle_tree(np.vstack(
            [rec, np.array([a.get_coord() for r in pep.values()
                            for a in r if a.element != 'H'])]))
        rec_only = _obstacle_tree(rec)   # linker is bonded to the peptide, so it is
                                         # tested against the receptor only
        print(f'\n--- {site} site, {label} ---')
        print(f'{"u":>3}{"SAM folded to":>15}{"f monomer":>12}{"cost":>9}'
              f'{"f dimer":>11}{"cost":>9}')
        for u in args.unwind:
            nflex, nbuild = u, u + 3
            rigid_hi = first - 1 - nflex
            anchor = [rigid_hi, rigid_hi - 1, rigid_hi - 2]
            ref = bb(SAMA, anchor) if rigid_hi - 2 >= min(SAMA) else None
            if ref is None:
                print(f'{u:>3}{rigid_hi:>15}{"n/a":>12}{"":>9}{"n/a":>11}')
                continue
            own = atoms(SAMA, min(SAMA), rigid_hi)
            par = atoms(SAMB, min(SAMB), 95) if SAMB else None
            okm = okd = 0
            for _ in range(args.tether_samples):
                ph, ps = sample(nbuild)
                N, CA, C = N0, CA0, C0
                grown = []
                for k in range(nbuild):
                    Cp = _place(C, CA, N, B_C_N, A_C_N_CA, ps[k])
                    CAp = _place(CA, N, Cp, B_CA_C, A_CA_C_N, np.pi)
                    Np = _place(N, Cp, CAp, B_N_CA, A_N_CA_C, ph[k])
                    grown.append((Np, CAp, Cp))
                    N, CA, C = Np, CAp, Cp
                got = []
                for res in anchor:
                    k = first - 1 - res
                    if not (0 <= k < len(grown)):
                        got = None; break
                    got.append(grown[k])
                if not got:
                    continue
                R, t = _kabsch(ref.reshape(-1, 3), np.array(got).reshape(-1, 3))
                if len(grown) > 1 and not _clashfree(
                        np.array([a for gg in grown[1:] for a in gg]), rec_only, 2.7):
                    continue                      # the linker itself must fit too
                if _clashfree(own @ R + t, obstacle):
                    okm += 1
                    if par is not None and _clashfree(par @ R + t, obstacle):
                        okd += 1
            fm, fd = okm / args.tether_samples, okd / args.tether_samples
            cm_ = f'{-RT*math.log(fm):7.2f}' if fm > 0 else '    inf'
            cd_ = f'{-RT*math.log(fd):7.2f}' if fd > 0 else '    inf'
            print(f'{u:>3}{rigid_hi:>15}{fm:>12.4f}{cm_:>9}{fd:>11.4f}{cd_:>9}')


# ================================================================= stage: analyse
def stage_analyse(args):
    import glob
    files = ([args.results] if getattr(args, 'results', None)
             else sorted(set(glob.glob('screen_results.tsv') +
                             glob.glob('screen_results_shard*.tsv'))))
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print('no results file'); return
    if len(files) > 1:
        print(f'merging {len(files)} shard files: {", ".join(files)}\n')
    rows = []
    for path in files:
      with open(path) as fh:
        hdr = fh.readline().rstrip('\n').split('\t')
        for l in fh:
            f = l.rstrip('\n').split('\t')
            if len(f) != len(hdr):
                continue
            d = dict(zip(hdr, f))
            for k in ('dG_total', 'dE_vdw', 'dE_coul', 'dG_polar', 'dG_nonpolar',
                      'dSASA'):
                d[k] = float(d[k])
            d['pep_charge'] = int(d['pep_charge'])
            rows.append(d)
    if not rows:
        print('no results'); return

    agg = defaultdict(list)
    for r in rows:
        agg[(r['paralogue'], r['site'], r['peptide'])].append(r)

    for para in sorted({r['paralogue'] for r in rows}):
        for site in ('major', 'minor'):
            keys = [k for k in agg if k[0] == para and k[1] == site]
            if not keys:
                continue
            print('\n' + '=' * 104)
            print(f'{para}   {site.upper()} SITE   MM-GBSA, kcal/mol (relative only)')
            print('=' * 104)
            print(f'{"peptide":<27}{"seq":<9}{"cls":<7}{"dG":>9}{"sd":>6}{"vdW":>8}'
                  f'{"Coul":>10}{"polar":>10}{"nonpol":>8}{"SB":>4}{"HB":>4}{"q":>4}')
            print('-' * 104)
            recs = []
            for k in keys:
                rs = agg[k]
                m = {c: float(np.mean([r[c] for r in rs])) for c in
                     ('dG_total', 'dE_vdw', 'dE_coul', 'dG_polar', 'dG_nonpolar')}
                recs.append((m['dG_total'], k[2], rs[0]['seq'], rs[0]['class'], m,
                             float(np.std([r['dG_total'] for r in rs])),
                             int(np.mean([int(r['n_saltbridge']) for r in rs])),
                             int(np.mean([int(r['n_hbond']) for r in rs])),
                             rs[0]['pep_charge']))
            for dg, pid, seq, cls, m, sd, sb, hb, q in sorted(recs):
                print(f'{pid:<27}{seq:<9}{cls:<7}{dg*KCAL:>9.1f}{sd*KCAL:>6.1f}'
                      f'{m["dE_vdw"]*KCAL:>8.1f}{m["dE_coul"]*KCAL:>10.1f}'
                      f'{m["dG_polar"]*KCAL:>10.1f}{m["dG_nonpolar"]*KCAL:>8.1f}'
                      f'{sb:>4}{hb:>4}{q:>+4d}')
            pos = [r[0] for r in recs if r[3] == 'pos']
            neg = [r[0] for r in recs if r[3] == 'neg']
            if pos and neg:
                print(f'\n  positives {np.mean(pos)*KCAL:7.1f}   '
                      f'negatives {np.mean(neg)*KCAL:7.1f}   '
                      f'separation {(np.mean(neg)-np.mean(pos))*KCAL:7.1f} kcal/mol')
            q = np.array([r[8] for r in recs], float)
            y = np.array([r[0] for r in recs], float) * KCAL
            if len(set(q)) > 1:
                A = np.vstack([q, np.ones_like(q)]).T
                sol = np.linalg.lstsq(A, y, rcond=None)[0]
                resid = y - A @ sol
                r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
                print(f'  charge regression: dG = {sol[0]:.1f}*q {sol[1]:+.1f}, '
                      f'R2 = {r2:.2f}  ->  {r2*100:.0f}% of the spread is net charge alone')
                print('\n  charge-corrected ranking (residual, more negative = better '
                      'than its charge predicts):')
                for i in np.argsort(resid):
                    print(f'    {resid[i]:+8.1f}  {recs[i][1]:<27}{recs[i][2]:<9}'
                          f'{recs[i][3]}')

    print('\n' + '=' * 104)
    print('COMPOSITION-MATCHED PAIRS -- the only charge-bias-free comparisons')
    print('=' * 104)
    for para in sorted({r['paralogue'] for r in rows}):
        for site in ('major', 'minor'):
            for a, b in MATCHED_PAIRS:
                ka, kb = (para, site, a), (para, site, b)
                if ka in agg and kb in agg:
                    ma = np.mean([r['dG_total'] for r in agg[ka]]) * KCAL
                    mb = np.mean([r['dG_total'] for r in agg[kb]]) * KCAL
                    nz = max(np.std([r['dG_total'] for r in agg[ka]]) * KCAL,
                             np.std([r['dG_total'] for r in agg[kb]]) * KCAL, 1.0)
                    v = 'REAL' if abs(ma - mb) > 2 * nz else 'in noise'
                    print(f'  {para} {site:<6} {a:<25}{ma:8.1f}   vs  {b:<25}{mb:8.1f}'
                          f'   diff {ma-mb:+7.1f}   [{v}]')


# ==================================================================== entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all',
                    choices=['all', 'pockets', 'screen', 'tether', 'analyse'])
    ap.add_argument('--datadir', default='.')
    ap.add_argument('--paralogues', nargs='+',
                    default=['KPNA1', 'KPNA2', 'KPNA3', 'KPNA4'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--equil', type=float, default=100.0, help='equilibration, ps')
    ap.add_argument('--md', type=float, default=400.0, help='production, ps')
    ap.add_argument('--frames', type=int, default=20, help='MM-GBSA frames')
    ap.add_argument('--platform', default=None,
                    choices=['CUDA', 'HIP', 'OpenCL', 'CPU'],
                    help='force an OpenMM platform (default: best available)')
    ap.add_argument('--energy-platform', default='auto',
                    choices=['auto', 'CPU', 'CUDA', 'OpenCL'],
                    help='platform for the MM-GBSA single points. "auto" uses the same '
                         'one as the MD; CPU keeps GPU memory free for the dynamics')
    ap.add_argument('--priority', nargs='*', default=[],
                    help='peptide-name substrings run first within this shard')
    ap.add_argument('--shard', default=None, metavar='I/N',
                    help='run only every Nth job, offset I (1-based): --shard 2/4 on '
                         'the second of four pods. Each pod writes its own results '
                         'file; concatenate them afterwards.')
    ap.add_argument('--results', default=None,
                    help='results filename (default screen_results.tsv, or '
                         'screen_results_shardIofN.tsv when --shard is used)')
    ap.add_argument('--max-fails', type=int, default=5,
                    help='abort after this many consecutive failed runs')
    ap.add_argument('--minimise', type=int, default=2000,
                    help='minimisation iterations before MD')
    ap.add_argument('--restraint', type=float, default=200.0,
                    help='receptor backbone restraint, kJ/mol/nm^2')
    ap.add_argument('--tether-samples', type=int, default=20000)
    ap.add_argument('--tether-seed', type=int, default=20260822)
    ap.add_argument('--unwind', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument('--trunc', type=int, nargs=2, default=list(RECRANGE),
                    metavar=('LO', 'HI'),
                    help='receptor residue range to keep; the default is the whole '
                         'construct, which is what you want on a GPU')
    ap.add_argument('--quick', action='store_true',
                    help='tiny run to check the install works')
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.equil, a.md, a.frames, a.minimise = 1, 5.0, 10.0, 3, 300
        a.paralogues = ['KPNA2']
        a.tether_samples = 2000
    if a.stage in ('all', 'pockets'):
        stage_pockets(a)
    if a.stage in ('all', 'screen'):
        stage_screen(a)
    if a.stage in ('all', 'tether'):
        stage_tether(a)
    if a.stage in ('all', 'analyse'):
        stage_analyse(a)


if __name__ == '__main__':
    main()
