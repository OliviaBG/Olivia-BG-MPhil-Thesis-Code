#!/usr/bin/env python3
"""
Extend the ACK1 NLS screen to beta-family import receptors.

Same method as the importin-alpha panel: thread the test 7-mer onto a peptide that a
crystal structure shows bound in the receptor's groove, rebuild side chains, run
restrained MD in Amber14/GBn2, score by ensemble MM-GBSA.

RECEPTORS INCLUDED, AND WHY
  importin-beta1   1M5N, HEAT 1-11 with the PTHrP non-classical NLS (67-94). The
                   basic core PGKKKKGKP sits in the beta1 groove as an extended
                   peptide -- the direct-beta1 import route, bypassing importin-alpha.
  transportin-1    5J3V, Kapbeta2 with the histone H3 tail (11-27) in the PY-NLS site.
                   H3 binds through a basic Epitope 1 WITHOUT the usual proline-tyrosine
                   motif, which is exactly ACK1's situation.

DELIBERATELY EXCLUDED
  importin-7  (6N88, 9QF0)  cargo is the FOLDED globular domain of H1.0 (residues
  importin-9  (6N1Z)        24-97), or the folded H2A-H2B core. These receptors do not
  importin-4  (7UNK)        bind a linear peptide in a groove, so threading a 7-mer
                            would produce a number with no physical meaning.
  importin-5  (6XTE, 6XU2)  human structures are apo -- no cargo to superpose onto.

A NOTE ON COMPARING ACROSS RECEPTORS
Absolute dG is NOT comparable between receptors: different interface sizes, different
templates, different burial. What IS comparable is each receptor's own matched-pair
margin -- ACK1 minus its own composition-matched scramble, within that receptor. Build
any cross-receptor claim on that, not on the raw ranking.

Usage:
    python3 beta_screen.py --stage screen --shard 1/6 --seeds 6 --md 1000
"""
import argparse, gc, json, os, sys, time
from collections import defaultdict

_THREADS = os.environ.get('ACK1_THREADS', '13')
for _v in ('OPENMM_CPU_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, _THREADS)

import numpy as np

import ack1_importin_gpu as G          # reuse EnergyBank, _make_system, _split, _sasa...

KCAL = 1 / 4.184
GAMMA = G.GAMMA

# receptor: (pdb, receptor chain, peptide chain, first residue of the 7-mer window)
RECEPTORS = {
    # p0 chosen so ACK1's own basic core (positions 3-5 of the 7-mer) lands on the
    # template positions that are actually buried, not merely on any basic residues.
    'IMPB1': dict(pdb='1M5N.pdb', rec='S', pep='Q', p0=86,
                  note='importin-beta1 + PTHrP NLS 86-92 PGKKKKG; ACK1 KRK maps to '
                       'K88-K89-K90'),
    'TNPO1': dict(pdb='5J3V.pdb', rec='A', pep='C', p0=11,
                  note='transportin-1 + histone H3 11-17 TGGKAPR (Epitope 1); ACK1 '
                       'KRK maps to G13-K14-A15, K14 the most buried residue'),
}

PEPTIDES = G.PEPTIDES
MATCHED_PAIRS = G.MATCHED_PAIRS
HDR = ('receptor\tsite\tpeptide\tseq\tclass\tseed\tdG_total\tdE_vdw\tdE_coul\tdG_polar'
       '\tdG_nonpolar\tdSASA\tn_frames\tn_saltbridge\tn_hbond\tpep_charge\truntime_s')


def carve(spec, path, datadir='.'):
    """receptor + the 7-residue peptide window; peptide relabelled chain P"""
    from Bio.PDB import PDBParser, PDBIO, Select
    p0 = spec['p0']

    class Keep(Select):
        def accept_chain(self, c):
            return c.id in (spec['rec'], spec['pep'])

        def accept_residue(self, r):
            if r.id[0] != ' ':
                return 0
            if r.get_parent().id == spec['rec']:
                return 1
            return 1 if p0 <= r.id[1] <= p0 + 6 else 0

        def accept_atom(self, a):
            return 0 if a.element == 'H' else 1

    st = PDBParser(QUIET=True).get_structure('t', os.path.join(datadir, spec['pdb']))
    # some entries have several copies of the complex; keep the first model only
    while len(st) > 1:
        st.detach_child(list(st)[-1].id)
    stage = f'{path}.{os.getpid()}.tmp'
    io_ = PDBIO(); io_.set_structure(st); io_.save(stage, Keep())
    out = []
    for l in open(stage):
        if l.startswith(('ATOM', 'HETATM')) and l[21] == spec['pep']:
            l = l[:21] + 'P' + l[22:]
        out.append(l)
    with open(stage, 'w') as fh:
        fh.writelines(out)
    os.replace(stage, path)


def template_seq(spec, datadir='.'):
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1 as t31
    m = PDBParser(QUIET=True).get_structure(
        't', os.path.join(datadir, spec['pdb']))[0]
    res = {r.id[1]: r for r in m[spec['pep']] if r.id[0] == ' '}
    return ''.join(t31.get(res[i].get_resname(), 'X')
                   for i in range(spec['p0'], spec['p0'] + 7))


def contacts(spec, datadir='.', cut=5.0):
    from Bio.PDB import PDBParser
    m = PDBParser(QUIET=True).get_structure(
        't', os.path.join(datadir, spec['pdb']))[0]
    rec = {r.id[1]: r for r in m[spec['rec']] if r.id[0] == ' '}
    pep = [r for r in m[spec['pep']] if r.id[0] == ' '
           and spec['p0'] <= r.id[1] <= spec['p0'] + 6]
    pc = np.array([a.get_coord() for r in pep for a in r if a.element != 'H'])
    out = []
    for i, r in rec.items():
        rc = np.array([a.get_coord() for a in r if a.element != 'H'])
        if np.linalg.norm(rc[:, None, :] - pc[None, :, :], axis=2).min() < cut:
            out.append(i)
    return sorted(out)


def run_one(args, rname, pid, seq, cls, seed, plat, props, cache):
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    from pdbfixer import PDBFixer

    t0 = time.time()
    spec = RECEPTORS[rname]
    src = f'_beta_{rname}.pdb'
    if os.path.exists(src) and not G._sane_pdb(src):
        os.remove(src)
    if not os.path.exists(src):
        carve(spec, src, args.datadir)
    tmpl = cache.setdefault(rname + '_seq', template_seq(spec, args.datadir))
    free = set(cache.setdefault(rname + '_ct', contacts(spec, args.datadir)))

    fx = PDBFixer(filename=src)
    mp = [f'{G.ONE2THREE[tmpl[k]]}-{spec["p0"] + k}-{G.ONE2THREE[seq[k]]}'
          for k in range(7) if seq[k] != tmpl[k]]
    if mp:
        fx.applyMutations(mp, 'P')
    fx.findMissingResidues(); fx.missingResidues = {}
    fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.4)
    top, pos = fx.topology, fx.positions

    system = G._make_system(top, gb=True)
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
    sim.step(int(args.equil / 0.002))

    nframes = max(1, args.frames)
    stride = max(1, int((args.md / 0.002) / nframes))
    acc = defaultdict(list)
    bank = G.EnergyBank(plat, props)
    try:
        for _ in range(nframes):
            sim.step(stride)
            P = sim.context.getState(getPositions=True).getPositions()
            rtop, rpos = G._split(top, P, False)
            ptop, ppos = G._split(top, P, True)
            E = {}
            for tag, (t, p) in (('c', (top, P)), ('r', (rtop, rpos)),
                                ('p', (ptop, ppos))):
                for k, v in bank.terms(tag, t, p).items():
                    E[tag + k] = v
            d_vdw = E['cvw'] - E['rvw'] - E['pvw']
            d_mm = E['cmm'] - E['rmm'] - E['pmm']
            d_coul = d_mm - d_vdw
            d_pol = (E['cgb'] - E['rgb'] - E['pgb']) - d_mm
            ds = G._sasa(top, P) - G._sasa(rtop, rpos) - G._sasa(ptop, ppos)
            nsb, nhb = G._contacts(top, P)
            acc['vdw'].append(d_vdw); acc['coul'].append(d_coul)
            acc['pol'].append(d_pol); acc['sasa'].append(ds)
            acc['np'].append(GAMMA * ds); acc['sb'].append(nsb); acc['hb'].append(nhb)
    finally:
        bank.close()
        try:
            del sim.context
        except Exception:
            pass
        del sim, system, integ
        gc.collect()

    m = {k: float(np.mean(v)) for k, v in acc.items()}
    q = sum({'K': 1, 'R': 1, 'D': -1, 'E': -1}.get(c, 0) for c in seq)
    tot = m['vdw'] + m['coul'] + m['pol'] + m['np']
    return dict(dG_total=tot, dE_vdw=m['vdw'], dE_coul=m['coul'], dG_polar=m['pol'],
                dG_nonpolar=m['np'], dSASA=m['sasa'], n_frames=nframes,
                n_saltbridge=int(m['sb']), n_hbond=int(m['hb']), pep_charge=q,
                runtime_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='screen')
    ap.add_argument('--datadir', default='.')
    ap.add_argument('--receptors', nargs='+', default=list(RECEPTORS))
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--equil', type=float, default=150.0)
    ap.add_argument('--md', type=float, default=1000.0)
    ap.add_argument('--frames', type=int, default=25)
    ap.add_argument('--minimise', type=int, default=2000)
    ap.add_argument('--restraint', type=float, default=200.0)
    ap.add_argument('--shard', default=None)
    ap.add_argument('--priority', nargs='*', default=[],
                    help='peptide-name substrings; matching runs are moved to the '
                         'front of the queue. Use this when one pod has to do a long '
                         'queue and you want the decisive comparisons first.')
    ap.add_argument('--results', default=None)
    ap.add_argument('--platform', default=None)
    ap.add_argument('--max-fails', type=int, default=5)
    a = ap.parse_args()

    out = a.results or (
        'beta_results_shard{}of{}.tsv'.format(*a.shard.split('/')) if a.shard
        else 'beta_results.tsv')

    plat, props, pname = G._platform(a.platform)
    print(f'OpenMM platform: {pname}   thread cap {os.environ.get("OPENMM_CPU_THREADS")}',
          flush=True)
    for r in a.receptors:
        print(f'  {r}: {RECEPTORS[r]["note"]}', flush=True)

    import glob
    done = set()
    for path in sorted(set(glob.glob('beta_results*.tsv'))):
        try:
            for l in open(path):
                f = l.rstrip('\n').split('\t')
                if len(f) > 5 and f[0] != 'receptor':
                    done.add((f[0], f[2], f[5]))
        except OSError:
            pass
    if done:
        print(f'already complete: {len(done)} runs', flush=True)
    if not os.path.exists(out):
        open(out, 'w').write(HDR + '\n')

    todo = [(r, pid, seq, cls, sd)
            for r in a.receptors
            for pid, seq, cls, _ in PEPTIDES
            for sd in range(a.seeds)
            if (r, pid, str(sd)) not in done]
    if a.shard:
        i, n = (int(x) for x in a.shard.split('/'))
        todo = [t for k, t in enumerate(todo) if k % n == i - 1]
        print(f'shard {i} of {n}', flush=True)

    if a.priority:
        hot = [k for k, t in enumerate(todo)
               if any(s in t[1] for s in a.priority)]
        hotset = set(hot)
        todo = ([todo[k] for k in hot] +
                [t for k, t in enumerate(todo) if k not in hotset])
        print(f'priority: {len(hot)} runs moved to the front '
              f'({", ".join(a.priority)})', flush=True)

    print(f'{len(todo)} runs to do ({a.md} ps, {a.frames} frames, {a.seeds} seeds)',
          flush=True)

    cache, consecutive = {}, 0
    for k, (r, pid, seq, cls, sd) in enumerate(todo, 1):
        try:
            res = run_one(a, r, pid, seq, cls, sd, plat, props, cache)
            consecutive = 0
        except Exception as e:
            consecutive += 1
            print(f'  FAIL {r} {pid} seed{sd}: {type(e).__name__}: {e}', flush=True)
            if consecutive == 1:
                import traceback; traceback.print_exc(); sys.stdout.flush()
            if consecutive >= a.max_fails:
                print(f'\n{consecutive} consecutive failures - stopping.', flush=True)
                raise SystemExit(1)
            continue
        row = [r, 'groove', pid, seq, cls, str(sd)] + [
            (f'{res[c]:.2f}' if isinstance(res[c], float) else str(res[c]))
            for c in ('dG_total', 'dE_vdw', 'dE_coul', 'dG_polar', 'dG_nonpolar',
                      'dSASA', 'n_frames', 'n_saltbridge', 'n_hbond', 'pep_charge',
                      'runtime_s')]
        open(out, 'a').write('\t'.join(row) + '\n')
        print(f'[{k}/{len(todo)}] {r:<6} {pid:<26} seed{sd} '
              f'dG={res["dG_total"] * KCAL:8.1f} kcal/mol  ({res["runtime_s"]:.0f}s)',
              flush=True)


if __name__ == '__main__':
    main()
