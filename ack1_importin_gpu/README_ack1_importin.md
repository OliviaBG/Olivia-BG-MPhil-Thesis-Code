# ACK NLS vs importin-α -- structural modelling and physics scoring

No AlphaFold anywhere in this pipeline. Crystallographic templates, Amber14 force field,
implicit-solvent MD, MM-GBSA.

---

## Deploying to the GPU pod -- from WSL

The files land in the Windows folder, which WSL sees at
`ack1_importin_gpu/` inside this repository. Don't work from there --
`/mnt/c` ignores `chmod`, which breaks SSH keys, and it round-trips line endings. Make a
WSL-native copy first:

```bash
cd <this repository>/ack1_importin_gpu
bash wsl_setup.sh
```

That copies everything to `~/AlphaFold/ack1_importin_gpu`, strips any CRLF, sets the
execute bits, checks `ssh`/`scp` are installed, and checks your key exists with mode 600
-- telling you exactly what to run if it doesn't. Then:

```bash
cd ~/AlphaFold/ack1_importin_gpu
bash deploy_to_pod.sh
```

**The SSH key must live in the WSL home**, not on `/mnt/c`. If yours is on the Windows
side:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp /mnt/c/Users/<windows-username>/.ssh/id_ed25519 ~/.ssh/
chmod 600 ~/.ssh/id_ed25519
```

`deploy_to_pod.sh` checks and fixes the key mode itself, and warns if the key is
somewhere `chmod` can't take effect.

That copies everything to `/workspace/ack1` on the pod and then, on the pod, installs
Miniforge, creates the `ack1` conda environment, installs OpenMM/PDBFixer/Biopython/
NumPy/SciPy, caps every thread pool at **13**, prints the GPU and platform check, and
runs a smoke test. Five to fifteen minutes the first time; re-running it is harmless.

Defaults are baked in and overridable:

```bash
POD_HOST=<pod-host> POD_PORT=13087 POD_USER=root \
KEY=~/.ssh/id_ed25519 REMOTE_DIR=/workspace/ack1 THREADS=13 bash deploy_to_pod.sh
```

The thread cap is written to `$CONDA_PREFIX/etc/conda/activate.d/ack1_threads.sh` and to
`~/.bashrc`, so it applies to every future shell on the pod, not just this one. It covers
`OPENMM_CPU_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` -- NumPy's BLAS would otherwise
oversubscribe the box regardless of what OpenMM does. The pipeline also sets these itself
before importing NumPy or OpenMM, so it stays capped even if something is run outside the
env.

## Running it

Drive the pod from WSL without logging in -- `pod.sh` stays on your side:

```bash
bash pod.sh start        # launches under nohup, returns immediately
bash pod.sh status       # alive?, rows done, last log lines, GPU utilisation
bash pod.sh follow       # live log; Ctrl-C stops watching, not the job
bash pod.sh analyse      # summary; part-way through is fine
bash pod.sh fetch        # copy screen_results.tsv / analysis.txt back here
bash pod.sh stop         # kill it; finished runs are kept, start resumes
```

Or by hand on the pod:

```bash
ssh -i ~/.ssh/id_ed25519 -p 13087 root@<pod-host>
cd /workspace/ack1
setsid nohup python -u ack1_importin_gpu.py --stage screen > screen.log 2>&1 < /dev/null &
echo $! > screen.pid
exit                                    # the job keeps running
```

`setsid` detaches the process from the SSH channel, `< /dev/null` stops it blocking on
stdin, and `-u` keeps Python unbuffered so `screen.log` is readable while it runs. Plain
`nohup ... &` without those will often make SSH hang on logout waiting for the channel to
close.

Check back later:

```bash
ssh -i ~/.ssh/id_ed25519 -p 13087 root@<pod-host> \
  'tail -n 20 /workspace/ack1/screen.log'
```

```bash
python ack1_importin_gpu.py --quick               # 10 min, checks the install works
python ack1_importin_gpu.py --stage analyse       # summary, any time, part-way is fine
```

Pull results back into WSL, then across to Windows if you want them there:

```bash
cd ~/AlphaFold/ack1_importin_gpu
scp -i ~/.ssh/id_ed25519 -P 13087 root@<pod-host>:/workspace/ack1/screen_results.tsv .
cp screen_results.tsv <this repository>/ack1_importin_gpu/
```

### If you would rather set it up by hand

```bash
conda create -n ack1 -c conda-forge python=3.11 openmm pdbfixer biopython numpy scipy
conda activate ack1
export OPENMM_CPU_THREADS=13 OMP_NUM_THREADS=13 ACK1_THREADS=13
python -m openmm.testInstallation        # must list CUDA
```

The pipeline is self-contained -- one `.py` file, no local imports. It needs these
alongside it:

```
ack1_importin_gpu.py     the pipeline
1EJL.pdb                 importin-α ΔIBB with SV40 NLS bound in BOTH sites
sam_dimer_fixed.pdb      the ACK SAM dimer structure -- NOT included in this
                         repository; supply your own (see note below)
kpna_seqs.fasta          KPNA1-4 human + Kpna2 mouse
```

The default run is 4 paralogues × 2 sites × 17 peptides × 3 seeds = 408 MD runs of
100 ps equilibration + 400 ps production. On a decent GPU that is roughly overnight.
Results append to `screen_results.tsv` and completed runs are skipped on restart, so
you can kill it and resume freely.

Useful flags:

```
--paralogues KPNA2              just one receptor
--seeds 5 --md 1000             more sampling
--trunc 95 290                  carve the receptor down (faster; major site only)
--minimise 5000                 harder minimisation before MD
--stage tether                  the SAM-occlusion calculation alone (CPU, ~5 min)
--platform CUDA                 force a platform instead of auto-detecting
```

`--priority` moves peptides whose names contain the given substrings to the front of a
shard's queue. This matters when topping a panel up to more seeds: the matched-pair
peptides determine the statistics, so the decisive runs should finish first.

`ACK1_GPU=1` selects a different CUDA device if the pod has more than one; run several
`--paralogues` in parallel on separate GPUs by pointing each at its own working
directory, since they all append to `screen_results.tsv`.

---

## What the four stages do

**`pockets`** -- finds every receptor residue within 5 Å of the bound NLS peptide in each
site and works out which of them differ between KPNA1/2/3/4. Writes `pocket_map.json`.

**`screen`** -- for each (paralogue, site, peptide): thread the peptide onto the
crystallographic NLS, mutate the receptor first shell to that paralogue, rebuild side
chains and hydrogens at pH 7.4, restrained MD in Amber14/GBn2, ensemble MM-GBSA over the
production frames.

**`tether`** -- the steric question. Holds the NLS in the bound conformation, grows the
chain backwards from it sampling φ/ψ, docks the folded SAM domain onto the grown stretch
and tests for clashes against the receptor -- in the monomer and in the SAM dimer. Answer
is a fraction of viable conformations *versus how many residues of SAM α5 are allowed to
unwind*. CPU-only, minutes.

**`analyse`** -- ranking, control calibration, charge regression, matched-pair test.

---

## How to read the MM-GBSA numbers -- important

They are **relative scores, not affinities**. For a +3/+4 peptide binding an acidic ARM
groove, MM-GBSA systematically over-rewards net charge. Three things are built in so this
doesn't fool either of us:

1. **Composition-matched scrambles.** `NEG_scramble_of_71reg` (WKSLRCK) has exactly the
   same amino acids as `ACK1_71-73_K71atP2` (LCKRKSW), and likewise for the 64-67
   register. A difference between a sequence and its own scramble cannot be a charge
   artefact -- it is real register-dependent complementarity. These are the comparisons
   to quote.

2. **Charge regression.** `analyse` fits ΔG against net peptide charge and reports R².
   If R² is 0.9, then 90% of the apparent spread is just counting lysines, and the
   charge-corrected residual ranking is what actually matters.

3. **A control ladder.** Three positives (SV40, nucleoplasmin, c-Myc) and five negatives.
   If the positives do not separate from the negatives, the panel has failed its own
   calibration and nothing else in it should be believed -- exactly the failure mode the
   AlphaFold3 co-folding panel hit.

Your own constructs are in the panel as in-silico counterparts: `MUT_71KRK73QQQ`,
`MUT_64QQQQ67`, `MUT_64EEEE67`.

---

## Results already obtained

### Pocket map -- the mouse-template caveat is gone

Mouse importin-α2 and human KPNA2 are **identical at all 56 first-shell positions**
(28 major + 28 minor). For pocket purposes 1EJL *is* a human KPNA2 structure.

| | major site | minor site |
|---|---|---|
| KPNA2 | identical to template | identical to template |
| KPNA1 | R106K, K108P, T151N, N239G | T279S, T324D |
| KPNA3 | R106S, E107D, K108R | P282G, S406I, G407S |
| KPNA4 | R106S, E107D, K108R, N239H | G281A, P282G, S406I, G407S |

The Trp/Asn ladder -- W142, W184, W231, W357, W399, N146, N188, N235, N361, N403 -- is
**invariant across all four paralogues**. So the P2 anchor chemistry is identical
everywhere and any selectivity must be flank-driven, at 106-108 in the major site and
279-282 / 406-407 in the minor site. Prediction: weak paralogue selectivity, and if any
exists it will show up in the minor site.

### Tether -- this is the substantive result

`ack1_tether_results.txt`. Fraction of viable conformations, monomer / dimer:

| register | u=0 | u=1 | u=2 | u=3 | u=4 |
|---|---|---|---|---|---|
| **major, K71 at P2** | 0 / 0 | 0.0003 / 0 | 0.008 / 0.006 | 0.038 / 0.014 | 0.047 / 0.022 |
| **minor, K71 at P2** | 0 / 0 | **0.113 / 0.111** | **0.207 / 0.192** | 0.276 / 0.216 | 0.329 / 0.264 |
| major, K64 at P2 | 0 / 0 | 0 / 0 | 0 / 0 | 0.009 / **0** | 0.030 / **0.001** |
| minor, R58 at P2 | 0 / 0 | 0.069 / **0** | 0.151 / **0.0003** | 0.311 / **0.003** | 0.263 / **0.007** |

Three conclusions:

**1. SAM α5 must unwind. Not a hypothesis any more.** With the domain folded through
residue 68/69 there are *zero* viable conformations in every register tested. At least
one to two residues of α5 have to fray before importin-α can engage anything.

**2. The minor site is 15-375× more permissive than the major site** for K71-R72-K73, at
every unwinding depth. At u=1 the ratio is 375×; at u=2 it is 28×. Most monopartite
cNLSs prefer the major site -- ACK's proximity to a folded domain would push it to the
minor site instead. That is the falsifiable prediction, and it is separable
experimentally: minor-site binding depends on W357/N361/E396/W399, major-site on
W142/W184/N188/D192.

**3. Dimerisation suppresses the competing sites and leaves 71-73 alone.** This is the
part that speaks directly to your mutagenesis. For K71-R72-K73 the dimer costs almost
nothing (0.207 0.192 at u=2, minor site). For K64-K67 and for R57-R58 the dimer is
close to lethal -- R58 goes from 0.151 to 0.0003, a 500× penalty; K64 from 0.009 to zero.

So the model predicts that in the SAM dimer, **71-73 is the only basic cluster that is
usable at all**. That is a mechanistic explanation for why 64-67 threads better onto the
importin register (4/5 pockets vs 3/5) yet is functionally silent in cells, and why
57RRQQ58 is silent too -- R58 is part of the End-Helix dimer interface, so it cannot be an
NLS anchor and a dimerisation residue at the same time.

---

## Caveats to keep attached to the tether numbers

- Backbone + Cβ only; linker side chains are not modelled, so *f* is an **over**estimate
  and the real cost is higher than quoted.
- φ/ψ come from coarse Ramachandran basins, not sequence-specific propensities, and the
  linker is not tested for self-avoidance (only against the receptor).
- The enthalpic cost of unwinding α5 is **not included**. The table is conformational
  availability only. A proper helix-coil calculation (AGADIR or Lifson-Roig) would add
  roughly 1 kcal/mol per residue unwound on top.
- The rigid SAM core is assumed intact at every *u*, which stops being physical once *u*
  is large -- at u≥4 in the R58 register most of α5 is gone and the domain would not fold
  or dimerise at all. That reinforces rather than weakens conclusion 3.
- The SAM dimer coordinates come from an experimental NMR structure, solved in
  solution rather than crystallographically.
- 3.0 Å is a hard-clash criterion, not an energy.

---

## The SAM dimer structure is not included

`sam_dimer_fixed.pdb` is an unpublished structure and is not distributed
with this repository. The scripts are otherwise unmodified and still expect
it, so `--stage tether` and `../ack1_nls_af3/helix_cost.py` will not run,
and the deployment scripts here list it among the files they copy to a pod
and will report it missing (`deploy_to_pod.sh` checks first and stops).
`pod_bootstrap.sh` also runs `--stage tether` on the pod, so that step fails
there as well.

The `pockets`, `screen` and `analyse` stages do not read it. The MM-GBSA
screen reproduces in full without it. Supply your own SAM dimer coordinates
under that filename to run the rest.

## Files

| file | what |
|---|---|
| `ack1_importin_gpu.py` | the whole pipeline, self-contained |
| `1EJL.pdb`, `kpna_seqs.fasta` | inputs, included |
| `sam_dimer_fixed.pdb` | **not included** in this repository -- see the note below |
| `pocket_map.json` | written by the `pockets` stage |
| `screen_results.tsv` | written by the `screen` stage, one row per run |
| `ack1_tether_results.txt` | tether output from a previous run |
| `alpha_topup_pod.sh` | extends the importin-alpha panel from 3 seeds to 6, one pod per shard |
| `check_alpha.sh`, `pull_alpha.sh` | status and result collection for the top-up pods |
