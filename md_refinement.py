"""
Molecular dynamics refinement of NES candidates docked into CRM1.

Provides restrained/flexible MD of a candidate NES peptide in the CRM1
Cys528 hydrophobic groove, together with the trajectory analysis used to
score the resulting poses: RMSD, radius of gyration, hydrogen bonds,
residue-residue contact maps, MM-GBSA-style interaction energies, DSSP
secondary structure and SASA burial.

The public entry point is NESMDRefiner; see its advanced-analysis helper
methods and the md_metrics dictionary returned by _run_crm1_docking.

OpenMM is required for the MD path; mdtraj is an optional dependency that
enables the DSSP and SASA metrics only.
"""

import numpy as np
import re
import tempfile
import os
import json
import time
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from io import StringIO

# OpenMM imports at module level
try:
    from openmm.app import PDBFile, ForceField, Modeller, Simulation, StateDataReporter
    from openmm import LangevinIntegrator, Platform, unit, CustomExternalForce, Vec3, NonbondedForce
    from openmm.app import PME, HBonds, NoCutoff, CutoffNonPeriodic
    import openmm.app as app
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("Warning: OpenMM not available. MD refinement will be disabled.")

# mdtraj is optional -- only used for the "advanced" analysis metrics
# (DSSP secondary structure, SASA/buried-SASA). Everything else in this
# module works identically without it; advanced_metrics() callers just get
# those two fields omitted (see _compute_advanced_analysis) rather than a
# crash, so this stays a soft dependency like OpenMM's own optional pieces.
try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("mdtraj not available -- secondary structure (DSSP) and SASA "
          "metrics will be skipped. Install with: pip install mdtraj")

# Every sample this many production steps apart also gets the "advanced"
# per-frame analysis (RMSD, Rg, backbone H-bonds, contact map, MM-GBSA-style
# binding energy) run on it, instead of at the full base sampling density
# (sample_interval in _run_crm1_docking, typically every 10 ps). These are
# each cheap individually but this keeps their combined cost bounded on long
# runs while still giving a meaningful time series (a stride of 5 against a
# 10 ps base interval means one advanced-analysis point every 50 ps).
ADVANCED_ANALYSIS_STRIDE = 5

# DSSP/SASA of the free peptide are cheap (tiny system) and run at the same
# stride as the rest of the "advanced" per-frame analysis. Buried SASA and
# the per-residue interaction-energy decomposition are much more expensive
# per frame (whole CRM1 groove shell for SASA; O(peptide_atoms x
# groove_atoms) for the energy decomposition) and are instead only computed
# on this many representative frames from the END of the production run
# (i.e. assumed post-equilibration, once the peptide has had time to settle
# into its bound pose) rather than the whole trajectory.
N_REPRESENTATIVE_FRAMES = 10

# OpenMM's Coulomb constant, 1/(4*pi*epsilon0), in kJ*nm/(mol*e^2) -- used
# by _residue_interaction_energy for the analytic pairwise decomposition.
ONE_4PI_EPS0 = 138.935456

# Ceiling used to decide whether the CRM1 groove shell (alone, before any
# NES peptide is merged in) is actually safe to cache. _check_structure_energy
# can fail outright (e.g. a broken CUDA install) or "succeed" while still
# reporting a catastrophically high energy after its own relax attempt --
# either way, caching that structure means every future run silently reuses
# a never-actually-relaxed shell (this is exactly what happened once here:
# a CUDA error broke the energy check while building this pod's cache, the
# code cached the unrelaxed shell anyway, and every docking run afterward
# inherited that bad starting structure even after CUDA got fixed). Below
# this ceiling is treated as safe to cache; at/above it, or on an outright
# failure (None), caching is skipped so the next run retries from scratch.
CRM1_SHELL_CACHE_ENERGY_CEILING_KJ_MOL = 1.0e5

# Nonbonded distance cutoff for the CRM1+Ran(+peptide) system. NoCutoff means
# every atom pair interacts with every other pair - fine for a small peptide,
# but for a ~20,000-atom system (CRM1+Ran+peptide) that's ~400 million
# pairwise interactions per force evaluation, which is impractical on CPU.
# A distance cutoff drops interactions beyond this range (negligible at this
# distance anyway) and typically cuts cost by 1-2 orders of magnitude with
# minimal accuracy impact.
NONBONDED_CUTOFF_NM = 1.5

# --- Side-chain relaxation phase --------------------------------
#
# Motivated directly by crystal_sanity_check.py: run against PDB 3NBY's real,
# experimentally-solved PKI-NES-CRM1 complex, the pipeline correctly scored
# the genuine crystallographic pose as a strong binder (raw_binding_score=1.0,
# anchor_occupancy_score=0.85) and its scrambled control as weak (0.03 / 0.0)
# -- proof the scoring/simulation methodology itself works when given a
# correctly-packed starting structure. That result also isolated what's
# actually different about real candidate starting poses: 'idealized_helix'
# Kabsch-fits the peptide's BACKBONE onto the same (independently validated,
# 1.19 Angstrom accurate) sub-pocket centroids the crystal check confirmed
# are geometrically correct -- but its SIDE CHAINS come from PDBFixer's
# generic rotamer-filling, not an energy-optimized fit, unlike the crystal
# structure's real, evolved rotamers already nestled into the pockets. A
# short (2 ns) production trajectory starting from generic rotamers may not
# have time to find good packing on its own, while hard negatives (rigid,
# pre-folded real proteins with their OWN already-optimized, if generically
# hydrophobic, side chains) don't have this handicap regardless of whether
# they're aimed at the correct pocket -- consistent with the specificity-
# control finding that negatives look "better" by raw packing metrics.
#
# _relax_peptide_sidechains() (used by _run_crm1_docking, all starting
# conformations / scramble settings alike, so this is a uniform pipeline
# improvement rather than a special case that would itself become a new
# confound between compared runs) restrains the peptide's own backbone atoms
# (N/CA/C/O) to their current -- Kabsch-registered, idealized-helix, or real
# crystal -- positions via a CustomExternalForce with a GLOBAL restraint
# constant (so it can be released via context.setParameter without rebuilding
# the Simulation), then minimizes + runs a short restrained MD segment so
# side chains can settle/thermally sample rotamers against the actual local
# CRM1 pocket environment, before the restraint is released and the existing
# minimization/equilibration/production protocol proceeds unchanged.
SIDECHAIN_RELAX_RESTRAINT_K_KJ_MOL_NM2 = 5000.0
SIDECHAIN_RELAX_MINIMIZE_ITERATIONS = 300
SIDECHAIN_RELAX_MD_STEPS = 2500  # 2 fs/step -> 5 ps of restrained sampling

# Radius (nm) used by _truncate_to_groove_shell() to decide which CRM1
# residues to keep. Also used as part of the groove-shell cache key (see
# NESMDRefiner._try_load_shell_cache/_save_shell_cache) - if this changes,
# any existing cache is automatically treated as stale and rebuilt.
GROOVE_SHELL_RADIUS_NM = 3.0

# Bumped whenever the on-disk groove shell cache's JSON schema changes (see
# _try_load_shell_cache/_save_shell_cache) -- any cache written under an
# older version is treated as stale and rebuilt, the same way a source-file
# or radius change is, rather than silently loading with new fields (like
# crm1_subpockets) left empty.
CACHE_FORMAT_VERSION = 6  # bumped: pre-round-trip pristine-position
# capture fix for the residue.id/residue.index mismatch bug (see
# _capture_pristine_reference_positions) changes how Cys528 and the
# groove-lining sub-pocket residues get located -- any cache written by the
# old, broken lookup logic must be treated as stale and rebuilt.

# Used by _build_idealized_helix_pdb() to write real residue names into the
# idealized-helix starting structure it constructs from a 1-letter sequence.
ONE_LETTER_TO_THREE_LETTER_AA = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
    'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
    'Y': 'TYR', 'V': 'VAL',
}

# Idealized alpha-helix backbone dihedrals/geometry (standard Engh & Huber-
# style textbook values), used by _build_idealized_helix_pdb() to construct
# a literature-consistent starting pose for the canonical helical NES-CRM1
# binding mode (Dong 2009 / Guttler 2010), as an alternative to whatever
# (often non-helical) conformation AlphaFold predicted for this stretch in
# the ISOLATED, unbound full-length protein -- see that function's
# docstring for the full rationale.
IDEAL_HELIX_PHI_DEG = -57.0
IDEAL_HELIX_PSI_DEG = -47.0
IDEAL_HELIX_OMEGA_DEG = 180.0
IDEAL_BOND_LENGTH_N_CA = 1.458   # Angstrom
IDEAL_BOND_LENGTH_CA_C = 1.525
IDEAL_BOND_LENGTH_C_N = 1.329
IDEAL_BOND_LENGTH_C_O = 1.231
IDEAL_BOND_ANGLE_N_CA_C = 111.0  # degrees
IDEAL_BOND_ANGLE_CA_C_N = 117.2
IDEAL_BOND_ANGLE_C_N_CA = 121.7
IDEAL_BOND_ANGLE_CA_C_O = 120.8

# Idealized EXTENDED/polyproline-II-like backbone dihedrals, the
# same role for _build_extended_pdb() that IDEAL_HELIX_PHI_DEG/PSI_DEG play
# for _build_idealized_helix_pdb() -- a second literature-informed starting
# hypothesis, alongside the helical one, since not every real NES binds
# helically (see crystal_sanity_check.py's 3NBZ entry: HIV-1 Rev NES binds
# in an extended, proline-containing conformation, not alpha-helical).
# phi=-75/psi=145 is the standard PPII region center (Chou-Fasman-era
# textbook values, e.g. Adzhubei et al. 2013 Annu Rev Biophys review of PPII
# geometry) -- close to but not identical to a flat antiparallel beta-strand
# (~-120/120), and a better match to how Rev-type NES binding is actually
# described in the literature than a generic beta-strand would be.
IDEAL_EXTENDED_PHI_DEG = -75.0
IDEAL_EXTENDED_PSI_DEG = 145.0
IDEAL_EXTENDED_OMEGA_DEG = 180.0

# Chou-Fasman beta-sheet/extended-strand propensity (Pb) values, same
# reference era/style as the helix_propensity (Pa) table inside
# _analyze_helix_propensity -- used by _predict_likely_conformation() to
# give a cheap, sequence-only prior on helical vs extended binding mode
# BEFORE any MD is run, rather than defaulting to idealized_helix for every
# candidate regardless of sequence composition.
BETA_SHEET_PROPENSITY = {
    'V': 1.70, 'I': 1.60, 'Y': 1.47, 'C': 1.19, 'W': 1.37, 'F': 1.38,
    'L': 1.30, 'T': 1.19, 'M': 1.05, 'A': 0.83, 'R': 0.93, 'G': 0.75,
    'D': 0.54, 'K': 0.74, 'S': 0.75, 'H': 0.87, 'N': 0.89, 'Q': 1.10,
    'P': 0.55, 'E': 0.37,
}

# Simulated-annealing ramp used by the use_simulated_annealing
# option on _run_crm1_docking -- see that parameter's docstring for the
# Rev-NES (3NBZ) motivating result. Kept modest (450K, not 500-600K some
# protein-folding annealing protocols use) since this is a small, already-
# minimized peptide+groove-shell system, not a full unfolding/refolding
# study -- the goal is just enough extra thermal energy to cross backbone
# dihedral barriers a 300K trajectory might not, not to unfold side chains
# or disrupt the CRM1 shell itself.
ANNEALING_HIGH_TEMP_K = 450.0
ANNEALING_HEAT_STEPS = 10000    # 20 ps ramp 300K -> 450K
ANNEALING_HOLD_STEPS = 15000    # 30 ps held at 450K
ANNEALING_COOL_STEPS = 10000    # 20 ps ramp 450K -> 300K
ANNEALING_RAMP_SUBSTEPS = 10    # temperature changed in this many discrete increments per ramp

# --- NES Phi-anchor <-> CRM1 sub-pocket geometry ----------------------------
#
# CRM1's NES-binding groove is not one generic pocket: it's formed by the
# outer A-helices of HEAT repeats 11 and 12, subdivided into five sequential
# hydrophobic sub-pockets -- conventionally P0 through P4 -- each engaging one
# Phi-anchor residue of a bound NES (Phi0-Phi4). Four of the five (Phi1-Phi4)
# were first resolved in Dong et al. 2009, Nature 458:1136-1141 (CRM1-
# Snurportin1-RanGTP complex, PDB 3GJX); the fifth (Phi0) was added in
# Guttler et al. 2010, Nat Struct Mol Biol 17:1367-1376 (PKI-NES/Rev-NES
# complexes, PDB family incl. 3GB8).
#
# What's actually stated in the text of those papers (checked,
# not assumed from memory) is narrower than a full per-pocket residue table:
#   - Ala541 sits at the base of the Phi3 (P3) pocket (Guttler 2010).
#   - Cys528's side-chain Hgamma makes NOE contacts with the Phi3/Phi4 anchor
#     methyls of a bound PKI-NES peptide (Guttler 2010) -- i.e. Cys528 sits
#     at the P3/P4 boundary, not centered in one pocket. (This is also why
#     _identify_binding_groove() below already uses Cys528, not a pocket
#     centroid, as its reference point -- it's the one residue with an
#     unambiguous published position, off-center in the groove rather than
#     in its middle.)
# A clean table of which residues line P0/P1/P2/P4 individually is NOT given
# as text in either paper -- getting that would mean reading coordinates back
# out of the deposited structures (3GJX/3GB8), which this module doesn't do.
#
# Instead, _identify_nes_subpockets() partitions the ~19 residues Guttler
# 2010 names as lining the groove into 5 sequential spatial clusters along
# the groove's own long axis, computed at runtime from real 3D coordinates
# (PCA of the groove-lining-residue CA cloud) and oriented/labeled using the
# two facts above as calibration anchors. This is a defensible geometric
# inference from real structure, not a second independent literature source
# for P0/P1/P2/P4 -- treat those three pockets' exact residue membership as
# best-effort and geometry-derived, flagged as such in
# _identify_nes_subpockets()'s own sanity check (which verifies Ala541
# actually lands in the P3 bin the geometry produced).
CRM1_GROOVE_LINING_RESIDUES_1INDEXED = [514, 518, 521, 525, 528, 534, 537, 538,
                                         541, 544, 545, 554, 558, 561, 564, 565,
                                         568, 572, 575]
# Expected residue TYPE at each position above, in this project's own
# crm1_reference/CRM1_Ran_only.pdb (verified directly against that file
# Every one of the 19 matches this exactly -- K514 V518 I521
# L525 C528 K534 K537 A538 A541 I544 M545 F554 H558 F561 T564 V565 K568
# F572 E575 -- so this PDB's numbering genuinely matches the literature's,
# confirmed rather than assumed). Used by _identify_nes_subpockets()'s
# post-truncation re-location to prefer a type-matching CA over a
# type-blind "nearest CA of any kind", which was found to occasionally
# lock onto the wrong atom near a PDBFixer re-cap seam (residue 528 itself
# was observed landing in the wrong sub-pocket bin because of exactly this
# -- see that method's docstring for the full story).
CRM1_GROOVE_LINING_RESIDUE_NAMES = {
    514: 'LYS', 518: 'VAL', 521: 'ILE', 525: 'LEU', 528: 'CYS', 534: 'LYS',
    537: 'LYS', 538: 'ALA', 541: 'ALA', 544: 'ILE', 545: 'MET', 554: 'PHE',
    558: 'HIS', 561: 'PHE', 564: 'THR', 565: 'VAL', 568: 'LYS', 572: 'PHE',
    575: 'GLU',
}
CRM1_P3_CALIBRATION_RESIDUE_1INDEXED = 541   # Guttler 2010: base of the Phi3 pocket
CRM1_CYS528_RESIDUE_1INDEXED = 528           # Guttler 2010 NOE data: P3/P4 boundary
N_NES_SUBPOCKETS = 5
SUBPOCKET_LABELS = ['P0', 'P1', 'P2', 'P3', 'P4']

# Phi-anchor register regex: same spacer convention as PSSM_ANCHOR_RE in
# nes_ml_predictor_improved.py (Phi1-X(1-3)-Phi2-X(1-3)-Phi3-X(1-2)-Phi4,
# per Kosugi et al. 2008 Traffic's NES classes), but with capture groups so
# each anchor's own sequence position can be recovered -- that module's
# regex deliberately avoids this (its docstring: "no real positional
# register... can't say 'this residue occupies Phi2' with real confidence",
# true for a heuristic scorer with no structure to check itself against).
# Here the register IS checked against real 3D geometry (the Kabsch fit
# below either lands the anchors near their assigned sub-pockets or it
# doesn't), so a wrong register choice shows up as a bad fit rather than
# being silently trusted.
PHI_ANCHOR_VOCAB = set('LIVFM')
PHI_REGISTER_RE = re.compile(r'([LIVFM]).{1,3}([LIVFM]).{1,3}([LIVFM]).{1,2}([LIVFM])')
PHI0_LOOKBACK = 3  # Guttler 2010: Phi0 sits a few residues N-terminal of Phi1

# Relaxed 3-of-4 register match. Snurportin1's real,
# crystallographically-confirmed NES-like segment (3GJX/3GB8 ground truth,
# this project -- core SQALASSFSVS) has L/F/V sitting at exactly the gap
# spacing PHI_REGISTER_RE requires, but only 3 Phi-vocab residues, not 4 --
# so the strict all-4 regex never matches it at all, and it was being
# treated identically to "no register-level signal here whatsoever", which
# is too strong a conclusion from one missing anchor. Per this project's
# explicit direction: a 3-of-4 match at the CORRECT spacing is still real,
# if lower-confidence, signal -- not nothing. Each pattern below allows
# exactly one anchor slot to be any residue while the other 3 keep BOTH the
# vocabulary and spacing constraints -- this is still fully spacing-
# constrained, not "any 3 hydrophobics anywhere in the sequence".
_PHI_RELAXED_PATTERNS = [
    ('P1', re.compile(r'(.).{1,3}([LIVFM]).{1,3}([LIVFM]).{1,2}([LIVFM])')),
    ('P2', re.compile(r'([LIVFM]).{1,3}(.).{1,3}([LIVFM]).{1,2}([LIVFM])')),
    ('P3', re.compile(r'([LIVFM]).{1,3}([LIVFM]).{1,3}(.).{1,2}([LIVFM])')),
    ('P4', re.compile(r'([LIVFM]).{1,3}([LIVFM]).{1,3}([LIVFM]).{1,2}(.)')),
]

# Class-4 register -- a genuinely DIFFERENT 5-anchor spacing
# pattern, Phi0-x(2)-Phi1-x(3)-Phi2-x(2)-Phi3-x(3)-Phi4, identified by Fung
# & Chook 2017 (eLife 6:e23961) specifically for X11L2's NES (a "helix-beta-
# turn" structure, distinct from both the canonical helical class-1/2 and
# the extended/proline class-3 patterns PHI_REGISTER_RE above already
# covers). Unlike PHI_REGISTER_RE, Phi0 here is a REQUIRED 5th anchor at a
# FIXED spacing, not an optional N-terminal lookback -- a structurally
# different register, not a looser version of the same one.
#
# Motivation: 5UWS (X11L2, this project's crystal ground truth for this
# exact structure) was, until now, this project's one confirmed real
# no-Phi-register-at-all case -- window expansion up to 6 residues each
# side (extract_crystal_references.py, still found nothing,
# because the standard pattern genuinely isn't there. But verified
# directly against this project's own broader sequence context
# (nes_data_pipeline/nes_dataset.json's X11L2 entry, residues 55-73:
# SSLQELVQQFEALPGDLVG) that THIS pattern matches cleanly at residues
# 57/60/64/67/71 (L/L/F/L/L) -- a real register 5UWS's own narrow
# extraction (65-72) was simply too short to contain (it starts 8 residues
# after the true Phi0 anchor). Confirmed via a standalone regex test
# against the literature sequence before wiring this in.
PHI_REGISTER_CLASS4_RE = re.compile(
    r'([LIVFM]).{2}([LIVFM]).{3}([LIVFM]).{2}([LIVFM]).{3}([LIVFM])')


def _insert_ter_at_chain_breaks_by_distance(pdb_path, ca_distance_threshold_angstrom=4.5):
    """
     Companion to extract_crystal_references.py's
    insert_ter_at_chain_breaks() -- same underlying problem (a real chain
    break with no TER record, causing OpenMM to try to bond residues that
    aren't actually adjacent), but that function's approach (detect breaks
    by non-consecutive residue NUMBERS) doesn't work here: confirmed
    directly that openmm.app.PDBFile.writeFile() renumbers every residue
    SEQUENTIALLY from 1, discarding the original numbering entirely -- so
    a genuine gap (e.g. from _truncate_to_groove_shell's residues_to_delete)
    becomes numerically invisible in the file this writes, even though the
    physical break is still there. This is exactly why 5DHF/5DIF kept
    failing ("No template found ... residue 7 (TYR)") even after the
    extract_crystal_references.py-level fix: that fix corrects the
    ORIGINAL input file, but _truncate_to_groove_shell writes its OWN new
    PDB text (via PDBFile.writeFile, not Bio.PDB) after deleting residues,
    and that write/reread round trip reintroduces the same class of bug
    independently.

    Coordinates, unlike residue numbers, survive the round trip -- so this
    detects breaks by actual 3D distance instead: a normal consecutive
    CA-CA spacing is ~3.8 Angstrom; anything past this threshold cannot be
    a real peptide bond and marks a genuine break, regardless of what the
    (possibly renumbered) residue IDs say. Call this on any PDB text file
    written by PDBFile.writeFile() before handing it back to PDBFixer.

     FIXED a real, severe bug in the original single-pass
    version of this function, found while debugging a catastrophic ~10.6
    BILLION kJ/mol starting energy on every single candidate's docking run
    (traced back to CRM1_Ran_only.pdb's groove shell -- the general-purpose
    reference used for real-candidate docking, not one of the specific
    crystal ground-truth files fixed earlier in this project). The original
    version only inspected each residue's CA line to decide whether to
    insert a TER, but a residue's OTHER atoms (N, H, ...) normally appear
    BEFORE its CA in a standard PDB atom ordering and had already been
    appended to the output by the time the CA-line check fired. That meant
    the TER record landed in the MIDDLE of the boundary residue's own atom
    block -- splitting one real residue into two incomplete halves (e.g.
    "...N, H" ending up in the fragment BEFORE the break, "CA, C, O, side
    chain..." starting the fragment AFTER it). PDBFixer's downstream
    addMissingAtoms()/addMissingHydrogens() then independently re-capped
    each incomplete half as if it were a genuinely separate, complete
    residue -- confirmed directly: this produced a full, physically
    duplicated copy of the boundary residue (e.g. GLN174 appearing twice,
    ~0.3-1.3 A apart, one C-terminally-capped copy ending one fragment, one
    N-terminally-capped copy starting the next) at EVERY TER insertion,
    which is exactly the kind of severe, near-zero-distance atomic overlap
    that blows up potential energy into the billions and can't be resolved
    by a normal-strength minimization.

    Fixed by grouping lines into whole per-residue blocks FIRST (via
    itertools.groupby on consecutive (chain, resid) line groups -- valid
    because a well-formed single-model PDB always keeps one residue's atoms
    contiguous), finding each block's own CA position for the distance
    decision, and only ever emitting complete blocks -- a TER can now only
    ever land BETWEEN two residues' atom blocks, never inside one.
    """
    lines = Path(pdb_path).read_text().splitlines(keepends=True)

    def _residue_key(line):
        if line.startswith(('ATOM', 'HETATM')):
            return (line[21], line[22:27])  # (chain_id, resid incl. insertion code)
        return None

    blocks = [(key, list(group)) for key, group in itertools.groupby(lines, key=_residue_key)]

    out_lines = []
    prev_chain = None
    prev_ca = None
    prev_resname = None
    prev_resid_field = None
    n_inserted = 0

    for key, block_lines in blocks:
        if key is not None:
            chain_id, resid = key
            ca_xyz = None
            for line in block_lines:
                if line[12:16].strip() == 'CA':
                    try:
                        ca_xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    except ValueError:
                        ca_xyz = None
                    break

            if prev_chain == chain_id and prev_ca is not None and ca_xyz is not None:
                dist = (sum((a - b) ** 2 for a, b in zip(ca_xyz, prev_ca))) ** 0.5
                if dist > ca_distance_threshold_angstrom:
                    ter_serial = n_inserted + 1
                    ter_line = (f"TER   {ter_serial:>5}      {prev_resname:>3} "
                                f"{prev_chain}{prev_resid_field}"
                                + " " * 53 + "\n")
                    out_lines.append(ter_line)
                    n_inserted += 1

            if ca_xyz is not None:
                prev_chain, prev_ca = chain_id, ca_xyz
                prev_resname = block_lines[0][17:20]
                prev_resid_field = block_lines[0][22:26]

        out_lines.extend(block_lines)  # whole residue block, atomically -- never split

    if n_inserted:
        Path(pdb_path).write_text(''.join(out_lines))
        print(f"      Inserted {n_inserted} TER record(s) at chain break(s) detected by "
              f"CA-CA distance (>{ca_distance_threshold_angstrom} Angstrom) -- residue "
              f"renumbering from the write/reread round trip made the usual numbering-gap "
              f"check blind to these")
    return n_inserted


def _detect_usable_cpu_count():
    """
    Real usable CPU-core-equivalents for THIS process, accounting for
    container CPU quota throttling (cgroup CFS bandwidth control) that
    os.cpu_count() does NOT reflect -- it (like os.sched_getaffinity())
    only reports cpuset/affinity restrictions, not a CFS bandwidth quota.
    A container can report a full host core count (e.g. `nproc` == 32,
    untouched affinity) while actually being throttled to a much smaller
    CPU-TIME budget per scheduling period -- e.g. a pod with
    /sys/fs/cgroup/cpu.max == "1360000 100000" gets 1,360,000us of CPU time
    per 100,000us (100ms) period, i.e. ~13.6 cores' worth, regardless of
    how many cores are visible. Spawning 32 OpenMM worker threads against
    a 13.6-core budget causes massive scheduling contention/oversubscription
    rather than a speedup -- confirmed directly: an equilibration step that
    should take low-single-digit minutes took over an hour on exactly this
    kind of pod before this fix, with `htop` unavailable to see it directly
    but /sys/fs/cgroup/cpu.max telling the real story.

    Tries cgroup v2 (/sys/fs/cgroup/cpu.max: "<quota> <period>", or "max"
    for no limit) first, then cgroup v1
    (cpu.cfs_quota_us/cpu.cfs_period_us, quota of -1 == no limit). Falls
    back to os.cpu_count() if neither file exists or parses (bare metal, a
    container with no CPU limit set, or a non-Linux host) -- never raises.
    """
    try:
        cgroup_v2 = Path('/sys/fs/cgroup/cpu.max')
        if cgroup_v2.exists():
            quota_str, period_str = cgroup_v2.read_text().split()
            if quota_str != 'max':
                quota, period = int(quota_str), int(period_str)
                if period > 0 and quota > 0:
                    return max(1, quota // period)
    except Exception:
        pass

    try:
        quota_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        period_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if quota_path.exists() and period_path.exists():
            quota = int(quota_path.read_text().strip())
            period = int(period_path.read_text().strip())
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except Exception:
        pass

    return os.cpu_count() or 1


def _select_fast_platform():
    """
    Explicitly select the fastest available OpenMM platform, rather than
    relying on auto-selection - which can silently land on the slow,
    single-threaded "Reference" platform, or on "CPU" without using all
    available cores, depending on the install.

    Only ever picks a platform that OpenMM reports as actually available on
    this machine (never hardcodes an assumption), so this cannot fail with
    a "platform not found" error - at worst it falls back to whatever
    OpenMM would have auto-selected anyway.

    Returns:
        (platform, properties) tuple to pass into Simulation(...)
    """
    preferred_order = ['CUDA', 'OpenCL', 'CPU', 'Reference']
    available = {}
    for i in range(Platform.getNumPlatforms()):
        p = Platform.getPlatform(i)
        available[p.getName()] = p

    for name in preferred_order:
        if name not in available:
            continue

        platform = available[name]
        properties = {}

        if name == 'CPU':
            usable = _detect_usable_cpu_count()
            total = os.cpu_count() or usable
            threads = str(usable)
            properties['Threads'] = threads
            if usable < total:
                print(f"    OpenMM platform: CPU ({threads} threads -- capped from "
                      f"{total} visible cores by a detected container CPU quota)")
            else:
                print(f"    OpenMM platform: CPU ({threads} threads)")
        else:
            print(f"    OpenMM platform: {name}")

        return platform, properties

    # Should be unreachable - Reference is always registered - but fall back
    # to plain auto-selection rather than raising, just in case.
    print("    Warning: No known platform found via explicit selection, "
          "falling back to OpenMM auto-selection")
    return None, {}


def estimate_md_time(num_candidates: int, duration_ns: float) -> float:
    """
    Estimate the time required for CRM1 docking MD simulation

    Args:
        num_candidates: Number of NES candidates to simulate
        duration_ns: Duration of each simulation in nanoseconds

    Returns:
        Estimated time in minutes
    """
    # Performance estimate (ns/minute) for the complex CRM1-NES system,
    # based on typical GPU performance
    ns_per_minute = 0.5

    # Calculate simulation time
    total_ns = num_candidates * duration_ns
    simulation_minutes = total_ns / ns_per_minute

    # Add overhead for setup, analysis, etc. (30 seconds per candidate)
    overhead_minutes = num_candidates * 0.5

    total_minutes = simulation_minutes + overhead_minutes

    return total_minutes


class NESMDRefiner:
    """
    Molecular Dynamics refinement for NES candidates

    Features:
    - Flexibility analysis via implicit solvent MD
    - Transition state sampling for helix formation
    - CRM1 docking simulation with Cys528 hydrophobic groove binding
    """

    def __init__(self, crm1_pdb_path: Optional[str] = None):
        """
        Initialize MD refiner

        Args:
            crm1_pdb_path: Optional path to CRM1 structure PDB for docking simulations
        """
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is required for MD refinement. Install with: conda install -c conda-forge openmm")

        self.crm1_pdb_path = crm1_pdb_path
        self.crm1_structure = None
        self.crm1_groove_residues = None
        self.crm1_cys528_position = None  # Track Cys528 position
        self.crm1_cys528_atom_index = None  # Cys528 CA atom's index within self.crm1_structure
        self.crm1_full_centroid = None  # Centroid of the FULL (pre-truncation) CRM1+Ran structure
        self.crm1_subpockets = {}  # label ('P0'-'P4') -> {'residue_numbers_1indexed': [...], 'centroid_nm': np.array}
        # Pre-truncation CA positions of the groove-lining reference residues
        # (CRM1_GROOVE_LINING_RESIDUES_1INDEXED), keyed by 1-indexed residue
        # number -- captured once on the full structure so they can be
        # re-located (nearest-atom-to-remembered-point, same trick used for
        # Cys528) after _truncate_to_groove_shell() renumbers everything.
        self._groove_lining_reference_positions = {}
        # Cys528's position captured on the VERY FIRST, completely
        # unprocessed load of the source PDB (before non-standard-residue
        # removal, PDBFixer's missing-residue/atom filling, or hydrogen
        # addition touch anything) -- see _capture_pristine_reference_positions.
        self._pristine_cys528_position = None

        # Load CRM1 structure if provided
        if crm1_pdb_path and os.path.exists(crm1_pdb_path):
            self._load_crm1_structure()

    def _capture_pristine_reference_positions(self, topology, positions):
        """
        Look up Cys528 and every CRM1_GROOVE_LINING_RESIDUES_1INDEXED
        residue by their PDB-preserved residue ID (atom.residue.id) on a
        topology/positions pair known to still have that numbering intact,
        and remember their CA positions for later spatial re-location.

        MUST be called on the VERY FIRST, completely unprocessed load of
        the source PDB (e.g. right after
        `fixer = PDBFixer(filename=self.crm1_pdb_path)`, or right after
        `pdb = PDBFile(self.crm1_pdb_path)` in the fallback branch --
        before anything else runs on it).

        Why this exists (found , against this project's REAL
        CRM1 structure, not a synthetic test): every downstream lookup
        that tried to identify these residues by residue.id or
        residue.index on the FINAL, fully-processed self.crm1_structure
        was unreliable, for two separate reasons:
          - residue.index is a purely positional, topology-wide counter
            (spans every chain), not tied to PDB numbering at all --
            confirmed directly: atom.residue.id == '528' correctly finds
            CYS at atom.residue.index == 684, an offset of 157.
          - residue.id itself, which DOES correctly mean "PDB resSeq"
            immediately after a fresh, unprocessed load, gets SCRAMBLED by
            the write-to-tempfile-then-PDBFixer-reread round trip this
            module uses to strip non-standard residues (GTP/MG/etc. --
            see NONSTANDARD_RESIDUES below) before refeeding PDBFixer.
            Confirmed directly: before that round trip, id='528' -> CYS;
            after it, id='528' -> SER, id='514' -> CYS, id='541' -> LEU --
            a completely different mapping. PDBFixer's __init__ also has
            no way to accept an in-memory Topology/positions directly
            (only filename/pdbfile/pdbxfile/url/pdbid), so the round trip
            itself can't just be skipped.

        So: capture positions HERE, on the one topology where
        atom.residue.id is still guaranteed to mean what it says, then let
        the EXISTING nearest-CA-to-remembered-point re-location logic in
        _identify_binding_groove(near_position=...) and
        _identify_nes_subpockets(near_positions=...) -- already proven to
        work for the post-truncation case (e.g. the "0.00 Å from its
        original position" match seen in testing) -- do the real lookup
        against the final, processed structure. Existing atom coordinates
        don't move during non-standard-residue removal, missing-atom
        filling, or hydrogen addition (only new atoms get added), so this
        nearest-point search should land at or near 0.00 Å here too.

        Sets self._pristine_cys528_position (an OpenMM position, or None
        if Cys528 wasn't found) and self._groove_lining_reference_positions
        (dict, may be missing a few residues -- same graceful degradation
        the downstream re-location logic already handles).
        """
        self._pristine_cys528_position = None
        self._groove_lining_reference_positions = {}

        for atom in topology.atoms():
            if (atom.name == 'CA' and atom.residue.name == 'CYS' and
                    atom.residue.id == str(CRM1_CYS528_RESIDUE_1INDEXED)):
                self._pristine_cys528_position = positions[atom.index]
                break

        for resnum in CRM1_GROOVE_LINING_RESIDUES_1INDEXED:
            expected_name = CRM1_GROOVE_LINING_RESIDUE_NAMES.get(resnum)
            for atom in topology.atoms():
                if atom.name != 'CA' or atom.residue.id != str(resnum):
                    continue
                if expected_name is None or atom.residue.name == expected_name:
                    self._groove_lining_reference_positions[resnum] = positions[atom.index]
                    break

        n_found = len(self._groove_lining_reference_positions)
        print(f"  Captured {n_found}/{len(CRM1_GROOVE_LINING_RESIDUES_1INDEXED)} "
              f"groove-lining reference positions by PDB residue ID "
              f"(Cys528 {'found' if self._pristine_cys528_position is not None else 'NOT found'}) "
              f"before any processing that could disturb residue numbering")

    def _load_crm1_structure(self):
        """
        Load CRM1 structure and identify the Cys528 hydrophobic groove
        Also removes non-standard residues (GTP, GDP, ions, etc.)
        """
        # Non-standard residues to remove (not in AMBER forcefield)
        NONSTANDARD_RESIDUES = {
            'GTP', 'GDP', 'GNP', 'ATP', 'ADP', 'ANP',  # Nucleotides
            'NAG', 'MAN', 'BMA', 'FUC',  # Sugars
            'HOH', 'WAT',  # Water
            'NA', 'CL', 'MG', 'CA', 'K', 'ZN', 'FE', 'MN',  # Ions
            'SO4', 'PO4', 'ACT', 'EDO', 'PEG',  # Common artifacts
            # GOL (glycerol, a standard cryoprotectant) was
            # confirmed missing here -- present in 5DHF/5DIF (and 3NC0's
            # AGOL/BGOL altloc-split naming) and directly reproduced,
            # locally, as a second real cause of "No template found" on
            # top of the chain-break issue fixed in extract_crystal_
            # references.py's insert_ter_at_chain_breaks(). Verified
            # directly: ForceField.createSystem() succeeds on 5DHF only
            # once GOL is excluded here AND the TER fix is applied --
            # neither fix alone was sufficient.
            'GOL', 'AGOL', 'BGOL',
        }

        try:
            print("  Loading CRM1 structure...")

            if self._try_load_shell_cache():
                return

            # Try PDBFixer for best results
            try:
                from pdbfixer import PDBFixer

                print("  Using PDBFixer for robust terminal handling...")
                fixer = PDBFixer(filename=self.crm1_pdb_path)

                # Capture Cys528 + groove-lining reference positions by PDB
                # residue ID NOW, on this completely fresh/unprocessed
                # topology -- before the non-standard-residue removal round
                # trip below scrambles residue.id. See
                # _capture_pristine_reference_positions's docstring.
                self._capture_pristine_reference_positions(fixer.topology, fixer.positions)

                # CRITICAL: Remove non-standard residues FIRST
                print("  Checking for non-standard residues...")
                residues_to_remove = []
                for residue in fixer.topology.residues():
                    if residue.name in NONSTANDARD_RESIDUES:
                        residues_to_remove.append(residue)

                if residues_to_remove:
                    print(f"  Removing {len(residues_to_remove)} non-standard residues")
                    for res in residues_to_remove:
                        print(f"     - {res.name} (index {res.index})")

                    # Use Modeller to delete them
                    temp_modeller = Modeller(fixer.topology, fixer.positions)
                    temp_modeller.delete(residues_to_remove)

                    # Create a new fixer with the cleaned structure
                    import tempfile
                    from openmm.app import PDBFile as PDBFileWriter
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                        PDBFileWriter.writeFile(temp_modeller.topology, temp_modeller.positions, tmp)
                        tmp_path = tmp.name

                    # Reload with PDBFixer
                    fixer = PDBFixer(filename=tmp_path)
                    os.unlink(tmp_path)
                    print("  Non-standard residues removed")

                # Find and add missing residues (including terminals)
                fixer.findMissingResidues()

                # Confirmed necessary for 5DHF/5DIF (Fung, Fu,
                # Chook 2015's engineered ScCRM1* construct): its own
                # Methods section documents an intentional internal
                # deletion (Delta377-413, 37 residues -- removed to help
                # crystallization, genuinely absent from the expressed
                # protein) PLUS a large disordered/unresolved loop near the
                # engineered V441D mutation (~residues 441-459, per the
                # paper's own "residues modeled... 1-440 and 460-1053").
                # Both show up identically to PDBFixer as "missing
                # residues" alongside completely normal, small (1-8
                # residue) crystallographic disorder gaps that exist in
                # nearly every structure and SHOULD be filled in as usual.
                # Left alone, addMissingAtoms() below tries to homology-
                # model 20-40 residue loops to bridge these -- geometry
                # that has no real answer (one gap doesn't exist in the
                # real molecule at all; the other is genuinely disordered,
                # not just unmodeled) and was confirmed to produce a broken
                # chain junction downstream ("No template found for
                # residue N (ILE)... missing 1 C atom"), causing every MD
                # attempt on these two structures to fail identically. Same
                # fix already proven out in _truncate_to_groove_shell()
                # below for a different cause (OUR OWN deliberate residue
                # deletion) -- here applied by gap LENGTH instead, since
                # there's no explicit deleted-residues list for the
                # source structure's own inherent gaps. Threshold of 10 is
                # generous for genuine single-loop crystallographic
                # disorder (typically 1-8 residues) while still catching
                # both gaps here (38 and ~19 residues) -- every other
                # structure in this project is unmodified wild-type-length
                # human/native CRM1 with no engineered deletions, so this
                # is not expected to change their behavior at all.
                LARGE_GAP_RESIDUE_THRESHOLD = 10
                if fixer.missingResidues:
                    large_gaps = {k: v for k, v in fixer.missingResidues.items()
                                  if len(v) > LARGE_GAP_RESIDUE_THRESHOLD}
                    if large_gaps:
                        gap_residue_count = sum(len(v) for v in large_gaps.values())
                        print(f"  Ignoring {len(large_gaps)} large gap(s) ({gap_residue_count} "
                              f"residues, >{LARGE_GAP_RESIDUE_THRESHOLD} residues each) PDBFixer "
                              f"flagged as 'missing' -- too large to be normal crystallographic "
                              f"disorder; treating as genuine chain breaks (engineered deletion "
                              f"and/or truly disordered loop) rather than fabricating loop "
                              f"geometry for them. Smaller gaps, if any, are still filled in "
                              f"normally below.")
                        for k in large_gaps:
                            del fixer.missingResidues[k]

                # Find and add missing atoms
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()

                # Add hydrogens
                fixer.addMissingHydrogens(7.0)

                # Get the fixed structure
                self.crm1_structure = Modeller(fixer.topology, fixer.positions)

                print("  CRM1 structure fixed with PDBFixer")
                self._capture_full_centroid()
                # Use the pristine (pre-round-trip) reference positions
                # captured above, NOT the id/index-based "None" path --
                # residue.id on THIS (final, processed) topology is no
                # longer reliable (see _capture_pristine_reference_positions).
                self._identify_binding_groove(near_position=self._pristine_cys528_position)
                self._identify_nes_subpockets(near_positions=self._groove_lining_reference_positions)
                self._truncate_to_groove_shell()
                shell_energy = self._check_structure_energy(
                    self.crm1_structure, label="CRM1 groove shell", relax=True)
                # _check_structure_energy(relax=True) may have just moved
                # every atom in self.crm1_structure (minimizeEnergy(), to
                # resolve exactly the kind of severe clashes that triggered
                # this relax branch in the first place) -- Cys528's position
                # and every sub-pocket's centroid/atom indices were computed
                # BEFORE that happened, against the pre-relax geometry, and
                # would otherwise silently go stale relative to what
                # actually gets simulated (and cached) from here on. Re-run
                # both against the now-final positions before deciding
                # whether to cache.
                self._identify_binding_groove(near_position=self.crm1_cys528_position)
                self._identify_nes_subpockets(near_positions=self._groove_lining_reference_positions)
                if (shell_energy is not None and np.isfinite(shell_energy)
                        and shell_energy < CRM1_SHELL_CACHE_ENERGY_CEILING_KJ_MOL):
                    self._save_shell_cache()
                else:
                    print(f"  Warning: NOT caching this shell: energy check "
                          f"{'failed outright' if shell_energy is None else f'is {shell_energy:,.1f} kJ/mol, still above the safety ceiling'} "
                          f"-- caching it would mean every future run silently reuses an unrelaxed "
                          f"structure. Will rebuild from scratch on the next run instead.")
                return

            except (ImportError, Exception) as e:
                print(f"  Warning: PDBFixer failed: {e}")
                import traceback
                traceback.print_exc()
                print("  Trying direct load...")

            # Fallback: Load and clean manually
            pdb = PDBFile(self.crm1_pdb_path)
            modeller = Modeller(pdb.topology, pdb.positions)

            # Same reasoning as the PDBFixer branch above: capture
            # reference positions by PDB residue ID now, before any
            # processing (this branch has no round-trip reload, but
            # capturing here anyway keeps both branches consistent and
            # avoids relying on residue.id/.index later regardless).
            self._capture_pristine_reference_positions(pdb.topology, pdb.positions)

            print("  Checking for non-standard residues...")
            residues_to_remove = []
            for residue in modeller.topology.residues():
                if residue.name in NONSTANDARD_RESIDUES:
                    residues_to_remove.append(residue)

            if residues_to_remove:
                print(f"  Removing {len(residues_to_remove)} non-standard residues")
                for res in residues_to_remove:
                    print(f"     - {res.name} (index {res.index})")
                modeller.delete(residues_to_remove)
                print("  Non-standard residues removed")

            self.crm1_structure = modeller
            print("  CRM1 structure loaded")
            self._capture_full_centroid()
            self._identify_binding_groove(near_position=self._pristine_cys528_position)
            self._identify_nes_subpockets(near_positions=self._groove_lining_reference_positions)
            self._truncate_to_groove_shell()
            shell_energy = self._check_structure_energy(
                self.crm1_structure, label="CRM1 groove shell", relax=True)
            # Same reasoning as the PDBFixer branch above: relax=True may
            # have just moved every atom, so re-run both re-locations
            # against the final positions before caching.
            self._identify_binding_groove(near_position=self.crm1_cys528_position)
            self._identify_nes_subpockets(near_positions=self._groove_lining_reference_positions)
            if (shell_energy is not None and np.isfinite(shell_energy)
                    and shell_energy < CRM1_SHELL_CACHE_ENERGY_CEILING_KJ_MOL):
                self._save_shell_cache()
            else:
                print(f"  Warning: NOT caching this shell: energy check "
                      f"{'failed outright' if shell_energy is None else f'is {shell_energy:,.1f} kJ/mol, still above the safety ceiling'} "
                      f"-- caching it would mean every future run silently reuses an unrelaxed "
                      f"structure. Will rebuild from scratch on the next run instead.")

        except Exception as e:
            print(f"  Warning: Error loading CRM1: {e}")
            import traceback
            traceback.print_exc()
            self.crm1_structure = None

    def _groove_shell_cache_path(self):
        base, _ = os.path.splitext(self.crm1_pdb_path)
        return base + '_groove_shell_cache.json'

    def _try_load_shell_cache(self):
        """
        Try to load a previously-computed, already-minimized groove shell
        from disk instead of redoing PDBFixer truncation/re-capping plus a
        full minimization on every app startup - that minimization is the
        slow part, and none of it depends on anything per-candidate, so it
        only ever needs to happen once for a given source PDB + radius.

        Automatically invalidated (falls through to a normal rebuild) if
        the source PDB file's mtime/size changed, GROOVE_SHELL_RADIUS_NM
        changed, or the cache predates the sub-pocket-partitioning format
        (CACHE_FORMAT_VERSION) added below -- so there's no risk of
        silently reusing a stale shell, or of a pre-existing cache from
        before this feature existed loading with crm1_subpockets left
        empty and nobody noticing.
        """
        cache_path = self._groove_shell_cache_path()
        if not os.path.exists(cache_path):
            return False
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)

            src_stat = os.stat(self.crm1_pdb_path)
            if (cache.get('source_mtime') != src_stat.st_mtime or
                    cache.get('source_size') != src_stat.st_size or
                    cache.get('radius_nm') != GROOVE_SHELL_RADIUS_NM or
                    cache.get('cache_format_version') != CACHE_FORMAT_VERSION):
                print("  Groove shell cache is stale (source file, radius, or cache "
                      "format changed) - rebuilding")
                return False

            pdb = PDBFile(StringIO(cache['pdb_text']))
            self.crm1_structure = Modeller(pdb.topology, pdb.positions)
            self.crm1_cys528_atom_index = cache['cys528_atom_index']
            self.crm1_groove_residues = cache['groove_residues']
            self.crm1_full_centroid = (np.array(cache['full_centroid'])
                                        if cache.get('full_centroid') is not None else None)
            self.crm1_cys528_position = self.crm1_structure.positions[self.crm1_cys528_atom_index]

            subpockets = {}
            for label, entry in (cache.get('subpockets') or {}).items():
                subpockets[label] = {
                    'residue_numbers_1indexed': list(entry['residue_numbers_1indexed']),
                    'centroid_nm': np.array(entry['centroid_nm']),
                    'atom_indices': list(entry.get('atom_indices', [])),
                }
            self.crm1_subpockets = subpockets

            atom_count = len(list(self.crm1_structure.topology.atoms()))
            print(f"  Loaded pre-minimized CRM1 groove shell from cache "
                  f"({atom_count:,} atoms, {len(subpockets)} sub-pockets) - "
                  f"skipping truncation and minimization")
            return True
        except Exception as e:
            print(f"  Warning: Could not load groove shell cache, rebuilding: {e}")
            return False

    def _save_shell_cache(self):
        """
        Persist the finished (truncated, re-capped, minimized) groove shell
        to disk so future app restarts can load it straight from
        _try_load_shell_cache() instead of redoing the whole pipeline.
        Best-effort: a failure here just means the next startup rebuilds it
        again, no worse than today's behaviour.
        """
        if self.crm1_structure is None or self.crm1_cys528_atom_index is None:
            return
        try:
            pdb_string = StringIO()
            PDBFile.writeFile(self.crm1_structure.topology, self.crm1_structure.positions, pdb_string)

            src_stat = os.stat(self.crm1_pdb_path)
            subpockets_serializable = {
                label: {
                    'residue_numbers_1indexed': entry['residue_numbers_1indexed'],
                    'centroid_nm': list(map(float, entry['centroid_nm'])),
                    'atom_indices': [int(i) for i in entry.get('atom_indices', [])],
                }
                for label, entry in (self.crm1_subpockets or {}).items()
            }
            cache = {
                'source_mtime': src_stat.st_mtime,
                'source_size': src_stat.st_size,
                'radius_nm': GROOVE_SHELL_RADIUS_NM,
                'cache_format_version': CACHE_FORMAT_VERSION,
                'pdb_text': pdb_string.getvalue(),
                'cys528_atom_index': self.crm1_cys528_atom_index,
                'groove_residues': list(self.crm1_groove_residues) if self.crm1_groove_residues else [],
                'full_centroid': (list(map(float, self.crm1_full_centroid))
                                   if self.crm1_full_centroid is not None else None),
                'subpockets': subpockets_serializable,
            }
            cache_path = self._groove_shell_cache_path()
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
            print(f"  Cached groove shell ({len(subpockets_serializable)} sub-pockets) to "
                  f"{os.path.basename(cache_path)} for faster startup next time")
        except Exception as e:
            print(f"  Warning: Could not save groove shell cache (non-fatal): {e}")

    def _identify_binding_groove(self, near_position=None):
        """
        Identify the hydrophobic groove near Cys528 in CRM1

        Literature indicates 4-5 hydrophobic anchor residues of NES bind in the
        hydrophobic groove near Cys528 of CRM1

        Args:
            near_position: Optional previously-known Cys528 position (an
                OpenMM position/Quantity). When given, re-locates Cys528 by
                finding the closest CYS CA atom to that 3D point instead of
                using sequence-index heuristics - required after
                _truncate_to_groove_shell() runs, since residue indices get
                completely renumbered by that process and no longer mean
                anything relative to the original numbering.
        """
        try:
            print("  Identifying Cys528 hydrophobic groove...")

            # Extract positions from topology
            positions = self.crm1_structure.positions
            topology = self.crm1_structure.topology

            # Find Cys528
            cys528_idx = None
            cys528_pos = None

            if near_position is not None:
                # Re-locating after truncation: find the CYS CA atom closest
                # to the previously known position, rather than relying on
                # sequence indices (which are now meaningless).
                near = np.array([near_position.x, near_position.y, near_position.z])
                best_dist = None
                for atom in topology.atoms():
                    if atom.residue.name == 'CYS' and atom.name == 'CA':
                        pos = positions[atom.index]
                        dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - near)
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            cys528_idx = atom.index
                            cys528_pos = pos
                if cys528_pos is not None:
                    print(f"  Re-located Cys528 in truncated structure "
                          f"({best_dist * 10:.2f} Å from its original position)")
            else:
                # Match by the PDB's own preserved residue sequence number
                # (atom.residue.id, a string carried through from the
                # source PDB's resSeq field) rather than atom.residue.index
                # (a purely positional, topology-wide 0-based counter).
                # Found this project's real CRM1 structure
                # breaks the index-based assumption -- PDBFixer's
                # findMissingResidues()/addMissingAtoms() step (which runs
                # before this, filling in unmodeled loop residues) inserts
                # residues into the topology, shifting every index()
                # downstream of the insertion point away from "PDB resSeq
                # - 1" even though residue.id stays correct. Confirmed
                # directly against crm1_reference/CRM1_Ran_only.pdb's raw
                # ATOM records: residue 528 IS really CYS at PDB numbering,
                # but atom.residue.index == 527 was finding nothing,
                # forcing this exact fallback branch on every run.
                for atom in topology.atoms():
                    if (atom.residue.name == 'CYS' and
                        atom.residue.id == str(CRM1_CYS528_RESIDUE_1INDEXED) and
                        atom.name == 'CA'):
                        cys528_idx = atom.index
                        cys528_pos = positions[cys528_idx]
                        break

                if cys528_pos is None:
                    print("  Warning: Cys528 not found by PDB residue ID, scanning for a nearby cysteine...")
                    # Still ID-based (not index-based) -- looks for any
                    # cysteine whose PDB residue number is close to 528,
                    # in case of a small numbering discrepancy (e.g. an
                    # off-by-a-few construct/isoform difference) rather
                    # than the wholesale index-vs-id mismatch above.
                    for atom in topology.atoms():
                        if atom.name != 'CA' or atom.residue.name != 'CYS':
                            continue
                        try:
                            rid = int(atom.residue.id)
                        except (TypeError, ValueError):
                            continue
                        if abs(rid - CRM1_CYS528_RESIDUE_1INDEXED) <= 7:
                            cys528_idx = atom.index
                            cys528_pos = positions[cys528_idx]
                            print(f"  Found cysteine at PDB residue {rid}")
                            break

                if cys528_pos is None:
                    # Last-resort fallback: the OLD positional-index
                    # heuristic, kept only for a structure where
                    # residue.id turns out to be missing/unusable entirely
                    # (e.g. a non-standard loader that didn't preserve it).
                    print("  Warning: Cys528 not found by ID either, falling back to positional-index heuristic...")
                    for atom in topology.atoms():
                        if (atom.residue.name == 'CYS' and
                            atom.residue.index == CRM1_CYS528_RESIDUE_1INDEXED - 1 and
                            atom.name == 'CA'):
                            cys528_idx = atom.index
                            cys528_pos = positions[cys528_idx]
                            break

            if cys528_pos is not None:
                self.crm1_cys528_position = cys528_pos
                self.crm1_cys528_atom_index = cys528_idx

                # Identify hydrophobic groove (residues within 12Å of Cys528)
                groove_residues = []
                hydrophobic_aas = ['LEU', 'ILE', 'VAL', 'PHE', 'MET', 'TRP', 'TYR', 'ALA']

                for residue in topology.residues():
                    if residue.name in hydrophobic_aas:
                        # Check if any atom is within 12Å of Cys528
                        for atom in residue.atoms():
                            if atom.name == 'CA':
                                pos = positions[atom.index]
                                distance = np.linalg.norm(
                                    np.array([pos.x, pos.y, pos.z]) -
                                    np.array([cys528_pos.x, cys528_pos.y, cys528_pos.z])
                                )
                                if distance < 1.2:  # 12Å in nm
                                    groove_residues.append(residue.index)
                                    break

                self.crm1_groove_residues = groove_residues
                print(f"  Identified {len(groove_residues)} hydrophobic groove residues near Cys528")
            else:
                print("  Warning: Could not locate Cys528 region")
                self.crm1_groove_residues = []

        except Exception as e:
            print(f"  Warning: Error identifying groove: {e}")
            self.crm1_groove_residues = []

    def _identify_nes_subpockets(self, near_positions=None):
        """
        Partition the CRM1 NES-binding groove into 5 sequential hydrophobic
        sub-pockets (P0-P4), one per Phi-anchor position of a bound NES.
        See the CRM1_GROOVE_LINING_RESIDUES_1INDEXED block near the top of
        this module for exactly which literature facts this is grounded in,
        and where it's geometric inference rather than a second published
        source.

        Must run after _identify_binding_groove() (needs a valid
        self.crm1_cys528_position/self.crm1_structure).

        Geometry:
          1. Locate the CA position of each residue in
             CRM1_GROOVE_LINING_RESIDUES_1INDEXED. On the first (pre-
             truncation) call, `near_positions` is None and residues are
             found by their original PDB numbering. After
             _truncate_to_groove_shell() renumbers everything, pass the
             positions remembered from that first call (self.
             _groove_lining_reference_positions) so each one can be
             re-located by nearest-CA-to-remembered-point instead -- the
             same trick _identify_binding_groove(near_position=...) already
             uses for Cys528 alone, generalized to this whole residue set.
          2. Fit the principal axis (first PCA component, via SVD) of the
             located CA coordinates -- the groove's own long axis, running
             roughly along the HEAT11/HEAT12 repeat direction.
          3. Project each residue's CA onto that axis.
          4. Orient the axis using the two calibration facts (Ala541 -> P3;
             Cys528, a separate independently-located point, sits on the
             P4 side of Ala541).
          5. Split the oriented axis into 5 equal-WIDTH bins spanning the
             full observed range of projections (min to max among all
             located residues), and assign every located residue to a bin
             by which one its projection falls into.

             Two earlier approaches were tried and rejected against this
             project's REAL CRM1 structure (not just synthetic geometry):
             equal-COUNT bins put Ala541 in the wrong bin whenever it
             wasn't near the exact rank-median of the located residues
             ; boundaries extrapolated purely from the
             Ala541/Cys528 pair distance overshot the ENTIRE rest of the
             data on the real structure, since those two residues turned
             out to sit much farther apart than a uniform-pocket-width
             assumption predicts -- collapsing P0/P1/P2 to completely
             empty . Equal-width bins over the real observed
             range can't overshoot past real data by construction.
             Whether Ala541 actually lands in the resulting P3 bin is
             checked afterward (calibration_ok) as an honest pass/fail
             signal, not guaranteed by how the bins are built.

        Sets self.crm1_subpockets = {'P0': {'residue_numbers_1indexed': [...],
        'centroid_nm': np.array([x, y, z])}, ...}. Sets it to {} on any
        failure or if there isn't enough located reference geometry to form
        5 bins -- this is a refinement layered on top of the working
        Cys528-groove logic, never allowed to break it.
        """
        try:
            if self.crm1_cys528_position is None or self.crm1_structure is None:
                self.crm1_subpockets = {}
                return

            positions = self.crm1_structure.positions
            topology = self.crm1_structure.topology

            residue_ca = {}       # 1-indexed original resnum -> Vec3 position (current frame)
            residue_ca_idx = {}   # 1-indexed original resnum -> atom index IN self.crm1_structure
            # (residue_ca_idx is only meaningful/kept once this call is the
            # LAST one to run against a given self.crm1_structure object --
            # true for the post-truncation call, or for the pre-truncation
            # call if truncation is skipped/fails. See _run_crm1_docking,
            # which offsets these by nes_peptide_atom_count to track each
            # sub-pocket's REAL simulated position every production frame,
            # the same way Cys528's combined-system index is already used,
            # instead of trusting a single position captured once at load
            # time to still be right after minimization/equilibration moves
            # things.)

            if near_positions is None:
                # Fresh, pre-truncation structure: match by the PDB's own
                # preserved residue sequence number (atom.residue.id)
                # rather than atom.residue.index (a purely positional,
                # topology-wide counter).: found this project's
                # real CRM1 structure breaks the index-based assumption --
                # PDBFixer's findMissingResidues()/addMissingAtoms() step
                # (which runs before this, filling in unmodeled loop
                # residues) inserts residues into the topology, shifting
                # every index() downstream of the insertion point away
                # from "PDB resSeq - 1", even though residue.id stays
                # correct. This branch used to have ZERO tolerance for
                # that -- unlike Cys528's own lookup (see
                # _identify_binding_groove), which at least had a nearby-
                # residue fallback scan, this one would silently accept
                # WHATEVER atom happened to sit at the wrong index with no
                # verification at all. Falls back to the old index-based
                # lookup only if id-based matching finds nothing, and even
                # then verifies residue TYPE first (via
                # CRM1_GROOVE_LINING_RESIDUE_NAMES) rather than accepting
                # blindly.
                for resnum in CRM1_GROOVE_LINING_RESIDUES_1INDEXED:
                    expected_name = CRM1_GROOVE_LINING_RESIDUE_NAMES.get(resnum)
                    found = False
                    for atom in topology.atoms():
                        if atom.name != 'CA' or atom.residue.id != str(resnum):
                            continue
                        if expected_name is None or atom.residue.name == expected_name:
                            residue_ca[resnum] = positions[atom.index]
                            residue_ca_idx[resnum] = atom.index
                            found = True
                            break
                    if found:
                        continue

                    # Last-resort fallback: old positional-index heuristic,
                    # but still type-checked rather than accepted blindly.
                    target_idx = resnum - 1
                    for atom in topology.atoms():
                        if (atom.residue.index == target_idx and atom.name == 'CA' and
                                (expected_name is None or atom.residue.name == expected_name)):
                            residue_ca[resnum] = positions[atom.index]
                            residue_ca_idx[resnum] = atom.index
                            break
            else:
                # Post-truncation: residue indices were completely
                # renumbered by PDBFixer's re-capping -- re-locate each
                # remembered pre-truncation position against the nearest
                # surviving CA instead of trusting any index.
                #
                # Prefer a residue-TYPE-matching CA (via
                # CRM1_GROOVE_LINING_RESIDUE_NAMES) over the type-blind
                # nearest CA of any kind. Found empirically,
                # real structure, not a synthetic test) that residue 528
                # itself landed on a different, non-CYS atom near a
                # PDBFixer re-cap seam under the old type-blind search --
                # _identify_binding_groove()'s own Cys528 search already
                # avoided this by restricting to CYS residues; this
                # generalizes that same protection to all 19 reference
                # residues instead of just the one. Falls back to the old
                # type-blind nearest-CA if no type-matching atom is within
                # tolerance, so this never loses coverage, only adds a
                # preference where a type match exists.
                for resnum, remembered_pos in near_positions.items():
                    target = np.array([remembered_pos.x, remembered_pos.y, remembered_pos.z])
                    expected_name = CRM1_GROOVE_LINING_RESIDUE_NAMES.get(resnum)

                    best_idx, best_dist = None, None
                    best_typed_idx, best_typed_dist = None, None
                    for atom in topology.atoms():
                        if atom.name != 'CA':
                            continue
                        pos = positions[atom.index]
                        dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - target)
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_idx = atom.index
                        if expected_name is not None and atom.residue.name == expected_name:
                            if best_typed_dist is None or dist < best_typed_dist:
                                best_typed_dist = dist
                                best_typed_idx = atom.index

                    # Tight tolerance -- this should be the SAME atom,
                    # just re-indexed/re-capped, not a genuinely different
                    # residue that happens to be nearby.
                    if best_typed_idx is not None and best_typed_dist <= 0.35:
                        residue_ca[resnum] = positions[best_typed_idx]
                        residue_ca_idx[resnum] = best_typed_idx
                    elif best_idx is not None and best_dist <= 0.35:
                        residue_ca[resnum] = positions[best_idx]
                        residue_ca_idx[resnum] = best_idx

            if near_positions is None:
                # Remember these (pre-truncation) positions for the
                # post-truncation re-location call that
                # _truncate_to_groove_shell() makes later.
                self._groove_lining_reference_positions = dict(residue_ca)

            found_resnums = sorted(residue_ca.keys())
            if len(found_resnums) < N_NES_SUBPOCKETS:
                print(f"  Warning: Only {len(found_resnums)}/{len(CRM1_GROOVE_LINING_RESIDUES_1INDEXED)} "
                      f"groove-lining reference residues located -- skipping NES "
                      f"sub-pocket partitioning (need >= {N_NES_SUBPOCKETS} for 5 bins)")
                self.crm1_subpockets = {}
                return

            coords = np.array([[residue_ca[r].x, residue_ca[r].y, residue_ca[r].z]
                                for r in found_resnums])
            centroid = coords.mean(axis=0)
            centered = coords - centroid

            # Groove long axis = first principal component (SVD of the
            # centered coordinates).
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            axis = vt[0]
            projections = centered @ axis

            # Orient using the calibration facts. Ala541 -> P3; Cys528 must
            # sit on the SAME side as P4 relative to Ala541 (Guttler 2010
            # NOE: Cys528 is at the P3/P4 boundary, not centered in P3).
            cys528_local = np.array([self.crm1_cys528_position.x,
                                      self.crm1_cys528_position.y,
                                      self.crm1_cys528_position.z]) - centroid
            cys528_proj = cys528_local @ axis

            ala541_proj = None
            if CRM1_P3_CALIBRATION_RESIDUE_1INDEXED in found_resnums:
                ala541_proj = projections[found_resnums.index(CRM1_P3_CALIBRATION_RESIDUE_1INDEXED)]

            if ala541_proj is not None and cys528_proj < ala541_proj:
                # Flip so increasing projection runs P0 -> P4 (i.e. the
                # Cys528/P4 end is the HIGH end of the axis).
                axis = -axis
                projections = -projections
                cys528_proj = -cys528_proj
                ala541_proj = -ala541_proj

            # Bin boundaries: equal-WIDTH bins spanning the full observed
            # range of projections (min to max among all located residues).
            #
            # History: the original approach here tried to
            # derive exact boundary POSITIONS from the two calibration
            # points alone -- place the P3/P4 boundary exactly at Cys528's
            # projection, the P2/P3 boundary symmetric to it around
            # Ala541's projection, then extrapolate that same half-width
            # outward for the remaining boundaries. Tested against this
            # project's REAL CRM1 structure (not just the synthetic
            # geometry it was originally verified against) and found to
            # fail badly: Ala541 and Cys528 turned out to sit much farther
            # apart, relative to the other 17 lining residues' actual
            # spread, than a uniform-pocket-width assumption predicts --
            # the extrapolated P2/P3 boundary undershot EVERY other
            # residue's projection, putting 14 of 18 located residues in P3
            # and the rest in P4, with P0/P1/P2 completely empty. A
            # follow-up attempt to rescale just the P0-P2 span to whatever
            # data existed below that boundary didn't help either, because
            # there was nothing at all below it to rescale -- the boundary
            # itself, not just its downstream spacing, was wrong.
            #
            # Equal-width bins over the real observed range can't overshoot
            # past real data by construction, so this failure mode isn't
            # possible here. The calibration facts are still used to
            # ORIENT the axis (so P0 is reliably "far from Cys528", not an
            # arbitrary PCA sign choice) -- only the exact boundary
            # POSITIONS are no longer extrapolated from just two points.
            # Whether Ala541 lands in the resulting P3 bin is now checked
            # (calibration_ok below) rather than enforced by construction --
            # an honest pass/fail signal instead of one guaranteed to pass.
            min_proj, max_proj = float(np.min(projections)), float(np.max(projections))
            bin_width = (max_proj - min_proj) / N_NES_SUBPOCKETS
            boundaries = [min_proj + bin_width * i for i in range(1, N_NES_SUBPOCKETS)]
            calibration_mode = ('equal-width bins (axis calibrated via Ala541/Cys528)'
                                 if ala541_proj is not None
                                 else 'equal-width bins (axis uncalibrated -- Ala541 not located this run)')

            def _label_for(proj):
                for i, b in enumerate(boundaries):
                    if proj < b:
                        return SUBPOCKET_LABELS[i]
                return SUBPOCKET_LABELS[-1]

            bin_members = {label: [] for label in SUBPOCKET_LABELS}
            for resnum, proj in zip(found_resnums, projections):
                bin_members[_label_for(proj)].append(resnum)

            subpockets = {}
            for label, bin_resnums in bin_members.items():
                if not bin_resnums:
                    continue
                bin_coords = np.array([[residue_ca[r].x, residue_ca[r].y, residue_ca[r].z]
                                        for r in bin_resnums])
                subpockets[label] = {
                    'residue_numbers_1indexed': [int(r) for r in bin_resnums],
                    'centroid_nm': bin_coords.mean(axis=0),
                    # Atom index of each member residue's CA WITHIN
                    # self.crm1_structure (valid as long as this
                    # NESMDRefiner's self.crm1_structure keeps this same
                    # topology, i.e. for the rest of this object's
                    # lifetime -- truncation only happens once). Lets
                    # _run_crm1_docking track this pocket's REAL simulated
                    # position every production frame (offset by
                    # nes_peptide_atom_count once merged with the peptide),
                    # instead of trusting centroid_nm to still be right
                    # after minimization/equilibration moves things.
                    'atom_indices': [residue_ca_idx[r] for r in bin_resnums if r in residue_ca_idx],
                }

            self.crm1_subpockets = subpockets

            calibration_ok = (
                'P3' in subpockets and ala541_proj is not None and
                CRM1_P3_CALIBRATION_RESIDUE_1INDEXED in subpockets['P3']['residue_numbers_1indexed']
            )
            sizes = ', '.join(f"{k}:{len(v['residue_numbers_1indexed'])}" for k, v in subpockets.items())
            print(f"  {'' if calibration_ok else 'Warning: '} Partitioned NES groove into "
                  f"{len(subpockets)} sub-pockets ({sizes}), {calibration_mode} -- "
                  f"Ala541 calibration check "
                  f"{'PASSED (in P3)' if calibration_ok else 'FAILED (not in expected P3 bin) -- treat pocket labels with extra caution this run'}")

        except Exception as e:
            print(f"  Warning: Error identifying NES sub-pockets: {e}")
            self.crm1_subpockets = {}

    def _capture_full_centroid(self):
        """
        Record the centroid of the FULL (pre-truncation) CRM1+Ran structure.

        _place_peptide_near_groove() needs a "core of the protein" reference
        point to compute an outward approach direction toward Cys528. Once
        _truncate_to_groove_shell() replaces self.crm1_structure with a
        small local shell, that shell's own centroid sits right next to
        Cys528 by construction (it was built by radius from that point) -
        useless for telling "outward" from "inward". So this must be
        captured from the full structure, before truncation happens.
        """
        try:
            coords = np.array([
                [p.x, p.y, p.z]
                for p in self.crm1_structure.positions.value_in_unit(unit.nanometer)
            ])
            self.crm1_full_centroid = coords.mean(axis=0)
        except Exception as e:
            print(f"  Warning: Could not compute full-structure centroid: {e}")
            self.crm1_full_centroid = None

    def _truncate_to_groove_shell(self, radius_nm=GROOVE_SHELL_RADIUS_NM):
        """
        Replace self.crm1_structure with a much smaller structure containing
        only residues near the NES-binding groove, instead of the full
        ~17,000-atom CRM1+Ran complex.

        Why: CRM1's backbone gets rigidly restrained during docking anyway
        (see _run_crm1_docking), so the vast majority of its atoms - far
        from the groove - never meaningfully interact with the peptide, but
        every one of them still has to be included in every force
        evaluation regardless of the nonbonded cutoff (the cutoff only
        reduces which PAIRS get evaluated, not how many atoms are in the
        system to begin with). For a ~20,000-atom system on CPU, that's
        still far too slow to be practical.

        Keeping a local shell of residues around the groove (anything with
        any atom within radius_nm of Cys528) preserves the actual local
        binding environment the peptide interacts with, while cutting the
        atom count by roughly an order of magnitude. Ran isn't specially
        preserved here - its role was to hold CRM1 in the RanGTP-bound
        conformation, which is already baked into these fixed, restrained
        coordinates; it binds a distinct site ~3 nm from Cys528 and isn't
        otherwise interacting with the peptide directly.

        Deleting residues out of the middle of a folded protein creates
        several separate broken chain fragments (the groove is formed by
        multiple non-contiguous stretches of sequence folding together in
        3D) - PDBFixer is run again afterward to properly cap each
        fragment's new termini, the same approach used for the NES peptide
        elsewhere in this file.

        Must be called AFTER _identify_binding_groove() (needs
        self.crm1_cys528_position) and _capture_full_centroid() (so the
        approach-direction reference point survives truncation). Re-runs
        groove identification afterward against the new, renumbered
        structure.
        """
        if self.crm1_cys528_position is None:
            print("  Warning: No Cys528 position known - skipping shell truncation")
            return

        try:
            print(f"  Truncating to a {radius_nm} nm shell around the binding groove...")
            original_atom_count = len(list(self.crm1_structure.topology.atoms()))

            center = np.array([
                self.crm1_cys528_position.x,
                self.crm1_cys528_position.y,
                self.crm1_cys528_position.z
            ])
            positions = self.crm1_structure.positions.value_in_unit(unit.nanometer)

            residues_to_delete = []
            for residue in self.crm1_structure.topology.residues():
                keep = False
                for atom in residue.atoms():
                    pos = positions[atom.index]
                    if np.linalg.norm(np.array([pos[0], pos[1], pos[2]]) - center) <= radius_nm:
                        keep = True
                        break
                if not keep:
                    residues_to_delete.append(residue)

            if not residues_to_delete:
                print("  Warning: Shell radius covers the entire structure - nothing to truncate")
                return

            # Confirmed necessary for 5DHF/5DIF: a sphere cut
            # through a folded protein can leave behind ISOLATED fragments
            # of only 1-2 residues (a scrap of loop that happened to swing
            # within the radius while everything sequence-adjacent to it
            # didn't) -- these are the kept residues right next to a
            # deletion boundary, on BOTH sides, close together. Confirmed
            # directly this is what breaks PDBFixer's terminal capping
            # ("No template found... residue N (LYS)... has 2 H atoms
            # too many"): a fragment this short needs an N-terminal AND a
            # C-terminal patch within a residue or two of each other, and
            # PDBFixer's capping doesn't handle that combination cleanly.
            # These fragments also aren't scientifically meaningful on
            # their own (a disconnected 1-2 residue scrap contributes
            # nothing coherent to a binding-groove simulation), so drop
            # them explicitly rather than trying to force a valid capping
            # for them.
            MIN_FRAGMENT_RESIDUES = 3
            residues_to_delete_set = set(residues_to_delete)
            extra_delete = []
            for chain in self.crm1_structure.topology.chains():
                fragment = []
                for residue in chain.residues():
                    if residue in residues_to_delete_set:
                        if fragment and len(fragment) < MIN_FRAGMENT_RESIDUES:
                            extra_delete.extend(fragment)
                        fragment = []
                    else:
                        fragment.append(residue)
                if fragment and len(fragment) < MIN_FRAGMENT_RESIDUES:
                    extra_delete.extend(fragment)
            if extra_delete:
                print(f"  Also dropping {len(extra_delete)} residue(s) in isolated fragments "
                      f"shorter than {MIN_FRAGMENT_RESIDUES} residues (within the shell radius "
                      f"but disconnected from anything sequence-adjacent -- not meaningfully "
                      f"cappable or scientifically useful on their own)")
                residues_to_delete.extend(extra_delete)

            shell_modeller = Modeller(self.crm1_structure.topology, self.crm1_structure.positions)
            shell_modeller.delete(residues_to_delete)

            # Strip existing hydrogens before re-capping. The kept residues
            # already have hydrogens from the FIRST addMissingHydrogens(7.0)
            # pass on the full structure (before truncation) - re-running
            # PDBFixer's hydrogen placement below on top of those, relying on
            # it to recognize every single existing H atom by name and skip
            # it, is exactly the kind of thing that silently adds a second,
            # near-overlapping hydrogen if even one name doesn't match its
            # internal template. Removing them all first and letting PDBFixer
            # re-protonate the whole shell fresh, in one consistent pass,
            # makes that class of duplicate-atom clash impossible rather than
            # just unlikely.
            hydrogens = [atom for atom in shell_modeller.topology.atoms()
                         if atom.element is not None and atom.element.symbol == 'H']
            if hydrogens:
                shell_modeller.delete(hydrogens)

            # Re-cap the resulting fragments (broken chain ends where
            # residues were removed from the middle of the protein) with
            # PDBFixer - same approach used for the NES peptide elsewhere.
            from pdbfixer import PDBFixer

            tmp_path = None
            try:
                pdb_string = StringIO()
                PDBFile.writeFile(shell_modeller.topology, shell_modeller.positions, pdb_string)
                pdb_string.seek(0)

                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                    tmp.write(pdb_string.getvalue())
                    tmp_path = tmp.name

                # PDBFile.writeFile() just renumbered every
                # residue sequentially from 1, above -- which erases the
                # numbering gap that would otherwise mark where
                # residues_to_delete just created a real break, and
                # doesn't write a TER there either (Modeller.delete()
                # doesn't split the Chain object). Left alone this causes
                # exactly the kind of mis-bonded-template failure this
                # session hit on 5DHF/5DIF ("No template found... residue
                # 7 (TYR)"). Fix by distance instead of residue numbering,
                # since coordinates (unlike residue IDs) survive the round
                # trip intact -- see _insert_ter_at_chain_breaks_by_distance's
                # docstring for the full story.
                _insert_ter_at_chain_breaks_by_distance(tmp_path)

                fixer = PDBFixer(filename=tmp_path)
                fixer.findMissingResidues()

                # Every gap findMissingResidues() finds here is a stretch WE
                # just deleted on purpose (residues_to_delete above) - not
                # real missing crystallographic density like it would be for
                # a freshly-loaded structure. PDBFixer can't tell the
                # difference: left alone, addMissingAtoms() below will try to
                # rebuild those deleted stretches back in with crude
                # loop-modeling geometry, which doesn't know the loop is
                # physically impossible to close for gaps this large and
                # produces exactly the kind of severe clashes seen here -
                # partially undoing the truncation in the process. Clearing
                # it out keeps addMissingAtoms()/addMissingHydrogens() doing
                # only what we actually want: filling in missing atoms on
                # residues we kept, and capping the new chain termini.
                if fixer.missingResidues:
                    gap_count = len(fixer.missingResidues)
                    residue_count = sum(len(v) for v in fixer.missingResidues.values())
                    print(f"  Ignoring {gap_count} gap(s) ({residue_count} residues) "
                          f"PDBFixer flagged as 'missing' - these are residues we "
                          f"deliberately deleted, not real missing density")
                    fixer.missingResidues = {}

                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(7.0)

                self.crm1_structure = Modeller(fixer.topology, fixer.positions)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            new_atom_count = len(list(self.crm1_structure.topology.atoms()))
            print(f"  Shell truncation: {original_atom_count:,} -> {new_atom_count:,} atoms "
                  f"({100 * new_atom_count / original_atom_count:.1f}% of original)")

            # Residue indices are completely renumbered now - re-locate
            # Cys528 and the groove residues against the new structure.
            self._identify_binding_groove(near_position=self.crm1_cys528_position)
            # Same renumbering problem applies to the sub-pocket reference
            # residues -- re-locate each by nearest-CA-to-remembered-point
            # (self._groove_lining_reference_positions was captured before
            # truncation, above).
            self._identify_nes_subpockets(near_positions=self._groove_lining_reference_positions)

        except Exception as e:
            print(f"  Warning: Shell truncation failed, continuing with full structure: {e}")
            import traceback
            traceback.print_exc()

    def _check_structure_energy(self, modeller, label="structure", relax=False, max_iterations=1000):
        """
        One-time diagnostic: build a minimal system for the given structure
        ALONE and report its potential energy, before it's ever merged with
        an NES peptide. This isolates two very different problems that both
        show up as "huge starting energy" later on:
          - the reference structure itself has severe internal clashes
            (e.g. from PDBFixer placing added/missing atoms badly) - a
            problem with the reference file/loading, unrelated to docking
          - the reference structure is fine on its own, and the huge energy
            only appears after merging with the peptide - a problem with
            placement/merging instead

        Runs once at load time (not per-candidate), so the one-off cost of
        building a system for the full CRM1+Ran structure is acceptable.

        If relax=True and the energy indicates real clashes, minimizes this
        structure IN PLACE (updates modeller.positions) and reports the
        before/after energy. This exists as a safety net for the truncated
        groove shell specifically: even with the findMissingResidues() gap
        fix above, PDBFixer re-capping many new chain termini at once can
        still leave some local strain at the cut points (e.g. a capping
        hydrogen placed slightly too close to a neighboring fragment). Doing
        this once here, at load time, means it's resolved a single time
        rather than needing to resolve (or failing to resolve) the same
        strain freshly on every candidate's docking run afterward.
        """
        try:
            print(f"  Checking {label} standalone energy (diagnostic)...")
            forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=CutoffNonPeriodic,
                nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer,
                constraints=HBonds
            )
            integrator = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
            platform, properties = _select_fast_platform()
            if platform is not None:
                sim = Simulation(modeller.topology, system, integrator, platform, properties)
            else:
                sim = Simulation(modeller.topology, system, integrator)
            sim.context.setPositions(modeller.positions)
            state = sim.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            print(f"  {label} standalone potential energy: {energy:,.1f} kJ/mol")

            if energy > 1e6:
                if relax:
                    print(f"  Warning: {label} has severe internal clashes ALL ON ITS OWN - "
                          f"minimizing once now to resolve this permanently, rather than "
                          f"leaving it to fail on every candidate's docking run afterward...")
                    sim.minimizeEnergy(maxIterations=max_iterations)
                    state = sim.context.getState(getEnergy=True, getPositions=True)
                    new_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    modeller.positions = state.getPositions()
                    print(f"  {label} relaxed: {energy:,.1f} -> {new_energy:,.1f} kJ/mol")
                    if new_energy > 1e6:
                        print(f"  Warning: {label} is still high after relaxing - there may be a "
                              f"structural problem this minimization alone can't fix "
                              f"(e.g. a genuinely mismodeled region).")
                    energy = new_energy
                else:
                    print(f"  Warning: {label} has severe internal clashes ALL ON ITS OWN - this is "
                          f"NOT caused by peptide placement/docking. The reference structure "
                          f"itself needs attention (likely from how missing atoms/residues got "
                          f"filled in during loading).")
            else:
                print(f"  {label} looks structurally sound on its own - if docking runs "
                      f"still show huge starting energy, the problem is in how the peptide "
                      f"gets placed/merged, not in this reference structure.")
            return energy
        except Exception as e:
            print(f"  Warning: Could not compute standalone energy for {label}: {e}")
            return None

    def _place_peptide_near_groove(self, peptide_modeller):
        """
        Rigidly translate the NES peptide so it starts near the CRM1 Cys528
        hydrophobic groove, instead of keeping its original coordinates from
        the source AlphaFold model - which have no relationship to CRM1's
        coordinate frame, and reliably cause severe atomic overlap (millions
        of kJ/mol of clash energy) when the two structures are naively
        merged as-is.

        This is a simple rigid-body translation, not a real docking search:
        it points the peptide's centroid at a small clearance distance from
        Cys528, along the line from CRM1's own center of mass through
        Cys528 (i.e. approaching from "outside" the protein, where the
        groove is surface-exposed), and leaves the flexible minimization /
        equilibration / production steps to refine the actual binding pose
        from there.

        Mutates peptide_modeller.positions in place. No-op if we don't have
        a CRM1 structure/groove to place the peptide relative to.
        """
        if self.crm1_cys528_position is None or self.crm1_structure is None:
            return

        old_positions = peptide_modeller.positions.value_in_unit(unit.nanometer)
        coords = np.array([[p.x, p.y, p.z] for p in old_positions])
        centroid = coords.mean(axis=0)
        local_coords = coords - centroid  # peptide's own geometry, centroid-relative

        crm1_coords = np.array([
            [p.x, p.y, p.z]
            for p in self.crm1_structure.positions.value_in_unit(unit.nanometer)
        ])
        # Use the FULL (pre-truncation) structure's centroid as the "core of
        # the protein" reference for the approach direction - the truncated
        # shell's own centroid sits right next to Cys528 by construction
        # (it was built by radius from that point), so it can't tell
        # "outward" from "inward" anymore. Falls back to the shell's own
        # centroid only if the full centroid wasn't captured for some reason.
        crm1_centroid = (
            self.crm1_full_centroid if self.crm1_full_centroid is not None
            else crm1_coords.mean(axis=0)
        )
        cys528 = np.array([
            self.crm1_cys528_position.x,
            self.crm1_cys528_position.y,
            self.crm1_cys528_position.z
        ])

        # Approach vector: from CRM1's core outward through Cys528 - the
        # direction a peptide binding the surface-exposed groove would
        # plausibly approach from.
        approach_dir = cys528 - crm1_centroid
        norm = np.linalg.norm(approach_dir)
        approach_dir = approach_dir / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

        # A fixed clearance measured from the peptide's centroid isn't
        # enough on its own: if the peptide extends well beyond its
        # centroid in some direction (e.g. an extended, non-compact
        # conformation as extracted from the source model), and we only
        # translate (no rotation), part of it can plunge straight into
        # CRM1's densely packed interior even though the centroid placement
        # looks perfectly reasonable - this is exactly what was happening
        # (atoms landing 0.017 nm apart - essentially exact overlap).
        #
        # base the clearance on the peptide's own actual radius (its
        # farthest atom from its own centroid) plus a safety margin, then
        # VERIFY with real atom-to-atom distances and push further out if
        # still too close, instead of trusting a single guessed distance.
        peptide_radius = np.linalg.norm(local_coords, axis=1).max()
        safety_margin = 0.6  # nm, beyond the peptide's own size
        clearance_nm = peptide_radius + safety_margin

        # min_safe_distance was originally 0.25 nm, checking only the SINGLE
        # closest atom pair. Two problems with that in practice (root cause
        # of starting energies in the millions of kJ/mol that minimization
        # then can't actually resolve, "converging" while stuck instead of
        # relaxing, and NaN positions once equilibration dynamics run on top
        # of that): (1) 0.25 nm is inside the repulsive wall for a typical
        # heavy-atom pair (LJ sigma ~0.3-0.34 nm for C/N/O), so "clearing"
        # 0.25 nm can still mean severe steric overlap; (2) checking only
        # the global minimum distance says nothing about how many OTHER
        # pairs are also nearly as close -- a peptide sliding flat against
        # the groove surface can have dozens of pairs simultaneously near
        # the threshold, whose repulsive energies all stack up even though
        # each individually "passes". Now requires BOTH a safer minimum
        # distance AND that no more than a handful of pairs sit inside a
        # tighter severe-clash radius.
        min_safe_distance = 0.40   # nm - minimum acceptable closest-atom-pair distance
        severe_clash_radius = 0.20  # nm - pairs this close are treated as real overlap
        max_severe_clashes = 0      # zero tolerance for near-overlapping pairs
        max_attempts = 14
        push_step_nm = 0.4
        translated_coords = None
        min_dist = None
        n_severe = None
        attempt = 0

        for attempt in range(max_attempts):
            target = cys528 + approach_dir * clearance_nm
            translated_coords = local_coords + target

            diffs = translated_coords[:, None, :] - crm1_coords[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            min_dist = dists.min()
            n_severe = int(np.sum(dists < severe_clash_radius))

            if min_dist >= min_safe_distance and n_severe <= max_severe_clashes:
                break

            # Still too close (or too many near-overlapping pairs) - push
            # further out along the approach direction and re-check, rather
            # than trusting the first guess.
            clearance_nm += push_step_nm

        new_positions = [Vec3(*row) for row in translated_coords] * unit.nanometer
        peptide_modeller.positions = new_positions

        print(f"    Placed NES peptide near Cys528 groove "
              f"(peptide radius {peptide_radius:.2f} nm, clearance {clearance_nm:.2f} nm, "
              f"{attempt + 1} attempt(s), closest approach {min_dist:.3f} nm, "
              f"{n_severe} severe-clash pair(s) < {severe_clash_radius} nm)")
        if min_dist < min_safe_distance or n_severe > max_severe_clashes:
            print(f"    Warning: Could not reach a clash-free placement after {max_attempts} "
                  f"attempts (closest approach still {min_dist:.3f} nm) - minimization "
                  f"may still start with some overlap.")

    def _find_phi_register(self, sequence: str) -> Dict[str, Optional[int]]:
        """
        Locate this NES candidate's Phi-anchor register within `sequence`,
        for mapping onto CRM1's P0-P4 sub-pockets (see the module-level
        PHI_REGISTER_RE/PHI_ANCHOR_VOCAB block for the literature and
        conventions this follows).

        Phi1-Phi4 use the SAME "rightmost regex match wins" convention as
        _find_pssm_anchor() in nes_ml_predictor_improved.py, so this agrees
        with what the rest of the pipeline already treats as "the" register
        for a given sequence rather than introducing a second, possibly
        conflicting one. Phi0 is independently optional (Guttler et al.
        2010's fifth, later-discovered pocket) -- found as the nearest
        Phi-vocabulary residue within PHI0_LOOKBACK positions N-terminal of
        Phi1, or left as None if there isn't one.

        Returns {'P0': idx or None, 'P1': idx, 'P2': idx, 'P3': idx,
        'P4': idx, 'register_match_type': 'full'|'partial'|'none',
        'pattern_class': 'standard'|'class4'} -- 0-indexed positions
        within `sequence`.

         Class-4 is now tried BEFORE the standard pattern, not
        after -- reversed from the original implementation once real data
        (X11L2/5UWS crystal_full_grid_check.py backfill) showed why that
        ordering was wrong. Class-4 needs a minimum 15-residue span (5
        anchors, gaps of 2+3+2+3); the standard pattern needs only 7-12.
        Whenever a candidate window is long enough to contain a genuine
        class-4 register (e.g. X11L2's widened native_range), it is ALSO
        long enough for the standard pattern to find a shorter, purely
        coincidental 4-anchor match somewhere inside that same window --
        and under the old "standard first, return on first hit" ordering,
        that coincidental match was found first and the real class-4
        register was never even attempted. Checking class-4 first avoids
        this: it cannot match any sequence shorter than 15 residues at
        all, so this reordering is a no-op for every normal 8-12 residue
        candidate (the vast majority of real NES windows) -- it only
        changes behaviour for the rare, intentionally-widened windows
        where a full class-4 register is structurally possible, which is
        exactly the case it needs to fix. A 'full' class-4 match is
        reported with pattern_class='class4' and P0 set directly as a
        real matched anchor (not found via lookback -- class4 always
        supplies its own real P0). If class-4 doesn't match, this falls
        back to the strict standard pattern (PHI_REGISTER_RE, all 4 of
        P1-P4 hydrophobic), and if THAT doesn't match either, to the BEST
        relaxed 3-of-4 match on the standard pattern (see
        _PHI_RELAXED_PATTERNS) -- exactly one of P1-P4 will be None in
        that case, the other 3 are real positions. 'register_match_type'
        is 'none' only when none of the three match. P0 is scored
        independently for the standard-pattern cases, and only looked up
        relative to P1 when P1 itself is present (skipped, not guessed, if
        P1 is the missing anchor in a partial match).
        """
        seq = sequence.upper()

        # Class-4 first (see docstring above for why) -- same "rightmost
        # match wins" convention as every other tier here.
        last_class4 = None
        for m in PHI_REGISTER_CLASS4_RE.finditer(seq):
            last_class4 = m
        if last_class4 is not None:
            register = {
                'P0': last_class4.start(1),
                'P1': last_class4.start(2),
                'P2': last_class4.start(3),
                'P3': last_class4.start(4),
                'P4': last_class4.start(5),
            }
            register['register_match_type'] = 'full'
            register['pattern_class'] = 'class4'
            return register

        last_match = None
        for m in PHI_REGISTER_RE.finditer(seq):
            last_match = m

        if last_match is not None:
            register = {
                'P1': last_match.start(1),
                'P2': last_match.start(2),
                'P3': last_match.start(3),
                'P4': last_match.start(4),
            }
            register['register_match_type'] = 'full'
            register['pattern_class'] = 'standard'
        else:
            # Neither strict pattern matched -- try the 4 relaxed variants
            # of the STANDARD pattern (one per anchor slot allowed to be
            # non-hydrophobic) and keep whichever match starts furthest
            # toward the C-terminus, same "most C-terminal wins" convention
            # as the strict case, applied across all 4 variants together so
            # this doesn't depend on which slot happens to be the missing
            # one. (No relaxed variant of class4 -- not enough real ground
            # truth yet to know which anchor, if any, is safe to loosen for
            # that pattern the way Snurportin1's case justified for the
            # standard one.)
            best = None  # (match_start, missing_label, match_obj)
            for missing_label, pattern in _PHI_RELAXED_PATTERNS:
                for m in pattern.finditer(seq):
                    if best is None or m.start() >= best[0]:
                        best = (m.start(), missing_label, m)
            if best is None:
                register = {label: None for label in SUBPOCKET_LABELS}
                register['register_match_type'] = 'none'
                register['pattern_class'] = None
                return register
            _, missing_label, m = best
            register = {}
            for i, label in enumerate(('P1', 'P2', 'P3', 'P4'), start=1):
                register[label] = None if label == missing_label else m.start(i)
            register['register_match_type'] = 'partial'
            register['pattern_class'] = 'standard'

        phi1_pos = register.get('P1')
        phi0_pos = None
        if phi1_pos is not None:
            for i in range(max(0, phi1_pos - PHI0_LOOKBACK), phi1_pos):
                if seq[i] in PHI_ANCHOR_VOCAB:
                    phi0_pos = i
                    break
        register['P0'] = phi0_pos
        return register

    def _find_reversed_phi_register(self, sequence: str, register: Dict) -> Dict:
        """
         The 'minus-direction' interpretation of an already-
        computed forward register -- confirmed necessary by real crystal
        ground truth this project: crystal_sanity_check.py's grid on 5DIF
        (CPEB4 NES, Fung/Fu/Chook 2015) showed anchor_occupancy_score
        BACKWARDS (scrambled registration scoring higher than correct),
        while raw_binding_score showed the expected correct-beats-
        scrambled direction. Root cause: _find_phi_register (and every
        anchor-to-pocket assignment downstream of it) assumes every NES
        binds N-terminus-first at CRM1's wide end (P0) and C-terminus-last
        at the narrow end (P4) -- true for PKI/Rev/Snurportin1/Paxillin,
        the ONLY structures this pipeline had ground truth for until this
        session added 5DHF/5DIF. But Fung/Fu/Chook 2015 (eLife 4:e10034)
        is specifically about hRio2NES and CPEB4NES binding in the OPPOSITE
        ("minus") orientation -- C-terminus at the wide end instead. A
        sequence-only regex can't distinguish these (the spacing pattern
        alone doesn't encode which physical end is which), so
        _find_phi_register alone assigning P1-P4 in N-to-C sequence order
        assigns CPEB4's real anchors to the WRONG physical pockets even
        though it happens to find a valid spacing match.

        This builds the alternative interpretation: the SAME matched
        sequence positions as `register`, but assigned to physical pockets
        in the OPPOSITE order. Caller (see _best_orientation_matched)
        tries both this and the forward register via a real Kabsch fit
        and keeps whichever actually fits the groove geometry better,
        rather than assuming a direction.

         Generalized to the class-4 case. This function
        originally only ever handled the standard pattern, where P0 is
        NEVER a real matched anchor -- always a bonus, independently
        looked-up position N-terminal of P1 (see _find_phi_register). So
        the original 4-anchor version below (P1<->P4, P2<->P3 swap, THEN
        derive a fresh P0 via C-terminal lookback from the new P1) was
        correct for that case: there was never a real P0 to preserve
        through the reversal, only a fresh one to look up on the other
        side. But class-4's P0 (see PHI_REGISTER_CLASS4_RE) IS a real,
        directly matched anchor, on equal footing with P1-P4 -- not a
        bonus lookback position. Running it through the old 4-anchor-only
        logic silently discarded that real P0 and replaced it with an
        unrelated, independently re-derived lookback position instead,
        which is how X11L2 (5UWS)'s corrected class-4 register was still
        only reaching the physical groove with 4 of its 5 real anchors
        even after the registration-ordering fix (see
        crystal_full_grid_check.py's 5UWS re-run, report).
        For a genuine 5-anchor register, the faithful reversal mirrors
        the whole chain end-to-end -- P0<->P4, P1<->P3, P2 unchanged (the
        middle anchor of an odd-length register is its own mirror point)
        -- with no lookback needed at all, since every position involved
        was already a real match.
        """
        if register.get('pattern_class') == 'class4':
            p0, p1, p2, p3, p4 = (register.get('P0'), register.get('P1'),
                                   register.get('P2'), register.get('P3'), register.get('P4'))
            return {
                'P4': p0, 'P3': p1, 'P2': p2, 'P1': p3, 'P0': p4,
                'register_match_type': register.get('register_match_type'),
                'pattern_class': 'class4',
            }

        seq = sequence.upper()
        p1, p2, p3, p4 = register.get('P1'), register.get('P2'), register.get('P3'), register.get('P4')
        reversed_register = {
            'P4': p1, 'P3': p2, 'P2': p3, 'P1': p4,
            'register_match_type': register.get('register_match_type'),
            'pattern_class': register.get('pattern_class'),
        }
        # P0 is always physically the widest pocket, adjacent to P1 -- but
        # WHICH sequence direction is "further toward the wide end" flips
        # with orientation. In reversed mode, physical P1 is occupied by
        # `p4` (the forward register's P4 anchor -- see reversed_register
        # above), and the wide end (P0) is on the sequence's C-terminal
        # side of THAT anchor (reversed_register['P1'] == p4), mirroring
        # the forward case's N-terminal lookback from its own P1 anchor.
        # (Bug fixed: this used to look back from `p1` --
        # forward's P1, not reversed's own P1 -- which for some sequences
        # coincidentally landed ON another already-matched anchor's own
        # index, e.g. producing P0 == P3 for MHSLESSL/CPEB4NES. Caught by
        # a synthetic geometry unit test before this reached the pod.)
        phi0_pos = None
        if p4 is not None:
            for i in range(p4 + 1, min(len(seq), p4 + 1 + PHI0_LOOKBACK)):
                if seq[i] in PHI_ANCHOR_VOCAB:
                    phi0_pos = i
                    break
        reversed_register['P0'] = phi0_pos
        return reversed_register

    def _best_orientation_matched(self, sequence: str, residues, old_positions,
                                   trust_input_positions: bool = False):
        """
         Try both the forward (plus-direction: PKI/Rev/
        Snurportin1/Paxillin-style) and reversed (minus-direction:
        hRio2NES/CPEB4NES-style, see _find_reversed_phi_register) anchor-
        to-pocket assignments for `sequence`'s matched Phi-register, and
        let the ACTUAL groove geometry decide which one is right for this
        candidate, rather than this pipeline silently assuming every NES
        binds N-terminus-first the way it did before this project's real
        ground truth (5DIF) showed that assumption backwards for a real,
        minus-direction binder.

        trust_input_positions: (added after the FIRST version
        of this fix -- a Kabsch-fit-only decision -- was deployed and
        tested against the real 5DIF crystal structure on the pod and
        found to still pick the WRONG orientation there.) Real production
        data exposed a genuine flaw in using preliminary Kabsch-fit RMSD
        as the decision metric: with only 3 anchors matched (this
        pipeline's 3-of-4 relaxed register match) and CRM1's sub-pockets
        laid out close to a single line (this module's own equal-width-
        bin partitioning), a rigid-body fit has enough ROTATIONAL freedom
        to find some rotation that makes the WRONG point correspondence
        look almost as good as the right one -- confirmed on the real pod
        run (5DIF: forward 2.96 Å vs reversed 3.04 Å, a near-tie that
        happened to go the wrong way) and reproduced/confirmed locally
        with noisy synthetic data mimicking that near-tie (Kabsch RMSD:
        forward 1.77 Å vs reversed 1.83 Å -- both "fine", picks wrong
        answer; same data's RAW un-rotated anchor-to-target distance:
        forward 26.06 Å vs reversed 4.20 Å -- decisively, correctly
        favors reversed). A rigid fit is the right tool for comparing
        SHAPES when there's no reference frame yet (a not-yet-placed
        candidate peptide, still in its own arbitrary local coordinates --
        that's the trust_input_positions=False / default path, used when
        the caller is about to physically move the peptide, i.e.
        apply_transform=True upstream). But when old_positions are ALREADY
        the real, correctly-placed coordinates in the SAME frame as
        self.crm1_subpockets (a real crystal structure's own pose, e.g.
        the crystal-sanity-check ground-truth use case -- upstream's
        apply_transform=False, meaning "don't move this, it's already
        right"), letting the decision use a free rotation throws away the
        one thing that actually makes the test meaningful: whether each
        REAL anchor is already sitting near its REAL assigned pocket, with
        no rotation available to paper over a wrong correspondence. So
        when trust_input_positions=True, orientation is instead decided by
        mean raw (un-rotated, un-translated) anchor-to-target distance --
        a much more decisive, appropriate signal for that specific case.
        The final Kabsch fit/RMSD reported below (and actually applied,
        when apply_transform=True) is unaffected either way; only the
        forward-vs-reversed DECISION method changes.

        Returns (matched, orientation_label, register_used) where matched
        is the same [(label, seq_idx), ...] list format
        _place_peptide_via_subpocket_registration already expects,
        orientation_label is 'forward' or 'reversed' (for logging/
        diagnostics), or (None, None, None) if NEITHER orientation has
        >= 3 matched anchors.
        """
        register = self._find_phi_register(sequence)
        reversed_register = self._find_reversed_phi_register(sequence, register)

        candidates = []
        for orientation, reg in (('forward', register), ('reversed', reversed_register)):
            matched = [(label, idx) for label, idx in reg.items()
                       if idx is not None and label in self.crm1_subpockets]
            if len(matched) < 3:
                continue
            mobile_points, target_points = [], []
            for label, seq_idx in matched:
                if seq_idx >= len(residues):
                    continue
                residue = residues[seq_idx]
                ca_atom = next((a for a in residue.atoms() if a.name == 'CA'), None)
                if ca_atom is None:
                    continue
                pos = old_positions[ca_atom.index]
                mobile_points.append([pos[0], pos[1], pos[2]])
                target_points.append(self.crm1_subpockets[label]['centroid_nm'])
            if len(mobile_points) < 3:
                continue
            mobile_arr, target_arr = np.array(mobile_points), np.array(target_points)

            if trust_input_positions:
                # Decisive, rotation-free signal -- see docstring above.
                # This is the ONLY thing used to pick the orientation;
                # the Kabsch fit below is still computed (both branches
                # need it) purely for the diagnostic/reported fit RMSD.
                decision_score = float(np.mean(np.linalg.norm(mobile_arr - target_arr, axis=1)))
            R, t = self._kabsch_transform(mobile_arr, target_arr)
            fit_rmsd = self._rmsd((R @ mobile_arr.T).T + t, target_arr)
            if not trust_input_positions:
                decision_score = fit_rmsd
            candidates.append((decision_score, fit_rmsd, orientation, matched))

        if not candidates:
            return None, None, None

        candidates.sort(key=lambda c: c[0])
        best_score, best_fit_rmsd, best_orientation, best_matched = candidates[0]
        metric_name = "raw anchor-target distance" if trust_input_positions else "Kabsch fit RMSD"
        if len(candidates) == 2:
            other_score, _, other_orientation, _ = candidates[1]
            print(f"    Orientation check [{metric_name}] ({other_orientation} "
                  f"{other_score * 10:.2f} Å vs {best_orientation} {best_score * 10:.2f} Å): "
                  f"using {best_orientation}-direction anchor-to-pocket assignment")
        return best_matched, best_orientation, register

    def classify_nes_binding_mode(self, sequence: str) -> Dict:
        """
         Sequence-only classification of which starting-pose
        method (native vs idealized_helix) is more likely trustworthy for
        this candidate, and how much confidence to put in anchor_occupancy_
        score at all -- used instead of picking a starting conformation by
        whichever happened to score highest on THIS run (see refine_nes_
        candidates' best_tag logic), per the collated-comparison report's
        recommendation: report native and idealized_helix separately and
        use sequence type, not score, to decide which to trust.

        WHY SEQUENCE-ONLY: no MD required, so this can run before -- or
        instead of -- any docking, and it can't be circular (it doesn't use
        anything idealized_helix/native docking produced).

        CALIBRATION CAVEAT (be honest about this): the only ground truth
        available is 5 real crystal structures (3 unique NES/NES-like
        cases, 2 with independent replicate crystal forms) from this
        session's crystal_sanity_check.py / idealized_helix_vs_crystal_
        check.py work. That is NOT enough to fit a real classifier, so this
        uses two mechanistically-grounded rules instead of a trained model:

          1. Does _find_phi_register even match a register? If not, this
             mirrors exactly what made anchor_occupancy_score uncomputable
             (None) for Snurportin1's real NES-like segment in BOTH crystal
             forms tested (3GJX, 3GB8) -- that candidate type is a known,
             currently-unaddressed blind spot (see the collated report,
             Section 2/6), not something this function can resolve. Neither
             native nor idealized_helix's raw_binding_score was reliable
             for that case either (both saturated at the same ceiling for
             correct AND scrambled registration), so this is flagged
             low-confidence rather than silently defaulting to one method.
          2. Given a register DOES match, does the core sequence contain a
             Proline? Guttler et al. 2010 identifies a "critical proline"
             as the literature-documented mechanism by which HIV-1 Rev's
             NES binds in an EXTENDED conformation instead of PKI-NES's
             alpha-helical one -- and this project's idealized_helix_vs_
             crystal_check.py directly confirmed the consequence: forcing
             an idealized alpha helix onto the real Rev-NES sequence
             (present in BOTH the 3NBZ and 3NC0 crystal forms) converged
             8-15 Angstrom from the real bound pose, versus 3-5 Angstrom
             for the genuinely helical PKI-NES case. IMPORTANT LIMITATION:
             this rule was validated on exactly one real proline-containing
             case (replicated across 2 crystals of the same underlying
             complex, so really n=1 independent case, not n=2) -- treat
             'medium', not 'high', confidence as accurate, and expect it to
             specifically MISS non-proline atypical cases like Snurportin1
             (which is why rule 1 exists as a separate, earlier check --
             Snurportin1's sequence has zero Pro/Gly breakers and would
             otherwise be misclassified as "likely helical").

        Returns:
            {
              'binding_mode_class': 'no_register_match' | 'extended_atypical' | 'likely_helical',
              'recommended_primary_method': 'native' | 'idealized_helix' | 'extended' | 'low_confidence_both',
              'confidence': 'low' | 'medium',
              'phi_register_matched': bool,
              'contains_proline': bool,
              'helix_breakers': int,
              'rationale': str,
            }

        UPDATE: 'extended_atypical' now recommends 'extended'
        (the literal PPII-geometry starting structure built by
        _build_extended_pdb, added) instead of 'native'. When
        this rule was first written , 'extended' didn't exist
        yet as an option, so 'native' (AlphaFold's isolated-state
        prediction) was the best available fallback for proline-containing/
        atypical candidates -- but this rule's OWN rationale already flags
        native as an unreliable stand-in (AlphaFold's isolated prediction
        for a disordered, pre-binding stretch is frequently not actually
        extended either). 'extended' is a strictly closer mechanistic match
        to the real Rev-NES bound pose than native ever was. See
        recommend_starting_conformation() below for the unified, single-
        recommendation entry point that combines this with the sequence-
        propensity check for the pre-MD "which one conformation should I
        even run" decision.
        """
        sequence = sequence.upper()
        register = self._find_phi_register(sequence)
        match_type = register.get('register_match_type', 'none')
        phi_register_matched = match_type in ('full', 'partial')

        contains_proline = 'P' in sequence
        helix_breakers = sequence.count('P') + sequence.count('G')

        if not phi_register_matched:
            return {
                'binding_mode_class': 'no_register_match',
                'recommended_primary_method': 'low_confidence_both',
                'confidence': 'low',
                'phi_register_matched': False,
                'register_match_type': match_type,
                'contains_proline': contains_proline,
                'helix_breakers': helix_breakers,
                'rationale': (
                    "No Phi-anchor register found for this sequence -- not even a relaxed 3-of-4 match. "
                    "anchor_occupancy_score will not be computable for EITHER starting conformation. raw_"
                    "binding_score is also unreliable at this sample size, so treat any score for this "
                    "candidate as low-confidence and consider manual/expert review rather than trusting "
                    "either method's automated score."
                ),
            }
        elif match_type == 'partial':
            missing_anchor = next((lbl for lbl in ('P1', 'P2', 'P3', 'P4') if register.get(lbl) is None), None)
            # Was 'native' -- updated for the same reason as the
            # full-match extended_atypical branch below: AlphaFold's
            # isolated-state prediction is not a reliable stand-in for an
            # extended/PPII pose, and 'extended' (the literal PPII-geometry
            # starting structure from _build_extended_pdb) is a closer
            # mechanistic match for a proline-driven atypical candidate,
            # partial register match or not. Caught by real testing: P42566
            # (766-800, 13 Pro/Gly helix-breakers) only gets a partial (3-of-4)
            # register match, so it landed here rather than in the full-match
            # extended_atypical branch -- this branch needed the same fix.
            recommended = 'extended' if (contains_proline or helix_breakers >= 2) else 'idealized_helix'
            return {
                'binding_mode_class': 'partial_register_match',
                'recommended_primary_method': recommended,
                'confidence': 'low',
                'phi_register_matched': True,
                'register_match_type': 'partial',
                'missing_anchor': missing_anchor,
                'contains_proline': contains_proline,
                'helix_breakers': helix_breakers,
                'rationale': (
                    f"Only 3 of 4 Phi-anchor positions are hydrophobic at the expected spacing (missing "
                    f"anchor: {missing_anchor}) -- the same pattern as Snurportin1's real NES-like segment "
                    f"(SQALASSFSVS) and, confirmed independently this project, Paxillin's real NES (5UWH, "
                    f"Fung et al. 2017, LMASLSDF). Paxillin's crystal_sanity_check.py grid is the first real "
                    f"MD evidence for a partial-match case: anchor_occupancy_score WAS computable (unlike "
                    f"Snurportin1's still-untested case) and showed a clear correct-vs-scrambled gap "
                    f"(+0.52 no-relax, +0.46 relaxed -- comparable in size to the full-match structures), "
                    f"confirming the relaxed 3-of-4 placement isn't just a sequence-level label, it produces "
                    f"real discriminating MD signal. Still flagged LOW confidence, not medium -- this is one "
                    f"real MD-confirmed case (same standard this function applies to the proline rule below), "
                    f"and two OTHER real sequences tested the same way (SMAD4/5UWU, X11L2/5UWS) found no "
                    f"register at all (not even partial) and fell back to the known-uninformative raw_"
                    f"binding_score ceiling -- so a partial match is real signal when found, but by no means "
                    f"guaranteed for every atypical NES. Recommend manual review alongside whichever method's "
                    f"score this returns; don't treat a low score here as confident evidence against binding."
                ),
            }
        elif contains_proline or helix_breakers >= 2:
            return {
                'binding_mode_class': 'extended_atypical',
                'recommended_primary_method': 'extended',
                'confidence': 'medium',
                'phi_register_matched': True,
                'register_match_type': 'full',
                'contains_proline': contains_proline,
                'helix_breakers': helix_breakers,
                'rationale': (
                    "Contains a Proline (or >=2 helix-breaking Pro/Gly residues) alongside a matched Phi-"
                    "register -- the same signature as HIV-1 Rev's real, experimentally-solved NES, which "
                    "binds CRM1 in an EXTENDED conformation, not the canonical alpha helix. This project's "
                    "idealized_helix_vs_crystal_check.py showed forcing an idealized helix onto that real "
                    "sequence converges 8-15 Angstrom from the true bound pose (vs 3-5 Angstrom for the "
                    "genuinely helical PKI-NES case) and collapses its score toward zero -- i.e. idealized_"
                    "helix would likely UNDERSCORE a real binder of this type. ( : now recommends "
                    "'extended' -- the literal PPII-geometry starting structure from _build_extended_pdb -- "
                    "rather than 'native', since AlphaFold's isolated-state prediction for a disordered, "
                    "pre-binding stretch is frequently not actually extended either; 'extended' is a closer "
                    "mechanistic match to the real Rev-NES bound pose.) Treat a low idealized_helix score for "
                    "this candidate as inconclusive, not evidence against binding. Calibrated on one real "
                    "proline-containing ground-truth case (2 independent crystal forms of the same complex) "
                    "-- medium, not high, confidence."
                ),
            }
        else:
            return {
                'binding_mode_class': 'likely_helical',
                'recommended_primary_method': 'idealized_helix',
                'confidence': 'medium',
                'phi_register_matched': True,
                'register_match_type': 'full',
                'contains_proline': contains_proline,
                'helix_breakers': helix_breakers,
                'rationale': (
                    "Matched Phi-register with no Proline and <2 helix-breaking residues -- consistent with "
                    "PKI-NES, the real, experimentally-solved classic alpha-helical NES, for which this "
                    "session's idealized_helix_vs_crystal_check.py showed idealized_helix converges "
                    "reasonably close to the true bound pose (3-5 Angstrom backbone RMSD). Trust the "
                    "idealized_helix result as primary; use native as a secondary cross-check. IMPORTANT: "
                    "this rule cannot detect atypical-but-proline-free cases like Snurportin1's real NES-like "
                    "segment, which also lacks Pro/Gly breakers yet still saw idealized_helix converge 14-22 "
                    "Angstrom off target in this project's check -- calibrated on one real ground-truth case, "
                    "medium confidence, and this is a known residual risk of this specific rule."
                ),
            }

    def recommend_starting_conformation(self, sequence: str) -> Dict:
        """
         Unified pre-MD decision of which SINGLE starting
        conformation -- 'native' | 'idealized_helix' | 'extended' -- to
        actually dock a candidate from, so a caller can pick one and run
        one trajectory instead of testing all three and picking a winner
        after the fact. Motivated directly by the Rev-NES/3NBZ helix-
        trapping finding this project: candidates were previously always
        started from idealized_helix regardless of sequence, and a real
        proline-containing NES got kinetically stuck in that wrong basin
        for the whole trajectory.

        Combines two independent, differently-grounded signals rather than
        picking one:
          1. classify_nes_binding_mode(sequence) -- mechanistic, calibrated
             against real crystal ground truth (PKI-NES helical, Rev-NES
             extended/proline), gated on whether a Phi-anchor register is
             even found. This is the PRIMARY signal -- it's the only one
             validated against real bound structures.
          2. _predict_likely_conformation(sequence) -- cheap Chou-Fasman
             helix (Pa) vs beta (Pb) propensity comparison, with a proline-
             count>=2 override. Sequence-only, doesn't require a register
             match, so it's used as a SECONDARY cross-check/tie-breaker,
             and specifically to flag disagreement on the 'likely_helical'
             case (downgrades confidence rather than silently overriding --
             the register-based rule is the one with real ground truth
             behind it, this is corroboration, not a veto).

        NOT a replacement for classify_nes_binding_mode() -- that method
        remains the source of truth used inside refine_nes_candidates() to
        pick which of several ALREADY-RUN variants to trust. This method is
        for the different, upstream question: which ONE conformation should
        even be run, before paying for any MD at all.

        Returns:
            {
              'recommended_starting_conformation': 'native' | 'idealized_helix' | 'extended',
              'confidence': 'low' | 'medium',
              'rationale': str,
              'binding_mode_classification': <full classify_nes_binding_mode() dict>,
              'propensity_prediction': <full _predict_likely_conformation() dict>,
            }
        """
        binding_mode = self.classify_nes_binding_mode(sequence)
        propensity = self._predict_likely_conformation(sequence)
        cls = binding_mode['binding_mode_class']

        if cls == 'no_register_match':
            # Neither geometry-specific method has a mechanistic basis to
            # prefer here -- fall back to the cheap propensity check alone,
            # explicitly flagged low confidence either way.
            recommended = propensity['predicted_conformation']
            confidence = 'low'
            rationale = (
                f"No Phi-anchor register match (see classify_nes_binding_mode) -- the mechanistic, "
                f"crystal-calibrated rule has no basis to prefer a conformation here. Falling back to "
                f"sequence-propensity only: predicted={propensity['predicted_conformation']} "
                f"(mean_helix_pa={propensity['mean_helix_pa']:.3f}, mean_beta_pb={propensity['mean_beta_pb']:.3f}, "
                f"proline_count={propensity['proline_count']}). Treat this candidate's result as "
                f"low-confidence overall regardless of which conformation is used."
            )
        elif cls == 'extended_atypical':
            recommended = 'extended'
            confidence = binding_mode['confidence']
            rationale = binding_mode['rationale']
        elif cls == 'partial_register_match':
            recommended = binding_mode['recommended_primary_method']
            confidence = 'low'
            rationale = binding_mode['rationale']
        elif cls == 'likely_helical':
            recommended = 'idealized_helix'
            confidence = binding_mode['confidence']
            rationale = binding_mode['rationale']
            if propensity['predicted_conformation'] == 'extended' and propensity['confidence'] > 0.3:
                confidence = 'low'
                rationale += (
                    f" CAUTION: sequence-propensity cross-check disagrees (predicts 'extended', "
                    f"confidence={propensity['confidence']:.2f}, proline_count={propensity['proline_count']}) "
                    f"-- downgrading confidence from the register-only rule's default; consider testing "
                    f"'extended' as well for this candidate rather than trusting idealized_helix alone."
                )
        else:
            recommended = 'idealized_helix'
            confidence = 'low'
            rationale = f"Unrecognized binding_mode_class '{cls}' -- defaulting conservatively."

        print(f"    Pre-MD recommendation: {recommended} (confidence={confidence}, "
              f"binding_mode={cls}, propensity={propensity['predicted_conformation']})")

        return {
            'recommended_starting_conformation': recommended,
            'confidence': confidence,
            'rationale': rationale,
            'binding_mode_classification': binding_mode,
            'propensity_prediction': propensity,
        }

    def _place_peptide_via_subpocket_registration(self, peptide_modeller, sequence: str,
                                                   scramble_registration: bool = False,
                                                   apply_transform: bool = True) -> bool:
        """
        Anchor-aware initial docking pose: superpose the NES peptide's own
        matched Phi-anchor C-alpha positions onto their assigned CRM1
        sub-pocket centroids (self.crm1_subpockets, from
        _identify_nes_subpockets) via a single rigid-body Kabsch fit, rather
        than the single-approach-vector placement in
        _place_peptide_near_groove().

        This is a real, if still simplified, docking step -- it uses the
        actual predicted per-anchor register-to-pocket correspondence
        (which pocket engages which residue), not just "somewhere near the
        groove, some orientation". It's still rigid-body (no conformational
        search over registers or side-chain rotamers) and still hands off
        to the existing flexible minimization/equilibration to refine the
        pose from there, same as before.

        scramble_registration: negative-control / specificity-test mode
        (added, see the eval-script docstring for the full
        rationale). When True, each matched anchor is superposed onto a
        DIFFERENT sub-pocket than the one its own register assigns it to
        -- a fixed cyclic shift over SUBPOCKET_LABELS (P0->P1->P2->P3->
        P4->P0) is applied to every matched label before the Kabsch fit,
        so e.g. the anchor that would normally target P2 instead targets
        P3. This exists to separate two things that a single "how well
        did this end up packed into the groove" score can't distinguish:
        a real NES motif specifically fitting ITS OWN correct anchor
        register, versus a peptide that's just a rigid, generically
        hydrophobic/sticky helix that packs reasonably well into ANY
        plausible 4-pocket layout regardless of whether the correspondence
        is the biologically real one. Comparing a candidate's correct-
        registration run against this scrambled one is the actual
        specificity test -- a real binder is expected to show a much
        bigger correct-vs-scrambled gap than a stable coiled-coil/
        leucine-zipper decoy. Everything else about placement/tracking
        below is identical between the two modes; the returned
        anchor_seq_positions dict is keyed by the (possibly scrambled)
        TARGET label, so _run_crm1_docking's downstream per-frame anchor-
        to-pocket tracking automatically follows whichever pocket this
        call actually targeted -- no separate tracking code path needed
        for the scrambled case.

        apply_transform:  when False, computes and RETURNS
        everything above exactly as normal (matched anchors,
        self._last_subpocket_registration, the diagnostic fit_rmsd_nm)
        but does NOT actually move peptide_modeller's atoms -- for a
        caller that already has the peptide in a known-correct pose (e.g.
        real crystallographic coordinates already in the same frame as
        self.crm1_structure) and wants the anchor-to-pocket TRACKING
        machinery downstream in _run_crm1_docking's production loop
        without overwriting that pose with our own geometrically-inferred
        Kabsch fit. Only meaningful when scramble_registration=False --
        the scrambled control's entire point is to intentionally displace
        the peptide, so it always applies its transform regardless of
        this flag.

        Returns False (caller should fall back to
        _place_peptide_near_groove()) whenever:
          - self.crm1_subpockets isn't usefully populated (no CRM1
            structure, or sub-pocket partitioning failed/was skipped)
          - fewer than 3 Phi anchors were matched to a sub-pocket (Kabsch
            needs >= 3 non-collinear points for a well-determined rotation;
            2 points leave rotation about their own connecting axis
            undefined) -- checked both before AND after the scramble shift,
            since shifting can't change the count (it's a bijection over
            SUBPOCKET_LABELS) but is re-checked defensively anyway
          - the peptide's topology doesn't have as many residues as
            `sequence` implies (shouldn't happen, but never trusted
            silently)

        On success, mutates peptide_modeller.positions in place (applying
        the SAME rigid transform to every atom, not just the matched
        anchors) and records the match quality in
        self._last_subpocket_registration for the caller to fold into
        md_metrics. Does not itself run clash-avoidance -- callers should
        follow this with _push_out_until_clash_free().
        """
        self._last_subpocket_registration = None

        if not self.crm1_subpockets or len(self.crm1_subpockets) < 3:
            return False

        residues = list(peptide_modeller.topology.residues())
        if len(residues) < len(sequence):
            print(f"    Warning: Peptide topology has fewer residues ({len(residues)}) than "
                  f"the candidate sequence ({len(sequence)}) -- skipping sub-pocket "
                  f"registration, falling back to generic groove placement")
            return False

        old_positions = peptide_modeller.positions.value_in_unit(unit.nanometer)

        # Try both the plus-direction (N-term at wide/P0 end --
        # PKI/Rev/Snurportin1/Paxillin-style) and minus-direction (C-term at
        # wide end -- hRio2NES/CPEB4NES-style, confirmed real via 5DIF this
        # session) anchor-to-pocket assignments, and let the actual groove
        # geometry decide which one is correct for THIS candidate, rather
        # than assuming plus-direction the way this pipeline did before
        # 5DIF's crystal_sanity_check.py run showed that assumption
        # backwards for a real minus-direction binder.
        #
        # trust_input_positions = (not apply_transform): when
        # apply_transform is False, the CALLER is telling us old_positions
        # are already a known-correct pose (e.g. a real crystal structure
        # already in the same frame as self.crm1_subpockets) that we must
        # NOT move -- in that case a free-rotation Kabsch fit is the wrong
        # decision tool (it can make a wrong correspondence look almost as
        # good as the right one when anchors are sparse/near-collinear;
        # confirmed on the real 5DIF pod run and reproduced locally with
        # synthetic noisy data -- see _best_orientation_matched's
        # docstring), so we use raw un-rotated anchor-to-target distance
        # instead. When apply_transform is True (a not-yet-placed
        # candidate, no real frame to compare against yet), the Kabsch-fit
        # approach is the only one that makes sense and stays the default.
        matched, orientation, register = self._best_orientation_matched(
            sequence, residues, old_positions, trust_input_positions=not apply_transform)
        if matched is None:
            return False

        if scramble_registration:
            def _shifted_label(label):
                i = SUBPOCKET_LABELS.index(label)
                return SUBPOCKET_LABELS[(i + 1) % len(SUBPOCKET_LABELS)]
            matched = [(_shifted_label(label), idx) for label, idx in matched]
            # Re-filter for target availability -- defensive only; the shift
            # is a bijection over the fixed SUBPOCKET_LABELS list so this
            # shouldn't actually drop anything self.crm1_subpockets already
            # has all 5 labels for, but never trusted silently.
            matched = [(label, idx) for label, idx in matched if label in self.crm1_subpockets]
            if len(matched) < 3:
                return False
            # The scrambled control's whole point is to intentionally displace
            # the peptide onto the wrong targets -- never silently skip that
            # just because a caller passed apply_transform=False for some
            # OTHER (unscrambled) variant's sake.
            apply_transform = True

        mobile_points, target_points, used_labels = [], [], []
        anchor_seq_positions = {}  # label -> 0-indexed position within `sequence`/residues
        for label, seq_idx in matched:
            residue = residues[seq_idx]
            ca_atom = next((a for a in residue.atoms() if a.name == 'CA'), None)
            if ca_atom is None:
                continue
            pos = old_positions[ca_atom.index]
            mobile_points.append([pos[0], pos[1], pos[2]])
            target_points.append(self.crm1_subpockets[label]['centroid_nm'])
            used_labels.append(label)
            anchor_seq_positions[label] = seq_idx

        if len(mobile_points) < 3:
            return False

        mobile_arr = np.array(mobile_points)
        target_arr = np.array(target_points)
        R, t = self._kabsch_transform(mobile_arr, target_arr)

        if apply_transform:
            all_coords = np.array(old_positions)
            new_coords = (R @ all_coords.T).T + t
            peptide_modeller.positions = [Vec3(*row) for row in new_coords] * unit.nanometer

        fit_rmsd_nm = self._rmsd((R @ mobile_arr.T).T + t, target_arr)
        scramble_note = " [SCRAMBLED -- specificity control]" if scramble_registration else ""
        transform_note = "" if apply_transform else " [NOT APPLIED -- keeping caller's own coordinates, " \
                                                      "e.g. a real crystal structure's already-correct pose; " \
                                                      "fit RMSD reported below is diagnostic only]"
        print(f"    Anchor-registered placement{scramble_note}{transform_note}: {len(used_labels)}/5 Phi "
              f"anchors matched to sub-pockets ({', '.join(used_labels)}), "
              f"anchor-fit RMSD {fit_rmsd_nm * 10:.2f} Å")

        self._last_subpocket_registration = {
            'matched_pockets': used_labels,
            'n_anchors_matched': len(used_labels),
            'anchor_fit_rmsd_nm': float(fit_rmsd_nm),
            # label -> 0-indexed position within the candidate's sequence,
            # so _run_crm1_docking's production loop can look up each
            # anchor's CA atom (via the same index into nes_ca_indices) and
            # track its REAL simulated distance to that pocket every frame,
            # not just at this initial placement. Keyed by the (possibly
            # scrambled) TARGET label -- see scramble_registration's
            # docstring above for why that's what makes the downstream
            # tracking follow the scrambled targets automatically.
            'anchor_seq_positions': anchor_seq_positions,
            'scramble_registration': scramble_registration,
            # 'reversed' means the minus-direction (hRio2NES/CPEB4NES-style)
            # anchor-to-pocket assignment fit the real groove geometry
            # better than the plus-direction default -- see
            # _best_orientation_matched. Purely diagnostic/reporting; the
            # tracking machinery above already works off `matched`/
            # `used_labels`, whichever orientation those came from.
            'orientation': orientation,
        }
        return True

    @staticmethod
    def _build_peptide_backbone_restraint_force(nes_modeller, nes_peptide_atom_count):
        """
        Builds (but does not add to any System) a CustomExternalForce that
        harmonically restrains the peptide's own backbone atoms (N/CA/C/O,
        identified by atom NAME rather than any capping-aware residue logic
        -- capping-group atoms don't carry these names, so they're
        naturally excluded without needing to know which residues are real
        caps) to their CURRENT positions in nes_modeller (i.e. wherever
        STEP 2/3 placement -- native extraction, idealized-helix
        construction, Kabsch anchor registration, or a real crystal pose --
        already put them).

        Uses a GLOBAL parameter ('k_bb_restraint') for the spring constant,
        not a per-particle one, specifically so the caller can add this
        force to the System BEFORE constructing the Simulation/Context
        (same pattern the existing CRM1-restraint force already uses a few
        lines below this call site) and later switch it off via
        context.setParameter('k_bb_restraint', 0.0) without needing to
        rebuild or reinitialize anything. See the SIDECHAIN_RELAX_* module
        constants' comment block for the full rationale (crystal_sanity_check.py finding) and _run_crm1_docking for where
        this gets used: minimize + a short restrained MD segment right
        after the Simulation is created, letting side chains settle
        against the real local CRM1 environment while the backbone stays
        pinned near its placed position, THEN released before the
        existing minimization/equilibration/production protocol proceeds
        exactly as it did before this existed.
        """
        force = CustomExternalForce("k_bb_restraint*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k_bb_restraint", SIDECHAIN_RELAX_RESTRAINT_K_KJ_MOL_NM2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        backbone_names = ('N', 'CA', 'C', 'O')
        n_restrained = 0
        for atom in nes_modeller.topology.atoms():
            if atom.index < nes_peptide_atom_count and atom.name in backbone_names:
                pos = nes_modeller.positions[atom.index]
                force.addParticle(atom.index, [pos.x, pos.y, pos.z])
                n_restrained += 1
        # Own force group, separate from the forcefield's own forces (group
        # 0) and the existing CRM1 restraint (group 1) -- same reasoning as
        # that force's own force-group comment: keeps this restraint's
        # contribution separable from the "real" forcefield energy for any
        # future energy-decomposition analysis, even though nothing
        # currently queries group 2 specifically.
        force.setForceGroup(2)
        return force, n_restrained

    def _push_out_until_clash_free(self, peptide_modeller, label="placement"):
        """
        Safety net for _place_peptide_via_subpocket_registration(): checks
        the peptide (already positioned by the Kabsch anchor fit) for
        severe atom-atom overlap against self.crm1_structure, and if found,
        translates the WHOLE peptide along a LOCAL escape direction --
        derived from whichever peptide/CRM1 atom pairs are actually
        clashing THIS iteration, recomputed every step -- until clash-free
        or max_attempts runs out. Translation-only, so it never disturbs
        the anchor-registered ORIENTATION the Kabsch fit just computed
        (unlike calling _place_peptide_near_groove() itself here, which
        would silently throw that orientation away and re-derive a
        completely different, generic one).

        History : this used to push along a single GLOBAL
        direction (CRM1's overall centroid -> peptide centroid) instead.
        Empirically (evaluate_anchor_occupancy_signal.py, real MD, 16 real
        labeled examples) that global direction was landing EVERY single
        candidate -- positives and negatives alike, indistinguishable by
        label -- at 1.5-2.5 nm average anchor-to-pocket distance over the
        production trajectory, despite the initial Kabsch fit itself being
        good (RMSD 0.11-0.33 nm, i.e. anchors DO land on their assigned
        pockets before this runs). That uniform, label-independent
        overshoot is exactly what you'd expect from a push direction
        that's geometrically unrelated to the local clash: CRM1's own
        centroid isn't near the groove surface a registered peptide sits
        against, so escaping "toward it, reversed" requires traveling much
        farther than the clash itself demands, and does so in the same
        detached-from-the-pocket direction for every candidate regardless
        of how good the underlying registration is. A direction computed
        from the actual clashing atom pairs should need a much smaller net
        displacement to clear, since it's answering "what's actually
        overlapping" rather than "which way is roughly outward overall".
        """
        if self.crm1_structure is None:
            return

        crm1_coords = np.array([[p.x, p.y, p.z] for p in
                                 self.crm1_structure.positions.value_in_unit(unit.nanometer)])
        crm1_centroid = (self.crm1_full_centroid if self.crm1_full_centroid is not None
                          else crm1_coords.mean(axis=0))

        min_safe_distance = 0.40   # nm -- same as _place_peptide_near_groove
        severe_clash_radius = 0.20  # nm
        max_severe_clashes = 0
        # Smaller step + more attempts than the old global-direction version
        # (which used 0.4 nm x 14 = 5.6 nm max): a LOCAL escape direction
        # should need much less net travel to clear a real clash, so finer
        # steps let this settle near the actual clash-free boundary instead
        # of jumping straight past the 0.5-1.5 nm anchor-occupancy window
        # in one or two 0.4 nm steps the way the old version did.
        max_attempts = 40
        push_step_nm = 0.1

        attempt = 0
        for attempt in range(max_attempts):
            coords = np.array(peptide_modeller.positions.value_in_unit(unit.nanometer))
            diffs = coords[:, None, :] - crm1_coords[None, :, :]  # (n_pep, n_crm1, 3)
            dists = np.linalg.norm(diffs, axis=2)                  # (n_pep, n_crm1)
            min_dist = dists.min()
            clash_mask = dists < min_safe_distance
            n_severe = int(np.sum(dists < severe_clash_radius))

            if min_dist >= min_safe_distance and n_severe <= max_severe_clashes:
                if attempt > 0:
                    print(f"    {label}: reached clash-free after {attempt} "
                          f"push-out step(s), closest approach {min_dist:.3f} nm")
                return

            # Local escape direction: mean of (peptide_atom - crm1_atom)
            # over every currently-violating pair, i.e. "away from whatever
            # I'm actually overlapping with right now" -- recomputed fresh
            # each iteration since which atoms are clashing changes as the
            # peptide moves.
            pep_idx, crm1_idx = np.where(clash_mask)
            away = None
            if len(pep_idx) > 0:
                escape_vectors = diffs[pep_idx, crm1_idx]
                summed = escape_vectors.sum(axis=0)
                norm = np.linalg.norm(summed)
                if norm > 1e-6:
                    away = summed / norm
            if away is None:
                # No clashing pairs found (shouldn't normally happen given
                # the return above already checks this), or the local
                # vectors canceled out exactly (rare, symmetric overlap) --
                # fall back to the old global direction so this can't get
                # stuck in place.
                centroid = coords.mean(axis=0)
                fallback = centroid - crm1_centroid
                fnorm = np.linalg.norm(fallback)
                away = fallback / fnorm if fnorm > 1e-6 else np.array([1.0, 0.0, 0.0])

            coords = coords + away * push_step_nm
            peptide_modeller.positions = [Vec3(*row) for row in coords] * unit.nanometer

        coords = np.array(peptide_modeller.positions.value_in_unit(unit.nanometer))
        diffs = coords[:, None, :] - crm1_coords[None, :, :]
        min_dist = np.linalg.norm(diffs, axis=2).min()
        if min_dist < min_safe_distance:
            print(f"    Warning: {label}: could not reach a clash-free placement after "
                  f"{max_attempts} push-out attempts (closest approach still "
                  f"{min_dist:.3f} nm) - minimization may still start with some overlap.")

    def _analyze_helix_propensity(self, sequence: str, full_protein_seq: str = None,
                                  seq_start_idx: int = None) -> dict:
        """
        Analyze α-helix formation propensity for NES sequence

        This is CRITICAL - NES must form amphipathic helix to bind CRM1

        Args:
            sequence: The core NES sequence to analyze
            full_protein_seq: Optional full protein sequence for extended helix analysis
            seq_start_idx: Start position of sequence in full protein (0-indexed)

        Strategy:
            - Helix propensity: Analyze extended window (±2-3 residues) since helices
              typically extend beyond the core NES motif
            - Amphipathic scoring: Focus only on the core NES sequence since this is
              the functional binding region
        """
        # Chou-Fasman helix propensity values
        helix_propensity = {
            'A': 1.42, 'L': 1.21, 'M': 1.45, 'E': 1.51, 'K': 1.16,
            'F': 1.13, 'Q': 1.11, 'I': 1.08, 'W': 1.08, 'V': 1.06,
            'D': 1.01, 'H': 1.00, 'R': 0.98, 'T': 0.83, 'S': 0.77,
            'C': 0.70, 'Y': 0.69, 'N': 0.67, 'P': 0.57, 'G': 0.57
        }

        if len(sequence) < 4:
            return {'helix_propensity': 0.0, 'amphipathic_score': 0.0,
                   'helix_breakers': 0, 'combined_score': 0.0,
                   'extended_helix_propensity': 0.0, 'helix_window': sequence}

        # HELIX PROPENSITY: Use extended window (±2-3 residues)
        extended_seq = sequence
        if full_protein_seq and seq_start_idx is not None:
            left_extend = 3
            right_extend = 3

            start = max(0, seq_start_idx - left_extend)
            end = min(len(full_protein_seq), seq_start_idx + len(sequence) + right_extend)

            extended_seq = full_protein_seq[start:end]
            print(f"    Extended helix window: {extended_seq} (±3 residues)")

        # Calculate helix propensity on extended sequence
        extended_scores = [helix_propensity.get(aa, 0.8) for aa in extended_seq]
        extended_avg_propensity = np.mean(extended_scores)

        # Also calculate for core sequence
        core_scores = [helix_propensity.get(aa, 0.8) for aa in sequence]
        core_avg_propensity = np.mean(core_scores)

        # AMPHIPATHIC SCORING: Only use core NES sequence
        hydrophobic = set('AILMFVPW')
        polar = set('STNQCYH')
        charged = set('DEKR')

        hydro_face_count = 0
        polar_face_count = 0

        for i in range(len(sequence)):
            if sequence[i] in hydrophobic:
                neighbors = []
                for offset in [3, 4, -3, -4]:
                    j = i + offset
                    if 0 <= j < len(sequence):
                        neighbors.append(sequence[j])
                hydro_neighbors = sum(1 for n in neighbors if n in hydrophobic)
                if hydro_neighbors >= 2:
                    hydro_face_count += 1
            if sequence[i] in polar or sequence[i] in charged:
                polar_face_count += 1

        hydro_ratio = hydro_face_count / len(sequence)
        polar_ratio = polar_face_count / len(sequence)

        if 0.3 <= hydro_ratio <= 0.6 and 0.3 <= polar_ratio <= 0.6:
            amphipathic_score = min(hydro_ratio, polar_ratio) * 2.0
        else:
            amphipathic_score = max(hydro_ratio, polar_ratio) * 0.5

        # Helix breakers
        core_breakers = sequence.count('P') + sequence.count('G')
        extended_breakers = extended_seq.count('P') + extended_seq.count('G')
        breaker_penalty = min(1.0, extended_breakers / len(extended_seq))

        # Use extended helix propensity for combined score
        combined = (extended_avg_propensity * 0.4 + amphipathic_score * 0.6) * (1.0 - breaker_penalty * 0.3)

        return {
            'helix_propensity': float(core_avg_propensity),
            'extended_helix_propensity': float(extended_avg_propensity),
            'amphipathic_score': float(amphipathic_score),
            'hydrophobic_face_ratio': float(hydro_ratio),
            'helix_breakers': core_breakers,
            'extended_helix_breakers': extended_breakers,
            'helix_window': extended_seq,
            'combined_score': float(combined)
        }

    @staticmethod
    def _predict_likely_conformation(sequence: str) -> dict:
        """
        Cheap, sequence-only prior on whether this candidate looks more
        helix-like or extended/beta-like, computed BEFORE any MD is run.

        WHY THIS EXISTS : _run_crm1_docking previously always
        started every candidate from starting_conformation='idealized_helix'
        by default, regardless of sequence composition -- reasonable as a
        single default given helical is the most common NES binding mode,
        but not a real prediction. Now that _build_extended_pdb() offers a
        second starting hypothesis, there should be a cheap way to decide
        which one is actually more likely for a GIVEN sequence before
        committing GPU time to either, rather than treating idealized_helix
        as the default for everything.

        Compares mean Chou-Fasman helix propensity (Pa, the same table
        _analyze_helix_propensity uses) against mean beta-sheet/extended
        propensity (Pb, BETA_SHEET_PROPENSITY) across the core sequence.
        This is a real, decades-old sequence-based heuristic (Chou & Fasman
        1974), not a novel classifier -- treat the output as a prior/
        starting hypothesis to inform which starting_conformation to try
        first, not a substitute for the MD result itself. Proline content
        is reported separately since it's a strong, well-established
        structural signal on its own (prolines cannot adopt helical phi and
        are common in PPII/extended NES motifs like Rev-type).
        """
        helix_propensity = {
            'A': 1.42, 'L': 1.21, 'M': 1.45, 'E': 1.51, 'K': 1.16,
            'F': 1.13, 'Q': 1.11, 'I': 1.08, 'W': 1.08, 'V': 1.06,
            'D': 1.01, 'H': 1.00, 'R': 0.98, 'T': 0.83, 'S': 0.77,
            'C': 0.70, 'Y': 0.69, 'N': 0.67, 'P': 0.57, 'G': 0.57
        }
        if not sequence:
            return {'predicted_conformation': 'idealized_helix', 'confidence': 0.0,
                    'mean_helix_pa': 0.0, 'mean_beta_pb': 0.0, 'proline_count': 0}

        mean_pa = float(np.mean([helix_propensity.get(aa, 0.8) for aa in sequence]))
        mean_pb = float(np.mean([BETA_SHEET_PROPENSITY.get(aa, 0.8) for aa in sequence]))
        proline_count = sequence.count('P')

        # Margin-based confidence rather than a bare label -- a Pa/Pb
        # difference of 0.02 is noise, a difference of 0.5+ is a real
        # signal. Normalized against the larger of the two scores so it's
        # roughly comparable across sequences of different overall
        # propensity magnitude.
        diff = mean_pa - mean_pb
        confidence = float(min(1.0, abs(diff) / max(mean_pa, mean_pb, 1e-6)))
        predicted = 'idealized_helix' if diff >= 0 else 'extended'

        # Two or more prolines in a short (~10 residue) NES core is a
        # strong independent extended/PPII signal (proline's ring-
        # constrained phi ~-65 is compatible with PPII but not with a
        # continuous alpha helix) -- override a weak/marginal helix call if
        # so, since Pa/Pb alone under-weight proline's structural effect.
        if proline_count >= 2 and predicted == 'idealized_helix' and confidence < 0.5:
            predicted = 'extended'
            confidence = max(confidence, 0.5)

        return {
            'predicted_conformation': predicted,
            'confidence': confidence,
            'mean_helix_pa': mean_pa,
            'mean_beta_pb': mean_pb,
            'proline_count': proline_count,
        }

    # -----------------------------------------------------------------
    # Advanced trajectory analysis helpers (RMSD, Rg, H-bonds, contact
    # map, MM-GBSA-style binding energy, per-residue decomposition,
    # DSSP/SASA). Added to give _run_crm1_docking's md_metrics output a
    # fuller, thesis-ready set of standard MD analysis figures on top of
    # the original energy/distance/contact/RMSF traces. Every method here
    # is a pure function of positions/parameters already available from
    # the docking run -- none of them re-run or extend the simulation
    # itself, they only analyze frames it already sampled.
    # -----------------------------------------------------------------

    @staticmethod
    def _kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Optimal rigid-body superposition of `mobile` (N,3) onto `target`
        (N,3) via the Kabsch algorithm (SVD of the cross-covariance
        matrix). Returns `mobile`'s coordinates rotated+translated into
        `target`'s frame, ready for a translation/rotation-free RMSD.
        """
        target_mean = target.mean(axis=0)
        mobile_c = mobile - mobile.mean(axis=0)
        target_c = target - target_mean
        H = mobile_c.T @ target_c
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        # Translate back using the TARGET's actual centroid (target_mean),
        # not target_c's own mean -- target_c is already centered, so its
        # mean is ~0 and using it here would silently drop the alignment's
        # translation component (caught by unit-testing this against a
        # known rotation+translation, see verification notes).
        return (R @ mobile_c.T).T + target_mean

    @staticmethod
    def _kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same fit as _kabsch_align, but returns the rotation matrix and
        translation vector separately, (R, t), such that `R @ point + t`
        maps a point from `mobile`'s frame into `target`'s frame -- so the
        SAME rigid-body transform fit from a small point set (e.g. a
        peptide's 3-5 Phi-anchor CAs) can be applied to a LARGER point set
        (e.g. every atom of that same peptide) than the one used to compute
        the fit. Used by _place_peptide_via_subpocket_registration().
        """
        target_mean = target.mean(axis=0)
        mobile_mean = mobile.mean(axis=0)
        mobile_c = mobile - mobile_mean
        target_c = target - target_mean
        H = mobile_c.T @ target_c
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = target_mean - R @ mobile_mean
        return R, t

    @staticmethod
    def _nerf_place_atom(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                          bond_length: float, bond_angle_deg: float,
                          dihedral_deg: float) -> np.ndarray:
        """
        NERF (Natural Extension Reference Frame) placement: given the
        previous three atoms in a chain (a, b, c, in bonding order) and the
        desired c-d bond length, b-c-d bond angle, and a-b-c-d dihedral,
        return the Cartesian position of the next atom d. Standard
        internal-coordinates-to-Cartesian construction (Parsons et al.
        2005) -- the same general approach purpose-built peptide-geometry
        tools (e.g. PeptideBuilder) use; implemented directly here (pure
        numpy, no extra dependency) rather than adding a new package to
        install on top of everything else this pipeline already needs.

        Verified against known ideal alpha-helix invariants
        before use in _build_idealized_helix_pdb(): bond lengths reproduce
        exactly, CA(i)-CA(i+4) landed at 6.42 Angstrom (textbook value for
        an alpha helix is ~6.0-6.5 Angstrom) and axial rise/residue at
        ~1.57 Angstrom (ideal is 1.5), consistently across a 15-residue
        test chain.
        """
        bond_angle = np.radians(bond_angle_deg)
        dihedral = np.radians(dihedral_deg)
        bc = c - b
        bc_hat = bc / np.linalg.norm(bc)
        ab = b - a
        n = np.cross(ab, bc_hat)
        n_hat = n / np.linalg.norm(n)
        m_hat = np.cross(n_hat, bc_hat)
        d_local = np.array([
            -bond_length * np.cos(bond_angle),
            bond_length * np.sin(bond_angle) * np.cos(dihedral),
            bond_length * np.sin(bond_angle) * np.sin(dihedral),
        ])
        M = np.column_stack([bc_hat, m_hat, n_hat])
        return c + M @ d_local

    def _build_idealized_helix_pdb(self, sequence: str, chain_id: str = 'A') -> str:
        """
        Build a backbone-only (N, CA, C, O) PDB structure for `sequence`
        along an idealized alpha helix (phi=-57 deg, psi=-47 deg,
        omega=180 deg, standard bond lengths/angles -- see the IDEAL_*
        module constants), via NERF placement (_nerf_place_atom).

        WHY THIS EXISTS : _run_crm1_docking's peptide starting
        conformation used to come exclusively from whatever AlphaFold
        predicted for this stretch of residues in the ISOLATED, unbound
        full-length protein (see the 'native' branch in _run_crm1_docking's
        peptide-preparation step). For a real NES, which is very often
        located in an intrinsically disordered region, that prediction is
        frequently NOT already helical -- AlphaFold single-sequence
        prediction handles disorder-to-order-on-binding poorly, and NES
        engagement of CRM1's groove is a textbook example of exactly that
        (coupled folding and binding). Asking a single ~5 ns MD trajectory
        to BOTH fold a plausibly-disordered stretch into a helix AND
        correctly dock it into the groove, starting from an arbitrary
        non-helical conformation, is a much harder sampling problem than
        starting already close to the literature-documented dominant
        bound-state conformation (Dong et al. 2009 Nature 458:1136-1141;
        Guttler et al. 2010 Nat Struct Mol Biol 17:1367-1376) and letting
        MD refine/relax from there.

        This is offered as an ALTERNATIVE starting hypothesis, not a
        replacement -- the literature is clear that helical engagement is
        the most common but not the only NES-CRM1 binding mode, so
        _run_crm1_docking runs BOTH this and the original AlphaFont-native
        starting conformation and reports both, rather than assuming the
        helical hypothesis is always correct.

        Side chains are NOT built here -- this returns backbone-only
        (N/CA/C/O) atoms; the caller feeds this through PDBFixer
        (findMissingAtoms/addMissingAtoms/addMissingHydrogens), which
        fills in side-chain atoms from standard template geometry relative
        to the given backbone, the same way it already does for missing
        atoms elsewhere in this file. Verified in a standalone test that
        PDBFixer's side-chain fill preserves the
        idealized backbone geometry (CA(i)-CA(i+4) still 6.42-6.43
        Angstrom after filling) and produces a normal, minimizable
        starting energy (~3,700 kJ/mol for a 14-residue test peptide --
        nothing like the multi-million kJ/mol seen for genuinely
        clashing/overlapping structures elsewhere in this module).
        """
        n_res = len(sequence)
        if n_res < 2:
            raise ValueError("Need at least 2 residues to build a helix backbone")

        # Bootstrap the first residue's N/CA/C manually (NERF needs 3
        # preceding atoms, which don't exist yet for residue 0) -- N at the
        # origin, CA along +x at the ideal N-CA bond length, C placed in
        # the xy-plane using the ideal N-CA-C bond angle.
        n0 = np.array([0.0, 0.0, 0.0])
        ca0 = np.array([IDEAL_BOND_LENGTH_N_CA, 0.0, 0.0])
        bootstrap_angle = np.radians(180.0 - IDEAL_BOND_ANGLE_N_CA_C)
        c0 = ca0 + IDEAL_BOND_LENGTH_CA_C * np.array(
            [np.cos(bootstrap_angle), np.sin(bootstrap_angle), 0.0])

        n_pos, ca_pos, c_pos, o_pos = [n0], [ca0], [c0], []
        for i in range(n_res):
            ni, cai, ci = n_pos[i], ca_pos[i], c_pos[i]
            # Carbonyl O: placed opposite the next residue's N across the
            # planar amide bond (dihedral offset by 180 deg from psi).
            o_pos.append(self._nerf_place_atom(
                ni, cai, ci, IDEAL_BOND_LENGTH_C_O, IDEAL_BOND_ANGLE_CA_C_O,
                IDEAL_HELIX_PSI_DEG - 180.0))
            if i < n_res - 1:
                n_next = self._nerf_place_atom(
                    ni, cai, ci, IDEAL_BOND_LENGTH_C_N, IDEAL_BOND_ANGLE_CA_C_N,
                    IDEAL_HELIX_PSI_DEG)
                ca_next = self._nerf_place_atom(
                    cai, ci, n_next, IDEAL_BOND_LENGTH_N_CA, IDEAL_BOND_ANGLE_C_N_CA,
                    IDEAL_HELIX_OMEGA_DEG)
                c_next = self._nerf_place_atom(
                    ci, n_next, ca_next, IDEAL_BOND_LENGTH_CA_C, IDEAL_BOND_ANGLE_N_CA_C,
                    IDEAL_HELIX_PHI_DEG)
                n_pos.append(n_next)
                ca_pos.append(ca_next)
                c_pos.append(c_next)

        lines = []
        atom_idx = 1
        for i, aa in enumerate(sequence):
            resname = ONE_LETTER_TO_THREE_LETTER_AA.get(aa.upper(), 'ALA')
            resnum = i + 1
            for name, pos in (('N', n_pos[i]), ('CA', ca_pos[i]),
                               ('C', c_pos[i]), ('O', o_pos[i])):
                lines.append(
                    f"ATOM  {atom_idx:5d}  {name:<3s} {resname:>3s} {chain_id}{resnum:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {name[0]:>2s}")
                atom_idx += 1
        lines.append("TER")
        lines.append("END")
        return "\n".join(lines) + "\n"

    def _build_extended_pdb(self, sequence: str, chain_id: str = 'A') -> str:
        """
        Same NERF-based backbone construction as _build_idealized_helix_pdb,
        but along an idealized EXTENDED/polyproline-II-like backbone
        (IDEAL_EXTENDED_PHI_DEG/PSI_DEG) instead of an alpha helix. Second
        starting hypothesis for starting_conformation='extended', for
        candidates where _predict_likely_conformation (or a known real
        extended-mode NES like Rev-type) suggests helical isn't the right
        starting guess. Backbone-only (N/CA/C/O), same as the helix builder
        -- side chains filled in later by PDBFixer from this backbone, same
        downstream path as idealized_helix.
        """
        n_res = len(sequence)
        if n_res < 2:
            raise ValueError("Need at least 2 residues to build an extended backbone")

        n0 = np.array([0.0, 0.0, 0.0])
        ca0 = np.array([IDEAL_BOND_LENGTH_N_CA, 0.0, 0.0])
        bootstrap_angle = np.radians(180.0 - IDEAL_BOND_ANGLE_N_CA_C)
        c0 = ca0 + IDEAL_BOND_LENGTH_CA_C * np.array(
            [np.cos(bootstrap_angle), np.sin(bootstrap_angle), 0.0])

        n_pos, ca_pos, c_pos, o_pos = [n0], [ca0], [c0], []
        for i in range(n_res):
            ni, cai, ci = n_pos[i], ca_pos[i], c_pos[i]
            o_pos.append(self._nerf_place_atom(
                ni, cai, ci, IDEAL_BOND_LENGTH_C_O, IDEAL_BOND_ANGLE_CA_C_O,
                IDEAL_EXTENDED_PSI_DEG - 180.0))
            if i < n_res - 1:
                n_next = self._nerf_place_atom(
                    ni, cai, ci, IDEAL_BOND_LENGTH_C_N, IDEAL_BOND_ANGLE_CA_C_N,
                    IDEAL_EXTENDED_PSI_DEG)
                ca_next = self._nerf_place_atom(
                    cai, ci, n_next, IDEAL_BOND_LENGTH_N_CA, IDEAL_BOND_ANGLE_C_N_CA,
                    IDEAL_EXTENDED_OMEGA_DEG)
                c_next = self._nerf_place_atom(
                    ci, n_next, ca_next, IDEAL_BOND_LENGTH_CA_C, IDEAL_BOND_ANGLE_N_CA_C,
                    IDEAL_EXTENDED_PHI_DEG)
                n_pos.append(n_next)
                ca_pos.append(ca_next)
                c_pos.append(c_next)

        lines = []
        atom_idx = 1
        for i, aa in enumerate(sequence):
            resname = ONE_LETTER_TO_THREE_LETTER_AA.get(aa.upper(), 'ALA')
            resnum = i + 1
            for name, pos in (('N', n_pos[i]), ('CA', ca_pos[i]),
                               ('C', c_pos[i]), ('O', o_pos[i])):
                lines.append(
                    f"ATOM  {atom_idx:5d}  {name:<3s} {resname:>3s} {chain_id}{resnum:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {name[0]:>2s}")
                atom_idx += 1
        lines.append("TER")
        lines.append("END")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    def _compute_rmsd_trace(self, ca_positions_over_time: List[np.ndarray]) -> List[float]:
        """
        CA-atom RMSD (nm) of each sampled production frame vs. the FIRST
        sampled frame (i.e. the structure right after equilibration),
        after optimal Kabsch superposition. A standard, simplified
        (CA-only rather than all-atom) peptide conformational-stability
        metric: a near-constant, low RMSD indicates the peptide has
        converged to a stable bound pose; a rising or noisy trace
        indicates continued drift/unfolding.
        """
        if not ca_positions_over_time:
            return []
        ref = ca_positions_over_time[0]
        return [self._rmsd(self._kabsch_align(frame, ref), ref) for frame in ca_positions_over_time]

    @staticmethod
    def _compute_radius_of_gyration_trace(positions_over_time: List[np.ndarray],
                                           masses: np.ndarray) -> List[float]:
        """
        Mass-weighted radius of gyration (nm) of the NES peptide per
        frame -- a standard compactness metric. Used alongside RMSD: a
        folded amphipathic helix should settle into a lower, more stable
        Rg than an extended or unfolding conformation.
        """
        if not positions_over_time:
            return []
        masses = np.asarray(masses)
        total_mass = masses.sum()
        trace = []
        for frame in positions_over_time:
            com = (frame * masses[:, None]).sum(axis=0) / total_mass
            sq_dev = np.sum((frame - com) ** 2, axis=1)
            trace.append(float(np.sqrt((masses * sq_dev).sum() / total_mass)))
        return trace

    @staticmethod
    def _identify_backbone_no_atoms(topology, n_peptide_atoms: int) -> Dict[int, Dict[str, int]]:
        """
        Map each NES peptide residue index -> its backbone N and O atom
        indices (within the combined peptide+CRM1 topology), used by
        _count_backbone_hbonds for the classic alpha-helix i->i+4 (and
        3-10 helix i->i+3) backbone hydrogen bond count. Only includes
        residues where both atoms were actually found.
        """
        backbone = {}
        for atom in topology.atoms():
            if atom.index >= n_peptide_atoms:
                break
            if atom.name in ('N', 'O'):
                backbone.setdefault(atom.residue.index, {})[atom.name] = atom.index
        return {k: v for k, v in backbone.items() if 'N' in v and 'O' in v}

    @staticmethod
    def _count_backbone_hbonds(positions_nm: np.ndarray, backbone_map: Dict[int, Dict[str, int]],
                                min_offset: int = 3, max_offset: int = 4,
                                distance_cutoff_nm: float = 0.35) -> int:
        """
        Count backbone O(i)...N(j) heavy-atom hydrogen bonds for j-i in
        {3, 4} (3-10 helix and alpha-helix backbone H-bond patterns
        respectively) in a single frame, using the standard <3.5 A
        O...N distance proxy (angle-independent -- a full angle-aware
        criterion is applied separately via DSSP for the headline
        "percent helix" metric; this is a cheap per-frame companion
        count that doesn't require mdtraj).
        """
        count = 0
        for i, atoms_i in backbone_map.items():
            for offset in range(min_offset, max_offset + 1):
                j = i + offset
                if j in backbone_map:
                    d = np.linalg.norm(positions_nm[atoms_i['O']] - positions_nm[backbone_map[j]['N']])
                    if d < distance_cutoff_nm:
                        count += 1
        return count

    @staticmethod
    def _build_residue_atom_map(topology, atom_index_filter) -> Dict[int, List[int]]:
        """Group atom indices by residue index, restricted to atoms for
        which atom_index_filter(atom) is True. Shared helper for building
        the NES-residue and groove-residue atom-index groups used by the
        contact matrix and the interaction-energy decomposition."""
        out: Dict[int, List[int]] = {}
        for atom in topology.atoms():
            if atom_index_filter(atom):
                out.setdefault(atom.residue.index, []).append(atom.index)
        return out

    @staticmethod
    def _count_contacts_matrix(positions, nes_residue_atoms: Dict[int, List[int]],
                                groove_residue_atoms: Dict[int, List[int]],
                                cutoff: float = 0.6) -> np.ndarray:
        """
        Per-(NES residue, groove residue) contact boolean matrix for one
        frame: entry [i, j] = True if any atom of NES residue i is within
        `cutoff` nm of any atom of groove residue j. The caller
        accumulates these across sampled frames and divides by the sample
        count to get a contact FREQUENCY map (0-1 per residue pair) --
        which specific anchor residues persistently contact which
        specific groove residues, rather than just the aggregate contact
        count already tracked elsewhere.
        """
        nes_ids = sorted(nes_residue_atoms.keys())
        groove_ids = sorted(groove_residue_atoms.keys())
        mat = np.zeros((len(nes_ids), len(groove_ids)), dtype=bool)

        nes_pos = {r: np.array([positions[a].value_in_unit(unit.nanometer) for a in atoms])
                   for r, atoms in nes_residue_atoms.items()}
        groove_pos = {r: np.array([positions[a].value_in_unit(unit.nanometer) for a in atoms])
                      for r, atoms in groove_residue_atoms.items()}

        for ni, nres in enumerate(nes_ids):
            for gi, gres in enumerate(groove_ids):
                diffs = nes_pos[nres][:, None, :] - groove_pos[gres][None, :, :]
                if np.linalg.norm(diffs, axis=2).min() < cutoff:
                    mat[ni, gi] = True
        return mat, nes_ids, groove_ids

    def _build_isolated_energy_context(self, topology, positions):
        """
        Build a standalone OpenMM Simulation/Context for a topology in
        isolation, using the SAME forcefield/implicit-solvent settings as
        the main docking system (amber14-all + GBn2), for MM-GBSA-style
        single-point energy evaluations. Built ONCE per candidate and
        reused across every sampled frame (just setPositions + getState)
        rather than rebuilding per frame -- the expensive part is
        createSystem, not the energy evaluation itself.
        """
        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        integrator = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
        platform, properties = _select_fast_platform()
        if platform is not None:
            sim = Simulation(topology, system, integrator, platform, properties)
        else:
            sim = Simulation(topology, system, integrator)
        sim.context.setPositions(positions)
        return sim

    @staticmethod
    def _extract_nonbonded_params(system):
        """
        Pull per-particle (charge, sigma, epsilon) from the System's
        NonbondedForce once, for the analytic pairwise interaction-energy
        decomposition in _residue_interaction_energy (avoids re-querying
        OpenMM's NonbondedForce per frame per residue).
        """
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                n = force.getNumParticles()
                charges = np.zeros(n)
                sigmas = np.zeros(n)
                epsilons = np.zeros(n)
                for i in range(n):
                    q, sigma, eps = force.getParticleParameters(i)
                    charges[i] = q.value_in_unit(unit.elementary_charge)
                    sigmas[i] = sigma.value_in_unit(unit.nanometer)
                    epsilons[i] = eps.value_in_unit(unit.kilojoule_per_mole)
                return charges, sigmas, epsilons
        return None, None, None

    @staticmethod
    def _residue_interaction_energy(positions_nm: np.ndarray, charges: np.ndarray,
                                     sigmas: np.ndarray, epsilons: np.ndarray,
                                     nes_residue_atoms: Dict[int, List[int]],
                                     groove_residue_atoms: Dict[int, List[int]],
                                     cutoff_nm: float = 1.2) -> Dict[int, float]:
        """
        Analytic pairwise Coulomb + Lennard-Jones interaction energy
        (vacuum electrostatics -- this does NOT include the implicit
        GBn2 polarization/desolvation term, which is a per-atom Born-radii
        effect that doesn't decompose additively into residue-pair
        contributions the same way) between each NES residue's atoms and
        the full groove-residue atom set, for one frame. Returns
        {nes_residue_index: energy_kj_mol}: a relative per-residue RANKING
        of which anchor residues drive the hydrophobic-groove contact, not
        an absolute binding free energy (see the system-level
        mmgbsa_binding_energy_trace_kj_mol in md_metrics for the estimate
        that DOES include implicit solvent). Uses OpenMM's own Coulomb
        constant and standard Lorentz-Berthelot combining rules
        (sigma_ij = mean, epsilon_ij = geometric mean), matching how
        NonbondedForce itself combines these parameters internally.
        """
        groove_atoms = sorted({a for atoms in groove_residue_atoms.values() for a in atoms})
        if not groove_atoms:
            return {}
        g_pos = positions_nm[groove_atoms]
        g_q, g_sig, g_eps = charges[groove_atoms], sigmas[groove_atoms], epsilons[groove_atoms]

        out = {}
        for res_idx, atom_idxs in nes_residue_atoms.items():
            p_pos = positions_nm[atom_idxs]
            p_q, p_sig, p_eps = charges[atom_idxs], sigmas[atom_idxs], epsilons[atom_idxs]

            diffs = p_pos[:, None, :] - g_pos[None, :, :]
            r = np.linalg.norm(diffs, axis=2)
            mask = (r < cutoff_nm) & (r > 1e-6)
            if not mask.any():
                out[res_idx] = 0.0
                continue

            qq = p_q[:, None] * g_q[None, :]
            sig_ij = 0.5 * (p_sig[:, None] + g_sig[None, :])
            eps_ij = np.sqrt(np.clip(p_eps[:, None] * g_eps[None, :], 0, None))

            r_safe = np.where(mask, r, np.inf)
            coulomb = ONE_4PI_EPS0 * qq / r_safe
            sr6 = np.where(mask, (sig_ij / r_safe) ** 6, 0.0)
            lj = 4 * eps_ij * (sr6 ** 2 - sr6)

            out[res_idx] = float(np.where(mask, coulomb, 0.0).sum() + lj.sum())
        return out

    @staticmethod
    def _compute_dssp_helix_fraction(peptide_pdb_text: str, frames_nm: List[np.ndarray]) -> Optional[List[float]]:
        """
        Build an independent mdtraj Trajectory for the free NES peptide
        (using a topology snapshot taken BEFORE it was merged with CRM1,
        so it's unaffected by the combined topology/atom-index scheme
        used everywhere else in this file) from a list of per-frame
        peptide coordinates, then run simplified DSSP per frame and
        return the fraction of residues assigned 'H' (helix) at each
        frame. Returns None if mdtraj isn't installed.
        """
        if not MDTRAJ_AVAILABLE or not frames_nm:
            return None
        try:
            pdb = PDBFile(StringIO(peptide_pdb_text))
            top = md.Topology.from_openmm(pdb.topology)
            xyz = np.stack(frames_nm, axis=0)
            traj = md.Trajectory(xyz=xyz, topology=top)
            codes = md.compute_dssp(traj, simplified=True)  # (n_frames, n_residues) of 'H'/'E'/'C'/'NA'
            return (codes == 'H').mean(axis=1).tolist()
        except Exception as e:
            print(f"    Warning: DSSP calculation failed (non-fatal): {e}")
            return None

    @staticmethod
    def _compute_ramachandran_trace(peptide_pdb_text: str, frames_nm: List[np.ndarray]) -> Optional[Dict]:
        """
        Per-frame, per-residue backbone phi/psi dihedral angles (degrees)
        for the free NES peptide -- same independent-topology-snapshot
        pattern as _compute_dssp_helix_fraction (built from a pre-merge
        peptide topology + a list of per-frame coordinates), computed on
        the same strided frame subset so it lines up 1:1 with the DSSP
        trace and other advanced-analysis time series.

        WHY THIS IS SEPARATE FROM DSSP: DSSP's 'coil' bucket doesn't
        distinguish genuinely disordered/random backbone sampling from a
        well-defined non-helical basin (e.g. polyproline-II/extended --
        relevant for Rev-type NES, which the literature describes as an
        extended, proline-containing conformation, not just "not
        helical"). Keeping the continuous phi/psi angles per frame lets
        that distinction actually be checked instead of assumed.

        Returns a dict with parallel-length residue-label lists and
        (n_frames x n_angles) angle arrays for phi and psi separately --
        phi excludes the first residue (needs the preceding C), psi
        excludes the last (needs the following N), same convention
        mdtraj itself uses. Returns None if mdtraj isn't installed or
        the peptide has too few residues to have any phi/psi angles.
        """
        if not MDTRAJ_AVAILABLE or not frames_nm:
            return None
        try:
            pdb = PDBFile(StringIO(peptide_pdb_text))
            top = md.Topology.from_openmm(pdb.topology)
            xyz = np.stack(frames_nm, axis=0)
            traj = md.Trajectory(xyz=xyz, topology=top)

            phi_indices, phi_angles = md.compute_phi(traj)  # (n_frames, n_phi) radians
            psi_indices, psi_angles = md.compute_psi(traj)  # (n_frames, n_psi) radians

            def _label(res):
                return f"{res.name}{res.resSeq}"

            # phi quartet = (C[i-1], N[i], CA[i], C[i]) -- last atom is residue i
            phi_residue_labels = [_label(top.atom(int(row[-1])).residue) for row in phi_indices]
            # psi quartet = (N[i], CA[i], C[i], N[i+1]) -- first atom is residue i
            psi_residue_labels = [_label(top.atom(int(row[0])).residue) for row in psi_indices]

            return {
                'phi_residue_labels': phi_residue_labels,
                'psi_residue_labels': psi_residue_labels,
                'phi_deg': np.degrees(phi_angles).tolist(),
                'psi_deg': np.degrees(psi_angles).tolist(),
            }
        except Exception as e:
            print(f"    Warning: Ramachandran (phi/psi) calculation failed (non-fatal): {e}")
            return None

    @staticmethod
    def _compute_residue_sasa_nm2(pdb_text: str, frames_nm: List[np.ndarray]) -> Optional[Dict[int, List[float]]]:
        """
        Same Shrake-Rupley SASA calculation as _compute_sasa_nm2, but
        returned PER RESIDUE (residue.index -> per-frame SASA list, nm^2)
        instead of summed over the whole structure. Used by the anchor-
        burial check (see 'anchor_burial_nm2' in _run_crm1_docking's
        metrics) to confirm a Phi-anchor residue that's CA-registered to a
        sub-pocket has actually lost solvent-accessible surface area (i.e.
        is genuinely buried in that pocket) rather than just sitting near
        it while still solvent-exposed -- backbone/CA registration alone
        doesn't guarantee the side chain is actually inserted. Returns None
        if mdtraj isn't installed.
        """
        if not MDTRAJ_AVAILABLE or not frames_nm:
            return None
        try:
            pdb = PDBFile(StringIO(pdb_text))
            top = md.Topology.from_openmm(pdb.topology)
            xyz = np.stack(frames_nm, axis=0)
            traj = md.Trajectory(xyz=xyz, topology=top)
            atom_sasa = md.shrake_rupley(traj, mode='atom')  # (n_frames, n_atoms) nm^2
            n_residues = top.n_residues
            residue_of_atom = np.array([a.residue.index for a in top.atoms])
            per_residue = {}
            for res_idx in range(n_residues):
                mask = residue_of_atom == res_idx
                per_residue[res_idx] = atom_sasa[:, mask].sum(axis=1).tolist()
            return per_residue
        except Exception as e:
            print(f"    Warning: Per-residue SASA calculation failed (non-fatal): {e}")
            return None

    @staticmethod
    def _compute_sasa_nm2(pdb_text: str, frames_nm: List[np.ndarray]) -> Optional[List[float]]:
        """
        Total Shrake-Rupley SASA (nm^2) per frame for a structure given as
        PDB text + a list of per-frame coordinate arrays matching that
        topology's atom order/count. Used for both the free-peptide SASA
        trace and the (expensive, representative-frames-only) complex/
        CRM1-alone SASA used to compute buried surface area. Returns None
        if mdtraj isn't installed.
        """
        if not MDTRAJ_AVAILABLE or not frames_nm:
            return None
        try:
            pdb = PDBFile(StringIO(pdb_text))
            top = md.Topology.from_openmm(pdb.topology)
            xyz = np.stack(frames_nm, axis=0)
            traj = md.Trajectory(xyz=xyz, topology=top)
            return md.shrake_rupley(traj).sum(axis=1).tolist()
        except Exception as e:
            print(f"    Warning: SASA calculation failed (non-fatal): {e}")
            return None

    def refine_nes_candidates(self, pdb_content: str, candidates: List[Dict],
                            duration_ns: float = 10.0,
                            test_both_conformations: bool = False,
                            test_specificity_control: bool = False) -> List[Dict]:
        """
        Refine NES candidates using MD simulation (CRM1 docking)

        NOTE: This module only performs full CRM1 docking MD. Fast, sequence-only
        structural triage (helix propensity / amphipathic moment / Phi-spacing)
        lives separately in quick_helix_analysis.py, used by the "Quick Analysis"
        UI path (POST /api/quick-analysis) - it is not part of this MD pipeline.

        Args:
            pdb_content: PDB file content as string
            candidates: List of NES candidate dictionaries
            duration_ns: Simulation duration in nanoseconds
            test_both_conformations: if True , runs
                _run_crm1_docking for BOTH starting conformations -- native
                (AlphaFold's isolated, unbound-state prediction) and
                idealized_helix (see _build_idealized_helix_pdb) -- instead
                of just native. Motivation: NES motifs are frequently in
                intrinsically disordered regions, where AlphaFold's
                isolated-state prediction is often not already helical,
                even though helical engagement is the literature-documented
                DOMINANT (not universal) NES-CRM1 binding mode -- a single
                ~5-10 ns trajectory starting from a non-helical
                conformation may not have time to both fold AND correctly
                dock, understating real binders. Roughly DOUBLES the MD
                cost per candidate on its own.
            test_specificity_control: if True , for each
                starting conformation being tested, ALSO runs
                _run_crm1_docking with scramble_registration=True (see
                that method's docstring) -- a negative control that
                anchor-registers the peptide against a cyclically-WRONG
                sub-pocket assignment instead of its own correct one.
                Motivation: evaluation at n=43 real examples found every
                packing-quality metric (anchor_occupancy_score, raw
                contacts, distances) running BACKWARDS -- hard negatives
                (coiled-coil/leucine-zipper fragments, which are rigid,
                pre-folded, generically hydrophobic helices) packed
                TIGHTER than real NES motifs on short unbiased MD, plausibly
                because they pack well into ANY plausible pocket layout
                regardless of whether it's the correct one -- something a
                single "how tight is the final pose" score can't detect.
                Comparing a candidate's correct-registration result against
                its scrambled-registration result is the actual specificity
                test: a real binder should show a much bigger correct-vs-
                scrambled gap than a decoy that isn't really responding to
                which specific pocket it's aimed at. Also roughly DOUBLES
                the MD cost per starting conformation being tested --
                combined with test_both_conformations=True this means 4
                total _run_crm1_docking calls per candidate (4x base cost).
                Both flags default to False and are independently opt-in,
                so a caller only pays for what it explicitly asks for
                (e.g. app.py's normal single-run behavior is unaffected).

        Returns:
            Enhanced candidates with MD metrics. When either flag above is
            True, candidate['md_metrics']/['md_enhanced_score'] still hold
            a single primary result for backward compatibility. As of
            WHICH unscrambled variant is picked as that primary
            result is decided by classify_nes_binding_mode(sequence) (see
            its docstring) -- a sequence-only rule for whether this
            candidate looks like a canonical helical NES (trust
            idealized_helix) or an atypical/extended one (trust native) --
            rather than simply whichever scored highest on this run, per
            the collated-comparison report's recommendation. Falls back to
            the old max-binding_score pick only when the classifier itself
            is low-confidence (no Phi-register match at all -- a case
            where neither method is known to be reliable). Every candidate
            also gets candidate['nes_binding_mode_classification'] with the
            full classification + rationale, and the complete per-variant
            breakdown remains available in candidate['md_metrics_by_variant']
            (keyed e.g. 'native', 'native_scrambled', 'idealized_helix',
            'idealized_helix_scrambled') -- so the non-primary method's
            score is never hidden, only de-prioritized.
        """
        enhanced_candidates = []

        for idx, candidate in enumerate(candidates):
            try:
                print(f"\n  Processing candidate {idx + 1}/{len(candidates)}")
                print(f"    Sequence: {candidate['sequence']}")
                print(f"    Position: {candidate['start']}-{candidate['end']}")

                binding_mode = self.classify_nes_binding_mode(candidate['sequence'])
                candidate['nes_binding_mode_classification'] = binding_mode
                print(f"    Binding mode: {binding_mode['binding_mode_class']} "
                      f"(recommend={binding_mode['recommended_primary_method']}, "
                      f"confidence={binding_mode['confidence']})")

                if test_both_conformations or test_specificity_control:
                    conformations = ['native', 'idealized_helix'] if test_both_conformations else ['native']
                    # Also test 'extended' (the literal PPII-geometry
                    # starting structure, see _build_extended_pdb) when
                    # classify_nes_binding_mode flags a proline/atypical
                    # signature -- see recommend_starting_conformation()'s
                    # docstring for the full motivation. Additive, not a
                    # replacement, so this stays backward-compatible with
                    # anything reading exactly the native/idealized_helix
                    # tags (e.g. the reference-set eval scripts) -- it only
                    # adds a third variant for candidates flagged this way,
                    # at the cost of one extra trajectory for those candidates only.
                    # Key off recommended_primary_method itself
                    # rather than one specific binding_mode_class -- caught
                    # by real testing (P42566, 766-800) that a proline-rich
                    # candidate can land in EITHER 'extended_atypical' (full
                    # register match) or 'partial_register_match' (3-of-4)
                    # depending on register geometry, and both branches now
                    # recommend 'extended' for the same proline-driven
                    # reason. Checking the recommendation directly (instead
                    # of duplicating the classification-branch logic here)
                    # means this can't drift out of sync with
                    # classify_nes_binding_mode() again the way it just did.
                    if binding_mode.get('recommended_primary_method') == 'extended' and 'extended' not in conformations:
                        conformations = conformations + ['extended']
                        print(f"    Also testing 'extended' starting conformation "
                              f"(binding_mode={binding_mode.get('binding_mode_class')}, "
                              f"proline/atypical NES signature detected)")
                    scrambles = [False, True] if test_specificity_control else [False]
                    variants = [(conf, scr) for conf in conformations for scr in scrambles]
                    print(f"    Testing {len(variants)} variant(s): " +
                          ", ".join(f"{c}{'+scrambled' if s else ''}" for c, s in variants))

                    # _run_crm1_docking mutates and returns the SAME
                    # candidate dict (not a copy) each call -- capture the
                    # md_metrics/md_enhanced_score reference right after
                    # each call, before the NEXT call reassigns those keys
                    # to a brand-new dict/value. Reassignment doesn't touch
                    # the object a variable already points to, so this is
                    # safe without an explicit deep copy.
                    variant_results = {}
                    for conf, scr in variants:
                        tag = conf if not scr else f"{conf}_scrambled"
                        print(f"    -- {tag} --")
                        result = self._run_crm1_docking(
                            pdb_content, candidate, duration_ns,
                            starting_conformation=conf, scramble_registration=scr)
                        variant_results[tag] = {
                            'metrics': result.get('md_metrics', {}) or {},
                            'score': result.get('md_enhanced_score', 0.5),
                        }

                    # Primary/backward-compatible result. As of,
                    # prefer classify_nes_binding_mode's sequence-based
                    # recommendation (see refine_nes_candidates' docstring)
                    # over picking whichever variant happened to score
                    # highest -- only fall back to the old max-binding_score
                    # pick when that classifier is itself low-confidence
                    # (no Phi-register match) or its recommended method
                    # wasn't actually one of the conformations tested this
                    # call (e.g. test_both_conformations=False).
                    unscrambled_tags = [conf for conf, scr in variants if not scr]
                    recommended_method = binding_mode.get('recommended_primary_method')
                    if (binding_mode.get('confidence') != 'low'
                            and recommended_method in unscrambled_tags):
                        best_tag = recommended_method
                        print(f"    Primary variant chosen by sequence-type classification: "
                              f"{best_tag} ({binding_mode['binding_mode_class']})")
                    else:
                        best_tag = max(unscrambled_tags, key=lambda t: (
                            variant_results[t]['metrics'].get('binding_score', 0.0) or 0.0))
                        print(f"    Primary variant chosen by fallback (max binding_score): {best_tag} "
                              f"(classifier confidence={binding_mode.get('confidence')}, "
                              f"recommended={recommended_method})")

                    candidate['md_metrics'] = variant_results[best_tag]['metrics']
                    candidate['md_enhanced_score'] = variant_results[best_tag]['score']
                    candidate['md_best_starting_conformation'] = best_tag
                    candidate['md_primary_variant_selection_method'] = (
                        'sequence_type_classification'
                        if (binding_mode.get('confidence') != 'low' and best_tag == recommended_method)
                        else 'max_binding_score_fallback'
                    )
                    candidate['md_metrics_by_variant'] = {
                        tag: v['metrics'] for tag, v in variant_results.items()
                    }
                    # Kept for backward compatibility with existing callers/
                    # eval scripts that read md_metrics_by_conformation
                    # specifically (predates the scrambled-control variants).
                    if test_both_conformations:
                        candidate['md_metrics_by_conformation'] = {
                            'native': variant_results.get('native', {}).get('metrics'),
                            'idealized_helix': variant_results.get('idealized_helix', {}).get('metrics'),
                        }
                        # 'extended' is only present when
                        # classify_nes_binding_mode flagged this candidate as
                        # extended_atypical (see conformations list above) --
                        # add it when it exists rather than always including
                        # a None entry for the common case.
                        if 'extended' in variant_results:
                            candidate['md_metrics_by_conformation']['extended'] = \
                                variant_results.get('extended', {}).get('metrics')

                    if test_specificity_control:
                        for conf in conformations:
                            correct = variant_results.get(conf, {}).get('metrics', {}) or {}
                            scrambled = variant_results.get(f"{conf}_scrambled", {}).get('metrics', {}) or {}
                            c_occ = correct.get('anchor_occupancy_score')
                            s_occ = scrambled.get('anchor_occupancy_score')
                            if c_occ is not None and s_occ is not None:
                                print(f"    Specificity gap ({conf}): correct anchor_occupancy="
                                      f"{c_occ:.3f}  scrambled={s_occ:.3f}  gap={c_occ - s_occ:+.3f}")

                    summary_bits = ", ".join(
                        f"{tag}={v['metrics'].get('binding_score', 0.0):.3f}"
                        for tag, v in variant_results.items())
                    print(f"    Best variant: {best_tag}  ({summary_bits})")

                    enhanced_candidates.append(candidate)
                else:
                    enhanced = self._run_crm1_docking(pdb_content, candidate, duration_ns)
                    enhanced_candidates.append(enhanced)

            except Exception as e:
                print(f"    Warning: Error processing candidate: {e}")

                # Even if simulation fails, try to get helix metrics
                try:
                    sequence = candidate['sequence']
                    full_protein_seq = candidate.get('full_sequence', None)
                    start_idx = candidate['start'] - 1

                    helix_metrics = self._analyze_helix_propensity(
                        sequence,
                        full_protein_seq=full_protein_seq,
                        seq_start_idx=start_idx
                    )

                    # Return candidate with helix-based scoring
                    candidate['md_enhanced_score'] = (
                        candidate.get('combined_score', 0.5) * 0.6 +
                        helix_metrics['combined_score'] * 0.4
                    )
                    candidate['md_metrics'] = {
                        'helix_propensity': helix_metrics['helix_propensity'],
                        'extended_helix_propensity': helix_metrics.get('extended_helix_propensity', helix_metrics['helix_propensity']),
                        'amphipathic_score': helix_metrics['amphipathic_score'],
                        'helix_combined_score': helix_metrics['combined_score'],
                        'binding_score': helix_metrics['combined_score'],
                        'binding_category': 'helix_based_fallback',
                        'error': str(e),
                        'simulation_failed': True,
                        'note': 'MD failed - using helix analysis only'
                    }
                    print(f"    Helix fallback score: {helix_metrics['combined_score']:.3f}")
                except Exception as helix_error:
                    # Complete failure - use defaults
                    print(f"    Warning: Helix analysis also failed: {helix_error}")
                    candidate['md_enhanced_score'] = candidate.get('combined_score', 0.5)
                    candidate['md_metrics'] = {
                        'error': str(e),
                        'helix_error': str(helix_error),
                        'simulation_failed': True
                    }

                enhanced_candidates.append(candidate)

        return enhanced_candidates

    def _run_crm1_docking(self, pdb_content: str, candidate: Dict, duration_ns: float,
                           starting_conformation: str = 'native',
                           scramble_registration: bool = False,
                           crystal_pdb_text: Optional[str] = None,
                           relax_sidechains: bool = True,
                           save_final_peptide_pdb_path: Optional[str] = None,
                           save_final_complex_pdb_path: Optional[str] = None,
                           save_best_anchor_frame_pdb_path: Optional[str] = None,
                           save_final_state_path: Optional[str] = None,
                           resume_from_state_path: Optional[str] = None,
                           use_simulated_annealing: bool = True) -> Dict:
        """
        Run flexible CRM1 docking with helix formation analysis

        Analyzes:
        - α-helix formation propensity (with extended window)
        - Amphipathic helix formation (core sequence only)
        - Binding affinity to hydrophobic groove
        - Flexible conformational sampling

        Args:
            starting_conformation: 'native' (default) starts the peptide
                from whatever conformation AlphaFold predicted for this
                stretch in the ISOLATED, unbound full-length protein (the
                original behavior of this function). 'idealized_helix'
                starts it instead from a literature-informed idealized
                alpha helix (see _build_idealized_helix_pdb) -- the
                canonical, most common NES-CRM1 binding mode (Dong 2009 /
                Guttler 2010), offered as an alternative starting
                hypothesis for candidates whose real bound conformation
                may not resemble AlphaFold's isolated-state prediction
                ( ; see refine_nes_candidates, which runs BOTH
                and reports both rather than assuming one is correct).
                Only the peptide-preparation step (this function's STEP 2)
                differs between the two -- Kabsch anchor registration,
                clash resolution, minimization, equilibration, production
                MD, and all metrics computation are identical code paths
                either way. A third value, 'crystal', is also accepted
                 -- see crystal_pdb_text below.
            scramble_registration:  specificity-control mode --
                passed straight through to
                _place_peptide_via_subpocket_registration (see its
                docstring for the full rationale). When True, the peptide
                is anchor-registered against a cyclically-shifted, WRONG
                sub-pocket assignment instead of its own correct one, and
                every downstream metric (contacts, distances, occupancy)
                is computed exactly the same way but now describes how
                well the peptide packs into pockets it does NOT actually
                belong in. Comparing a candidate's normal run against its
                scrambled run is the actual test: a real NES motif should
                fit its own correct registration much better than a
                scrambled one, while a rigid, generically hydrophobic
                decoy (coiled-coil/leucine-zipper) is expected to pack
                comparably well either way, since it isn't really
                responding to which specific pocket it's aimed at.
            crystal_pdb_text: required when starting_conformation ==
                'crystal' -- raw PDB text for the peptide ONLY, already in
                the same coordinate frame as self.crm1_structure (e.g.
                extracted directly from the same source PDB used to build
                the CRM1 reference this NESMDRefiner was constructed with).
                Ignored for 'native'/'idealized_helix'.
            relax_sidechains:  whether to run the restrained-
                backbone/free-sidechain relaxation phase (see SIDECHAIN_
                RELAX_* module constants) before the main minimization/
                equilibration/production protocol. Defaults to True (the
                pipeline-wide improvement this was designed to be) --
                pass False to reproduce the OLD (pre- ) behavior
                exactly, e.g. for a controlled with-vs-without comparison
                rather than only having "before this existed" vs "after,"
                which would confound the relaxation phase itself with
                whatever else changed between those two points in time.
            save_final_peptide_pdb_path:  if given, writes the
                peptide-only atoms' FINAL positions (post-production, same
                coordinate frame as self.crm1_structure) to this path once
                the full minimize/equilibrate/produce protocol completes.
                Added specifically so a converged idealized_helix (or
                native) starting pose can be directly RMSD-compared against
                a real crystal structure's own coordinates -- see
                idealized_helix_vs_crystal_check.py. Only written on the
                full-success path (production actually ran); silently
                skipped on any fallback/failure path, since there's no
                converged MD state to save in that case.
            save_final_complex_pdb_path:  same trigger/timing
                as save_final_peptide_pdb_path, but writes the FULL final
                pose instead -- peptide AND CRM1 together, in their actual
                post-production positions (CRM1 is not held rigid during
                minimize/equilibrate/produce, so its final coordinates
                differ from the input structure too). Added so the bound
                complex can be opened directly in a molecular viewer
                (PyMOL/ChimeraX/etc.) to see how a candidate actually sits
                in the groove, rather than only being able to inspect
                numeric contact/distance metrics. Independent of
                save_final_peptide_pdb_path -- pass either, both, or
                neither.
            save_best_anchor_frame_pdb_path:  writes the
                complex (peptide + CRM1) pose from whichever ONE of the
                N_REPRESENTATIVE_FRAMES end-of-production samples has the
                MOST Phi-anchor residues crossing the WELL_BURIED_THRESHOLD_
                NM2 hydrophobic-burial bar (see the anchor_burial_nm2 block
                below) -- i.e. the most-engaged snapshot from the same
                already-converged window anchor_burial_fraction_well_buried
                itself is averaged over, not the literal last integrator
                step save_final_complex_pdb_path writes. Added because the
                final-frame pose can visually look like a peptide only
                loosely perched at the groove edge even when the run's own
                trusted metrics (anchor_occupancy_score, anchor burial)
                show real, sustained engagement -- this gives an honest
                "best moment from the window already being scored" snapshot
                instead, without applying any steering/restraint force
                (which would change what the run is evidence OF). Silently
                skipped if fewer than 1 anchor position was registered or
                the representative-frame SASA calculation failed (same
                failure conditions as anchor_burial_nm2 itself).
            save_final_state_path:  if given, saves the full
                OpenMM simulation state (positions + velocities + box
                vectors, via Simulation.saveState -- standard XML
                serialization) at the end of production. Pairs with
                resume_from_state_path below to let a caller screen several
                short independent replicates, pick whichever one is
                actually going well, and CONTINUE that exact trajectory
                further rather than starting a brand-new run from scratch
                (which would just re-roll a new random seed and could as
                easily land on a worse outcome as a better one -- see the
                rank-1 ACK1 case that motivated this).
            resume_from_state_path:  if given, skips peptide
                placement, side-chain relaxation, minimization, simulated
                annealing, AND equilibration entirely -- the System/
                Simulation are still built normally (same topology/forces,
                required for the saved state to load into a compatible
                Context), but positions/velocities are then overwritten via
                Simulation.loadState(resume_from_state_path) instead of the
                normal fresh-start pipeline, and execution jumps straight
                to the production loop. duration_ns in this mode means
                ADDITIONAL production time from the resumed state, not
                total time from scratch -- e.g. resuming a run saved at 2 ns
                with duration_ns=18 continues it to 20 ns cumulative.
                starting_conformation/scramble_registration/relax_sidechains/
                use_simulated_annealing are still required arguments (needed
                to build the same topology) but their VALUES are irrelevant
                to the resumed trajectory's actual positions, which come
                entirely from the loaded state.
            use_simulated_annealing:  whether to run a brief
                heat-then-cool ramp (300K -> ANNEALING_HIGH_TEMP_K -> 300K,
                see module constants) immediately before the normal 300K
                equilibration. WHY: the Rev-NES (3NBZ) positive control run
                this project showed a real NES known to bind in an EXTENDED
                conformation staying almost entirely alpha-helical (73% DSSP
                helix fraction) across a full 50 ns production run when
                started from starting_conformation='idealized_helix' --
                evidence the peptide can get kinetically trapped near its
                artificial starting pose rather than genuinely relaxing
                toward whichever conformation is actually favorable, within
                the time a single unbiased 300K trajectory has to work with.
                A short high-temperature window gives the peptide enough
                thermal energy to cross backbone dihedral barriers it
                otherwise wouldn't, before settling back to 300K for the
                normal equilibration/production protocol -- NOT a
                replacement for genuinely testing both starting_conformation
                hypotheses (see _predict_likely_conformation /
                starting_conformation='extended'), just a way to make the
                MD itself less dependent on which one you happened to start
                from. Defaults to True; pass False to reproduce the exact
                old (pre- ) equilibration protocol, e.g. for a
                controlled with-vs-without comparison.
        """
        print(f"    Running flexible CRM1 docking (starting conformation: {starting_conformation}"
              f"{', SCRAMBLED registration' if scramble_registration else ''})...")

        start_idx = candidate['start'] - 1
        end_idx = candidate['end']
        sequence = candidate['sequence']

        # Get full protein sequence if available
        full_protein_seq = candidate.get('full_sequence', None)

        # STEP 1: Analyze helix propensity with extended window
        helix_metrics = self._analyze_helix_propensity(
            sequence,
            full_protein_seq=full_protein_seq,
            seq_start_idx=start_idx
        )
        print(f"    Helix propensity: {helix_metrics['helix_propensity']:.2f}")
        print(f"    Extended helix propensity: {helix_metrics.get('extended_helix_propensity', helix_metrics['helix_propensity']):.2f}")
        print(f"    Amphipathic score: {helix_metrics['amphipathic_score']:.2f}")

        # STEP 1b: cheap sequence-only prior on helical vs extended binding
        # mode, BEFORE any MD is run -- see _predict_likely_conformation's
        # docstring. Purely informational at this point (does NOT override
        # the caller's chosen starting_conformation) -- logged/reported so
        # it's visible whether the starting_conformation actually used
        # agrees or disagrees with this sequence-based prior, e.g. running
        # idealized_helix on a sequence this predicts as 'extended' is worth
        # knowing about even if that's still the run you want.
        conformation_prediction = self._predict_likely_conformation(sequence)
        agreement = ('n/a' if starting_conformation not in ('idealized_helix', 'extended')
                     else ('AGREES' if conformation_prediction['predicted_conformation'] == starting_conformation
                           else 'DISAGREES'))
        print(f"    Sequence-based conformation prior: {conformation_prediction['predicted_conformation']} "
              f"(confidence={conformation_prediction['confidence']:.2f}, "
              f"Pa={conformation_prediction['mean_helix_pa']:.2f}, Pb={conformation_prediction['mean_beta_pb']:.2f}, "
              f"prolines={conformation_prediction['proline_count']}) -- {agreement} with starting_conformation="
              f"'{starting_conformation}'")

        # Create temporary PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        full_modeller = None  # only built/used for starting_conformation=='native'
        try:
            # STEP 2: Prepare NES peptide with proper terminal caps
            print(f"    Preparing NES peptide ({starting_conformation})...")

            if starting_conformation == 'idealized_helix':
                # Literature-informed alternative starting hypothesis: build
                # the peptide along an idealized alpha helix instead of
                # extracting whatever (often non-helical) conformation
                # AlphaFold predicted for this stretch in the isolated,
                # unbound protein. See _build_idealized_helix_pdb's
                # docstring for the full rationale.
                pdb_string_text = self._build_idealized_helix_pdb(sequence)
            elif starting_conformation == 'extended':
                # Second literature-informed starting hypothesis, alongside
                # idealized_helix -- see _build_extended_pdb's docstring.
                # Not every real NES binds helically (Rev-type/3NBZ binds
                # extended, proline-containing); this lets a candidate be
                # tested from an extended starting guess too, rather than
                # only ever testing whether it can hold/reach a helix.
                pdb_string_text = self._build_extended_pdb(sequence)
            elif starting_conformation == 'crystal':
                # Ground-truth sanity check -- caller supplies
                # the REAL experimentally-solved peptide coordinates
                # directly (crystal_pdb_text), already in the SAME
                # coordinate frame as self.crm1_structure (both extracted
                # from the same source PDB, e.g. 3NBY -- see
                # crm1_reference/CRM1_Ran_3NBY.pdb +
                # PKI_NES_peptide_3NBY_chainB_4-13.pdb). No native/
                # idealized-helix construction needed; this exists purely
                # to test whether the pipeline's own scoring even
                # recognizes a peptide known, by direct crystallographic
                # evidence, to be correctly bound -- see STEP 3's
                # placement branch below, which (unlike native/
                # idealized_helix) skips the Kabsch anchor-registration
                # re-placement for this conformation when not scrambled,
                # since re-placing a peptide that's already correctly
                # positioned would throw away the one thing this test is
                # trying to isolate.
                if crystal_pdb_text is None:
                    raise ValueError("starting_conformation='crystal' requires crystal_pdb_text")
                pdb_string_text = crystal_pdb_text
            else:
                # Load the full structure
                pdb = PDBFile(tmp_path)
                full_modeller = Modeller(pdb.topology, pdb.positions)

                # Extract NES region
                residues_to_delete = []
                for residue in full_modeller.topology.residues():
                    if residue.index < start_idx or residue.index >= end_idx:
                        residues_to_delete.append(residue)

                full_modeller.delete(residues_to_delete)

                import io
                _pdb_string_io = io.StringIO()
                PDBFile.writeFile(full_modeller.topology, full_modeller.positions, _pdb_string_io)
                pdb_string_text = _pdb_string_io.getvalue()

            # CRITICAL FIX: Create properly capped peptide
            try:
                from pdbfixer import PDBFixer
                import io

                # Use PDBFixer to add missing atoms and fix terminals
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as peptide_tmp:
                    peptide_tmp.write(pdb_string_text)
                    peptide_tmp_path = peptide_tmp.name

                fixer = PDBFixer(filename=peptide_tmp_path)
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(7.0)

                nes_modeller = Modeller(fixer.topology, fixer.positions)
                os.unlink(peptide_tmp_path)

                print(f"    NES peptide prepared with PDBFixer (terminals fixed)")

            except Exception as fixer_error:
                print(f"    Warning: PDBFixer approach failed: {fixer_error}")
                if full_modeller is None:
                    # idealized_helix path -- the manual-capping fallback
                    # below only knows how to rebuild from full_modeller
                    # (the extracted-from-AlphaFold structure), which was
                    # never built on this path. Nothing sensible to fall
                    # back to here -- re-raise and let the outer
                    # except-Exception below return a helix-only result,
                    # the same safety net the native path already falls
                    # back to if BOTH of its own attempts fail.
                    raise
                print(f"    Trying manual terminal capping...")

                # Manual approach: build proper PDB with TER cards
                try:
                    forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

                    residues = list(full_modeller.topology.residues())
                    if len(residues) == 0:
                        raise Exception("No residues in peptide")

                    # Build PDB with proper formatting
                    import io
                    pdb_with_caps = io.StringIO()

                    atom_idx = 1
                    for residue in full_modeller.topology.residues():
                        for atom in residue.atoms():
                            pos = full_modeller.positions[atom.index]
                            pdb_with_caps.write(
                                f"ATOM  {atom_idx:5d}  {atom.name:<4s}{residue.name:>3s} A{residue.index+1:4d}    "
                                f"{pos[0]._value*10:8.3f}{pos[1]._value*10:8.3f}{pos[2]._value*10:8.3f}"
                                f"  1.00  0.00           {atom.element.symbol:>2s}  \n"
                            )
                            atom_idx += 1

                    pdb_with_caps.write("TER\n")
                    pdb_with_caps.write("END\n")

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as cap_tmp:
                        cap_tmp.write(pdb_with_caps.getvalue())
                        cap_tmp_path = cap_tmp.name

                    # Reload and add hydrogens with forcefield
                    capped_pdb = PDBFile(cap_tmp_path)
                    capped_modeller = Modeller(capped_pdb.topology, capped_pdb.positions)
                    capped_modeller.addHydrogens(forcefield, pH=7.0)

                    nes_modeller = capped_modeller
                    os.unlink(cap_tmp_path)

                    print(f"    NES peptide prepared with manual terminal handling")

                except Exception as manual_error:
                    print(f"    Warning: Manual capping also failed: {manual_error}")
                    raise Exception(f"All peptide preparation methods failed: {manual_error}")

        except Exception as e:
            # Peptide preparation failed - return helix-only analysis
            print(f"    Warning: All preparation methods failed: {e}")
            print(f"    Returning helix-based analysis only")

            md_metrics = {
                # Helix formation metrics (ALWAYS available)
                'helix_propensity': helix_metrics['helix_propensity'],
                'extended_helix_propensity': helix_metrics.get('extended_helix_propensity', helix_metrics['helix_propensity']),
                'conformation_prediction': conformation_prediction,
                'amphipathic_score': helix_metrics['amphipathic_score'],
                'hydrophobic_face_ratio': helix_metrics['hydrophobic_face_ratio'],
                'helix_breakers': helix_metrics['helix_breakers'],
                'helix_combined_score': helix_metrics['combined_score'],

                # No MD results
                'binding_score': helix_metrics['combined_score'],
                'binding_category': 'helix_based_prediction',
                'binding_likelihood': f"Helix score: {helix_metrics['combined_score']:.2f}",
                'starting_conformation': starting_conformation,
                'scramble_registration': scramble_registration,
                'relax_sidechains': relax_sidechains,
                'note': 'Peptide preparation failed - helix analysis only',
                'error': str(e)
            }

            enhanced_score = (candidate.get('combined_score', 0.5) + md_metrics['binding_score']) / 2.0
            candidate['md_enhanced_score'] = enhanced_score
            candidate['md_metrics'] = md_metrics

            print(f"    Helix-based score: {md_metrics['binding_score']:.3f}")
            return candidate

        try:
            # STEP 3: If CRM1 available, run flexible docking
            if self.crm1_structure is not None:
                print("    Creating flexible NES-CRM1 complex...")

                # Place the peptide before merging -- without this, it keeps
                # whatever coordinates it had inside the original AlphaFold
                # model, which have no relationship to CRM1's coordinate
                # frame, and naively merging the two reliably causes severe
                # atomic overlap (multi-million kJ/mol starting energy).
                #
                # Prefer an anchor-aware pose: superpose this candidate's
                # own matched Phi-anchor register onto its assigned CRM1
                # sub-pockets (P0-P4) via Kabsch fit, instead of a single
                # generic approach vector -- a real, if still rigid-body,
                # docking step rather than "somewhere near the groove, some
                # orientation". Falls back to the original generic
                # placement whenever the candidate's sequence doesn't match
                # a clean Phi register, or sub-pocket data isn't available
                # (e.g. calibration check failed at load time) -- never
                # left unplaced.
                # 'crystal' + not scrambled: the peptide's incoming
                # coordinates ARE the ground truth (already correctly
                # positioned relative to self.crm1_structure) -- don't let
                # our own geometrically-inferred Kabsch fit move it, only
                # use the registration call to populate anchor tracking.
                # See _place_peptide_via_subpocket_registration's
                # apply_transform docstring for the full rationale.
                apply_transform = not (starting_conformation == 'crystal' and not scramble_registration)
                subpocket_registered = self._place_peptide_via_subpocket_registration(
                    nes_modeller, sequence, scramble_registration=scramble_registration,
                    apply_transform=apply_transform)
                if subpocket_registered:
                    if apply_transform:
                        self._push_out_until_clash_free(nes_modeller, label="anchor-registered placement")
                    else:
                        print("    Keeping supplied coordinates as-is (apply_transform=False) -- "
                              "skipping clash push-out too, since a real crystal structure's own pose "
                              "shouldn't need it")
                else:
                    print("    No usable Phi-anchor register / sub-pocket data -- "
                          "falling back to generic groove-approach placement")
                    if starting_conformation == 'crystal':
                        print("    Warning: 'crystal' conformation couldn't register anchors for tracking, "
                              "but its coordinates are still real -- NOT falling back to generic "
                              "placement, which would overwrite them")
                    else:
                        self._place_peptide_near_groove(nes_modeller)
                subpocket_registration = self._last_subpocket_registration if subpocket_registered else None

                # Capture the peptide's own atom count BEFORE merging in CRM1 -
                # this, not the atom count of the original full-length source
                # protein, is the correct boundary between "NES" and "CRM1"
                # atoms in the merged structure below. Using the wrong count
                # here would misapply (or entirely skip) the CRM1 restraint,
                # and corrupt every NES-vs-CRM1 contact/distance calculation
                # later in this function.
                nes_peptide_atom_count = len(list(nes_modeller.topology.atoms()))
                # Same reasoning for residues: after PDBFixer rebuilds the
                # capped peptide, its residues are freshly numbered from 0 -
                # they no longer correspond to start_idx/end_idx (the NES's
                # residue numbers in the *original* full-length protein).
                nes_peptide_residue_count = len(list(nes_modeller.topology.residues()))

                # Snapshot the standalone peptide topology+positions as PDB
                # text BEFORE merging with CRM1 below -- nes_modeller.add()
                # mutates nes_modeller.topology/positions in place, so this
                # reference would otherwise silently become the COMBINED
                # structure by the time any post-hoc free-peptide analysis
                # (DSSP / SASA) runs after the production loop.
                _peptide_snapshot_io = StringIO()
                PDBFile.writeFile(nes_modeller.topology, nes_modeller.positions, _peptide_snapshot_io)
                peptide_only_pdb_text = _peptide_snapshot_io.getvalue()

                # Combine NES with CRM1 (NES is now flexible, CRM1 will be restrained)
                nes_modeller.add(self.crm1_structure.topology, self.crm1_structure.positions)

                # Setup forcefield with implicit solvent. Use a nonbonded
                # cutoff rather than NoCutoff - this system is ~20,000 atoms
                # (CRM1+Ran+peptide), and NoCutoff means every atom pair
                # interacts with every other (~400 million pairs per force
                # evaluation), which is impractical on CPU. A cutoff drops
                # negligible long-range interactions and cuts cost by 1-2
                # orders of magnitude.
                forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
                system = forcefield.createSystem(
                    nes_modeller.topology,
                    nonbondedMethod=CutoffNonPeriodic,
                    nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer,
                    constraints=HBonds
                )

                # Add soft restraints to keep CRM1 stable while NES samples
                force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
                force.addPerParticleParameter("k")
                force.addPerParticleParameter("x0")
                force.addPerParticleParameter("y0")
                force.addPerParticleParameter("z0")

                # Restrain CRM1 backbone
                for atom in nes_modeller.topology.atoms():
                    if atom.name == 'CA' and atom.index >= nes_peptide_atom_count:  # CRM1 atoms
                        pos = nes_modeller.positions[atom.index]
                        force.addParticle(atom.index, [
                            1000.0,  # kJ/mol/nm^2
                            pos.x, pos.y, pos.z
                        ])

                # Own force group for the restraint, separate from the
                # forcefield's own forces (default group 0). This lets the
                # MM-GBSA-style binding energy estimate below query
                # getState(..., groups={0}) to get the complex's forcefield
                # energy WITHOUT the restraint's contribution, which the
                # isolated peptide-alone/CRM1-alone comparison systems
                # (built with the same forcefield but no restraint) don't
                # have either -- without this separation the comparison
                # would be apples-to-oranges.
                force.setForceGroup(1)
                system.addForce(force)

                # Peptide backbone restraint for the side-chain relaxation
                # phase below -- added to the System now, same
                # as the CRM1 restraint just above, so it exists in the
                # Context from the start; its global parameter starts ACTIVE
                # (see _build_peptide_backbone_restraint_force) and gets
                # switched off further down once relaxation is done. See the
                # SIDECHAIN_RELAX_* module constants for the full rationale.
                peptide_backbone_force, n_backbone_restrained = self._build_peptide_backbone_restraint_force(
                    nes_modeller, nes_peptide_atom_count)
                system.addForce(peptide_backbone_force)

                # Run simulation
                integrator = LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtosecond)
                platform, platform_properties = _select_fast_platform()
                if platform is not None:
                    simulation = Simulation(nes_modeller.topology, system, integrator, platform, platform_properties)
                else:
                    simulation = Simulation(nes_modeller.topology, system, integrator)
                # Defined here, outside the fresh-start-only
                # block below, because resume_from_state_path skips
                # minimization entirely (nothing to re-minimize -- the
                # loaded state already came from a converged, previously-
                # equilibrated trajectory) but 'minimization_energy_trace'
                # in the returned metrics dict further down is unconditional.
                # None correctly signals "not applicable" for a resumed run,
                # same convention as this file's other Optional metric
                # fields (e.g. anchor_burial_nm2 or None).
                minimization_trace = None

                if resume_from_state_path is None:
                    simulation.context.setPositions(nes_modeller.positions)

                    # Report starting energy before touching anything. A huge
                    # positive value here (millions of kJ/mol) is a strong sign of
                    # severe atomic clashes from how the NES peptide was merged
                    # with CRM1 - that makes minimization numerically pathological
                    # (very slow, or never converging) rather than just
                    # computationally slow.
                    if not relax_sidechains:
                        # Old (pre- behavior: release the backbone
                        # restraint immediately, before it does anything -- the
                        # force still exists in the System (added unconditionally
                        # above for simplicity), but at k_bb_restraint=0 it
                        # contributes exactly zero energy/force, equivalent to it
                        # never having been added. Everything below then runs
                        # exactly as it did before this phase existed.
                        simulation.context.setParameter('k_bb_restraint', 0.0)

                    start_state = simulation.context.getState(getEnergy=True)
                    start_energy = start_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    print(f"    Starting potential energy: {start_energy:,.1f} kJ/mol")
                    if start_energy > 1e6:
                        print("    Warning: Extremely high starting energy - likely severe atomic "
                              "clashes in the NES-CRM1 complex. Minimization may be very slow "
                              "or fail to converge.")

                    if relax_sidechains:
                        # Side-chain relaxation phase: backbone held
                        # near its placed position (native/idealized_helix/Kabsch-
                        # registered/real crystal, whichever this run used) while
                        # side chains -- generic PDBFixer rotamers for everything
                        # except 'crystal', which already has real ones -- get a
                        # real chance to settle against the actual local CRM1
                        # environment before the timed production run starts. See
                        # the SIDECHAIN_RELAX_* module constants' comment block
                        # for the full rationale (crystal_sanity_check.py
                        # finding). Applied uniformly regardless of starting_
                        # conformation/scramble_registration whenever it's on at
                        # all, so it's a pipeline-wide improvement rather than a
                        # special case that would itself become a new confound
                        # -- relax_sidechains itself is the intentional, caller-
                        # controlled axis for an explicit with-vs-without
                        # comparison (see this function's docstring).
                        print(f"    Side-chain relaxation ({n_backbone_restrained} backbone atoms restrained, "
                              f"{SIDECHAIN_RELAX_MINIMIZE_ITERATIONS} min iterations + "
                              f"{SIDECHAIN_RELAX_MD_STEPS * 2.0 / 1000:.1f} ps restrained MD)...")
                        simulation.minimizeEnergy(maxIterations=SIDECHAIN_RELAX_MINIMIZE_ITERATIONS)
                        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
                        simulation.step(SIDECHAIN_RELAX_MD_STEPS)
                        relax_state = simulation.context.getState(getEnergy=True)
                        relax_energy = relax_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        print(f"      After side-chain relaxation: {relax_energy:,.1f} kJ/mol "
                              f"(was {start_energy:,.1f} before)")
                        # Release the backbone restraint -- global parameter, so
                        # this doesn't require touching the System or
                        # reinitializing the Context, unlike the CRM1 restraint
                        # (which stays active for the rest of the run, unchanged
                        # from before this existed).
                        simulation.context.setParameter('k_bb_restraint', 0.0)

                        # The relaxation phase above already resolved whatever
                        # portion of the initial clash energy was fixable by
                        # side-chain repacking -- the iteration-budget decision
                        # and convergence trace below should reflect what the
                        # MAIN minimization is actually starting from, not the
                        # pre-relaxation number (which would make an already-
                        # easier problem look like it still needs the large
                        # iteration budget the pre-relaxation energy implied).
                        post_relax_state = simulation.context.getState(getEnergy=True)
                        start_energy = post_relax_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                    # Minimize in small batches instead of one opaque blocking call,
                    # so progress is visible - lets us tell "slow but working" apart
                    # from "stuck/diverging" instead of guessing, and stop early
                    # once it's actually converged instead of always running the
                    # full iteration budget.
                    #
                    # A pathologically clashed start (millions of kJ/mol) needs
                    # more room to relax than a mild one - scale the iteration
                    # budget with how bad the starting energy is, instead of a
                    # flat 500 that was frequently not enough to escape a severe
                    # clash before minimizeEnergy's internal line search stalls
                    # out and reports a fake "converged" (zero further change)
                    # while still catastrophically high.
                    MINIMIZATION_ENERGY_CEILING_KJ_MOL = 1.0e5
                    total_min_iterations = 3000 if start_energy > 1e6 else 500
                    print(f"    Energy minimization (budget: {total_min_iterations} iterations)...")
                    batch_size = 50
                    prev_energy = start_energy
                    stuck_high = False
                    # Recorded for the report: a graph of this settling toward a
                    # plateau is good evidence the starting structure wasn't
                    # pathologically clashed and minimization actually converged.
                    minimization_trace = [{'iteration': 0, 'energy_kj_mol': float(start_energy)}]
                    for batch_start in range(0, total_min_iterations, batch_size):
                        simulation.minimizeEnergy(maxIterations=batch_size)
                        state = simulation.context.getState(getEnergy=True)
                        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        iterations_done = min(batch_start + batch_size, total_min_iterations)
                        delta = energy - prev_energy
                        print(f"      [minimize] after {iterations_done}/{total_min_iterations} "
                              f"iterations: {energy:,.1f} kJ/mol (Δ {delta:,.1f})")
                        minimization_trace.append({'iteration': iterations_done, 'energy_kj_mol': float(energy)})

                        if not np.isfinite(energy):
                            print("    Warning: Energy is NaN/Inf - system is numerically unstable, "
                                  "aborting minimization early")
                            break

                        if batch_start > 0 and abs(delta) < 1.0:
                            if energy > MINIMIZATION_ENERGY_CEILING_KJ_MOL:
                                # This is NOT real convergence -- the minimizer's
                                # step size has collapsed (near-zero further
                                # change) while still stuck at a catastrophic
                                # energy, almost always because the initial
                                # placement left atoms too close to genuinely
                                # resolve via steepest-descent/L-BFGS alone.
                                # Stop burning iterations on something that isn't
                                # moving, but do NOT treat this as success.
                                print(f"    Warning: Minimization stalled (no further change) but still "
                                      f"{energy:,.1f} kJ/mol, above the "
                                      f"{MINIMIZATION_ENERGY_CEILING_KJ_MOL:,.0f} kJ/mol sanity "
                                      f"ceiling - this is a stuck/clashed structure, not a "
                                      f"converged one. Stopping minimization.")
                                stuck_high = True
                            else:
                                print("    Energy has converged (no further meaningful change) - "
                                      "stopping minimization early")
                            break

                        prev_energy = energy

                    final_state = simulation.context.getState(getEnergy=True)
                    final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    print(f"    Minimization finished: {final_energy:,.1f} kJ/mol "
                          f"(started at {start_energy:,.1f})")

                    # Fail fast, with a clear diagnostic, rather than silently
                    # running equilibration/production dynamics on top of a
                    # structure that never actually relaxed - that's what was
                    # producing "Particle coordinate is NaN" deep into
                    # equilibration, burning a full run's worth of time before
                    # failing with a much less informative error. The existing
                    # `except Exception as docking_error` handler around this
                    # whole method already turns this into the same graceful
                    # helix-based fallback as any other docking failure, just
                    # faster and with a message that actually points at the
                    # cause (placement/minimization) instead of a bare NaN.
                    if not np.isfinite(final_energy) or final_energy > MINIMIZATION_ENERGY_CEILING_KJ_MOL:
                        raise RuntimeError(
                            f"Minimization did not reach a physically reasonable structure "
                            f"(final energy {final_energy:,.1f} kJ/mol"
                            f"{', stalled/non-finite' if stuck_high or not np.isfinite(final_energy) else ''}, "
                            f"ceiling {MINIMIZATION_ENERGY_CEILING_KJ_MOL:,.0f} kJ/mol). This means the "
                            f"initial peptide placement near Cys528 had steric clashes minimization "
                            f"couldn't resolve -- see _place_peptide_near_groove's clearance/severe-clash "
                            f"log line above for the attempted placement's closest-approach distance."
                        )

                    # Simulated annealing (optional, see use_simulated_annealing's
                    # docstring for the Rev-NES motivating result) -- runs BEFORE
                    # the normal 300K equilibration below, so any conformational
                    # change it enables has the full equilibration + production
                    # protocol afterward to settle from, exactly like any other
                    # starting pose would.
                    if use_simulated_annealing:
                        print(f"    Simulated annealing (300K -> {ANNEALING_HIGH_TEMP_K:.0f}K -> 300K)...")
                        t_anneal_start = time.time()

                        heat_substep = ANNEALING_HEAT_STEPS // ANNEALING_RAMP_SUBSTEPS
                        for i in range(1, ANNEALING_RAMP_SUBSTEPS + 1):
                            temp_k = 300.0 + (ANNEALING_HIGH_TEMP_K - 300.0) * (i / ANNEALING_RAMP_SUBSTEPS)
                            integrator.setTemperature(temp_k * unit.kelvin)
                            simulation.step(heat_substep)

                        simulation.step(ANNEALING_HOLD_STEPS)

                        cool_substep = ANNEALING_COOL_STEPS // ANNEALING_RAMP_SUBSTEPS
                        for i in range(1, ANNEALING_RAMP_SUBSTEPS + 1):
                            temp_k = ANNEALING_HIGH_TEMP_K - (ANNEALING_HIGH_TEMP_K - 300.0) * (i / ANNEALING_RAMP_SUBSTEPS)
                            integrator.setTemperature(temp_k * unit.kelvin)
                            simulation.step(cool_substep)

                        # Belt-and-braces: floating point on the last ramp
                        # substep should already land almost exactly on 300K,
                        # but set it explicitly rather than trust that, since
                        # every step downstream (equilibration, production,
                        # every energy/score calculation) assumes 300K exactly.
                        integrator.setTemperature(300.0 * unit.kelvin)

                        anneal_elapsed = time.time() - t_anneal_start
                        total_anneal_ps = (ANNEALING_HEAT_STEPS + ANNEALING_HOLD_STEPS + ANNEALING_COOL_STEPS) * 0.002
                        print(f"    Annealing complete ({total_anneal_ps:.0f} ps, {anneal_elapsed:.0f}s elapsed)")

                    # Equilibrate -- stepped in 10 ps chunks with a progress
                    # print per chunk instead of one silent 50,000-step call, so
                    # a slow platform (or a genuinely large system) doesn't look
                    # indistinguishable from a hung process for however long
                    # this takes. Same total step count/physics either way,
                    # simulation.step() just gets called 10 times instead of 1.
                    EQUILIBRATION_STEPS = 50000  # 100 ps
                    EQUILIBRATION_CHUNK_STEPS = 5000  # 10 ps per progress print
                    print("    Equilibration (100 ps)...")
                    t_equil_start = time.time()
                    for equil_done in range(0, EQUILIBRATION_STEPS, EQUILIBRATION_CHUNK_STEPS):
                        chunk = min(EQUILIBRATION_CHUNK_STEPS, EQUILIBRATION_STEPS - equil_done)
                        simulation.step(chunk)
                        ps_done = (equil_done + chunk) * 0.002
                        elapsed = time.time() - t_equil_start
                        print(f"    Equilibration: {ps_done:.0f}/100 ps "
                              f"({elapsed:.0f}s elapsed)")

                else:
                    print(f"    Resuming from saved state: {resume_from_state_path} "
                          f"(skipping placement/relax/minimize/anneal/equilibration)...")
                    simulation.loadState(resume_from_state_path)
                    integrator.setTemperature(300.0 * unit.kelvin)

                # Production
                print(f"    Production MD ({duration_ns} ns)...")
                steps = int(duration_ns * 500000)  # 2 fs timestep

                # Sample every 10 ps
                sample_interval = 5000

                binding_contacts = []
                groove_distances = []
                cys528_distances = []
                hydrophobic_contacts = []

                # Extra series recorded purely for reporting/graphing - not
                # used in the scoring math below, which still only needs the
                # means above.
                sample_times_ps = []
                production_energy_trace = []
                nes_ca_positions_over_time = []  # one (n_ca, 3) array per sample, for RMSF

                # NES peptide CA atom indices, in residue order, for RMSF.
                # nes_modeller's peptide residues come first (indices 0..
                # nes_peptide_residue_count-1) since the peptide is placed
                # before CRM1 gets added to the same Modeller - matches the
                # same 0..nes_peptide_residue_count convention already used
                # by _count_hydrophobic_contacts() below.
                nes_ca_indices = []
                nes_ca_residue_labels = []
                for residue in nes_modeller.topology.residues():
                    if residue.index >= nes_peptide_residue_count:
                        break
                    for atom in residue.atoms():
                        if atom.name == 'CA':
                            nes_ca_indices.append(atom.index)
                            nes_ca_residue_labels.append(f"{residue.name}{residue.index + 1}")
                            break

                # Cys528's index in the combined (peptide+CRM1) structure,
                # computed once here rather than re-searched every sample.
                # Uses the cached atom index from _identify_binding_groove()
                # (correct even after shell truncation renumbered
                # everything) instead of the old hardcoded "index==527"
                # heuristic, which only ever matched the ORIGINAL full
                # structure's numbering and would silently find the wrong
                # atom (or nothing) once the shell is in use.
                cys528_combined_idx = None
                if self.crm1_cys528_atom_index is not None:
                    cys528_combined_idx = nes_peptide_atom_count + self.crm1_cys528_atom_index

                # Phi-anchor <-> sub-pocket tracking setup: for every anchor
                # this candidate's registration actually matched (see
                # _place_peptide_via_subpocket_registration), record (a) the
                # anchor CA's combined-system index, and (b) the combined-
                # system indices of that pocket's OWN member-residue CAs, so
                # the production loop below can compute a REAL, per-frame
                # anchor-to-pocket distance -- both sides tracked live off
                # simulated positions, the same way cys528_combined_idx is
                # re-read every frame above, rather than trusting either
                # side to sit still after minimization/equilibration.
                anchor_pocket_tracking = {}  # label -> {'anchor_idx': int, 'pocket_atom_indices': [int,...]}
                if subpocket_registration is not None:
                    for label, seq_idx in subpocket_registration.get('anchor_seq_positions', {}).items():
                        if seq_idx >= len(nes_ca_indices):
                            continue
                        pocket = self.crm1_subpockets.get(label) if self.crm1_subpockets else None
                        if not pocket or not pocket.get('atom_indices'):
                            continue
                        anchor_pocket_tracking[label] = {
                            'anchor_idx': nes_ca_indices[seq_idx],
                            'pocket_atom_indices': [nes_peptide_atom_count + i
                                                     for i in pocket['atom_indices']],
                        }
                anchor_distance_traces = {label: [] for label in anchor_pocket_tracking}

                # ---- advanced-analysis setup (RMSD/Rg/H-bonds/contact map/
                #      MM-GBSA-style binding energy/DSSP/SASA) -- see
                #      ADVANCED_ANALYSIS_STRIDE / N_REPRESENTATIVE_FRAMES at
                #      module level for the sampling-density trade-offs. ----
                nes_backbone_map = self._identify_backbone_no_atoms(nes_modeller.topology, nes_peptide_atom_count)
                nes_residue_atoms = self._build_residue_atom_map(
                    nes_modeller.topology, lambda a: a.index < nes_peptide_atom_count)
                groove_residue_atoms = self._build_residue_atom_map(
                    nes_modeller.topology,
                    lambda a: a.index >= nes_peptide_atom_count and
                    (a.residue.index - nes_peptide_residue_count) in (self.crm1_groove_residues or []))
                # All atom indices (within the combined peptide+CRM1
                # topology) belonging to any groove residue -- built ONCE
                # here from groove_residue_atoms (residue-index-aware and
                # correct, see _build_residue_atom_map) for the per-frame
                # groove-contact count below.
                #
                # That per-frame count used to build its own
                # groove atom list separately, via
                # `positions[i + nes_peptide_atom_count] for i in
                # self.crm1_groove_residues` -- treating self.
                # crm1_groove_residues' entries (RESIDUE indices, see
                # _identify_binding_groove's `groove_residues.append(
                # residue.index)`) as if adding nes_peptide_atom_count to
                # one directly gave a valid ATOM index. It doesn't -- each
                # residue has multiple atoms, so a residue index and an
                # atom index are never interchangeable this way. That bug
                # silently picked out whichever atom happened to sit at
                # that (essentially arbitrary) numeric offset -- almost
                # always NOT one of the real identified groove residues --
                # instead of the intended groove atoms, making avg_contacts
                # (binding_score's first, multiplicative factor) close to
                # meaningless: found while investigating why binding_score
                # was reading ~0 for essentially every real candidate,
                # positives and negatives alike.
                groove_atom_indices_flat = sorted(
                    idx for atoms in groove_residue_atoms.values() for idx in atoms)
                nes_peptide_masses = np.array([
                    a.element.mass.value_in_unit(unit.dalton) if a.element is not None else 1.0
                    for a in nes_modeller.topology.atoms() if a.index < nes_peptide_atom_count
                ])
                charges, sigmas, epsilons = self._extract_nonbonded_params(system)

                n_crm1_atoms = len(list(self.crm1_structure.topology.atoms()))
                # Isolated peptide-alone / CRM1-alone energy contexts, built
                # ONCE and reused for every sampled frame's MM-GBSA-style
                # binding energy evaluation (single-point energy at that
                # frame's coordinates -- these contexts are never stepped).
                peptide_energy_ctx, crm1_energy_ctx = None, None
                try:
                    _peptide_snapshot_pdb = PDBFile(StringIO(peptide_only_pdb_text))
                    peptide_energy_ctx = self._build_isolated_energy_context(
                        _peptide_snapshot_pdb.topology, _peptide_snapshot_pdb.positions)
                    crm1_energy_ctx = self._build_isolated_energy_context(
                        self.crm1_structure.topology, self.crm1_structure.positions)
                except Exception as e:
                    print(f"    Warning: Could not build isolated energy contexts for MM-GBSA-style "
                          f"binding energy (skipping that metric, non-fatal): {e}")

                nes_peptide_positions_over_time = []  # all-atom peptide, full sample density (small system, cheap)
                hbond_trace = []
                mmgbsa_binding_energy_trace = []
                contact_freq_matrix = None
                contact_freq_nes_ids, contact_freq_groove_ids = [], []
                n_contact_samples = 0

                # Progress reporting for the production loop -- this used to
                # print nothing at all between "Production MD (X ns)..." and
                # the run finishing, which is indistinguishable from a hung
                # process on a slow platform. One line per sampled frame
                # (every sample_interval, i.e. every 10 ps of simulated
                # time) with an elapsed/estimated-remaining wall-clock
                # readout, not per integration step -- frequent enough to
                # show real movement, not so frequent it floods the log.
                total_frames = max(1, (steps + sample_interval - 1) // sample_interval)
                t_prod_start = time.time()

                for step in range(0, steps, sample_interval):
                    simulation.step(sample_interval)
                    state = simulation.context.getState(getPositions=True, getEnergy=True)
                    positions = state.getPositions(asNumpy=True)

                    frame_idx = step // sample_interval + 1
                    ps_done = (step + sample_interval) * 0.002
                    elapsed = time.time() - t_prod_start
                    rate_sec_per_frame = elapsed / frame_idx
                    remaining_sec = rate_sec_per_frame * (total_frames - frame_idx)
                    print(f"    Production MD: frame {frame_idx}/{total_frames}  "
                          f"{ps_done:.0f}/{duration_ns * 1000:.0f} ps simulated  "
                          f"({elapsed:.0f}s elapsed, ~{remaining_sec:.0f}s remaining)")

                    sample_times_ps.append((step + sample_interval) * 0.002)  # 2 fs/step
                    production_energy_trace.append(
                        float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    )
                    if nes_ca_indices:
                        nes_ca_positions_over_time.append(np.array([
                            positions[i].value_in_unit(unit.nanometer) for i in nes_ca_indices
                        ]))

                    # Analyze NES-CRM1 contacts
                    nes_com = self._get_center_of_mass(positions[:nes_peptide_atom_count])

                    # Distance to Cys528
                    if cys528_combined_idx is not None:
                        dist_cys528 = np.linalg.norm(
                            nes_com - positions[cys528_combined_idx].value_in_unit(unit.nanometer)
                        )
                        cys528_distances.append(dist_cys528)

                    # Phi-anchor <-> sub-pocket distance, this frame -- both
                    # the anchor CA and the pocket's own member-residue
                    # centroid are read from THIS frame's real simulated
                    # positions (not the static centroid_nm captured once
                    # at load time), so this reflects whether the anchor
                    # actually stayed registered to its assigned pocket
                    # through minimization/equilibration/production, not
                    # just where the initial Kabsch fit put it.
                    for label, track in anchor_pocket_tracking.items():
                        anchor_pos_nm = positions[track['anchor_idx']].value_in_unit(unit.nanometer)
                        pocket_now_nm = np.mean([
                            positions[i].value_in_unit(unit.nanometer)
                            for i in track['pocket_atom_indices']
                        ], axis=0)
                        anchor_distance_traces[label].append(
                            float(np.linalg.norm(np.array(anchor_pos_nm) - pocket_now_nm))
                        )

                    # Count contacts with groove residues -- uses
                    # groove_atom_indices_flat (correct, residue-index-
                    # aware atom set, see comment above where it's built)
                    # rather than looping through self.crm1_groove_residues
                    # directly. Vectorized with numpy rather than
                    # _count_contacts()'s plain nested-Python-loop
                    # implementation, since this now compares against every
                    # atom of every groove residue (previously, thanks to
                    # the bug above, it was accidentally comparing against
                    # only one essentially-arbitrary atom per residue, so
                    # the cost of doing this correctly is real and worth
                    # keeping fast at production-loop frequency).
                    if groove_atom_indices_flat:
                        pep_pos_nm = np.array(
                            positions[:nes_peptide_atom_count].value_in_unit(unit.nanometer))
                        groove_pos_nm = np.array([
                            positions[i].value_in_unit(unit.nanometer) for i in groove_atom_indices_flat
                        ])
                        diffs = pep_pos_nm[:, None, :] - groove_pos_nm[None, :, :]
                        contacts = int(np.sum(np.linalg.norm(diffs, axis=2) < 0.6))  # 6 Å
                    else:
                        contacts = 0
                    binding_contacts.append(contacts)

                    # Hydrophobic contacts
                    hydro_contacts = self._count_hydrophobic_contacts(
                        positions, nes_modeller.topology, 0, nes_peptide_residue_count
                    )
                    hydrophobic_contacts.append(hydro_contacts)

                    # All-atom peptide coordinates this frame (nm), for the
                    # Rg / DSSP / SASA analysis below.
                    peptide_frame_nm = np.array([
                        positions[i].value_in_unit(unit.nanometer) for i in range(nes_peptide_atom_count)
                    ])
                    nes_peptide_positions_over_time.append(peptide_frame_nm)

                    # ---- advanced per-frame analysis at reduced stride
                    #      (H-bonds, contact map, MM-GBSA-style binding
                    #      energy) -- each cheap alone but this bounds their
                    #      combined cost on long production runs. ----
                    if (len(sample_times_ps) - 1) % ADVANCED_ANALYSIS_STRIDE == 0:
                        if nes_backbone_map:
                            hbond_trace.append(
                                self._count_backbone_hbonds(peptide_frame_nm, nes_backbone_map))

                        if groove_residue_atoms and nes_residue_atoms:
                            mat, mat_nes_ids, mat_groove_ids = self._count_contacts_matrix(
                                positions, nes_residue_atoms, groove_residue_atoms)
                            if contact_freq_matrix is None:
                                contact_freq_matrix = mat.astype(int)
                                contact_freq_nes_ids, contact_freq_groove_ids = mat_nes_ids, mat_groove_ids
                            else:
                                contact_freq_matrix += mat.astype(int)
                            n_contact_samples += 1

                        if charges is not None and peptide_energy_ctx is not None and crm1_energy_ctx is not None:
                            try:
                                peptide_energy_ctx.context.setPositions(positions[:nes_peptide_atom_count])
                                e_peptide = peptide_energy_ctx.context.getState(getEnergy=True) \
                                    .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                                crm1_energy_ctx.context.setPositions(
                                    positions[nes_peptide_atom_count:nes_peptide_atom_count + n_crm1_atoms])
                                e_crm1 = crm1_energy_ctx.context.getState(getEnergy=True) \
                                    .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                                e_complex_no_restraint = simulation.context.getState(
                                    getEnergy=True, groups={0}).getPotentialEnergy() \
                                    .value_in_unit(unit.kilojoule_per_mole)
                                mmgbsa_binding_energy_trace.append(
                                    float(e_complex_no_restraint - e_peptide - e_crm1))
                            except Exception as e:
                                print(f"    Warning: MM-GBSA-style energy evaluation failed for "
                                      f"this frame (non-fatal): {e}")

                # RMSF per NES peptide residue (CA atoms) across the
                # production trajectory - how much each residue wobbled
                # around its own average position. Low RMSF at the
                # hydrophobic anchor residues alongside high RMSF at the
                # flexible ends is the classic signature of a peptide
                # settling into a stable bound pose rather than just
                # drifting freely.
                nes_peptide_rmsf = []
                if nes_ca_positions_over_time:
                    all_frames = np.array(nes_ca_positions_over_time)  # (n_samples, n_ca, 3)
                    mean_pos = all_frames.mean(axis=0)
                    sq_devs = np.sum((all_frames - mean_pos) ** 2, axis=2)
                    rmsf_nm = np.sqrt(sq_devs.mean(axis=0))
                    nes_peptide_rmsf = [
                        {'residue': label, 'rmsf_nm': float(val)}
                        for label, val in zip(nes_ca_residue_labels, rmsf_nm)
                    ]

                # ---- RMSD / Rg, computed post-loop from the already-stored
                #      full-density CA / all-atom peptide positions,
                #      subsampled to ADVANCED_ANALYSIS_STRIDE to match the
                #      other advanced-analysis time series above. ----
                advanced_sample_times_ps = sample_times_ps[::ADVANCED_ANALYSIS_STRIDE]
                strided_indices = list(range(0, len(nes_ca_positions_over_time), ADVANCED_ANALYSIS_STRIDE))
                ca_strided = [nes_ca_positions_over_time[i] for i in strided_indices]
                peptide_strided = [nes_peptide_positions_over_time[i] for i in strided_indices]

                rmsd_trace = self._compute_rmsd_trace(ca_strided)
                rg_trace = self._compute_radius_of_gyration_trace(peptide_strided, nes_peptide_masses)

                # ---- DSSP helix fraction + free-peptide SASA (mdtraj) on
                #      the same strided subset -- cheap since it's just the
                #      free peptide, a small system. ----
                dssp_helix_fraction_trace = self._compute_dssp_helix_fraction(peptide_only_pdb_text, peptide_strided)
                ramachandran_trace = self._compute_ramachandran_trace(peptide_only_pdb_text, peptide_strided)
                peptide_sasa_trace_nm2 = self._compute_sasa_nm2(peptide_only_pdb_text, peptide_strided)

                # ---- buried SASA + per-residue interaction-energy
                #      decomposition, restricted to a handful of
                #      REPRESENTATIVE frames from the END of production
                #      (assumed post-equilibration/bound state) -- both are
                #      too expensive to run on the whole trajectory (whole-
                #      shell SASA; O(peptide_atoms x groove_atoms) energy
                #      sums). CRM1's own coordinates are approximated by its
                #      reference (pre-docking) position for this block only,
                #      since only the PEPTIDE's per-frame positions were
                #      stored at full density -- a reasonable approximation
                #      given CRM1's backbone is under a stiff (1000 kJ/mol/
                #      nm^2) positional restraint throughout production, but
                #      an approximation nonetheless (documented here rather
                #      than silently assumed). ----
                buried_sasa_nm2 = None
                residue_interaction_energy_kj_mol = {}
                rep_frame_indices = list(range(
                    max(0, len(nes_peptide_positions_over_time) - N_REPRESENTATIVE_FRAMES),
                    len(nes_peptide_positions_over_time)
                ))
                crm1_ref_frame_nm = np.array([
                    self.crm1_structure.positions[i].value_in_unit(unit.nanometer)
                    for i in range(n_crm1_atoms)
                ])

                if rep_frame_indices and charges is not None and groove_residue_atoms:
                    try:
                        energies_per_frame = []
                        for fi in rep_frame_indices:
                            full_frame_nm = np.vstack([nes_peptide_positions_over_time[fi], crm1_ref_frame_nm])
                            energies_per_frame.append(self._residue_interaction_energy(
                                full_frame_nm, charges, sigmas, epsilons,
                                nes_residue_atoms, groove_residue_atoms))
                        all_res_ids = sorted(nes_residue_atoms.keys())
                        label_by_res_id = {res_id: (nes_ca_residue_labels[i] if i < len(nes_ca_residue_labels) else str(res_id))
                                            for i, res_id in enumerate(all_res_ids)}
                        residue_interaction_energy_kj_mol = {
                            label_by_res_id[res_id]: float(np.mean([e.get(res_id, 0.0) for e in energies_per_frame]))
                            for res_id in all_res_ids
                        }
                    except Exception as e:
                        print(f"    Warning: Per-residue interaction energy decomposition failed (non-fatal): {e}")

                    try:
                        rep_peptide_frames = [nes_peptide_positions_over_time[i] for i in rep_frame_indices]
                        rep_peptide_sasa = self._compute_sasa_nm2(peptide_only_pdb_text, rep_peptide_frames)

                        _crm1_snapshot_io = StringIO()
                        PDBFile.writeFile(self.crm1_structure.topology, self.crm1_structure.positions, _crm1_snapshot_io)
                        rep_crm1_sasa = self._compute_sasa_nm2(
                            _crm1_snapshot_io.getvalue(), [crm1_ref_frame_nm] * len(rep_frame_indices))

                        _complex_snapshot_io = StringIO()
                        PDBFile.writeFile(nes_modeller.topology, nes_modeller.positions, _complex_snapshot_io)
                        rep_complex_frames = [
                            np.vstack([nes_peptide_positions_over_time[i], crm1_ref_frame_nm])
                            for i in rep_frame_indices
                        ]
                        rep_complex_sasa = self._compute_sasa_nm2(_complex_snapshot_io.getvalue(), rep_complex_frames)

                        if rep_peptide_sasa and rep_crm1_sasa and rep_complex_sasa:
                            buried_per_frame = [p + c - x for p, c, x in
                                                 zip(rep_peptide_sasa, rep_crm1_sasa, rep_complex_sasa)]
                            buried_sasa_nm2 = float(np.mean(buried_per_frame))
                    except Exception as e:
                        print(f"    Warning: Buried SASA calculation failed (non-fatal): {e}")

                # ---- per-anchor hydrophobic-groove burial --
                #      confirms each Phi-anchor residue that got CA-
                #      registered to a sub-pocket has actually LOST solvent-
                #      accessible surface area by being in the complex, not
                #      just sitting near the pocket while still solvent-
                #      exposed. Answers "does it sit IN the groove, not just
                #      at the anchor positions" -- same representative end-
                #      of-production frames as buried_sasa_nm2 above, just
                #      decomposed per residue via _compute_residue_sasa_nm2
                #      instead of summed over the whole peptide. ----
                anchor_burial_nm2 = {}
                mean_anchor_burial_nm2 = None
                anchor_burial_fraction_well_buried = None
                best_anchor_frame_well_buried_count = None
                best_anchor_frame_sample_index = None
                if rep_frame_indices and subpocket_registration is not None:
                    try:
                        anchor_seq_positions = subpocket_registration.get('anchor_seq_positions', {}) or {}
                        if anchor_seq_positions:
                            rep_peptide_res_sasa = self._compute_residue_sasa_nm2(
                                peptide_only_pdb_text, rep_peptide_frames)
                            rep_complex_res_sasa = self._compute_residue_sasa_nm2(
                                _complex_snapshot_io.getvalue(), rep_complex_frames)
                            if rep_peptide_res_sasa and rep_complex_res_sasa:
                                # WELL_BURIED_THRESHOLD_NM2: a lone aliphatic
                                # side chain (Leu/Ile/Val-scale) losing
                                # roughly this much SASA on binding is a
                                # reasonable "genuinely inserted into a
                                # pocket" bar -- small enough not to demand
                                # near-total burial (the groove is a shell,
                                # not a fully enclosed cavity), large enough
                                # to exclude a residue that's merely CA-
                                # adjacent to a pocket while its side chain
                                # still faces solvent.
                                WELL_BURIED_THRESHOLD_NM2 = 0.15
                                n_well_buried = 0
                                # Per-representative-frame well-
                                # buried counts, local index (0..len(rep_
                                # frame_indices)-1) -> how many anchors cross
                                # the threshold AT THAT SPECIFIC FRAME -- lets
                                # save_best_anchor_frame_pdb_path pick the
                                # single most-engaged snapshot from this same
                                # window, rather than only ever reporting the
                                # window-mean fraction.
                                per_frame_well_buried_counts = None
                                for label, seq_idx in anchor_seq_positions.items():
                                    if seq_idx not in rep_peptide_res_sasa or seq_idx not in rep_complex_res_sasa:
                                        continue
                                    free_vals = rep_peptide_res_sasa[seq_idx]
                                    bound_vals = rep_complex_res_sasa[seq_idx]
                                    n = min(len(free_vals), len(bound_vals))
                                    if n == 0:
                                        continue
                                    burial_vals = [free_vals[i] - bound_vals[i] for i in range(n)]
                                    mean_burial = float(np.mean(burial_vals))
                                    anchor_burial_nm2[label] = mean_burial
                                    if mean_burial >= WELL_BURIED_THRESHOLD_NM2:
                                        n_well_buried += 1
                                    if per_frame_well_buried_counts is None:
                                        per_frame_well_buried_counts = [0] * n
                                    for i in range(min(n, len(per_frame_well_buried_counts))):
                                        if burial_vals[i] >= WELL_BURIED_THRESHOLD_NM2:
                                            per_frame_well_buried_counts[i] += 1
                                if anchor_burial_nm2:
                                    mean_anchor_burial_nm2 = float(np.mean(list(anchor_burial_nm2.values())))
                                    anchor_burial_fraction_well_buried = n_well_buried / len(anchor_burial_nm2)
                                if per_frame_well_buried_counts:
                                    best_local_idx = int(np.argmax(per_frame_well_buried_counts))
                                    best_anchor_frame_well_buried_count = per_frame_well_buried_counts[best_local_idx]
                                    best_anchor_frame_sample_index = rep_frame_indices[best_local_idx]

                                    if save_best_anchor_frame_pdb_path:
                                        try:
                                            best_peptide_nm = nes_peptide_positions_over_time[
                                                best_anchor_frame_sample_index]
                                            best_complex_positions = (
                                                [Vec3(*row) for row in best_peptide_nm]
                                                + [Vec3(*row) for row in crm1_ref_frame_nm]
                                            ) * unit.nanometer
                                            best_modeller = Modeller(nes_modeller.topology, best_complex_positions)
                                            with open(save_best_anchor_frame_pdb_path, 'w') as _best_pdb_f:
                                                PDBFile.writeFile(best_modeller.topology, best_modeller.positions,
                                                                   _best_pdb_f)
                                            print(f"    Saved best-anchor-frame complex pose "
                                                  f"({best_anchor_frame_well_buried_count}/"
                                                  f"{len(anchor_burial_nm2)} anchors well-buried, "
                                                  f"sample {best_anchor_frame_sample_index}) to "
                                                  f"{save_best_anchor_frame_pdb_path}")
                                        except Exception as e:
                                            print(f"    Warning: Best-anchor-frame PDB save failed (non-fatal): {e}")
                    except Exception as e:
                        print(f"    Warning: Anchor burial calculation failed (non-fatal): {e}")

                # ---- residue-pair contact frequency map (fraction of
                #      strided samples each NES-residue/groove-residue pair
                #      was in contact) ----
                residue_contact_map = None
                if contact_freq_matrix is not None and n_contact_samples > 0:
                    freq = contact_freq_matrix / n_contact_samples
                    residue_contact_map = {
                        # contact_freq_nes_ids holds NES residue.index values,
                        # which run contiguously 0..nes_peptide_residue_count-1
                        # in the same order nes_ca_residue_labels was built in
                        # (see the RMSF setup above), so a direct index lookup
                        # recovers each residue's human-readable label.
                        'nes_residues': [nes_ca_residue_labels[r] if r < len(nes_ca_residue_labels) else str(r)
                                          for r in contact_freq_nes_ids],
                        'groove_residues': [str(r) for r in contact_freq_groove_ids],
                        'frequency': freq.tolist(),
                    }

                # Calculate binding metrics
                avg_contacts = np.mean(binding_contacts)
                avg_cys528_dist = np.mean(cys528_distances) if cys528_distances else 2.0
                avg_hydro = np.mean(hydrophobic_contacts)

                # Binding score (0-1, higher = better binding)
                #
                # CONTACTS_SCALE / HYDRO_SCALE recalibrated from
                # real MD data (evaluate_anchor_occupancy_signal.py, 22
                # real docked examples, both classes, AFTER fixing two
                # separate pre-existing bugs: the residue-index-as-atom-
                # index mixup in the groove-contact count, and the CA-only
                # (rather than side-chain) definition of hydrophobic
                # contact -- see git history / comments on
                # _count_hydrophobic_contacts and the groove_atom_indices_
                # flat construction above). The OLD divisors (10.0, 5.0)
                # were tuned back when both bugs kept avg_contacts and
                # avg_hydro near-zero almost always; once genuinely fixed,
                # real avg_groove_contacts ranged ~89-779 (median ~390)
                # and real avg_hydrophobic_contacts ranged ~3.4-94 (median
                # ~30) across candidates that docked at all -- the old
                # divisors left EVERY one of those clipped at the
                # min(1.0,...) ceiling, exactly as uninformative as when
                # they were all pinned to 0 before the counting fixes.
                #
                # Anchored to roughly the 75th percentile of that observed
                # range (~570 contacts, ~40 hydrophobic contacts), not the
                # median, so a TYPICAL real docker lands in a moderate
                # middle range instead of already nearly saturating --
                # leaving headroom to actually distinguish good from
                # exceptional. Still an approximation from ~22 examples,
                # not a fully validated calibration -- flagged the same
                # way this codebase already flags its other hand-picked
                # constants (see CRM1_pocket_scoring_evaluation_2026-07-27.md);
                # revisit once a larger labeled MD dataset exists.
                CONTACTS_SCALE = 570.0
                HYDRO_SCALE = 40.0

                # Hard min(1.0,...) clip replaced with
                # tanh(0.5 * product). The 22-example real-candidate
                # calibration above never anticipated how tightly PACKED a
                # genuine crystal-conformation docking run can get:
                # crystal_full_grid_check.py's 10-structure ground-truth
                # grid showed the unclipped product reaching up to 8.3 (vs.
                # a real-candidate max of ~1.8), so >25% of crystal-grid
                # runs were hitting the ceiling -- and critically, BOTH the
                # correct AND scrambled registration runs saturated
                # together for 4 of those 10 structures (Paxillin 5UWH,
                # X11L2 5UWS, hRio2 5DHF, CPEB4 5DIF), erasing the
                # correct-vs-scrambled gap entirely rather than reflecting
                # an actual loss of discrimination. Verified OFFLINE, at no
                # GPU cost, directly against the already-recorded
                # avg_groove_contacts/avg_cys528_distance_nm/
                # avg_hydrophobic_contacts fields for all 108 crystal-grid
                # runs and all 52 real-candidate runs: switching to
                # tanh(0.5x) eliminates every double-saturation case (0/54
                # correct/scrambled pairs vs. 6/54 under the old clip), and
                # recovers a real, previously-hidden positive gap for 3 of
                # those 4 structures (5UWH +0.00 -> +0.22; 5DHF +0.00 ->
                # +0.08; 5DIF +0.00 -> +0.12, all crystal conformation,
                # relaxed). X11L2 (5UWS) stays at ~0 regardless of scoring
                # function, correctly -- its crystal conformation has no
                # Phi-register match at all (see _find_phi_register's
                # class-4 docstring), so there was never a registration-
                # dependent signal here to recover; this is a genuine,
                # separate limitation, not something a rescaled
                # raw_binding_score could fix. Because AUC is rank-based
                # and every candidate replacement (tanh at several k
                # values, 1-exp(-x)) is a strictly monotonic function of
                # the same product, real-candidate discrimination
                # (evaluate_anchor_occupancy_signal.py, n=52) is completely
                # unaffected either way (AUC identical to 6 decimal places
                # under every transform tested) -- this change only
                # affects runs that were previously hitting the ceiling,
                # which in practice means crystal-conformation runs, not
                # typical real-candidate ones.
                raw_binding_product = (avg_contacts / CONTACTS_SCALE) * (1.0 / max(0.5, avg_cys528_dist)) * (avg_hydro / HYDRO_SCALE)
                binding_score = float(np.tanh(0.5 * raw_binding_product))

                # Binding affinity estimate (approximate, in kcal/mol)
                # Based on number of contacts and proximity
                #
                # avg_contacts shares the same recalibration need as
                # binding_score above -- left unscaled, this was producing
                # wildly unphysical estimates (real protein-peptide binding
                # affinities are roughly -5 to -20 kcal/mol; this read in
                # the hundreds, e.g. -589 kcal/mol on one real candidate,
                # once avg_contacts was actually being computed correctly).
                # Scaled by CONTACTS_SCALE/10 (~57) so a typical real
                # docker's contact term lands around -7 to -14 kcal/mol --
                # a physically plausible range -- instead of an order of
                # magnitude too large. Same caveat as above: an
                # approximation from limited real data, not a validated
                # potential.
                binding_affinity = -1.0 * (avg_contacts / (CONTACTS_SCALE / 10.0)) - 3.0 * (1.0 / max(0.5, avg_cys528_dist))

                # Phi-anchor <-> sub-pocket occupancy, averaged over the
                # production trajectory (anchor_distance_traces, populated
                # per-frame above from REAL simulated positions on both
                # sides -- see the tracking-setup comment near
                # anchor_pocket_tracking). None if this candidate's
                # sequence never matched a usable Phi register (fell back
                # to generic groove placement, no per-anchor pocket to
                # track).
                #
                # Distance-to-score mapping: a matched anchor whose CA
                # averages <=0.5 nm from its assigned pocket's own
                # residue-CA centroid over the trajectory counts as fully
                # anchored (1.0); linearly down to 0.0 by 1.5 nm. Those two
                # numbers are a literature-informed but NOT empirically
                # fit starting point (typical CA-to-CA packing distances
                # for a buried hydrophobic side chain sitting in a pocket
                # formed by neighboring CAs) -- flagged here the same way
                # this codebase already flagged its other hand-picked
                # distance-based scoring constants (see
                # CRM1_pocket_scoring_evaluation_2026-07-27.md) pending a
                # real labeled-data pass with diagnose_feature_importance.py
                # or equivalent.
                ANCHOR_FULL_OCCUPANCY_NM = 0.5
                ANCHOR_ZERO_OCCUPANCY_NM = 1.5
                per_anchor_avg_distance_nm = {
                    label: float(np.mean(trace)) for label, trace in anchor_distance_traces.items() if trace
                }
                anchor_occupancy_score = None
                avg_anchor_distance_nm = None
                if per_anchor_avg_distance_nm:
                    avg_anchor_distance_nm = float(np.mean(list(per_anchor_avg_distance_nm.values())))
                    span = ANCHOR_ZERO_OCCUPANCY_NM - ANCHOR_FULL_OCCUPANCY_NM
                    anchor_occupancy_score = float(np.clip(
                        1.0 - (avg_anchor_distance_nm - ANCHOR_FULL_OCCUPANCY_NM) / span, 0.0, 1.0))

                # Binding category with helix consideration
                # CRITICAL: Even good binding is irrelevant if helix formation is poor
                helix_bonus = helix_metrics['combined_score']

                # Adjust binding score by helix propensity
                adjusted_binding_score = binding_score * (0.5 + 0.5 * helix_bonus)

                # Fold in anchor occupancy as a modest modulating factor
                # (up to +/-15%), same pattern as the helix adjustment
                # above and the anchor2_window binding-region bonus in
                # app.py's NES heuristic scorer -- not a dominant term,
                # and explicitly NOT applied at all (score/category
                # unchanged) when no clean Phi register was matched, rather
                # than penalizing candidates this feature simply couldn't
                # register.
                ANCHOR_OCCUPANCY_WEIGHT = 0.15
                if anchor_occupancy_score is not None:
                    adjusted_binding_score = min(1.0, adjusted_binding_score *
                                                  (1.0 - ANCHOR_OCCUPANCY_WEIGHT
                                                   + ANCHOR_OCCUPANCY_WEIGHT * anchor_occupancy_score))

                if adjusted_binding_score > 0.7:
                    category = 'strong_binder'
                    likelihood = 'High likelihood of CRM1 binding (good helix formation)'
                elif adjusted_binding_score > 0.4:
                    category = 'moderate_binder'
                    likelihood = 'Moderate likelihood of CRM1 binding'
                elif helix_bonus < 0.3:
                    category = 'weak_binder'
                    likelihood = 'Low likelihood - poor helix formation limits binding'
                else:
                    category = 'weak_binder'
                    likelihood = 'Low likelihood of CRM1 binding'

                # See save_final_peptide_pdb_path's docstring above -- lets a
                # caller (idealized_helix_vs_crystal_check.py) RMSD-compare
                # this converged pose directly against real crystal
                # coordinates. Peptide atoms are indices
                # [0, nes_peptide_atom_count) by construction (see STEP 2 /
                # _build_peptide_backbone_restraint_force's same convention).
                if save_final_state_path:
                    simulation.saveState(save_final_state_path)
                    print(f"    Saved simulation state (positions+velocities) to {save_final_state_path}")

                if save_final_peptide_pdb_path or save_final_complex_pdb_path:
                    final_positions = simulation.context.getState(getPositions=True).getPositions()
                    final_modeller = Modeller(nes_modeller.topology, final_positions)

                    # Write the full complex FIRST, before final_modeller is
                    # trimmed down to peptide-only below (Modeller.delete()
                    # mutates in place) -- see save_final_complex_pdb_path's
                    # docstring above.
                    if save_final_complex_pdb_path:
                        with open(save_final_complex_pdb_path, 'w') as _complex_pdb_f:
                            PDBFile.writeFile(final_modeller.topology, final_modeller.positions, _complex_pdb_f)
                        print(f"    Saved final complex (peptide + CRM1) pose to {save_final_complex_pdb_path}")

                    if save_final_peptide_pdb_path:
                        residues_to_drop = [
                            residue for residue in final_modeller.topology.residues()
                            if next(iter(residue.atoms())).index >= nes_peptide_atom_count
                        ]
                        final_modeller.delete(residues_to_drop)
                        with open(save_final_peptide_pdb_path, 'w') as _final_pdb_f:
                            PDBFile.writeFile(final_modeller.topology, final_modeller.positions, _final_pdb_f)
                        print(f"    Saved final peptide-only pose to {save_final_peptide_pdb_path}")

                md_metrics = {
                    # Helix formation metrics (CRITICAL)
                    'helix_propensity': helix_metrics['helix_propensity'],
                    'extended_helix_propensity': helix_metrics.get('extended_helix_propensity', helix_metrics['helix_propensity']),
                    'conformation_prediction': conformation_prediction,
                    'amphipathic_score': helix_metrics['amphipathic_score'],
                    'hydrophobic_face_ratio': helix_metrics['hydrophobic_face_ratio'],
                    'helix_breakers': helix_metrics['helix_breakers'],
                    'helix_combined_score': helix_metrics['combined_score'],

                    # Binding metrics
                    'binding_score': float(adjusted_binding_score),  # Adjusted by helix AND anchor occupancy
                    'raw_binding_score': float(binding_score),  # Neither adjustment applied
                    'binding_affinity_kcal_mol': float(binding_affinity),
                    'anchor_occupancy_score': anchor_occupancy_score,  # None if no Phi register matched
                    'avg_anchor_pocket_distance_nm': avg_anchor_distance_nm,
                    'per_anchor_avg_pocket_distance_nm': per_anchor_avg_distance_nm,  # {'P1': nm, 'P2': nm,...}
                    'avg_groove_contacts': float(avg_contacts),
                    'avg_cys528_distance_nm': float(avg_cys528_dist),
                    'avg_hydrophobic_contacts': float(avg_hydro),
                    'binding_category': category,
                    'binding_likelihood': likelihood,
                    'simulation_time_ns': duration_ns,
                    'starting_conformation': starting_conformation,  # 'native' or 'idealized_helix'
                    'scramble_registration': scramble_registration,
                    'relax_sidechains': relax_sidechains,

                    # Report/graphing data - not used in the scoring above.
                    'minimization_energy_trace': minimization_trace,
                    'production_time_series_ps': sample_times_ps,
                    'production_energy_trace_kj_mol': production_energy_trace,
                    'production_groove_contacts_trace': [int(c) for c in binding_contacts],
                    'production_cys528_distance_trace_nm': [float(d) for d in cys528_distances],
                    'production_hydrophobic_contacts_trace': [int(c) for c in hydrophobic_contacts],
                    'nes_peptide_rmsf': nes_peptide_rmsf,

                    # ---- advanced metrics (see class docstring / module
                    #      constants for stride and representative-frame
                    #      details) ----
                    'advanced_time_series_ps': advanced_sample_times_ps,
                    'peptide_rmsd_trace_nm': rmsd_trace,
                    'peptide_radius_of_gyration_trace_nm': rg_trace,
                    'backbone_hbond_count_trace': hbond_trace,
                    'mmgbsa_binding_energy_trace_kj_mol': mmgbsa_binding_energy_trace,
                    'dssp_helix_fraction_trace': dssp_helix_fraction_trace,
                    'ramachandran_trace': ramachandran_trace,
                    'peptide_sasa_trace_nm2': peptide_sasa_trace_nm2,
                    'buried_sasa_nm2': buried_sasa_nm2,
                    'anchor_burial_nm2': anchor_burial_nm2 or None,
                    'mean_anchor_burial_nm2': mean_anchor_burial_nm2,
                    'anchor_burial_fraction_well_buried': anchor_burial_fraction_well_buried,
                    'best_anchor_frame_well_buried_count': best_anchor_frame_well_buried_count,
                    'best_anchor_frame_sample_index': best_anchor_frame_sample_index,
                    'residue_interaction_energy_kj_mol': residue_interaction_energy_kj_mol,
                    'residue_contact_map': residue_contact_map,

                    # Phi-anchor <-> CRM1 sub-pocket (P0-P4) registration
                    # from _place_peptide_via_subpocket_registration() (the
                    # initial-placement fit) plus per-frame tracking over
                    # the whole trajectory (anchor_occupancy_score /
                    # avg_anchor_pocket_distance_nm / per_anchor_avg_pocket_
                    # distance_nm above, and the raw per-frame trace just
                    # below). None if this candidate's sequence didn't
                    # match a usable Phi register (generic groove placement
                    # was used instead -- score/category are UNCHANGED by
                    # this feature in that case, not penalized).
                    #
                    # anchor_occupancy_score IS folded into binding_score/
                    # category above (ANCHOR_OCCUPANCY_WEIGHT, modest +/-15%
                    # modulation) at the user's request. Flagging honestly
                    # anyway, the same way this codebase already flagged
                    # crm1_binding_affinity's blend in
                    # CRM1_pocket_scoring_evaluation_2026-07-27.md: the
                    # P0/P1/P2/P4 pocket membership behind this is a
                    # geometric inference calibrated against only two
                    # literature-confirmed points (see the module-level
                    # CRM1_GROOVE_LINING_RESIDUES_1INDEXED comment), and the
                    # 0.5/1.5 nm distance-to-score mapping is a first-pass
                    # guess, not fit against real labeled binding data the
                    # way that eval doc's other weights eventually were.
                    # Worth a similar empirical pass before trusting this
                    # weight at face value.
                    'subpocket_registration': subpocket_registration,
                    'anchor_pocket_distance_traces_nm': anchor_distance_traces,
                }

                print(f"    Binding score: {binding_score:.3f}")
                print(f"    Category: {category}")
                print(f"    Estimated affinity: {binding_affinity:.1f} kcal/mol")

            else:
                # No CRM1 structure - use helix formation analysis
                print("    Warning: No CRM1 structure, using helix-based scoring...")

                # Score based on helix propensity alone
                binding_score = helix_metrics['combined_score']

                if binding_score > 0.7:
                    category = 'predicted_strong_binder'
                    likelihood = 'High helix propensity suggests strong CRM1 binding'
                elif binding_score > 0.5:
                    category = 'predicted_moderate_binder'
                    likelihood = 'Moderate helix propensity suggests likely CRM1 binding'
                elif binding_score > 0.3:
                    category = 'predicted_weak_binder'
                    likelihood = 'Weak helix propensity suggests uncertain binding'
                else:
                    category = 'predicted_non_binder'
                    likelihood = 'Poor helix formation unlikely to bind CRM1'

                md_metrics = {
                    # Helix formation metrics
                    'helix_propensity': helix_metrics['helix_propensity'],
                    'extended_helix_propensity': helix_metrics.get('extended_helix_propensity', helix_metrics['helix_propensity']),
                    'conformation_prediction': conformation_prediction,
                    'amphipathic_score': helix_metrics['amphipathic_score'],
                    'hydrophobic_face_ratio': helix_metrics['hydrophobic_face_ratio'],
                    'helix_breakers': helix_metrics['helix_breakers'],
                    'helix_combined_score': helix_metrics['combined_score'],

                    # Predicted binding (no MD)
                    'binding_score': float(binding_score),
                    'binding_category': category,
                    'binding_likelihood': likelihood,
                    'starting_conformation': starting_conformation,
                    'scramble_registration': scramble_registration,
                    'relax_sidechains': relax_sidechains,
                    'note': 'Helix-based prediction (no CRM1 structure available)'
                }

                print(f"    Helix-based score: {binding_score:.3f}")
                print(f"    {likelihood}")

            # Enhanced score combines original and MD/helix analysis
            enhanced_score = (candidate.get('combined_score', 0.5) + md_metrics['binding_score']) / 2.0

            candidate['md_enhanced_score'] = enhanced_score
            candidate['md_metrics'] = md_metrics

            return candidate

        except Exception as docking_error:
            # CRM1 docking failed - return helix-based analysis
            print(f"    Warning: CRM1 docking failed: {docking_error}")
            print(f"    Returning helix-based analysis")

            md_metrics = {
                'helix_propensity': helix_metrics['helix_propensity'],
                'extended_helix_propensity': helix_metrics.get('extended_helix_propensity', helix_metrics['helix_propensity']),
                'conformation_prediction': conformation_prediction,
                'amphipathic_score': helix_metrics['amphipathic_score'],
                'hydrophobic_face_ratio': helix_metrics['hydrophobic_face_ratio'],
                'helix_breakers': helix_metrics['helix_breakers'],
                'helix_combined_score': helix_metrics['combined_score'],
                'binding_score': helix_metrics['combined_score'],
                'binding_category': 'helix_based_docking_failed',
                'binding_likelihood': f"Helix score: {helix_metrics['combined_score']:.2f}",
                'starting_conformation': starting_conformation,
                'scramble_registration': scramble_registration,
                'relax_sidechains': relax_sidechains,
                'note': 'CRM1 docking failed - helix analysis only',
                'error': str(docking_error)
            }

            enhanced_score = (candidate.get('combined_score', 0.5) + md_metrics['binding_score']) / 2.0
            candidate['md_enhanced_score'] = enhanced_score
            candidate['md_metrics'] = md_metrics

            print(f"    Helix fallback score: {md_metrics['binding_score']:.3f}")
            return candidate

        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _get_center_of_mass(self, positions):
        """Calculate center of mass of positions"""
        return np.mean([pos.value_in_unit(unit.nanometer) for pos in positions], axis=0)

    def _count_contacts(self, positions1, positions2, cutoff=0.6):
        """Count contacts between two sets of positions within cutoff (nm)"""
        contacts = 0
        for pos1 in positions1:
            p1 = pos1.value_in_unit(unit.nanometer)
            for pos2 in positions2:
                p2 = pos2.value_in_unit(unit.nanometer)
                if np.linalg.norm(p1 - p2) < cutoff:
                    contacts += 1
        return contacts

    def _count_hydrophobic_contacts(self, positions, topology, start_idx, end_idx):
        """
        Count hydrophobic side-chain contacts between the NES peptide
        (residues [start_idx, end_idx) by residue.index) and its partner
        (everything else in `topology`, i.e. the CRM1 groove shell here):
        number of (NES hydrophobic side-chain heavy-atom, partner
        hydrophobic side-chain heavy-atom) pairs within 0.6 nm.

        History : this used to compare CA-to-CA distance only
        (backbone alpha-carbons), not actual side-chain atoms -- found
        while diagnosing why binding_score's avg_hydro factor was reading
        ~0 for almost every real MD-docked candidate
        (evaluate_anchor_occupancy_signal.py), even ones with hundreds of
        real all-atom groove contacts. Real hydrophobic packing/burial is
        mediated by SIDE CHAINS (e.g. a Leu or Phe side-chain tip), which
        extend farther from the backbone than the CA itself -- CA-CA
        distance under 0.6 nm is a much rarer, stricter geometric event
        than genuine side-chain burial, so the old definition was
        systematically undercounting real hydrophobic contact, regardless
        of anything else in this file. Kept the SAME 0.6 nm cutoff this
        module already uses for its other contact definitions (see
        _run_crm1_docking's groove-contact count) -- only the ATOM
        SELECTION changed (side-chain heavy atoms, not CA), to avoid
        changing two guessed things (atom selection AND distance
        threshold) in the same pass. Hydrogens excluded (heavy atoms only)
        since they're far more numerous and would otherwise dominate a
        distance-based count without adding real chemical information.

        Vectorized (numpy) rather than the old nested-Python-loop version,
        since this now compares potentially dozens of NES side-chain atoms
        against hundreds of partner side-chain atoms, called every
        production-loop frame.
        """
        hydrophobic_aas = ('LEU', 'ILE', 'VAL', 'PHE', 'MET')
        backbone_names = ('N', 'CA', 'C', 'O')

        def is_hydrophobic_sidechain_atom(atom):
            return (atom.residue.name in hydrophobic_aas and
                    atom.name not in backbone_names and
                    (atom.element is None or atom.element.symbol != 'H'))

        nes_sidechain_pos, partner_sidechain_pos = [], []
        for atom in topology.atoms():
            if not is_hydrophobic_sidechain_atom(atom):
                continue
            pos = positions[atom.index].value_in_unit(unit.nanometer)
            if start_idx <= atom.residue.index < end_idx:
                nes_sidechain_pos.append(pos)
            else:
                partner_sidechain_pos.append(pos)

        if not nes_sidechain_pos or not partner_sidechain_pos:
            return 0

        nes_arr = np.array(nes_sidechain_pos)
        partner_arr = np.array(partner_sidechain_pos)
        diffs = nes_arr[:, None, :] - partner_arr[None, :, :]
        return int(np.sum(np.linalg.norm(diffs, axis=2) < 0.6))
