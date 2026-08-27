#!/usr/bin/env python3
"""
crystal_full_grid_check.py
============================================================
Combined ground-truth validation grid for the CRM1/NES MD pipeline --
thesis-results-section version. Merges what crystal_sanity_check.py
(starting_conformation='crystal', correct vs. scrambled registration x
relaxed vs. unrelaxed side-chains) and idealized_helix_vs_crystal_check.py
(idealized_helix vs. crystal backbone RMSD) each tested SEPARATELY, and
ADDS the third starting hypothesis this project uses everywhere else for
real candidates -- 'native' (AlphaFold's isolated, unbound-state
prediction for the NES motif's own parent protein) -- so all three
starting conformations get run against the SAME real, experimentally-
solved ground truth, each with a correct-vs-scrambled specificity control
and a relaxed/unrelaxed side-chain condition.

GRID: per structure, 3 conformations (native, idealized_helix, crystal) x
2 registration (correct, scrambled) x 2 relaxation (True, False) = 12 runs
(the crystal arm alone reproduces crystal_sanity_check.py's original 2x2
grid exactly).

*** IMPORTANT CAVEAT ON 'native' -- READ BEFORE CITING RESULTS ***
Several of these crystal structures are ENGINEERED CHIMERAS, not the NES's
own native protein context: Guttler et al. 2010's PKI-NES (3NBY) and
Rev-NES (3NBZ, 3NC0) structures graft a heterologous NES peptide onto a
SNURPORTIN-1 scaffold for crystallization -- confirmed directly from each
file's own PDB DBREF record (`grep ^DBREF`), which annotates chain B as
SPN1_HUMAN/O95149 (COMPND also says "MOLECULE: SNURPORTIN-1 ... ENGINEERED:
YES") even though the crystallized N-terminal sequence is really PKI-
alpha's or HIV-1 Rev's own NES, not Snurportin1's. 'native' therefore
always fetches the DONOR NES protein's own UniProt accession/full-length
AlphaFold model, never whatever chain the crystal structure's DBREF
labels internally for the SCAFFOLD -- see NATIVE_SOURCES below.

SAFETY / PROVENANCE OF NATIVE_SOURCES: this script assumes no live
UniProt/AlphaFold network access (only this pod does), so the accessions
below are one of:
  - 'verified': taken directly from the structure's own PDB DBREF record,
    or already independently established elsewhere in this project
    (3NBY's PKI-alpha P04541 numbering, already used in
    crystal_sanity_check.py's own comments).
  - 'guess' / 'guess_range_crosschecked' / 'guess_range_trusted': my best
    identification from the protein name/literature, NOT fetched live.
    Cross-checked where possible against nes_data_pipeline/nes_dataset.json
    (independently-curated NESbase entries) -- see each entry's comment.

Regardless of confidence label, this script NEVER trusts an accession
blindly: before running ANY 'native' condition, it fetches the AlphaFold
model, slices out [native_start, native_end], and requires that slice to
match this structure's own known crystallized peptide sequence EXACTLY
(see verify_native_slice()). If it doesn't match, 'native' is SKIPPED for
that structure with a loud printed warning -- crystal and idealized_helix
still run fine regardless. Check the printed "native verification" line
for every structure before trusting the 'native' column of your results;
a structure with accession=None or a failed verification simply has no
native data point, not wrong data.

USAGE:
    python3 crystal_full_grid_check.py --duration-ns 2.0
    python3 crystal_full_grid_check.py --duration-ns 2.0 --structures 3NBY,3GJX,3NBZ
    python3 crystal_full_grid_check.py --duration-ns 2.0 --skip-native   # crystal+idealized_helix only, 8 runs/structure
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from Bio.PDB import PDBParser

from md_refinement import NESMDRefiner
from crystal_sanity_check import CRYSTAL_STRUCTURES

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / 'nes_data_pipeline'))
from structural_dataset_v2_pipeline import fetch_alphafold_pdb  # noqa: E402

NATIVE_PDB_CACHE_DIR = THIS_DIR / 'crm1_reference' / 'native_pdb_cache'

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}

# Per-structure source for the 'native' (isolated, unbound, own-protein)
# starting conformation. accession=None means "not identified -- native is
# skipped for this structure, crystal/idealized_helix still run fine."
NATIVE_SOURCES = {
    '3NBY': {
        # PKI-alpha -- CORRECTED, TWICE. This project's own
        # reference data (crystal_sanity_check.py's comment AND
        # nes_dataset.json) both cited P04541 as human PKIA's accession --
        # confirmed WRONG via live web search when the AlphaFold fetch for
        # P04541 came back empty: P04541 is not human PKIA. Real accession
        # is P61925 (UniProt/GeneCards/NCBI Gene 5569 all agree). Second
        # pass: kept the old (37, 46) numbering at first, but runtime
        # self-verification caught a one-residue offset -- P61925 residues
        # 37-46 gave 'ELALKLAGLD', not the expected 'LALKLAGLDI'. Shifting
        # by +1 to (38, 47) exactly resolves it (drop the leading E, the
        # real sequence continues...LALKLAGLDI at 38-47) -- P61925's own
        # numbering is offset by +1 vs. the old P04541-based convention,
        # not just a different accession for identical numbering.
        'accession': 'P61925',
        'native_range': (38, 47),
        'confidence': 'verified',
    },
    '3GJX': {
        'accession': 'O95149',  # Snurportin1 -- VERIFIED via this
        'native_range': (4, 15),  # structure's own DBREF (chain B -> UNP
        'confidence': 'verified',  # O95149, NO numbering offset)
    },
    '3GB8': {
        'accession': 'O95149',  # same as 3GJX (replicate structure),
        'native_range': (4, 15),  # DBREF also confirms no offset
        'confidence': 'verified',
    },
    '3NBZ': {
        # HIV-1 Rev (REV_HV1C4) -- GUESS. Cross-referenced against TWO
        # independent nes_dataset.json/NESbase entries that agree on
        # sequence LPPLERLTL at residues 75-83 (one entry gives that exact
        # range+sequence directly; a second, longer entry's own sequence
        # LQLPPLERLTLD contains LPPLERLTL starting at its own position 75).
        # NOT confirmed against 3NBZ's own DBREF (chain B's DBREF describes
        # the Snurportin1 SCAFFOLD, not the grafted Rev insert -- see
        # module docstring). Self-verified at runtime regardless.
        'accession': 'P05865',
        'native_range': (75, 83),
        'confidence': 'guess_range_crosschecked',
    },
    '3NC0': {
        # Same donor protein as 3NBZ (replicate structure), but this
        # crystal form's peptide is ONE RESIDUE LONGER (LPPLERLTLS vs
        # LPPLERLTL) -- the +1 extension was NOT independently
        # cross-referenced (the NESbase entry used for 3NBZ ends in...D at
        # that position, not the S seen here, possibly a different Rev
        # isolate/strain numbering). LOWER confidence than 3NBZ; if
        # verification fails here specifically, that mismatch is the
        # likely reason, not a script bug.
        'accession': 'P05865',
        'native_range': (75, 84),
        'confidence': 'guess',
    },
    '5UWH': {
        # Paxillin (PXN_HUMAN) -- GUESS, accession not cross-referenced
        # against an external source this project.
        'accession': 'P49023',
        'native_range': (269, 278),
        'confidence': 'guess',
    },
    '5UWU': {
        # SMAD4 (SMA4_HUMAN) -- accession is a GUESS, but native_range was
        # cross-referenced against a nes_dataset.json human-SMAD4 entry
        # (sequence GIDLSGLTLQ at residues 140-149; this structure's 9-mer
        # IDLSGLTLQ is that same motif minus the leading G, i.e. 141-149).
        'accession': 'Q13485',
        'native_range': (141, 149),
        'confidence': 'guess_range_crosschecked',
    },
    '5UWS': {
        # X11L2 / APBA3 (human) -- accession is a GUESS. native_range
        # WIDENED, second pass) from the crystal-bounded
        # (65, 72) to (57, 71) -- the crystallized/resolved density only
        # covers 65-72 (confirmed by reading NES_peptide_5UWS_chainD.pdb
        # directly, no atoms present outside that range), so 'crystal'
        # conformation is stuck with the narrow window regardless. But
        # 'native'/'idealized_helix' don't need real crystal density at
        # all -- and the paper's own class-4 register (Fung & Chook 2017,
        # Phi0-x2-Phi1-x3-Phi2-x2-Phi3-x3-Phi4) needs 5 anchors spanning
        # 15 residues, which the narrow window can't contain (it's missing
        # the true Phi0/Phi1 anchors at 57/60 entirely). Verified via
        # nes_dataset.json's broader X11L2 entry (SSLQELVQQFEALPGDLVG at
        # 55-73) that residues 57/60/64/67/71 (L/L/F/L/L) match the class-4
        # pattern exactly -- see md_refinement.py's PHI_REGISTER_CLASS4_RE
        # comment for the full derivation. 'verify_against' overrides what
        # the fetched accession is checked against (the WIDE sequence, not
        # cfg['sequence'], which is still the narrow crystal one) -- still
        # self-verified at runtime, not trusted blind.
        'accession': 'O96018',
        'native_range': (57, 71),
        'verify_against': 'LQELVQQFEALPGDL',
        'confidence': 'guess_range_crosschecked',
    },
    '5DHF': {
        # RIOK2 / hRio2 (human) -- accession is a GUESS, but native_range
        # is HIGH confidence: crystal_sanity_check.py's own pre-existing
        # comment on this structure already states the range "matches
        # paper's own hRio2 numbering" (Fung, Fu, Chook 2015).
        'accession': 'Q9BVS4',
        'native_range': (394, 400),
        'confidence': 'guess_range_trusted',
    },
    '5DIF': {
        # CPEB4 (human) -- accession is a GUESS, native_range HIGH
        # confidence (same reasoning as 5DHF -- pre-existing comment
        # confirms this range matches the paper's own CPEB4 numbering).
        'accession': 'Q17RY0',
        'native_range': (383, 393),
        'confidence': 'guess_range_trusted',
    },
}


def get_native_pdb_text(accession):
    """Fetch (or reuse on-disk cache of) the AlphaFold isolated-state
    model for `accession`. Returns None on any failure -- caller must
    treat that as "native unavailable for this structure", not an error
    to propagate."""
    NATIVE_PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = NATIVE_PDB_CACHE_DIR / f'{accession}.pdb'
    if cache_file.exists():
        return cache_file.read_text()
    print(f"    fetching AlphaFold model for {accession}...")
    try:
        pdb_text = fetch_alphafold_pdb(accession)
    except Exception as e:
        print(f"    Warning: fetch failed for {accession}: {e}")
        return None
    if pdb_text is None:
        print(f"    Warning: no AlphaFold structure available for {accession}")
        return None
    cache_file.write_text(pdb_text)
    return pdb_text


def extract_sequence_slice(pdb_text, start, end):
    """Returns the 1-letter sequence for residues [start, end] (inclusive,
    1-indexed, matching the PDB's own residue numbering) from `pdb_text`,
    or None if any residue in that range is missing. Assumes a single
    protein chain (true for AlphaFold-DB models)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp.write(pdb_text)
        tmp_path = tmp.name
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('x', tmp_path)
    model = next(iter(structure))
    chain = next(iter(model))
    by_resnum = {}
    for residue in chain:
        if residue.id[0] != ' ':
            continue
        by_resnum[residue.id[1]] = residue.resname
    residues = [by_resnum.get(i) for i in range(start, end + 1)]
    if any(r is None for r in residues):
        return None
    return ''.join(THREE_TO_ONE.get(r, 'X') for r in residues)


def verify_native_slice(sid, expected_sequence):
    """Fetches this structure's native source (if any), slices it, and
    checks the slice EXACTLY matches expected_sequence (the structure's
    own known crystallized peptide sequence, from CRYSTAL_STRUCTURES).
    Returns (pdb_text, native_range) on success, (None, None) on any
    failure -- always printed either way, never silent."""
    src = NATIVE_SOURCES.get(sid)
    if src is None or src.get('accession') is None:
        print(f"    native: no source configured for {sid} -- skipping native conformation")
        return None, None
    accession = src['accession']
    native_range = src['native_range']
    pdb_text = get_native_pdb_text(accession)
    if pdb_text is None:
        print(f"    native: could not fetch {accession} -- skipping native conformation for {sid}")
        return None, None
    actual = extract_sequence_slice(pdb_text, *native_range)
    if actual is None:
        print(f"    native: {accession} has a gap across {native_range} -- skipping native conformation for {sid}")
        return None, None
    if actual != expected_sequence:
        print(f"    native: Warning: VERIFICATION FAILED for {sid} -- {accession} residues "
              f"{native_range[0]}-{native_range[1]} = '{actual}', expected '{expected_sequence}' "
              f"(confidence was '{src['confidence']}'). Skipping native conformation for {sid} -- "
              f"do NOT trust this accession/range without further checking.")
        return None, None
    print(f"    native: verified -- {accession} residues {native_range[0]}-{native_range[1]} "
          f"= '{actual}' (confidence label: '{src['confidence']}')")
    return pdb_text, native_range


def run_one(refiner, sid, cfg, conformation, duration_ns, scramble, relax_sidechains,
            native_pdb_text=None, native_range=None, native_sequence=None,
            wide_sequence=None, wide_range=None):
    """Dispatches to NESMDRefiner._run_crm1_docking with the right
    arguments for whichever of the three starting conformations is being
    tested this call. Returns the md_metrics dict (never raises -- errors
    are already handled/converted to a helix-fallback dict deep inside
    _run_crm1_docking itself).

    wide_sequence/wide_range : only set for structures whose
    NATIVE_SOURCES entry has a 'verify_against' override -- i.e. ones
    where the crystal-bounded cfg['sequence'] is too narrow to contain a
    full Phi-anchor register (e.g. 5UWS's class-4 register needs 57-71,
    the crystal only resolved 65-72). 'idealized_helix' has no real
    coordinates to respect either way (it's built purely from sequence),
    so it can safely use the wider, verified window instead of the
    crystal-bounded one. 'crystal' is deliberately NEVER given this --
    it's built from cfg['peptide_pdb'], which only physically has atoms
    for the narrow resolved range, so widening it here would do nothing
    but silently mismatch the sequence against the real coordinates."""
    if conformation == 'crystal':
        crystal_pdb_text = cfg['peptide_pdb'].read_text()
        candidate = {
            'sequence': cfg['sequence'],
            'start': cfg['residue_range'][0],
            'end': cfg['residue_range'][1],
            'full_sequence': None,
            'combined_score': 0.5,
        }
        result = refiner._run_crm1_docking(
            pdb_content=crystal_pdb_text,  # unused by the 'crystal' branch itself
            candidate=candidate,
            duration_ns=duration_ns,
            starting_conformation='crystal',
            scramble_registration=scramble,
            crystal_pdb_text=crystal_pdb_text,
            relax_sidechains=relax_sidechains,
        )
    elif conformation == 'idealized_helix':
        use_sequence = wide_sequence if wide_sequence is not None else cfg['sequence']
        use_start, use_end = wide_range if wide_range is not None else cfg['residue_range']
        candidate = {
            'sequence': use_sequence,
            'start': use_start,
            'end': use_end,
            'full_sequence': None,
            'combined_score': 0.5,
        }
        result = refiner._run_crm1_docking(
            pdb_content='',  # unused by the idealized_helix branch
            candidate=candidate,
            duration_ns=duration_ns,
            starting_conformation='idealized_helix',
            scramble_registration=scramble,
            relax_sidechains=relax_sidechains,
        )
    elif conformation == 'native':
        # native_sequence defaults to cfg['sequence'] -- for every
        # structure except 5UWS-style widened ones, native_range spans the
        # same length as cfg['sequence'] so this was previously a no-op.
        # For a widened structure, native_range is LONGER than
        # cfg['sequence'] fix) -- using cfg['sequence'] here
        # would silently mismatch length against native_range and corrupt
        # the candidate, so the actual verified sequence must be used.
        seq_for_native = native_sequence if native_sequence is not None else cfg['sequence']
        candidate = {
            'sequence': seq_for_native,
            'start': native_range[0],
            'end': native_range[1],
            'full_sequence': None,
            'combined_score': 0.5,
        }
        result = refiner._run_crm1_docking(
            pdb_content=native_pdb_text,
            candidate=candidate,
            duration_ns=duration_ns,
            starting_conformation='native',
            scramble_registration=scramble,
            relax_sidechains=relax_sidechains,
        )
    else:
        raise ValueError(f"unknown conformation {conformation!r}")

    return result.get('md_metrics', {}) or {}


def _summarize(label, metrics):
    print(f"\n  [{label}]")
    for key in ('anchor_occupancy_score', 'avg_anchor_pocket_distance_nm', 'anchor_fit_rmsd_nm',
                'raw_binding_score', 'binding_score', 'avg_groove_contacts',
                'avg_hydrophobic_contacts', 'avg_cys528_distance_nm', 'helix_combined_score'):
        print(f"    {key}: {metrics.get(key)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=2.0,
                     help='MD duration for each run. Same default used elsewhere in this project for comparability.')
    ap.add_argument('--structures', default=','.join(CRYSTAL_STRUCTURES.keys()),
                     help='Comma-separated subset of CRYSTAL_STRUCTURES keys to run (default: all).')
    ap.add_argument('--skip-native', action='store_true',
                     help='Only run crystal + idealized_helix (8 runs/structure instead of 12) -- '
                          'skips all AlphaFold fetching/verification entirely.')
    ap.add_argument('--out', default='crystal_full_grid_results.json')
    args = ap.parse_args()

    struct_ids = [s.strip() for s in args.structures.split(',') if s.strip()]

    missing = []
    for sid in struct_ids:
        cfg = CRYSTAL_STRUCTURES[sid]
        if not cfg['crm1_pdb'].exists() or not cfg['peptide_pdb'].exists():
            missing.append(sid)
    if missing:
        print(f"Missing reference files for: {missing} -- check crm1_reference/. Aborting.")
        return

    conformations = ['crystal', 'idealized_helix'] if args.skip_native else ['native', 'idealized_helix', 'crystal']

    # Resume/merge: if --out already exists (e.g. from an earlier partial
    # run covering a different --structures subset), load it and merge new
    # structures in rather than overwriting -- this is what makes "test on
    # 3 structures, look at it, then continue with the rest" safe to do as
    # two separate invocations with the SAME --out path. A structure
    # re-run this call (already present in the loaded file) is simply
    # replaced with its fresh result, not skipped -- if that's not what
    # you want, use a different --structures list or --out path.
    all_results = {}
    out_path = Path(args.out)
    if out_path.exists():
        try:
            all_results = json.loads(out_path.read_text())
            print(f"Resuming: loaded {len(all_results)} structure(s) already in {out_path} "
                  f"({', '.join(all_results.keys())}) -- new/re-run structures will be merged in.")
        except (json.JSONDecodeError, OSError) as e:
            backup = out_path.with_suffix(out_path.suffix + '.corrupt')
            print(f"    {out_path} is unreadable ({e}) -- saving it as {backup.name} and starting fresh.")
            out_path.rename(backup)
            all_results = {}

    for sid in struct_ids:
        cfg = CRYSTAL_STRUCTURES[sid]
        print("\n" + "#" * 70)
        print(f"# {sid}: {cfg['label']}")
        print("#" * 70)
        print(f"CRM1+RanGTP reference: {cfg['crm1_pdb'].name}")
        print(f"Crystal peptide: {cfg['peptide_pdb'].name}  sequence={cfg['sequence']}")

        # 'verify_against' lets a structure verify its native fetch against
        # a WIDER expected sequence than cfg['sequence'] (see 5UWS, added
        # - default is cfg['sequence'] so every other
        # structure's behavior is completely unchanged.
        src_cfg = NATIVE_SOURCES.get(sid, {})
        expected_seq = src_cfg.get('verify_against', cfg['sequence'])

        native_pdb_text, native_range = (None, None)
        if 'native' in conformations:
            print("  Verifying native-conformation source before running anything...")
            native_pdb_text, native_range = verify_native_slice(sid, expected_seq)

        # Only pass a wide window into idealized_helix if this structure
        # actually declared one (verify_against present) AND verification
        # against it succeeded -- otherwise wide_sequence/wide_range stay
        # None and idealized_helix falls back to the normal crystal-bounded
        # cfg['sequence']/residue_range, exactly as before this change.
        wide_sequence, wide_range = (None, None)
        if 'verify_against' in src_cfg and native_pdb_text is not None:
            wide_sequence, wide_range = expected_seq, native_range
            print(f"  idealized_helix will use the WIDENED window {wide_range} "
                  f"('{wide_sequence}') instead of the crystal-bounded {cfg['residue_range']}.")

        refiner = NESMDRefiner(crm1_pdb_path=str(cfg['crm1_pdb']))
        all_results[sid] = {}

        for conformation in conformations:
            if conformation == 'native' and native_pdb_text is None:
                print(f"\n  (skipping all native/* runs for {sid} -- see verification message above)")
                continue
            for relax in (False, True):
                for scramble in (False, True):
                    run_label = (f"{conformation}_{'scrambled' if scramble else 'correct'}_"
                                 f"{'relaxed' if relax else 'norelax'}")
                    print("\n" + "=" * 70)
                    print(f"{sid} / {run_label}")
                    print("=" * 70)
                    metrics = run_one(refiner, sid, cfg, conformation, args.duration_ns,
                                       scramble, relax,
                                       native_pdb_text=native_pdb_text, native_range=native_range,
                                       native_sequence=expected_seq,
                                       wide_sequence=wide_sequence, wide_range=wide_range)
                    _summarize(run_label, metrics)
                    all_results[sid][run_label] = metrics

        # Save incrementally after each structure finishes, so a crash/
        # interrupt partway through the full grid doesn't lose everything.
        Path(args.out).write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n(checkpoint saved to {args.out} after {sid})")

    print("\n" + "#" * 70)
    print("# VERDICT (per structure, per conformation, per relaxation condition)")
    print("#" * 70)
    # Iterate every structure ACCUMULATED in all_results so far (not just
    # struct_ids from this invocation) -- a continuation run's verdict
    # should cover everything in --out to date, including structures done
    # in an earlier invocation. Conformations actually run for a given
    # structure are inferred from its own result keys, since --skip-native
    # may have differed between invocations that share the same --out.
    all_sids_in_order = [sid for sid in CRYSTAL_STRUCTURES if sid in all_results]
    for sid in all_sids_in_order:
        cfg = CRYSTAL_STRUCTURES[sid]
        run_conformations = sorted({key.rsplit('_', 2)[0] for key in all_results[sid].keys()})
        print(f"\n{sid} ({cfg['label']}):")
        for conformation in run_conformations:
            for relax in (False, True):
                suffix = 'relaxed' if relax else 'norelax'
                c = all_results[sid].get(f'{conformation}_correct_{suffix}', {})
                s = all_results[sid].get(f'{conformation}_scrambled_{suffix}', {})
                c_occ, s_occ = c.get('anchor_occupancy_score'), s.get('anchor_occupancy_score')
                c_rbs, s_rbs = c.get('raw_binding_score'), s.get('raw_binding_score')
                if not c and not s:
                    continue
                print(f"  {conformation} / relax_sidechains={relax}:")
                if c_occ is not None and s_occ is not None:
                    print(f"    anchor_occupancy_score: correct={c_occ:.3f}  scrambled={s_occ:.3f}  "
                          f"gap={c_occ - s_occ:+.3f}")
                if c_rbs is not None and s_rbs is not None:
                    print(f"    raw_binding_score:      correct={c_rbs:.3f}  scrambled={s_rbs:.3f}  "
                          f"gap={c_rbs - s_rbs:+.3f}")


if __name__ == '__main__':
    main()
