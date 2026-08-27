#!/usr/bin/env python3
"""
crystal_sanity_check.py  (generalized )
============================================================
Ground-truth diagnostic for the CRM1/NES MD pipeline.

WHY THIS EXISTS: four independent tests earlier in this project (anchor_occupancy_score
at n=43, canonical Kosugi/LocNES spacing-class matching, initial-vs-final
pose drift, and the correct-vs-scrambled specificity control in
evaluate_anchor_occupancy_signal.py) all came back either flat or actively
BACKWARDS -- hard negatives (coiled-coil/leucine-zipper fragments) packed
tighter, made more contacts, and even showed a bigger "correct beats
scrambled" advantage than real NES motifs did. Before concluding anything
about which candidates are real binders, this script asks a more basic
question: does this pipeline's own scoring even recognize a peptide *known
by direct experimental evidence* to be correctly bound?

WHAT IT DOES (now covers THREE independent real crystal structures, not
just one, so the verdict below isn't resting on a single data point):

  1. PKI-alpha / 3NBY -- Guttler et al. 2010 (Nat Struct Mol Biol
     17:1367-1376). Classic class-1 alpha-helical NES, LALKLAGLDI
     (PKI-alpha P04541 residues 37-46). Sequence verified directly against
     the file's SEQRES/ATOM records earlier in this project.
  2. Snurportin1 / 3GJX -- Dong et al. 2009 (Nature 458:1136-1141). Real,
     full-length Snurportin1's own N-terminal groove-engaging segment,
     residues 5-15 = SQALASSFSVS -- identified geometrically (
     extract_crystal_references.py: sliding-window search for the chain B
     stretch with lowest average CA-CA distance to the CRM1 groove-lining
     residues, since a flat-cutoff approach fragmented the real segment).
     Documented in the literature as a non-canonical/atypical NES, so a
     less clean hydrophobic spacing pattern than PKI-NES is expected.
  3. HIV-1 Rev NES / 3NBZ -- Guttler et al. 2010, same paper as #1, "crystal
     I". Residues 6-14 = LPPLERLTL, identified the same geometric way.
     Notably this NES binds in an EXTENDED, proline-containing
     conformation, NOT the alpha-helical mode PKI-NES uses -- the found
     sequence matches the well-known Rev NES motif, an independent
     confirmation the geometric identification is finding something real.

For each structure, runs _run_crm1_docking across a small grid:
  - registration: correct vs. scrambled (cyclic-shift negative control)
  - relax_sidechains: True vs. False (the restrained-backbone/free-
    sidechain relaxation phase added, motivated by the original
    3NBY-only check showing native/idealized_helix starting poses have
    worse (generic PDBFixer-filled) rotamers than a real crystal structure
    -- but which the v2 regression check showed ALSO inflates the
    scrambled control's score, narrowing (not eliminating) the gap)

That's 3 structures x 2 registration x 2 relaxation = 12 runs total.

READING THE OUTPUT: if the pipeline's scoring is doing anything like what
it's supposed to, each structure's real crystal pose should score clearly
better than its own scrambled counterpart, in both relaxation conditions
(though the gap may narrow under relax_sidechains=True, per the earlier
3NBY-only finding). Consistent gaps across three independent structures is
much stronger evidence than one; if the gap only shows up for one of the
three, that's important to know too.

USAGE:
    python3 crystal_sanity_check.py --duration-ns 2.0
    python3 crystal_sanity_check.py --duration-ns 2.0 --structures 3NBY,3NBZ
"""

import argparse
import json
from pathlib import Path

from md_refinement import NESMDRefiner

THIS_DIR = Path(__file__).resolve().parent
REF_DIR = THIS_DIR / 'crm1_reference'

# One entry per real crystal ground-truth structure. 'sequence' MUST match
# the peptide file's actual residue count/order -- it's used by
# _place_peptide_via_subpocket_registration for anchor tracking even when
# apply_transform=False (crystal + not scrambled), so a mismatch would
# silently corrupt indexing rather than error out.
CRYSTAL_STRUCTURES = {
    '3NBY': {
        'label': 'PKI-alpha NES (Guttler et al. 2010)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_3NBY.pdb',
        'peptide_pdb': REF_DIR / 'PKI_NES_peptide_3NBY_chainB_4-13.pdb',
        'sequence': 'LALKLAGLDI',
        'residue_range': (37, 46),  # PKI-alpha P04541 numbering, reference only
    },
    # Sequence/residue_range corrected -- the original
    # extraction (SQALASSFSVS, residues 5-15) trimmed off a real, paper-
    # confirmed anchor residue (Leu4/Phi1; Dong et al. 2009's own Fig 1C,
    # cross-referenced in Guttler et al. 2010's Fig 1C, places Snurportin1's
    # real Phi0-Phi4 register at Met1/Leu4/Leu8/Phe12/Val14). Root cause:
    # extract_crystal_references.py's window picker minimized AVERAGE
    # distance-to-groove across the window, and an intervening linker
    # residue (Glu3, 9.81 A) dragged that average up enough that trimming
    # off the real Leu4 anchor scored better on paper -- even though Leu4
    # (7.22 A) was itself closer to the groove than several residues the
    # narrower window already included anyway. Fixed in
    # find_groove_contacting_segment (now expands the window when doing so
    # completes a real 4-anchor Phi-register match); re-extracted and
    # re-verified this recovers Leu4 and gives a full (not partial) match.
    '3GJX': {
        'label': 'Snurportin1 NES-like segment (Dong et al. 2009)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_3GJX.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_3GJX_chainB.pdb',
        'sequence': 'LSQALASSFSVS',
        'residue_range': (4, 15),  # local chain B numbering, reference only
    },
    '3NBZ': {
        'label': 'HIV-1 Rev NES (Guttler et al. 2010, crystal I)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_3NBZ.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_3NBZ_chainB.pdb',
        'sequence': 'LPPLERLTL',
        'residue_range': (6, 14),  # local chain B numbering, reference only
    },
    # Reproducibility-check structures -- NOT new independent
    # NES sequences, separate independently-solved crystal FORMS of the
    # same two complexes above (3NC0 replicates 3NBZ, 3GB8 replicates
    # 3GJX). Their raw per-residue distance profiles matched the originals
    # almost exactly when extracted (extract_crystal_references.py output,
    # - included here to test whether findings from the
    # single-crystal-form check (e.g. idealized_helix failing badly on
    # these non-canonical cases) hold up in an independent structure.
    '3NC0': {
        'label': 'HIV-1 Rev NES (Guttler et al. 2010, crystal II -- replicate of 3NBZ)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_3NC0.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_3NC0_chainB.pdb',
        'sequence': 'LPPLERLTLS',
        'residue_range': (6, 15),  # local chain B numbering, reference only
    },
    # Same window-trimming fix as 3GJX above -- re-extracted,
    # now recovers the identical corrected sequence/range as 3GJX (a good
    # consistency check, since this structure really is the same construct
    # in a different crystal form).
    '3GB8': {
        'label': 'Snurportin1 (Dong et al. 2009, alternate crystal form -- replicate of 3GJX)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_3GB8.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_3GB8_chainB.pdb',
        'sequence': 'LSQALASSFSVS',
        'residue_range': (4, 15),  # local chain B numbering, reference only
    },
    # Three NEW independent NES sequences (not replicates) from
    # Fung et al. 2017 eLife 6:e23961 -- added specifically to get more
    # real ground truth for testing the relaxed 3-of-4 Phi-anchor register
    # match (md_refinement.py's classify_nes_binding_mode). [
    # update: at the time this was written, Snurportin1 (3GJX/3GB8, above)
    # was this project's only real partial-match case -- since corrected;
    # see 3GJX's note above. It turned out to be a full match all along,
    # just extracted one residue short. 5UWS below is now the more
    # relevant real partial/no-match example, per the paper's own note
    # that it uses a different 5-anchor spacing pattern.] This deposition
    # batch uses a DIFFERENT chain lettering
    # (CRM1=C, peptide=D, not A/B) AND a different CRM1 residue numbering
    # (+11 offset vs this project's convention) than the Guttler/Dong-era
    # structures above -- extract_crystal_references.py auto-detected and
    # corrected both before writing these files out, so 'residue_range'
    # below is already in THIS project's numbering convention, same as the
    # others.
    # Sequence/residue_range corrected -- same window-
    # trimming bug as 3GJX/3GB8, expanded by 2 residues (through Lys277/
    # Phe278) to complete a full 4-anchor match.
    '5UWH': {
        'label': 'Paxillin NES (Fung et al. 2017)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_5UWH.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_5UWH_chainD.pdb',
        'sequence': 'LMASLSDFKF',
        'residue_range': (269, 278),  # local chain D numbering, reference only
    },
    # Sequence/residue_range corrected -- same fix, expanded
    # by 2 residues (Ile141/Asp142 added on the N-terminal side) to
    # complete a full 4-anchor match.
    '5UWU': {
        'label': 'SMAD4 NES (Fung et al. 2017)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_5UWU.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_5UWU_chainD.pdb',
        'sequence': 'IDLSGLTLQ',
        'residue_range': (141, 149),  # local chain D numbering, reference only
    },
    # 5UWS: sequence/residue_range UNCHANGED by the window fix --
    # tried expanding up to 6 residues each side (22 total) and still found
    # no full 4-anchor Phi-register match, consistent with the paper's own
    # description of this one as using a genuinely different 5-anchor
    # spacing pattern this project's regex isn't built to catch. Real
    # partial/no-match example, not a truncation artifact like the others.
    '5UWS': {
        'label': 'X11L2 NES (Fung et al. 2017, novel class 4 spacing pattern)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_5UWS.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_5UWS_chainD.pdb',
        'sequence': 'EALPGDLV',
        'residue_range': (65, 72),  # local chain D numbering, reference only
    },
    # HRio2NES and CPEB4NES (Fung, Fu, Chook 2015, eLife
    # 4:e10034) -- NOT more data on the two binding modes already covered
    # above (alpha-helical PKI-type, extended-proline Rev-type). These bind
    # the CRM1 groove in the OPPOSITE polarity ("minus" direction -- N/C
    # termini reversed relative to every other structure in this set) while
    # still using the same P0-P4 pockets, per the paper's central finding.
    # A genuinely third, structurally distinct binding mode.
    # CAUTION: deposited using an ENGINEERED S. cerevisiae CRM1 construct
    # (ScCRM1 1-1058, Delta377-413, V441D, groove region 537DLTVK541
    # mutated to GLCEQ), not wild-type human CRM1 like every other
    # structure here -- extract_crystal_references.py's exact 19-residue
    # groove fingerprint DID match (with the same +11 numbering offset as
    # the other Fung-lab depositions), and the extracted sequences (TEFNQAL,
    # MHSLESSL) are exact substrings of the paper's own reported full NES
    # sequences at the paper's own residue numbers -- real confirmation,
    # not just a numbering coincidence -- but this is still yeast CRM1, not
    # human, so treat any MD result here as slightly less directly
    # generalizable than the others.
    # 5DHF: sequence/residue_range UNCHANGED by the window fix --
    # tried expanding up to 6 residues each side and still found no full
    # 4-anchor Phi-register match. Real no-match case (also already known
    # from this project's MD runs: TEFNQAL never registers, flat
    # raw_binding_score, no anchor_occupancy_score at all), not a
    # truncation artifact.
    '5DHF': {
        'label': 'hRio2 NES (Fung, Fu, Chook 2015 -- minus/reverse-direction binding mode, engineered ScCRM1*)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_5DHF.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_5DHF_chainD.pdb',
        'sequence': 'TEFNQAL',
        'residue_range': (394, 400),  # matches paper's own hRio2 numbering
    },
    # Sequence/residue_range corrected -- same window-trimming
    # bug, expanded by 3 residues (through Asp392/Ile393) to complete a
    # full 4-anchor match. This was THE critical validation case for the
    # forward/reversed anchor-to-pocket orientation fix (see
    # md_refinement.py's _best_orientation_matched) -- re-verified locally
    # that the orientation logic still correctly picks 'reversed' for this
    # new, longer sequence before deploying this correction (register
    # indices shift when the window changes, so this needed re-checking,
    # not just assuming the old result still applied). The extra 3
    # residues (Ile391/Asp392/Ile393) are real, contiguous deposited
    # sequence, same as the rest -- not fabricated to force a match.
    '5DIF': {
        'label': 'CPEB4 NES (Fung, Fu, Chook 2015 -- minus/reverse-direction binding mode, engineered ScCRM1*)',
        'crm1_pdb': REF_DIR / 'CRM1_Ran_5DIF.pdb',
        'peptide_pdb': REF_DIR / 'NES_peptide_5DIF_chainD.pdb',
        'sequence': 'MHSLESSLIDI',
        'residue_range': (383, 393),  # matches paper's own CPEB4 numbering
    },
}


def _summarize(label, metrics):
    print(f"\n  [{label}]")
    for key in ('anchor_occupancy_score', 'avg_anchor_pocket_distance_nm', 'anchor_fit_rmsd_nm',
                'raw_binding_score', 'binding_score', 'avg_groove_contacts',
                'avg_hydrophobic_contacts', 'avg_cys528_distance_nm', 'helix_combined_score'):
        print(f"    {key}: {metrics.get(key)}")


def run_one(refiner, cfg, duration_ns, scramble, relax_sidechains):
    crystal_pdb_text = cfg['peptide_pdb'].read_text()
    candidate = {
        'sequence': cfg['sequence'],
        'start': cfg['residue_range'][0],
        'end': cfg['residue_range'][1],
        'full_sequence': None,
        'combined_score': 0.5,
    }
    result = refiner._run_crm1_docking(
        pdb_content=crystal_pdb_text,  # unused by the 'crystal' branch itself, harmless placeholder
        candidate=candidate,
        duration_ns=duration_ns,
        starting_conformation='crystal',
        scramble_registration=scramble,
        crystal_pdb_text=crystal_pdb_text,
        relax_sidechains=relax_sidechains,
    )
    return result.get('md_metrics', {}) or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=2.0,
                     help='MD duration for each run. Same default used elsewhere in this project for comparability.')
    ap.add_argument('--structures', default='3NBY,3GJX,3NBZ',
                     help='Comma-separated subset of CRYSTAL_STRUCTURES keys to run (default: all three).')
    ap.add_argument('--out', default='crystal_sanity_check_results_v3.json')
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

    all_results = {}

    for sid in struct_ids:
        cfg = CRYSTAL_STRUCTURES[sid]
        print("\n" + "#" * 70)
        print(f"# {sid}: {cfg['label']}")
        print("#" * 70)
        print(f"CRM1+RanGTP reference: {cfg['crm1_pdb'].name}")
        print(f"Crystal peptide: {cfg['peptide_pdb'].name}  sequence={cfg['sequence']}")

        refiner = NESMDRefiner(crm1_pdb_path=str(cfg['crm1_pdb']))
        all_results[sid] = {}

        for relax in (False, True):
            for scramble in (False, True):
                run_label = f"{'scrambled' if scramble else 'correct'}_{'relaxed' if relax else 'norelax'}"
                print("\n" + "=" * 70)
                print(f"{sid} / {run_label}  (registration={'scrambled' if scramble else 'correct'}, "
                      f"relax_sidechains={relax})")
                print("=" * 70)
                metrics = run_one(refiner, cfg, args.duration_ns, scramble, relax)
                _summarize(run_label, metrics)
                all_results[sid][run_label] = metrics

        # Save incrementally after each structure finishes, so a crash/interrupt
        # partway through the full grid doesn't lose everything already run.
        Path(args.out).write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n(checkpoint saved to {args.out} after {sid})")

    print("\n" + "#" * 70)
    print("# VERDICT (per structure, per relaxation condition)")
    print("#" * 70)
    for sid in struct_ids:
        cfg = CRYSTAL_STRUCTURES[sid]
        print(f"\n{sid} ({cfg['label']}):")
        for relax in (False, True):
            suffix = 'relaxed' if relax else 'norelax'
            c = all_results[sid].get(f'correct_{suffix}', {})
            s = all_results[sid].get(f'scrambled_{suffix}', {})
            c_occ, s_occ = c.get('anchor_occupancy_score'), s.get('anchor_occupancy_score')
            c_rbs, s_rbs = c.get('raw_binding_score'), s.get('raw_binding_score')
            print(f"  relax_sidechains={relax}:")
            if c_occ is not None and s_occ is not None:
                print(f"    anchor_occupancy_score: correct={c_occ:.3f}  scrambled={s_occ:.3f}  "
                      f"gap={c_occ - s_occ:+.3f}")
            if c_rbs is not None and s_rbs is not None:
                print(f"    raw_binding_score:      correct={c_rbs:.3f}  scrambled={s_rbs:.3f}  "
                      f"gap={c_rbs - s_rbs:+.3f}")

    print(f"\nFull results saved to {args.out}")
    print("\nConsistent positive gaps (correct > scrambled) across all three structures is much stronger "
          "evidence the pipeline's scoring recognizes genuine binding than the original single-structure "
          "(3NBY-only) check. If the gap holds for some structures but not others, that's informative too -- "
          "e.g. it might mean the scoring is tuned toward alpha-helical (PKI-like) binding and less sensitive "
          "to the extended/atypical modes (Rev-NES, Snurportin1).")


if __name__ == '__main__':
    main()
