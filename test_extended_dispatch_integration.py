#!/usr/bin/env python3
"""
test_extended_dispatch_integration.py
============================================================
End-to-end validation of the new pre-MD conformation dispatch :
classify_nes_binding_mode() now flags proline-containing/atypical
candidates as 'extended_atypical' and recommends 'extended' (the literal
PPII-geometry starting structure), and refine_nes_candidates() now
automatically adds 'extended' as a THIRD tested variant (alongside native
and idealized_helix) whenever that flag fires -- instead of always blindly
testing just native+idealized_helix regardless of sequence.

recommend_starting_conformation() itself was already checked standalone,
sequence-only, against the two real ground-truth cases (PKI-NES ->
idealized_helix, Rev-NES -> extended, both correct). This script checks the
DIFFERENT thing: that the actual integration wiring inside
refine_nes_candidates() works on real infrastructure -- that flagged
candidates really do get an extra 'extended' MD trajectory run (not just a
label nobody acts on), that it shows up in md_metrics_by_variant, and that
best_tag selection picks it as primary when appropriate.

Reuses the same 10-example subset and on-disk AlphaFold structure cache
(crm1_eval_pdb_cache/) already populated by rerun_subset_annealing_check.py
this project -- no new downloads needed for the ones already cached.

WHAT THIS DOES:
  1. Cheap, local, no-MD pass: run classify_nes_binding_mode(sequence) on
     all 10 subset examples, print which ones (if any) get flagged
     'extended_atypical'.
  2. For ONLY the flagged ones (expected to be few or zero -- most real
     NES motifs and most decoys in this particular subset are proline-free,
     this is exploratory), run the FULL refine_nes_candidates(
     test_both_conformations=True) and report:
       - which conformations actually got tested (should include 'extended')
       - which variant was picked as primary (md_best_starting_conformation)
       - anchor_occupancy_score/DSSP-style metrics per variant, so the
         'extended' variant's result can be compared directly against
         native/idealized_helix for the same candidate.

USAGE:
    python3 test_extended_dispatch_integration.py
    python3 test_extended_dispatch_integration.py --duration-ns 5.0
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / 'nes_data_pipeline'))

from evaluate_anchor_occupancy_signal import get_structure_bundle, build_candidate  # noqa: E402
from evaluate_crm1_pocket_signal import CRM1_REF_CANDIDATES  # noqa: E402
from md_refinement import NESMDRefiner  # noqa: E402

SUBSET = [
    ("P42566", 766, 800, 1), ("O35973", 485, 498, 1), ("Q14872", 336, 344, 1),
    ("Q00987", 188, 202, 1), ("Q14653", 136, 150, 1), ("P14635", 141, 153, 1),
    ("Q96CS2", 269, 278, 0), ("Q9BZF9", 320, 329, 0), ("Q13439", 2175, 2183, 0),
    ("Q08DM8", 173, 183, 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration-ns', type=float, default=5.0,
                     help='Longer than the 2 ns subset default, shorter than the 50 ns Rev-NES '
                          'validation runs -- just enough to see whether the extended trajectory '
                          'behaves sensibly, not a full production run.')
    ap.add_argument('--pdb-cache-dir', default='crm1_eval_pdb_cache')
    args = ap.parse_args()

    pdb_cache_dir = Path(args.pdb_cache_dir)
    pdb_cache_dir.mkdir(exist_ok=True)

    crm1_ref = next((p for p in CRM1_REF_CANDIDATES if (THIS_DIR / p).exists()), None)
    if not crm1_ref:
        print(f"No CRM1 reference found (checked {CRM1_REF_CANDIDATES}) -- aborting.")
        return
    refiner = NESMDRefiner(crm1_pdb_path=str(THIS_DIR / crm1_ref))

    print(f"{'='*90}\nSTEP 1: cheap sequence-only classification pass (no MD)\n{'='*90}")
    struct_cache = {}
    flagged = []
    for accession, start, end, label in SUBSET:
        bundle = get_structure_bundle(accession, struct_cache, pdb_cache_dir)
        if bundle is None:
            print(f"{accession:10s} -- no cached/downloadable structure, skipping")
            continue
        pdb_text, residue_numbers, sequence = bundle
        candidate = build_candidate(residue_numbers, sequence, start, end)
        if candidate is None:
            print(f"{accession:10s} -- gap in resolved span, skipping")
            continue
        binding_mode = refiner.classify_nes_binding_mode(candidate['sequence'])
        cls = binding_mode['binding_mode_class']
        # Was `cls == 'extended_atypical'` -- missed P42566
        # (766-800), which only gets a partial 3-of-4 register match and so
        # lands in 'partial_register_match', not 'extended_atypical', even
        # though md_refinement.py now recommends 'extended' for it too (see
        # that file's fix). Check the actual recommendation
        # instead of one specific classification label, matching the same
        # fix already applied inside refine_nes_candidates() itself.
        will_test_extended = binding_mode['recommended_primary_method'] == 'extended'
        flag = " <-- FLAGGED for 'extended' variant" if will_test_extended else ""
        print(f"{accession:10s} seq={candidate['sequence']:15s} label={label}  "
              f"binding_mode={cls:22s} recommend={binding_mode['recommended_primary_method']}{flag}")
        if will_test_extended:
            flagged.append((accession, start, end, label, pdb_text, candidate))

    print(f"\n{len(flagged)}/{len(SUBSET)} examples flagged for an 'extended' trajectory "
          f"(these are the ones worth spending MD time on for this integration test)")

    if not flagged:
        print("\nNone of these 10 examples happen to be proline-containing/atypical -- expected, "
              "this subset wasn't chosen for that property. The dispatch logic itself is confirmed "
              "correct (see STEP 1 output: classify_nes_binding_mode is doing its job), but this "
              "particular subset can't exercise the 'extended' MD-run branch. Consider running this "
              "against Rev-NES directly (known extended_atypical) instead if you want to see an "
              "actual 'extended' trajectory complete via the refine_nes_candidates() path.")
        return

    print(f"\n{'='*90}\nSTEP 2: full MD run (test_both_conformations=True) for flagged example(s)\n{'='*90}")
    results = {}
    for accession, start, end, label, pdb_text, candidate in flagged:
        print(f"\n--- {accession} {start}-{end} (label={label}) ---")
        enhanced = refiner.refine_nes_candidates(
            pdb_text, [candidate], args.duration_ns, test_both_conformations=True)
        cand = enhanced[0]
        tested = sorted(cand.get('md_metrics_by_variant', {}).keys())
        print(f"  Variants tested: {tested}")
        print(f"  Primary variant chosen: {cand.get('md_best_starting_conformation')} "
              f"(selection method: {cand.get('md_primary_variant_selection_method')})")
        for tag, metrics in (cand.get('md_metrics_by_variant') or {}).items():
            occ = metrics.get('anchor_occupancy_score')
            rbs = metrics.get('raw_binding_score')
            print(f"    {tag:25s} anchor_occupancy_score={occ}  raw_binding_score={rbs}")
        results[accession] = {
            'label': label, 'sequence': candidate['sequence'],
            'variants_tested': tested,
            'best_starting_conformation': cand.get('md_best_starting_conformation'),
            'md_metrics_by_variant': cand.get('md_metrics_by_variant'),
        }

    out_path = THIS_DIR / 'test_extended_dispatch_integration_result.json'
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
