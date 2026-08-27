#!/usr/bin/env python3
"""
run_ack1_v2_full_pipeline.py
============================================================
Re-runs the ACK1 (Q07912) LSSDFKRLGL (528-537) NES candidate through the
production refine_nes_candidates() path -- not a hand-picked single
starting conformation like the earlier run_ack1_nes_binding_pose.py /
run_rev_and_ack1_ramachandran.py scripts used -- so it picks up every
methodological change made this project automatically:

  1. Simulated annealing (300 K -> 450 K -> 300 K ramp before production)
     is now the DEFAULT inside _run_crm1_docking (use_simulated_annealing
     defaults True) -- no flag needed here, it just applies.
  2. classify_nes_binding_mode() + recommend_starting_conformation() now
     drive which starting conformation(s) actually get tested (previously
     always hardcoded to idealized_helix for this candidate) -- LSSDFKRLGL
     is expected to classify as 'likely_helical' (no proline, matched
     Phi-register) and so stick with native+idealized_helix, but this now
     happens via the real classifier rather than being assumed.
  3. Per-anchor hydrophobic-groove burial (anchor_burial_nm2,
     mean_anchor_burial_nm2, anchor_burial_fraction_well_buried) is now
     computed automatically inside _run_crm1_docking.
  4. Ramachandran (phi/psi) trace is now computed automatically inside
     _run_crm1_docking -- and, unlike the reference-set eval scripts, this
     one calls refine_nes_candidates() directly rather than going through
     evaluate_anchor_occupancy_signal.py's compute_features()/
     _extract_feature_fields() allowlist, so ramachandran_trace and the
     burial fields are NOT silently dropped this time.

REFERENCE STRUCTURE CHANGE FROM PRIOR ACK1 RUNS: prior ACK1 runs in this
project used CRM1.pdb (the generic reference), which this project's
sequence-alignment check confirmed is actually the full 3NBY biological
assembly -- it contains 2 copies of Snurportin-1 and Ran alongside CRM1,
not a CRM1-only structure. This run instead uses
crm1_reference/CRM1_Ran_only.pdb, confirmed by direct chain inspection
this project to contain ONLY chain A (CRM1, 1041 residues) and chain C
(Ran, 171 residues) -- no Snurportin-1. Because of this reference change,
this run's numbers are NOT expected to be identical to the earlier
LSSDFKRLGL results already in the report (Table 4-7) -- differences could
come from the annealing/dispatch changes above, the cleaner reference, or
both, and this script does not attempt to separate those two effects.

USAGE:
    python3 run_ack1_v2_full_pipeline.py
    python3 run_ack1_v2_full_pipeline.py --duration-ns 20.0
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ACK1_ACCESSION = "Q07912"
ACK1_SEQUENCE = "LSSDFKRLGL"
ACK1_START, ACK1_END = 528, 537


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=20.0,
                     help='20 ns matches the existing Table 7 comparison point in the report '
                          '(LSSDFKRLGL at 20 ns vs. PKI-NES at 50 ns) -- keeps this new run '
                          'comparable to a number already in the document.')
    ap.add_argument('--crm1-ref', default='crm1_reference/CRM1_Ran_only.pdb',
                     help='Clean CRM1+Ran reference (no Snurportin-1) -- see module docstring.')
    ap.add_argument('--test-specificity-control', action='store_true', default=False,
                     help='Also run scrambled-registration controls (doubles cost again). Off by default.')
    args = ap.parse_args()

    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app
    import requests

    crm1_path = THIS_DIR / args.crm1_ref
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path} -- aborting.")
        return

    print(f"{'='*100}\nACK1 (Q07912) LSSDFKRLGL (528-537) -- full refine_nes_candidates() re-run\n{'='*100}")
    print(f"Duration: {args.duration_ns} ns per variant")
    print(f"CRM1 reference: {crm1_path} (clean, no Snurportin-1 -- see docstring)")
    print(f"test_specificity_control: {args.test_specificity_control}\n")

    client = flask_app.test_client()
    print(f"Resolving AlphaFold model_id for {ACK1_ACCESSION} ...")
    resp = client.get(f"/api/models/{ACK1_ACCESSION}")
    if resp.status_code != 200:
        print(f"  /api/models/{ACK1_ACCESSION} returned HTTP {resp.status_code} -- aborting.")
        return
    models = resp.get_json() or []
    alphafold_models = [m for m in models if m.get("source") == "alphafold"]
    if not alphafold_models:
        print("  No AlphaFold entry found -- aborting.")
        return
    canonical_id = f"AF-{ACK1_ACCESSION}-F1"
    by_id = {m["model_id"]: m for m in alphafold_models}
    chosen = by_id.get(canonical_id) or max(
        (m for m in alphafold_models if (m.get("numResidues") or 0) >= ACK1_END),
        key=lambda m: m.get("numResidues") or 0, default=None,
    )
    if chosen is None:
        print("  No usable AlphaFold structure covers this residue range -- aborting.")
        return
    model_id = chosen["model_id"]
    print(f"  Using {model_id} ({chosen.get('numResidues')} aa, avg pLDDT: {chosen.get('avg_confidence')})")

    pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
    print(f"Downloading real structure: {pdb_url}")
    dl = requests.get(pdb_url, timeout=30)
    if dl.status_code != 200:
        print(f"  Could not download structure ({dl.status_code}) -- aborting.")
        return
    pdb_content = dl.text
    print(f"  {len(pdb_content)} bytes downloaded\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))
    candidate = {"sequence": ACK1_SEQUENCE, "start": ACK1_START, "end": ACK1_END, "full_sequence": None}

    binding_mode = refiner.classify_nes_binding_mode(ACK1_SEQUENCE)
    rec = refiner.recommend_starting_conformation(ACK1_SEQUENCE)
    print(f"Pre-MD classification: binding_mode={binding_mode['binding_mode_class']} "
          f"recommend={binding_mode['recommended_primary_method']} confidence={binding_mode['confidence']}")
    print(f"Unified recommendation: {rec['recommended_starting_conformation']} (confidence={rec['confidence']})\n")

    est_minutes = estimate_md_time(1, args.duration_ns) * 2  # native + idealized_helix at minimum
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h) for at least 2 variants\n")

    enhanced = refiner.refine_nes_candidates(
        pdb_content, [candidate], args.duration_ns,
        test_both_conformations=True,
        test_specificity_control=args.test_specificity_control,
    )
    cand = enhanced[0]

    print(f"\n{'='*100}\nACK1 v2 RE-RUN DONE\n{'='*100}")
    tested = sorted(cand.get('md_metrics_by_variant', {}).keys())
    print(f"Variants tested: {tested}")
    print(f"Primary variant chosen: {cand.get('md_best_starting_conformation')} "
          f"(selection method: {cand.get('md_primary_variant_selection_method')})")
    for tag, metrics in (cand.get('md_metrics_by_variant') or {}).items():
        print(f"\n  --- {tag} ---")
        print(f"    anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
        print(f"    raw_binding_score: {metrics.get('raw_binding_score')}")
        print(f"    avg_cys528_distance_nm: {metrics.get('avg_cys528_distance_nm')}")
        print(f"    avg_groove_contacts: {metrics.get('avg_groove_contacts')}")
        dssp_trace = metrics.get('dssp_helix_fraction_trace') or []
        if dssp_trace:
            print(f"    DSSP helix fraction (mean): {sum(dssp_trace)/len(dssp_trace):.4f}")
        print(f"    mean_anchor_burial_nm2: {metrics.get('mean_anchor_burial_nm2')}")
        print(f"    anchor_burial_fraction_well_buried: {metrics.get('anchor_burial_fraction_well_buried')}")
        conf_pred = metrics.get('conformation_prediction')
        if conf_pred:
            print(f"    pre-MD conformation_prediction: {conf_pred.get('predicted_conformation')} "
                  f"(confidence={conf_pred.get('confidence')})")
        rama = metrics.get('ramachandran_trace')
        print(f"    ramachandran_trace present: {rama is not None}")

    out_path = THIS_DIR / f"ack1_lssdfkrlgl_v2_{args.duration_ns:g}ns_result.json"
    out_path.write_text(json.dumps({
        'accession': ACK1_ACCESSION, 'sequence': ACK1_SEQUENCE, 'start': ACK1_START, 'end': ACK1_END,
        'duration_ns': args.duration_ns, 'model_id': model_id, 'crm1_reference': str(crm1_path),
        'binding_mode_classification': binding_mode, 'recommendation': rec,
        'md_best_starting_conformation': cand.get('md_best_starting_conformation'),
        'md_primary_variant_selection_method': cand.get('md_primary_variant_selection_method'),
        'md_metrics_by_variant': cand.get('md_metrics_by_variant'),
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
