#!/usr/bin/env python3
"""
run_rev_nes_extended_check.py
============================================================
Companion to run_rev_nes_helix_check.py -- that script deliberately docked
HIV-1 Rev-NES (3NBZ) starting from the WRONG assumption
(starting_conformation='idealized_helix') to test whether the pipeline's
scoring is helix-biased. This script docks it from its actual PREDICTED
starting conformation instead: recommend_starting_conformation('LPPLERLTL')
returns 'extended' (confidence=medium, binding_mode=extended_atypical) --
consistent with the real crystal structure, which is genuinely non-helical/
proline-containing.

This is simply the "most likely case" pose for this positive control --
the same treatment every ACK1 candidate got in the replicate study, applied
here to a KNOWN true positive for comparison. Produces the pose PDB that
was missing (only the deliberately-mismatched idealized_helix pose existed
before this).

Same protocol/conventions as run_rev_nes_helix_check.py: correct
registration, relax_sidechains=True, CRM1_Ran_3NBZ.pdb reference (confirmed
this project to NOT be meaningfully contaminated in practice for this
specific structure -- see rerun_3nc0_3gjx_clean.py's docstring/session notes),
50 ns default to match the existing idealized_helix run for a fair
comparison.

USAGE:
    python3 run_rev_nes_extended_check.py --duration-ns 50
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

SEQUENCE = "LPPLERLTL"
START, END = 6, 14  # local 3NBZ chain B numbering -- see run_rev_nes_helix_check.py's NOTE ON NUMBERING
LABEL = "HIV-1 Rev NES (3NBZ, Guttler et al. 2010, crystal I) -- most-likely-case (extended) pose"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration-ns", type=float, default=50.0)
    ap.add_argument("--crm1-pdb", default=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBZ.pdb"),
                     help="Crystal-matched CRM1 reference (default: crm1_reference/CRM1_Ran_3NBZ.pdb)")
    ap.add_argument("--out-json", default=None,
                     help="Where to save full metrics (default: 'rev_nes_extended_check_<duration>ns.json')")
    ap.add_argument("--out-pdb", default=None,
                     help="Where to save the final complex pose (default: 'REV_NES_3NBZ_extended_<duration>ns_complex.pdb')")
    args = ap.parse_args()

    crm1_path = Path(args.crm1_pdb)
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path}\n"
              f"Copy crm1_reference/CRM1_Ran_3NBZ.pdb over from your AlphaFold folder first.")
        sys.exit(1)

    dur_tag = f"{args.duration_ns:g}ns"
    out_json = Path(args.out_json) if args.out_json else THIS_DIR / f"rev_nes_extended_check_{dur_tag}.json"
    out_pdb = Path(args.out_pdb) if args.out_pdb else THIS_DIR / f"REV_NES_3NBZ_extended_{dur_tag}_complex.pdb"

    print(f"{LABEL}")
    print(f"Sequence: {SEQUENCE} ({START}-{END}, local 3NBZ chain B numbering)")
    print(f"CRM1 reference: {crm1_path}")
    print(f"Duration: {args.duration_ns} ns, starting_conformation=extended, correct registration\n")

    print("Importing md_refinement.py (checks for OpenMM/PDBFixer/mdtraj) ...\n")
    sys.path.insert(0, str(THIS_DIR))
    from md_refinement import NESMDRefiner, estimate_md_time

    est_minutes = estimate_md_time(1, args.duration_ns)
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h) -- ballpark only.\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))

    rec = refiner.recommend_starting_conformation(SEQUENCE)
    print(f"Pre-MD predictor check: recommended={rec['recommended_starting_conformation']} "
          f"(confidence={rec['confidence']}) -- {'matches' if rec['recommended_starting_conformation'] == 'extended' else 'DOES NOT MATCH'} this run's starting_conformation.\n")

    candidate = {"sequence": SEQUENCE, "start": START, "end": END,
                 "full_sequence": None, "combined_score": 0.5}

    print(f"\n{'=' * 100}")
    print("REV-NES (3NBZ) MOST-LIKELY-CASE (EXTENDED) POSE")
    print(f"{'=' * 100}")

    result = refiner._run_crm1_docking(
        pdb_content="",  # unused for extended -- built from sequence alone
        candidate=candidate,
        duration_ns=args.duration_ns,
        starting_conformation="extended",
        scramble_registration=False,
        relax_sidechains=True,
        save_final_complex_pdb_path=str(out_pdb),
    )

    metrics = result.get("md_metrics", {}) or {}
    helix_trace = metrics.get("dssp_helix_fraction_trace") or []

    print(f"\n{'=' * 100}")
    print("DONE")
    print(f"{'=' * 100}")
    print(f"anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
    print(f"avg_anchor_pocket_distance_nm: {metrics.get('avg_anchor_pocket_distance_nm')}")
    print(f"avg_groove_contacts:    {metrics.get('avg_groove_contacts')}")
    print(f"avg_hydrophobic_contacts: {metrics.get('avg_hydrophobic_contacts')}")
    print(f"raw_binding_score:      {metrics.get('raw_binding_score')}")
    print(f"binding_score:          {metrics.get('binding_score')}")
    if helix_trace:
        n_frames = len(helix_trace)
        n_any = sum(1 for h in helix_trace if h and h > 0)
        mean_h = sum(helix_trace) / n_frames
        print(f"DSSP helix fraction: mean={mean_h:.4f}  max={max(helix_trace):.4f}  "
              f"frames_with_any_helix={n_any}/{n_frames} ({100*n_any/n_frames:.1f}%)")
    else:
        print("DSSP helix fraction trace: not available (mdtraj missing?)")

    out_json.write_text(json.dumps({
        "label": LABEL,
        "sequence": SEQUENCE,
        "start": START,
        "end": END,
        "duration_ns": args.duration_ns,
        "crm1_reference": str(crm1_path),
        "pre_md_recommendation": rec,
        "md_metrics": metrics,
    }, indent=2, default=str))
    print(f"\nWrote {out_json}")
    if out_pdb.exists():
        print(f"Wrote {out_pdb}")
        print("Copy both files back -- the JSON has the full traces for plotting, "
              "the PDB is the final complex pose.")


if __name__ == "__main__":
    main()
