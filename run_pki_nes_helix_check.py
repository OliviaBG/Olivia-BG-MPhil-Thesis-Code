#!/usr/bin/env python3
"""
run_pki_nes_helix_check.py
============================================================
Positive control for the "does LSSDFKRLGL ever hold a helix" question --
runs this project's SAME production MD protocol (Kabsch anchor
registration, side-chain relaxation, minimization, equilibration,
production) on a real, experimentally-solved, genuinely alpha-helical NES
that has never been through a full multi-ns production run in this
project before: PKI-alpha's NES (PDB 3NBY, Guttler et al. 2010, Nat Struct
Mol Biol 17:1367-1376), UniProt P61925, sequence LALKLAGLDI.

WHY THIS ONE: 3NBY is already this project's own reference for "confirmed
canonical alpha-helical NES" (see crystal_sanity_check.py, quick_helix_
analysis.py's docstring, idealized_helix_vs_crystal_check.py). That last
script DID already dock this exact sequence via starting_conformation=
'idealized_helix', but only at 2 ns and only to check backbone RMSD
against the real crystal pose -- it never tracked DSSP secondary
structure over the trajectory, so there is no existing answer to "does it
actually STAY helical during production MD in this pipeline." P61925 is
also not among the 54 accessions in anchor_occupancy_eval_v4_clean.json,
so this is a genuinely held-out check, not a re-run of something already
scored.

If the pipeline's MD protocol (force field, relaxation, equilibration
protocol) can hold a known-real helix folded for a known-real helical
binder, but not for LSSDFKRLGL, that's evidence LSSDFKRLGL's extended/
coil result is a real, sequence-specific finding. If PKI-NES ALSO unwinds
under this same protocol, that would instead point to something
systematic in the simulation setup itself -- a much bigger finding, and
worth knowing either way.

Uses the crystal-matched CRM1 reference (crm1_reference/CRM1_Ran_3NBY.pdb,
extracted specifically to pair with this structure's own coordinate
frame -- see extract_crystal_references.py) rather than the generic
CRM1.pdb used for the ACK1 candidates, since that is the more precise
reference for this specific known complex. Needs no AlphaFold download or
Flask app -- idealized_helix docking only needs the sequence, not a
source structure (pdb_content is unused on this code path; see
idealized_helix_vs_crystal_check.py, which established this pattern).

REQUIREMENTS: same as the other MD scripts (OpenMM, PDBFixer, GPU
recommended). Needs crm1_reference/CRM1_Ran_3NBY.pdb present -- copy it
over from your AlphaFold folder if this pod doesn't already have it from
the earlier crystal-validation work.

USAGE:
    python3 run_pki_nes_helix_check.py --duration-ns 50
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Matches crystal_sanity_check.py's CRYSTAL_STRUCTURES['3NBY'] entry exactly.
SEQUENCE = "LALKLAGLDI"
START, END = 37, 46
LABEL = "PKI-alpha NES (3NBY, Guttler et al. 2010) -- confirmed canonical alpha-helical NES"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration-ns", type=float, default=50.0)
    ap.add_argument("--crm1-pdb", default=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBY.pdb"),
                     help="Crystal-matched CRM1 reference (default: crm1_reference/CRM1_Ran_3NBY.pdb)")
    ap.add_argument("--out-json", default=None,
                     help="Where to save full metrics (default: 'pki_nes_helix_check_<duration>ns.json')")
    ap.add_argument("--out-pdb", default=None,
                     help="Where to save the final complex pose (default: 'PKI_NES_3NBY_idealized_helix_<duration>ns_complex.pdb')")
    args = ap.parse_args()

    crm1_path = Path(args.crm1_pdb)
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path}\n"
              f"Copy crm1_reference/CRM1_Ran_3NBY.pdb over from your AlphaFold folder first "
              f"(it was already extracted during the earlier crystal-validation work).")
        sys.exit(1)

    dur_tag = f"{args.duration_ns:g}ns"
    out_json = Path(args.out_json) if args.out_json else THIS_DIR / f"pki_nes_helix_check_{dur_tag}.json"
    out_pdb = Path(args.out_pdb) if args.out_pdb else THIS_DIR / f"PKI_NES_3NBY_idealized_helix_{dur_tag}_complex.pdb"

    print(f"{LABEL}")
    print(f"Sequence: {SEQUENCE} ({START}-{END})")
    print(f"CRM1 reference: {crm1_path}")
    print(f"Duration: {args.duration_ns} ns, starting_conformation=idealized_helix, correct registration\n")

    print("Importing md_refinement.py (checks for OpenMM/PDBFixer/mdtraj) ...\n")
    sys.path.insert(0, str(THIS_DIR))
    from md_refinement import NESMDRefiner, estimate_md_time

    est_minutes = estimate_md_time(1, args.duration_ns)
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h) -- ballpark only.\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))

    candidate = {"sequence": SEQUENCE, "start": START, "end": END,
                 "full_sequence": None, "combined_score": 0.5}

    print(f"\n{'=' * 100}")
    print("PKI-NES (3NBY) HELIX-RETENTION CHECK")
    print(f"{'=' * 100}")

    result = refiner._run_crm1_docking(
        pdb_content="",  # unused for idealized_helix -- see idealized_helix_vs_crystal_check.py
        candidate=candidate,
        duration_ns=args.duration_ns,
        starting_conformation="idealized_helix",
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
    print(f"avg_cys528_distance_nm: {metrics.get('avg_cys528_distance_nm')}")
    print(f"avg_groove_contacts:    {metrics.get('avg_groove_contacts')}")
    print(f"helix_propensity (seq-only, t=0 idealized): {metrics.get('helix_propensity')}")
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
        "md_metrics": metrics,
    }, indent=2, default=str))
    print(f"\nWrote {out_json}")
    if out_pdb.exists():
        print(f"Wrote {out_pdb}")
        print("Copy both files back -- the JSON has the full DSSP trace for plotting, "
              "the PDB is the final complex pose.")


if __name__ == "__main__":
    main()
