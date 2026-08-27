#!/usr/bin/env python3
"""
run_rev_nes_helix_check.py
============================================================
Second positive control, deliberately different in kind from
run_pki_nes_helix_check.py -- not just a second data point, but a test of
a different failure mode.

The PKI-NES positive control (3NBY) proved this pipeline's MD protocol CAN
correctly fold and dock a real, confirmed NES when that NES happens to bind
in the alpha-helical mode (the same mode every candidate, including
LSSDFKRLGL, is blindly docked as via starting_conformation='idealized_helix').
But that leaves an open question: is the pipeline's scoring only sensitive to
helical engagement? If LSSDFKRLGL's real binding mode were extended/
non-helical rather than "just a weak NES", would this pipeline even be able
to recognize it as a genuine binder -- or does it structurally favor helices
regardless of whether that's the correct answer for a given sequence?

HIV-1 Rev NES (PDB 3NBZ, Guttler et al. 2010, same paper as the PKI
structure -- "crystal I") answers this directly: it is a real, experimentally
-confirmed NES that binds CRM1 in an EXTENDED, proline-containing
conformation, NOT alpha-helical (see crystal_sanity_check.py's own notes on
this structure). Sequence LPPLERLTL. Docked here the exact same way every
blind candidate is docked -- starting_conformation='idealized_helix', i.e.
this project's pipeline is given the WRONG starting assumption (that it's
helical) on purpose, same as it would be for any unknown candidate.

WHAT SUCCESS LOOKS LIKE HERE IS DIFFERENT FROM THE PKI CHECK: we are NOT
expecting a stable helix to form or persist (the real structure isn't
helical, so persisting in an idealized helix would itself be a sign the
peptide got stuck in an artificial starting pose rather than finding its
real binding mode). What matters is whether the pipeline still recognizes
real hydrophobic anchor engagement (anchor_occupancy_score, groove contacts,
per-pocket distances) despite the non-ideal starting guess and despite the
final conformation likely not being a clean helix. If it does, that is
strong evidence the pipeline's scoring is not helix-biased, which directly
strengthens (or weakens, if this comes back flat) the interpretation of
LSSDFKRLGL's own extended, low-engagement result as a genuine finding rather
than a pipeline limitation specific to non-helical peptides.

NOTE ON NUMBERING: unlike the PKI-NES check, 3NBZ's residue_range in
crystal_sanity_check.py (6, 14) is LOCAL CHAIN B NUMBERING from the crystal
file itself, not native HIV-1 Rev protein numbering -- see that file's own
comment on the 3NBZ entry. Reported here as-is for consistency with the rest
of this project's crystal reference data; do not treat these as native Rev
protein residue numbers without checking against the Rev sequence/UniProt
P69718 separately if that mapping is ever needed.

REQUIREMENTS: same as run_pki_nes_helix_check.py -- OpenMM, PDBFixer,
mdtraj (for the DSSP trace), GPU recommended. Needs
crm1_reference/CRM1_Ran_3NBZ.pdb present -- copy it over from your AlphaFold
folder if this pod doesn't already have it. No peptide reference file is
needed (idealized_helix docking builds the peptide from sequence alone, same
as the PKI check).

USAGE:
    python3 run_rev_nes_helix_check.py --duration-ns 50
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Matches crystal_sanity_check.py's CRYSTAL_STRUCTURES['3NBZ'] entry exactly.
SEQUENCE = "LPPLERLTL"
START, END = 6, 14  # local 3NBZ chain B numbering -- see NOTE ON NUMBERING above
LABEL = "HIV-1 Rev NES (3NBZ, Guttler et al. 2010, crystal I) -- confirmed EXTENDED-mode NES"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration-ns", type=float, default=50.0)
    ap.add_argument("--crm1-pdb", default=str(THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBZ.pdb"),
                     help="Crystal-matched CRM1 reference (default: crm1_reference/CRM1_Ran_3NBZ.pdb)")
    ap.add_argument("--out-json", default=None,
                     help="Where to save full metrics (default: 'rev_nes_helix_check_<duration>ns.json')")
    ap.add_argument("--out-pdb", default=None,
                     help="Where to save the final complex pose (default: 'REV_NES_3NBZ_idealized_helix_<duration>ns_complex.pdb')")
    args = ap.parse_args()

    crm1_path = Path(args.crm1_pdb)
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path}\n"
              f"Copy crm1_reference/CRM1_Ran_3NBZ.pdb over from your AlphaFold folder first.")
        sys.exit(1)

    dur_tag = f"{args.duration_ns:g}ns"
    out_json = Path(args.out_json) if args.out_json else THIS_DIR / f"rev_nes_helix_check_{dur_tag}.json"
    out_pdb = Path(args.out_pdb) if args.out_pdb else THIS_DIR / f"REV_NES_3NBZ_idealized_helix_{dur_tag}_complex.pdb"

    print(f"{LABEL}")
    print(f"Sequence: {SEQUENCE} ({START}-{END}, local 3NBZ chain B numbering)")
    print(f"CRM1 reference: {crm1_path}")
    print(f"Duration: {args.duration_ns} ns, starting_conformation=idealized_helix, correct registration")
    print(f"NOTE: real Rev-NES binds in an EXTENDED conformation, not helical -- we are NOT expecting/hoping")
    print(f"for a persistent helix here. Watch anchor_occupancy_score and groove contacts, not DSSP.\n")

    print("Importing md_refinement.py (checks for OpenMM/PDBFixer/mdtraj) ...\n")
    sys.path.insert(0, str(THIS_DIR))
    from md_refinement import NESMDRefiner, estimate_md_time

    est_minutes = estimate_md_time(1, args.duration_ns)
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h) -- ballpark only.\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))

    candidate = {"sequence": SEQUENCE, "start": START, "end": END,
                 "full_sequence": None, "combined_score": 0.5}

    print(f"\n{'=' * 100}")
    print("REV-NES (3NBZ) NON-HELICAL-BIAS CHECK")
    print(f"{'=' * 100}")

    result = refiner._run_crm1_docking(
        pdb_content="",  # unused for idealized_helix -- see run_pki_nes_helix_check.py
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
    print(f"avg_anchor_pocket_distance_nm: {metrics.get('avg_anchor_pocket_distance_nm')}")
    print(f"avg_groove_contacts:    {metrics.get('avg_groove_contacts')}")
    print(f"avg_hydrophobic_contacts: {metrics.get('avg_hydrophobic_contacts')}")
    print(f"raw_binding_score:      {metrics.get('raw_binding_score')}")
    print(f"binding_score:          {metrics.get('binding_score')}")
    if helix_trace:
        n_frames = len(helix_trace)
        n_any = sum(1 for h in helix_trace if h and h > 0)
        mean_h = sum(helix_trace) / n_frames
        print(f"DSSP helix fraction (informational only, NOT expected to be high): "
              f"mean={mean_h:.4f}  max={max(helix_trace):.4f}  "
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
        print("Copy both files back -- the JSON has the full traces for plotting, "
              "the PDB is the final complex pose.")


if __name__ == "__main__":
    main()
