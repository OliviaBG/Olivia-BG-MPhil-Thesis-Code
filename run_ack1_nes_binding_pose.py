#!/usr/bin/env python3
"""
run_ack1_nes_binding_pose.py
============================================================
Re-runs ONE specific CRM1-docking MD condition for an ACK1 (Q07912) NES
candidate and saves the converged FINAL COMPLEX (peptide + CRM1 together,
in their real post-production positions) to a single PDB file, so it can
be opened directly in a molecular viewer (PyMOL, ChimeraX, even a browser
3Dmol viewer) to see how the candidate actually sits in the CRM1 groove.

WHY THIS SCRIPT EXISTS: the earlier full refinement runs
(run_ack1_md_refinement.py / run_ack1_md_specificity_control.py) never
passed a save-pose path to _run_crm1_docking(), so none of those 24 runs
left a structure file behind -- only the numeric metrics were saved to the
result JSON. Re-deriving a structure requires re-running MD (the metrics
alone don't determine atomic coordinates), but a SINGLE 10 ns run takes
about 20 minutes on a GPU, so this only reruns the one condition worth
looking at rather than all 24.

DEFAULT CANDIDATE: LSSDFKRLGL (528-537), idealized_helix, CORRECT
registration -- this is the one candidate that showed a real, specificity-
confirmed signal in the full MD analysis (anchor_occupancy_score 0.347,
closest Cys528 approach of any candidate, ~20x groove contacts, and a
clean collapse to 0.000 under the scrambled-registration control). Use
--sequence/--start/--end/--conformation/--scrambled to look at a different
candidate or condition instead.

This uses md_refinement.NESMDRefiner._run_crm1_docking() directly -- the
same production code path as the other MD scripts, not a reimplementation
-- with the new save_final_complex_pdb_path parameter (added
specifically for this) that writes the full post-MD complex instead of
metrics-only.

REQUIREMENTS: same as run_ack1_md_refinement.py (OpenMM, PDBFixer, real
internet access, CRM1.pdb, GPU recommended). ~20-25 min for the default
10 ns run.

USAGE:
    python3 run_ack1_nes_binding_pose.py
    python3 run_ack1_nes_binding_pose.py --sequence LSSDFKRLGL --start 528 --end 537 --conformation idealized_helix
    python3 run_ack1_nes_binding_pose.py --scrambled   # the negative-control pose, for comparison
"""
import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def resolve_model_id(client, accession, min_residues):
    """Same fixed logic as run_ack1_md_refinement.py / run_ack1_md_specificity_control.py --
    always prefer the exact canonical model_id these candidates were generated against."""
    resp = client.get(f'/api/models/{accession}')
    if resp.status_code != 200:
        print(f"  /api/models/{accession} returned HTTP {resp.status_code}")
        return None
    models = resp.get_json()
    if not models:
        return None
    alphafold_models = [m for m in models if m.get('source') == 'alphafold']
    if not alphafold_models:
        return None

    canonical_id = f"AF-{accession}-F1"
    by_id = {m['model_id']: m for m in alphafold_models}
    chosen = by_id.get(canonical_id)
    if chosen is None:
        covering = [m for m in alphafold_models if (m.get('numResidues') or 0) >= min_residues]
        if not covering:
            print(f"  No AlphaFold entry for {accession} covers residue {min_residues}")
            return None
        chosen = max(covering, key=lambda m: m.get('numResidues') or 0)
        print(f"  NOTE: canonical model_id {canonical_id} not found; using {chosen['model_id']} instead.")

    n_res = chosen.get('numResidues') or 0
    if n_res < min_residues:
        print(f"  ERROR: {chosen['model_id']} only has {n_res} residues, need {min_residues}. Aborting.")
        return None

    print(f"  Using {chosen['model_id']} ({n_res} aa, avg pLDDT confidence: {chosen.get('avg_confidence')})")
    return chosen['model_id']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accession", default="Q07912")
    ap.add_argument("--sequence", default="LSSDFKRLGL", help="NES candidate sequence (default: the standout hit)")
    ap.add_argument("--start", type=int, default=528)
    ap.add_argument("--end", type=int, default=537)
    ap.add_argument("--conformation", default="idealized_helix", choices=["native", "idealized_helix"],
                     help="Starting conformation (default: idealized_helix, matching the primary result)")
    ap.add_argument("--scrambled", action="store_true",
                     help="Save the SCRAMBLED-registration (negative control) pose instead of the correct one")
    ap.add_argument("--duration-ns", type=float, default=10.0,
                     help="MD duration, in ns (default 10.0, matching the original run)")
    ap.add_argument("--out", default=None,
                     help="Output PDB path (default: '<accession>_<sequence>_<conformation>[_scrambled]_complex.pdb')")
    args = ap.parse_args()

    tag = f"{args.conformation}{'_scrambled' if args.scrambled else ''}"
    out_path = Path(args.out) if args.out else THIS_DIR / f"{args.accession}_{args.sequence}_{tag}_complex.pdb"

    candidate = {"sequence": args.sequence, "start": args.start, "end": args.end}
    print(f"Re-running MD for {args.sequence} ({args.start}-{args.end}), conformation={args.conformation}, "
          f"scrambled={args.scrambled}, duration_ns={args.duration_ns}")
    print(f"Output complex PDB: {out_path}\n")

    print("Importing app.py and md_refinement.py ...\n")
    sys.path.insert(0, str(THIS_DIR))
    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app

    est_minutes = estimate_md_time(1, args.duration_ns)
    print(f"Rough estimate: ~{est_minutes:.0f} min -- ballpark only, depends on your GPU.\n")

    client = flask_app.test_client()
    print(f"Resolving AlphaFold model_id for {args.accession} (must cover residue {args.end}) ...")
    model_id = resolve_model_id(client, args.accession, min_residues=args.end)
    if not model_id:
        print(f"No usable AlphaFold structure found for {args.accession}. Aborting.")
        sys.exit(1)

    import requests
    pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
    print(f"Downloading real structure: {pdb_url}")
    resp = requests.get(pdb_url, timeout=30)
    if resp.status_code != 200:
        print(f"Could not download structure ({resp.status_code}). Aborting.")
        sys.exit(1)
    pdb_content = resp.text
    print(f"  {len(pdb_content)} bytes downloaded\n")

    crm1_pdb_path = str(THIS_DIR / "CRM1.pdb")
    if not Path(crm1_pdb_path).exists():
        print(f"CRM1.pdb not found at {crm1_pdb_path}. Aborting.")
        sys.exit(1)
    refiner = NESMDRefiner(crm1_pdb_path=crm1_pdb_path)

    print(f"\n{'=' * 100}")
    print(f"NES BINDING POSE -- {args.accession} ({model_id}), {args.sequence} ({args.start}-{args.end}), "
          f"{args.conformation}{' [SCRAMBLED CONTROL]' if args.scrambled else ' [correct registration]'}")
    print(f"{'=' * 100}")

    result = refiner._run_crm1_docking(
        pdb_content, candidate, args.duration_ns,
        starting_conformation=args.conformation,
        scramble_registration=args.scrambled,
        save_final_complex_pdb_path=str(out_path),
    )

    metrics = result.get("md_metrics", {}) or {}
    print(f"\n{'=' * 100}")
    print("DONE")
    print(f"{'=' * 100}")
    print(f"anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
    print(f"avg_cys528_distance_nm: {metrics.get('avg_cys528_distance_nm')}")
    print(f"avg_groove_contacts:    {metrics.get('avg_groove_contacts')}")
    if out_path.exists():
        print(f"\nComplex structure written to: {out_path}")
        print("Copy this one file back and open it in PyMOL/ChimeraX (or any PDB viewer) to see the binding pose.")
    else:
        print(f"\nWARNING: {out_path} was not created -- this usually means the run hit a fallback/failure path "
              f"partway through (see the log above) rather than completing production MD.")


if __name__ == "__main__":
    main()
