#!/usr/bin/env python3
"""
run_rev_and_ack1_ramachandran.py
============================================================
Runs BOTH remaining MD jobs in one script, back to back, so this only needs
kicking off once on the pod:

  1. Rev-NES positive control (3NBZ, Guttler et al. 2010 crystal I) --
     see run_rev_nes_helix_check.py's docstring for the full rationale.
     Real, confirmed NES that binds CRM1 in an EXTENDED, proline-containing
     conformation (not helical), docked here the same blind way every
     candidate is (starting_conformation='idealized_helix'). Tests whether
     this pipeline's scoring is helix-biased.

  2. ACK1 (Q07912) LSSDFKRLGL candidate, re-run with the same settings as
     run_ack1_nes_binding_pose.py's default (idealized_helix, correct
     registration, real downloaded AlphaFold structure + CRM1.pdb) -- the
     standout ACK1 candidate this whole session has been examining.

BOTH runs now also compute a full per-frame, per-residue Ramachandran
(phi/psi backbone dihedral) trace, via the ramachandran_trace field added
to md_refinement.py's _run_crm1_docking() output this project (piggybacks
on the same in-memory peptide coordinate frames already used for the DSSP
trace, so it costs virtually nothing extra). This lets you distinguish
genuinely disordered backbone sampling from a well-defined non-helical
basin (e.g. polyproline-II/extended) for BOTH peptides -- particularly
relevant for Rev-NES, whose real bound conformation is specifically
extended/proline-kinked, not just "not a helix."

WHY BOTH IN ONE SCRIPT: no shared computation between the two runs (they
use different CRM1 references, different peptides) -- this is purely a
convenience wrapper so you only have to kick off one nohup job instead of
two. Each run is fully independent; if one fails, the other still
completes and saves its own output.

REQUIREMENTS: same as run_pki_nes_helix_check.py / run_ack1_nes_binding_pose.py
combined -- OpenMM, PDBFixer, mdtraj, GPU recommended, real internet access
(for the ACK1 AlphaFold download step only -- the Rev-NES step needs no
internet). Needs crm1_reference/CRM1_Ran_3NBZ.pdb AND CRM1.pdb both present.

USAGE:
    python3 run_rev_and_ack1_ramachandran.py
    python3 run_rev_and_ack1_ramachandran.py --rev-duration-ns 50 --ack1-duration-ns 50
    python3 run_rev_and_ack1_ramachandran.py --skip-rev        # only re-run ACK1
    python3 run_rev_and_ack1_ramachandran.py --skip-ack1       # only run Rev-NES
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

REV_SEQUENCE = "LPPLERLTL"
REV_START, REV_END = 6, 14  # local 3NBZ chain B numbering -- see run_rev_nes_helix_check.py NOTE
REV_LABEL = "HIV-1 Rev NES (3NBZ, Guttler et al. 2010, crystal I) -- confirmed EXTENDED-mode NES"

ACK1_ACCESSION = "Q07912"
ACK1_SEQUENCE = "LSSDFKRLGL"
ACK1_START, ACK1_END = 528, 537


def _print_ramachandran_summary(metrics):
    rama = metrics.get("ramachandran_trace")
    if not rama:
        print("Ramachandran trace: not available (mdtraj missing, or computation failed -- see warning above)")
        return
    n_frames = len(rama["phi_deg"])
    n_phi_res = len(rama["phi_residue_labels"])
    print(f"Ramachandran trace: {n_frames} frames x {n_phi_res} phi angles, "
          f"{len(rama['psi_residue_labels'])} psi angles -- residues: {', '.join(rama['phi_residue_labels'])}")


def run_rev_nes(duration_ns):
    from md_refinement import NESMDRefiner, estimate_md_time

    crm1_path = THIS_DIR / "crm1_reference" / "CRM1_Ran_3NBZ.pdb"
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path} -- skipping Rev-NES run.")
        return

    out_json = THIS_DIR / f"rev_nes_helix_check_{duration_ns:g}ns.json"
    out_pdb = THIS_DIR / f"REV_NES_3NBZ_idealized_helix_{duration_ns:g}ns_complex.pdb"

    print(f"\n{'#' * 100}")
    print(f"# RUN 1/2: {REV_LABEL}")
    print(f"{'#' * 100}")
    print(f"Sequence: {REV_SEQUENCE} ({REV_START}-{REV_END}, local 3NBZ chain B numbering)")
    print(f"Duration: {duration_ns} ns, starting_conformation=idealized_helix, correct registration")
    est_minutes = estimate_md_time(1, duration_ns)
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h)\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))
    candidate = {"sequence": REV_SEQUENCE, "start": REV_START, "end": REV_END,
                 "full_sequence": None, "combined_score": 0.5}

    result = refiner._run_crm1_docking(
        pdb_content="",
        candidate=candidate,
        duration_ns=duration_ns,
        starting_conformation="idealized_helix",
        scramble_registration=False,
        relax_sidechains=True,
        save_final_complex_pdb_path=str(out_pdb),
    )
    metrics = result.get("md_metrics", {}) or {}

    print(f"\n{'=' * 100}")
    print("REV-NES DONE")
    print(f"{'=' * 100}")
    print(f"anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
    print(f"avg_anchor_pocket_distance_nm: {metrics.get('avg_anchor_pocket_distance_nm')}")
    print(f"avg_groove_contacts:    {metrics.get('avg_groove_contacts')}")
    print(f"raw_binding_score:      {metrics.get('raw_binding_score')}")
    helix_trace = metrics.get("dssp_helix_fraction_trace") or []
    if helix_trace:
        n_frames = len(helix_trace)
        n_any = sum(1 for h in helix_trace if h and h > 0)
        print(f"DSSP helix fraction (informational, NOT expected high): mean={sum(helix_trace)/n_frames:.4f}  "
              f"frames_with_any_helix={n_any}/{n_frames}")
    _print_ramachandran_summary(metrics)

    out_json.write_text(json.dumps({
        "label": REV_LABEL, "sequence": REV_SEQUENCE, "start": REV_START, "end": REV_END,
        "duration_ns": duration_ns, "crm1_reference": str(crm1_path), "md_metrics": metrics,
    }, indent=2, default=str))
    print(f"\nWrote {out_json}")
    if out_pdb.exists():
        print(f"Wrote {out_pdb}")


def run_ack1(duration_ns):
    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app
    import requests

    out_json = THIS_DIR / f"Q07912_LSSDFKRLGL_idealized_helix_{duration_ns:g}ns_rama.json"
    out_pdb = THIS_DIR / f"Q07912_LSSDFKRLGL_idealized_helix_{duration_ns:g}ns_rama_complex.pdb"

    print(f"\n{'#' * 100}")
    print(f"# RUN 2/2: ACK1 (Q07912) {ACK1_SEQUENCE} ({ACK1_START}-{ACK1_END})")
    print(f"{'#' * 100}")
    print(f"Duration: {duration_ns} ns, starting_conformation=idealized_helix, correct registration")
    est_minutes = estimate_md_time(1, duration_ns)
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h)\n")

    client = flask_app.test_client()
    print(f"Resolving AlphaFold model_id for {ACK1_ACCESSION} ...")
    resp = client.get(f"/api/models/{ACK1_ACCESSION}")
    if resp.status_code != 200:
        print(f"  /api/models/{ACK1_ACCESSION} returned HTTP {resp.status_code} -- skipping ACK1 run.")
        return
    models = resp.get_json() or []
    alphafold_models = [m for m in models if m.get("source") == "alphafold"]
    if not alphafold_models:
        print("  No AlphaFold entry found -- skipping ACK1 run.")
        return
    canonical_id = f"AF-{ACK1_ACCESSION}-F1"
    by_id = {m["model_id"]: m for m in alphafold_models}
    chosen = by_id.get(canonical_id) or max(
        (m for m in alphafold_models if (m.get("numResidues") or 0) >= ACK1_END),
        key=lambda m: m.get("numResidues") or 0, default=None,
    )
    if chosen is None:
        print("  No usable AlphaFold structure covers this residue range -- skipping ACK1 run.")
        return
    model_id = chosen["model_id"]
    print(f"  Using {model_id} ({chosen.get('numResidues')} aa, avg pLDDT: {chosen.get('avg_confidence')})")

    pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
    print(f"Downloading real structure: {pdb_url}")
    dl = requests.get(pdb_url, timeout=30)
    if dl.status_code != 200:
        print(f"  Could not download structure ({dl.status_code}) -- skipping ACK1 run.")
        return
    pdb_content = dl.text
    print(f"  {len(pdb_content)} bytes downloaded\n")

    crm1_path = THIS_DIR / "CRM1.pdb"
    if not crm1_path.exists():
        print(f"CRM1.pdb not found at {crm1_path} -- skipping ACK1 run.")
        return
    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))
    candidate = {"sequence": ACK1_SEQUENCE, "start": ACK1_START, "end": ACK1_END}

    result = refiner._run_crm1_docking(
        pdb_content, candidate, duration_ns,
        starting_conformation="idealized_helix",
        scramble_registration=False,
        save_final_complex_pdb_path=str(out_pdb),
    )
    metrics = result.get("md_metrics", {}) or {}

    print(f"\n{'=' * 100}")
    print("ACK1 DONE")
    print(f"{'=' * 100}")
    print(f"anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
    print(f"avg_cys528_distance_nm: {metrics.get('avg_cys528_distance_nm')}")
    print(f"avg_groove_contacts:    {metrics.get('avg_groove_contacts')}")
    _print_ramachandran_summary(metrics)

    out_json.write_text(json.dumps({
        "accession": ACK1_ACCESSION, "sequence": ACK1_SEQUENCE, "start": ACK1_START, "end": ACK1_END,
        "duration_ns": duration_ns, "model_id": model_id, "md_metrics": metrics,
    }, indent=2, default=str))
    print(f"\nWrote {out_json}")
    if out_pdb.exists():
        print(f"Wrote {out_pdb}")
    else:
        print(f"WARNING: {out_pdb} was not created -- check the log above for a fallback/failure path.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev-duration-ns", type=float, default=50.0)
    ap.add_argument("--ack1-duration-ns", type=float, default=50.0)
    ap.add_argument("--skip-rev", action="store_true")
    ap.add_argument("--skip-ack1", action="store_true")
    args = ap.parse_args()

    print("Importing md_refinement.py (checks for OpenMM/PDBFixer/mdtraj) ...\n")
    sys.path.insert(0, str(THIS_DIR))

    if not args.skip_rev:
        run_rev_nes(args.rev_duration_ns)
    else:
        print("Skipping Rev-NES run (--skip-rev)")

    if not args.skip_ack1:
        run_ack1(args.ack1_duration_ns)
    else:
        print("Skipping ACK1 run (--skip-ack1)")

    print(f"\n{'#' * 100}")
    print("# ALL RUNS COMPLETE -- copy back any *_rama*.json / *_complex.pdb / rev_nes_*.json files that got created")
    print(f"{'#' * 100}")


if __name__ == "__main__":
    main()
