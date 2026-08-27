#!/usr/bin/env python3
"""
run_ack1_rank1_best_anchor_frame.py
============================================================
Re-runs ACK1 rank 1 (478-487, FPDRIDELYL), its replicate-study winning
conformation (idealized_helix, correct registration), at the same 20 ns
used for the Stage B production run -- but this time with the new
save_best_anchor_frame_pdb_path option (md_refinement.py, )
enabled alongside the usual save_final_complex_pdb_path.

WHY: the Stage B 20 ns pose (ack1_rank1_idealized_helix_20ns_complex.pdb)
looked, on visual inspection, like the peptide was only loosely perched at
the groove edge -- despite the run's own trusted metrics (anchor_occupancy_
score 0.445, 3/4 anchors well-buried on average) indicating real, sustained
engagement. That mismatch is expected: save_final_complex_pdb_path writes
the literal LAST integrator step, which is one arbitrary snapshot and can
catch the peptide between more/less engaged moments. save_best_anchor_
frame_pdb_path instead writes whichever of the same 10 end-of-production
representative frames (the same window anchor_burial_fraction_well_buried
is already averaged over) has the MOST anchors crossing the well-buried
threshold -- an honest "best moment from the already-converged window",
with no steering/restraint force applied (which would have changed what
the result is evidence of).

NOTE: because MD is stochastic and no random seed is fixed anywhere in
this pipeline, this is a NEW, independent trajectory -- its own
anchor_occupancy_score etc. may differ somewhat from the original 0.445
Stage B run. That's expected and consistent with everything already
documented about run-to-run variability in this project; the point of
this script is the frame-selection logic, not a repeat of the replicate
study.

USAGE:
    python3 run_ack1_rank1_best_anchor_frame.py
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ACK1_ACCESSION = "Q07912"
CANDIDATE = {"rank": 1, "start": 478, "end": 487, "sequence": "FPDRIDELYL"}
CONFORMATION = "idealized_helix"  # replicate-study winner for rank 1
DURATION_NS = 20.0


def main():
    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app
    import requests

    crm1_path = THIS_DIR / "crm1_reference" / "CRM1_Ran_only.pdb"
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path} -- aborting.")
        return

    final_pdb = THIS_DIR / f"ack1_rank1_{CONFORMATION}_{DURATION_NS:g}ns_v2_final_complex.pdb"
    best_pdb = THIS_DIR / f"ack1_rank1_{CONFORMATION}_{DURATION_NS:g}ns_best_anchor_frame_complex.pdb"
    out_json = THIS_DIR / "ack1_rank1_best_anchor_frame_result.json"

    print(f"{'='*100}\nACK1 rank 1 (478-487, FPDRIDELYL), {CONFORMATION}, {DURATION_NS} ns\n"
          f"Saving BOTH final-frame and best-anchor-frame complex poses\n{'='*100}")

    client = flask_app.test_client()
    print(f"Resolving AlphaFold model_id for {ACK1_ACCESSION} ...")
    resp = client.get(f"/api/models/{ACK1_ACCESSION}")
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} -- aborting.")
        return
    models = resp.get_json() or []
    alphafold_models = [m for m in models if m.get("source") == "alphafold"]
    canonical_id = f"AF-{ACK1_ACCESSION}-F1"
    by_id = {m["model_id"]: m for m in alphafold_models}
    chosen = by_id.get(canonical_id) or max(
        (m for m in alphafold_models if (m.get("numResidues") or 0) >= 1038),
        key=lambda m: m.get("numResidues") or 0, default=None,
    )
    if chosen is None:
        print("  No usable AlphaFold structure found -- aborting.")
        return
    model_id = chosen["model_id"]
    print(f"  Using {model_id} ({chosen.get('numResidues')} aa)")

    pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
    dl = requests.get(pdb_url, timeout=30)
    if dl.status_code != 200:
        print(f"  Download failed ({dl.status_code}) -- aborting.")
        return
    pdb_content = dl.text
    print(f"  {len(pdb_content)} bytes\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))
    est_min = estimate_md_time(1, DURATION_NS)
    print(f"Estimated time: ~{est_min:.0f} min\n")

    candidate = {"sequence": CANDIDATE['sequence'], "start": CANDIDATE['start'],
                 "end": CANDIDATE['end'], "full_sequence": None}

    result = refiner._run_crm1_docking(
        pdb_content=pdb_content,
        candidate=candidate,
        duration_ns=DURATION_NS,
        starting_conformation=CONFORMATION,
        scramble_registration=False,
        save_final_complex_pdb_path=str(final_pdb),
        save_best_anchor_frame_pdb_path=str(best_pdb),
    )
    metrics = result.get('md_metrics', {}) or {}

    print(f"\n{'='*100}\nDONE\n{'='*100}")
    print(f"anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
    print(f"mean_anchor_burial_nm2: {metrics.get('mean_anchor_burial_nm2')}")
    print(f"anchor_burial_fraction_well_buried: {metrics.get('anchor_burial_fraction_well_buried')}")
    print(f"best_anchor_frame_well_buried_count: {metrics.get('best_anchor_frame_well_buried_count')}")
    print(f"best_anchor_frame_sample_index: {metrics.get('best_anchor_frame_sample_index')}")
    if final_pdb.exists():
        print(f"Final-frame pose:      {final_pdb.name}")
    if best_pdb.exists():
        print(f"Best-anchor-frame pose: {best_pdb.name}")

    out_json.write_text(json.dumps({
        "rank": 1, "sequence": CANDIDATE['sequence'], "conformation": CONFORMATION,
        "duration_ns": DURATION_NS, "md_metrics": metrics,
        "final_pdb": final_pdb.name, "best_anchor_frame_pdb": best_pdb.name,
    }, indent=2, default=str))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
