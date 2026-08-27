#!/usr/bin/env python3
"""
run_ack1_rank1_continued_trajectory.py
============================================================
"Seed-screening then continue" protocol for ACK1 rank 1 (478-487,
FPDRIDELYL), idealized_helix, correct registration -- built in response to
watching this exact candidate/conformation swing from anchor_occupancy_
score 0.445 to 0.0 between two independent fresh 20 ns runs. A fresh 20 ns
run just re-rolls a new random seed every time; this instead:

  STAGE A (screen): runs 3 independent, SHORT (2 ns) replicates from
  scratch, each saving its final simulation state (positions + velocities,
  via the new save_final_state_path / Simulation.saveState) as well as its
  own anchor_occupancy_score. Checkpointed after every single screen run.

  STAGE B (continue): picks whichever of the 3 screens scored highest on
  anchor_occupancy_score, then RESUMES that exact trajectory (via the new
  resume_from_state_path / Simulation.loadState -- md_refinement.py, ) for 18 more ns of production, reaching 20 ns cumulative from
  the SAME continuous trajectory rather than a new independent run. This
  actually addresses the stochasticity problem instead of re-rolling it:
  whatever basin the winning screen found itself in after 2 ns, the
  extension continues exploring FROM there rather than starting over.

  Saves both the final-frame pose and the best-anchor-frame pose (same
  save_best_anchor_frame_pdb_path logic used for the single-run check
  earlier) from the Stage B extension, plus the full metrics dict
  (anchor_occupancy_score, anchor burial, DSSP/Ramachandran traces, etc.).

Checkpointed at every stage (every screen run individually, then the
extension) -- safe to interrupt and re-run; already-completed stages are
skipped.

USAGE:
    python3 run_ack1_rank1_continued_trajectory.py
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ACK1_ACCESSION = "Q07912"
CANDIDATE = {"rank": 1, "start": 478, "end": 487, "sequence": "FPDRIDELYL"}
CONFORMATION = "idealized_helix"  # replicate-study winner for rank 1
SCREEN_DURATION_NS = 2.0
N_SCREENS = 3
TARGET_TOTAL_NS = 20.0
EXTENSION_NS = TARGET_TOTAL_NS - SCREEN_DURATION_NS

CACHE_PATH = THIS_DIR / "ack1_rank1_continued_trajectory_result.json"
STATE_DIR = THIS_DIR / "ack1_rank1_screen_states"


def load_state():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"screens": [], "extension": None}


def save_state(state):
    CACHE_PATH.write_text(json.dumps(state, indent=2, default=str))


def main():
    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app
    import requests

    crm1_path = THIS_DIR / "crm1_reference" / "CRM1_Ran_only.pdb"
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path} -- aborting.")
        return
    STATE_DIR.mkdir(exist_ok=True)

    state = load_state()

    print(f"{'='*100}\nACK1 rank 1 (478-487, FPDRIDELYL), {CONFORMATION} -- "
          f"seed-screen (3x{SCREEN_DURATION_NS:g}ns) then continue winner to {TARGET_TOTAL_NS:g}ns\n{'='*100}")

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
    candidate = {"sequence": CANDIDATE['sequence'], "start": CANDIDATE['start'],
                 "end": CANDIDATE['end'], "full_sequence": None}

    # ---- STAGE A: screen 3 independent short replicates ----
    print(f"\n{'='*100}\nSTAGE A: {N_SCREENS}x {SCREEN_DURATION_NS:g}ns independent screens\n{'='*100}")
    est_screen_min = estimate_md_time(1, SCREEN_DURATION_NS)
    while len(state["screens"]) < N_SCREENS:
        i = len(state["screens"]) + 1
        state_path = STATE_DIR / f"rank1_screen{i}_state.xml"
        print(f"\n-- screen {i}/{N_SCREENS} (~{est_screen_min:.0f} min) --")
        result = refiner._run_crm1_docking(
            pdb_content=pdb_content,
            candidate=candidate,
            duration_ns=SCREEN_DURATION_NS,
            starting_conformation=CONFORMATION,
            scramble_registration=False,
            save_final_state_path=str(state_path),
        )
        metrics = result.get('md_metrics', {}) or {}
        occ = metrics.get('anchor_occupancy_score')
        print(f"   anchor_occupancy_score={occ}")
        state["screens"].append({
            "screen_index": i,
            "state_path": str(state_path),
            "anchor_occupancy_score": occ,
            "raw_binding_score": metrics.get('raw_binding_score'),
        })
        save_state(state)
        print(f"   Checkpointed ({len(state['screens'])}/{N_SCREENS} screens done)")

    # ---- pick winner ----
    scored = [s for s in state["screens"] if s.get("anchor_occupancy_score") is not None]
    if not scored:
        print("\nNo screen produced a valid anchor_occupancy_score -- cannot pick a winner. Aborting.")
        return
    winner = max(scored, key=lambda s: s["anchor_occupancy_score"])
    print(f"\n{'='*100}\nSTAGE A COMPLETE -- winner: screen {winner['screen_index']} "
          f"(anchor_occupancy_score={winner['anchor_occupancy_score']})\n{'='*100}")
    for s in state["screens"]:
        marker = "  <== WINNER" if s["screen_index"] == winner["screen_index"] else ""
        print(f"  screen {s['screen_index']}: anchor_occupancy_score={s['anchor_occupancy_score']}{marker}")

    # ---- STAGE B: continue the winning trajectory ----
    if state.get("extension") is not None:
        print(f"\nStage B already complete, skipping (see {CACHE_PATH}).")
    else:
        print(f"\n{'='*100}\nSTAGE B: continuing screen {winner['screen_index']}'s trajectory "
              f"{EXTENSION_NS:g} more ns (-> {TARGET_TOTAL_NS:g}ns cumulative)\n{'='*100}")
        est_ext_min = estimate_md_time(1, EXTENSION_NS)
        print(f"Estimated: ~{est_ext_min:.0f} min\n")

        final_pdb = THIS_DIR / f"ack1_rank1_{CONFORMATION}_continued_{TARGET_TOTAL_NS:g}ns_final_complex.pdb"
        best_pdb = THIS_DIR / f"ack1_rank1_{CONFORMATION}_continued_{TARGET_TOTAL_NS:g}ns_best_anchor_frame_complex.pdb"

        result = refiner._run_crm1_docking(
            pdb_content=pdb_content,
            candidate=candidate,
            duration_ns=EXTENSION_NS,
            starting_conformation=CONFORMATION,
            scramble_registration=False,
            resume_from_state_path=winner["state_path"],
            save_final_complex_pdb_path=str(final_pdb),
            save_best_anchor_frame_pdb_path=str(best_pdb),
        )
        metrics = result.get('md_metrics', {}) or {}

        print(f"\n{'='*100}\nSTAGE B COMPLETE\n{'='*100}")
        print(f"anchor_occupancy_score: {metrics.get('anchor_occupancy_score')}")
        print(f"mean_anchor_burial_nm2: {metrics.get('mean_anchor_burial_nm2')}")
        print(f"anchor_burial_fraction_well_buried: {metrics.get('anchor_burial_fraction_well_buried')}")
        print(f"best_anchor_frame_well_buried_count: {metrics.get('best_anchor_frame_well_buried_count')}")
        if final_pdb.exists():
            print(f"Final-frame pose:       {final_pdb.name}")
        if best_pdb.exists():
            print(f"Best-anchor-frame pose: {best_pdb.name}")

        state["extension"] = {
            "continued_from_screen": winner["screen_index"],
            "screen_occupancy": winner["anchor_occupancy_score"],
            "extension_duration_ns": EXTENSION_NS,
            "cumulative_duration_ns": TARGET_TOTAL_NS,
            "metrics": metrics,
            "final_pdb": final_pdb.name,
            "best_anchor_frame_pdb": best_pdb.name,
        }
        save_state(state)
        print(f"\nWrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
