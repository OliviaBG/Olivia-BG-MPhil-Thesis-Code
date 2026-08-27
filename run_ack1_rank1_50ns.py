#!/usr/bin/env python3
"""
run_ack1_rank1_50ns.py
============================================================
Extends the ACK1 rank 1 (478-487, FPDRIDELYL), idealized_helix, correct-
registration trajectory out to 50 ns cumulative by RESUMING a saved OpenMM
state (positions + velocities) rather than starting from scratch.

WHY IT RESUMES A 2 ns SCREEN AND NOT THE 20 ns RUN
------------------------------------------------------------
run_ack1_rank1_continued_trajectory.py's Stage B passed
resume_from_state_path / save_final_complex_pdb_path /
save_best_anchor_frame_pdb_path -- but NOT save_final_state_path. So no
positions+velocities XML was ever written at the 20 ns mark, and there is
nothing on disk to Simulation.loadState() from at 20 ns. The only saved
states for this candidate are the three 2 ns Stage A screens in
ack1_rank1_screen_states/.

This script resumes the SAME winning screen the 20 ns run resumed
(screen 2, anchor_occupancy_score 0.457) and continues it 48 more ns to
50 ns cumulative. Because no random seed is fixed anywhere in this
pipeline, that means:

  *** The resulting 50 ns trajectory shares only its first 2 ns with the
  existing 20 ns run, then diverges. It SUPERSEDES the 20 ns result
  (anchor_occupancy_score 0.586) rather than extending it. The two are
  independent continuations of the same starting basin, not the same
  trajectory at two lengths -- do not report them as such. ***

IF THE SCREEN STATES ARE GONE (--rebuild-screens)
------------------------------------------------------------
Those XMLs lived on the pod, not in the project folder, so a pod rebuild
destroys them. With --rebuild-screens this script re-runs Stage A itself
(3 independent 2 ns screens, ~12 min of GPU total), saves the states
properly, picks the winner by anchor_occupancy_score, and continues from
there. That is a faithful repeat of the original protocol -- but it is a
NEW screen set: a different screen may win, with a different occupancy,
and the resulting trajectory is a fresh sample rather than any kind of
continuation of previously reported numbers. Both the rebuilt screens and
which one won are recorded in the output JSON so the provenance is not
lost.

CHUNKED SO A LOST POD COSTS ONE CHUNK, NOT THE WHOLE RUN
------------------------------------------------------------
The 48 ns extension runs as consecutive chunks (default 8 ns), each
resuming the previous chunk's saved state and saving its own.
Checkpointed to ack1_rank1_50ns_result.json after every chunk -- safe to
interrupt and re-run this exact command; completed chunks are skipped.

Chunking is not just insurance. Each chunk returns its own full metrics
dict, so you get anchor_occupancy_score / anchor burial as a function of
simulated time across 2 -> 50 ns: direct evidence of whether the pose is
converged or still drifting, which is the reason for going to 50 ns at
all. Pass --chunk-ns 48 to run one continuous block instead, if you'd
rather have a single 48 ns averaging window directly comparable to the
existing run's 18 ns one.

NOTE ON METRIC WINDOWS: every metric a chunk reports is averaged over
THAT CHUNK's production window only, and each chunk's
production_time_series_ps restarts at 0. Each chunk record stores
cumulative_start_ns / cumulative_end_ns so traces can be stitched onto a
real 2 -> 50 ns axis afterwards. The headline number is the FINAL chunk's
anchor_occupancy_score (the most converged window), not a mean across
chunks.

Does NOT touch ack1_rank1_continued_trajectory_result.json or any of the
existing 20 ns PDBs -- new outputs only.

REQUIREMENTS: run on the pod (OpenMM with a working CUDA platform,
PDBFixer, mdtraj -- see pod_setup.sh), from the folder holding
md_refinement.py, app.py and crm1_reference/. Roughly 1.5-2 h of GPU time
for 48 ns at the pipeline's own 0.5 ns/min estimate.

USAGE:
    python3 run_ack1_rank1_50ns.py --dry-run          # check resumability, run nothing
    python3 run_ack1_rank1_50ns.py                    # resume existing screen -> 50 ns
    python3 run_ack1_rank1_50ns.py --rebuild-screens  # screens are gone: redo Stage A first
    python3 run_ack1_rank1_50ns.py --chunk-ns 48      # one continuous block
    python3 run_ack1_rank1_50ns.py --target-ns 100    # keep going further
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ACK1_ACCESSION = "Q07912"
CANDIDATE = {"rank": 1, "start": 478, "end": 487, "sequence": "FPDRIDELYL"}
CONFORMATION = "idealized_helix"   # replicate-study winner for rank 1
SCREEN_CUMULATIVE_NS = 2.0         # what a Stage A screen state represents
N_SCREENS = 3                      # matches the original Stage A protocol

SCREEN_RESULT_PATH = THIS_DIR / "ack1_rank1_continued_trajectory_result.json"
SCREEN_STATE_DIR = THIS_DIR / "ack1_rank1_screen_states"
CACHE_PATH = THIS_DIR / "ack1_rank1_50ns_result.json"
STATE_DIR = THIS_DIR / "ack1_rank1_50ns_states"


def load_state():
    if CACHE_PATH.exists():
        print(f"Resuming from checkpoint {CACHE_PATH.name}")
        return json.loads(CACHE_PATH.read_text())
    return {"resumed_from": None, "rebuilt_screens": [], "chunks": []}


def save_state(state):
    CACHE_PATH.write_text(json.dumps(state, indent=2, default=str))


def find_existing_screen():
    """Locate the highest-scoring original 2 ns screen and its state XML.

    Returns a dict on success, or None if the prior result or the state file
    itself is missing. Paths recorded in the prior result are absolute pod
    paths (/root/AlphaFold/...); honour them if they exist, else fall back to
    ack1_rank1_screen_states/ next to this script, so a moved folder or a
    rebuilt pod with restored states still works.
    """
    if not SCREEN_RESULT_PATH.exists():
        print(f"  {SCREEN_RESULT_PATH.name} not found.")
        return None

    prior = json.loads(SCREEN_RESULT_PATH.read_text())
    screens = [s for s in prior.get("screens", []) if s.get("anchor_occupancy_score") is not None]
    if not screens:
        print(f"  {SCREEN_RESULT_PATH.name} has no screen with a valid anchor_occupancy_score.")
        return None

    winner = max(screens, key=lambda s: s["anchor_occupancy_score"])
    idx = winner["screen_index"]
    print(f"  Prior winning screen: {idx} (anchor_occupancy_score="
          f"{winner['anchor_occupancy_score']:.4f}) -- the screen the 20 ns run continued.")

    for path, label in ((Path(winner["state_path"]), "recorded path"),
                        (SCREEN_STATE_DIR / f"rank1_screen{idx}_state.xml", "local screen-state dir")):
        if path.exists():
            print(f"  State file found via {label}: {path}")
            return {"origin": "original_stage_a",
                    "screen_index": idx,
                    "screen_occupancy": winner["anchor_occupancy_score"],
                    "state_path": str(path)}

    print(f"  State file NOT found. Tried:")
    print(f"    recorded : {winner['state_path']}")
    print(f"    local    : {SCREEN_STATE_DIR / f'rank1_screen{idx}_state.xml'}")
    return None


def screen_summary(screens):
    """Print the whole screen pool as a distribution, not just the winner.

    The winner alone is a biased view: 'best of N' is only meaningful if N and
    the spread are on the record. This is what makes a re-run with more seeds
    defensible rather than a quiet re-roll until the number looks good.
    """
    vals = sorted((s["anchor_occupancy_score"] for s in screens
                   if s.get("anchor_occupancy_score") is not None))
    if not vals:
        print("  (no screen produced a valid anchor_occupancy_score)")
        return
    n = len(vals)
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    nonzero = sum(1 for v in vals if v > 1e-9)
    print(f"  pool n={n}   min={vals[0]:.4f}   median={median:.4f}   max={vals[-1]:.4f}")
    print(f"  non-zero screens: {nonzero}/{n}")
    print(f"  all: {', '.join(f'{v:.4f}' for v in vals)}")


def rebuild_screens(refiner, pdb_content, state, target_n):
    """Run independent short screens until the pool holds target_n of them.

    APPENDS -- screens already in the checkpoint are kept and counted, so
    asking for more seeds later grows the pool rather than replacing it, and
    every attempt stays on the record. Checkpointed after every screen, so an
    interruption costs at most one.
    """
    from md_refinement import estimate_md_time

    SCREEN_STATE_DIR.mkdir(exist_ok=True)
    have = len(state["rebuilt_screens"])
    todo = max(0, target_n - have)
    est = estimate_md_time(1, SCREEN_CUMULATIVE_NS)
    print(f"\n{'='*100}")
    print(f"STAGE A SCREENS: pool target {target_n}, {have} already recorded, "
          f"{todo} to run @ {SCREEN_CUMULATIVE_NS:g} ns (~{est:.0f} min each)")
    print(f"{'='*100}")

    candidate = {"sequence": CANDIDATE["sequence"], "start": CANDIDATE["start"],
                 "end": CANDIDATE["end"], "full_sequence": None}

    while len(state["rebuilt_screens"]) < target_n:
        i = len(state["rebuilt_screens"]) + 1
        out_state = SCREEN_STATE_DIR / f"rank1_rebuilt_screen{i}_state.xml"
        print(f"\n-- rebuilt screen {i}/{N_SCREENS} @ {SCREEN_CUMULATIVE_NS:g} ns --")
        result = refiner._run_crm1_docking(
            pdb_content=pdb_content,
            candidate=candidate,
            duration_ns=SCREEN_CUMULATIVE_NS,
            starting_conformation=CONFORMATION,
            scramble_registration=False,
            save_final_state_path=str(out_state),
        )
        metrics = result.get("md_metrics", {}) or {}
        occ = metrics.get("anchor_occupancy_score")
        print(f"   anchor_occupancy_score={occ}")
        state["rebuilt_screens"].append({
            "screen_index": i,
            "state_path": str(out_state),
            "anchor_occupancy_score": occ,
            "raw_binding_score": metrics.get("raw_binding_score"),
        })
        save_state(state)
        print(f"   Checkpointed ({len(state['rebuilt_screens'])}/{target_n})")

    return pick_screen_winner(state)


def pick_screen_winner(state):
    """Highest-occupancy screen in the whole pool, with the distribution shown."""
    scored = [s for s in state["rebuilt_screens"] if s.get("anchor_occupancy_score") is not None]
    if not scored:
        print("\nNo screen produced a valid anchor_occupancy_score -- cannot pick a winner.")
        return None

    winner = max(scored, key=lambda s: s["anchor_occupancy_score"])
    print(f"\n{'='*100}")
    print(f"SCREEN POOL COMPLETE -- winner: screen {winner['screen_index']} "
          f"(anchor_occupancy_score={winner['anchor_occupancy_score']:.4f})")
    print(f"{'='*100}")
    for s in sorted(state["rebuilt_screens"], key=lambda x: x["screen_index"]):
        mark = "  <== WINNER" if s["screen_index"] == winner["screen_index"] else ""
        print(f"  screen {s['screen_index']}: anchor_occupancy_score={s['anchor_occupancy_score']}{mark}")
    print()
    screen_summary(state["rebuilt_screens"])
    if winner["anchor_occupancy_score"] < 0.2:
        print()
        print(f"  NOTE: the best seed in this pool is only "
              f"{winner['anchor_occupancy_score']:.4f}. For reference the original Stage A")
        print(f"  screens were 0.0000 / 0.4574 / 0.0000 -- so a weak pool is a plausible draw,")
        print(f"  but continuing a weak seed for 48 ns may simply confirm it stays weak.")
        print(f"  Consider --n-screens with a larger pool before committing the GPU hours.")

    return {"origin": "rebuilt_stage_a",
            "screen_index": winner["screen_index"],
            "screen_occupancy": winner["anchor_occupancy_score"],
            "state_path": winner["state_path"]}


def build_plan(chunk_ns, target_ns, done_chunks):
    """Consecutive chunk boundaries from the screen's 2 ns out to target_ns."""
    plan = []
    cursor = SCREEN_CUMULATIVE_NS
    while cursor < target_ns - 1e-9:
        length = min(chunk_ns, target_ns - cursor)
        plan.append({"index": len(plan) + 1,
                     "cumulative_start_ns": round(cursor, 6),
                     "cumulative_end_ns": round(cursor + length, 6),
                     "duration_ns": round(length, 6)})
        cursor += length
    for c in plan:
        c["done"] = any(d["index"] == c["index"] for d in done_chunks)
    return plan


def download_ack1_structure():
    from app import app as flask_app
    import requests

    client = flask_app.test_client()
    print(f"Resolving AlphaFold model_id for {ACK1_ACCESSION} ...")
    resp = client.get(f"/api/models/{ACK1_ACCESSION}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} resolving model")
    models = resp.get_json() or []
    alphafold_models = [m for m in models if m.get("source") == "alphafold"]
    canonical_id = f"AF-{ACK1_ACCESSION}-F1"
    by_id = {m["model_id"]: m for m in alphafold_models}
    chosen = by_id.get(canonical_id) or max(
        (m for m in alphafold_models if (m.get("numResidues") or 0) >= 1038),
        key=lambda m: m.get("numResidues") or 0, default=None,
    )
    if chosen is None:
        raise RuntimeError("No usable AlphaFold structure found")
    print(f"  Using {chosen['model_id']} ({chosen.get('numResidues')} aa)")
    dl = requests.get(f"https://alphafold.ebi.ac.uk/files/{chosen['model_id']}.pdb", timeout=30)
    if dl.status_code != 200:
        raise RuntimeError(f"Download failed ({dl.status_code})")
    print(f"  {len(dl.text)} bytes\n")
    return chosen["model_id"], dl.text


def report_platform():
    """Print which OpenMM platform will actually be used, before burning GPU hours."""
    try:
        from openmm import Platform
    except ImportError:
        print("  (openmm not importable yet -- platform check skipped)")
        return
    names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    print(f"  OpenMM platforms registered: {names}")
    if "CUDA" in names:
        print("  -> md_refinement._select_best_platform() will pick CUDA.")
    else:
        print("  -> WARNING: no CUDA platform. This will run on CPU and 48 ns will take days.")
        print("     Fix the install first (see pod_setup.sh) rather than starting this run.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-ns", type=float, default=50.0,
                    help="Cumulative trajectory length to reach, in ns (default 50.0)")
    ap.add_argument("--chunk-ns", type=float, default=8.0,
                    help="Length of each resumable chunk, in ns (default 8.0). Use --chunk-ns 48 "
                         "for one continuous block with a single averaging window.")
    ap.add_argument("--rebuild-screens", action="store_true",
                    help="If the original 2 ns screen states are missing (e.g. the pod they lived "
                         "on was rebuilt), re-run Stage A from scratch instead of aborting. "
                         "Produces a NEW screen set -- see the docstring.")
    ap.add_argument("--n-screens", type=int, default=N_SCREENS,
                    help=f"Target size of the screen pool (default {N_SCREENS}, the original "
                         f"protocol). Screens already recorded are KEPT and counted, so raising "
                         f"this runs only the shortfall and grows the pool -- every attempt stays "
                         f"on the record, so 'best of N' is reported with the real N.")
    ap.add_argument("--screens-only", action="store_true",
                    help="Run/complete the screen pool, print its distribution, then stop without "
                         "starting the long production chunks. Use this to look at the spread "
                         "before committing GPU hours to whichever seed won.")
    ap.add_argument("--reset-chunks", action="store_true",
                    help="Discard recorded production chunks so a NEW screen winner can be "
                         "continued from scratch. Required when the winner changes after adding "
                         "screens, since the existing chunk chain descends from the old winner. "
                         "Discarded chunks are kept in the JSON under 'discarded_chunks'.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report resumability, platform and the chunk plan, then exit without "
                         "running any MD.")
    args = ap.parse_args()

    print(f"{'='*100}")
    print(f"ACK1 rank 1 ({CANDIDATE['start']}-{CANDIDATE['end']}, {CANDIDATE['sequence']}), "
          f"{CONFORMATION} -> {args.target_ns:g} ns cumulative")
    print(f"{'='*100}\n")

    state = load_state()
    state.setdefault("rebuilt_screens", [])
    state.setdefault("chunks", [])

    if args.reset_chunks and state["chunks"]:
        print(f"--reset-chunks: moving {len(state['chunks'])} recorded chunk(s) to "
              f"'discarded_chunks' and starting the chain over.\n")
        state["discarded_chunks"] = state.get("discarded_chunks", []) + state["chunks"]
        state["chunks"] = []
        save_state(state)

    print("Checking for a resumable state ...")
    pool = state["rebuilt_screens"]
    screens_to_run = 0
    winner = None

    if pool:
        print(f"  Screen pool in checkpoint: {len(pool)} screen(s).")
        if len(pool) < args.n_screens:
            if not args.rebuild_screens:
                print(f"\nABORTING: --n-screens {args.n_screens} asks for a bigger pool than the "
                      f"{len(pool)} recorded,\nbut --rebuild-screens was not given. Re-run with "
                      f"both to add the missing {args.n_screens - len(pool)}.")
                sys.exit(1)
            screens_to_run = args.n_screens - len(pool)
        else:
            winner = pick_screen_winner(state)
    else:
        winner = find_existing_screen()
        if winner is None:
            if not args.rebuild_screens:
                print()
                print("ABORTING: there is no resumable state for this candidate.")
                print()
                print("The 20 ns run never saved a state of its own (Stage B omitted")
                print("save_final_state_path), so the 2 ns Stage A screens were the only")
                print("resumable point -- and they lived on the pod, not in this folder, so a pod")
                print("rebuild destroys them.")
                print()
                print("Either restore ack1_rank1_screen_states/ from a backup, or re-run Stage A:")
                print(f"    python3 {Path(__file__).name} --rebuild-screens")
                print(f"({N_SCREENS} x {SCREEN_CUMULATIVE_NS:g} ns, roughly 12 min of GPU. "
                      f"It is a NEW screen set, not the original one.)")
                sys.exit(1)
            screens_to_run = args.n_screens

    if screens_to_run:
        print(f"  -> will run {screens_to_run} more screen(s) to reach a pool of {args.n_screens}, "
              f"then continue the best of the whole pool.")

    plan = build_plan(args.chunk_ns, args.target_ns, state["chunks"])
    print(f"\nPlan ({len(plan)} chunk(s), {args.chunk_ns:g} ns each, "
          f"{args.target_ns - SCREEN_CUMULATIVE_NS:g} ns of new production MD):")
    for c in plan:
        print(f"  chunk {c['index']}: {c['cumulative_start_ns']:g} -> {c['cumulative_end_ns']:g} ns "
              f"({c['duration_ns']:g} ns)" + ("   [already done]" if c["done"] else ""))
    if screens_to_run:
        print(f"  ...preceded by {screens_to_run} x {SCREEN_CUMULATIVE_NS:g} ns Stage A screen(s).")
    if args.screens_only:
        print("  (--screens-only: the chunks above will NOT run this invocation.)")

    crm1_path = THIS_DIR / "crm1_reference" / "CRM1_Ran_only.pdb"
    if not crm1_path.exists():
        print(f"\nCRM1 reference not found: {crm1_path}")
        print("Aborting -- it must be the SAME reference the screens used, or a saved state "
              "will not load into a compatible Context.")
        sys.exit(1)
    print(f"\nCRM1 reference: {crm1_path.name}")

    print("\nPlatform:")
    report_platform()

    if args.dry_run:
        print("\n--dry-run: nothing was run. The plan above is what would execute.")
        return

    from md_refinement import NESMDRefiner, estimate_md_time

    remaining = 0.0 if args.screens_only else sum(c["duration_ns"] for c in plan if not c["done"])
    remaining += screens_to_run * SCREEN_CUMULATIVE_NS
    est_min = estimate_md_time(1, remaining) if remaining else 0.0
    print(f"\nEstimated: ~{est_min:.0f} min (~{est_min/60:.1f} h) for the {remaining:g} ns "
          f"still to run.\n")

    model_id, pdb_content = download_ack1_structure()
    STATE_DIR.mkdir(exist_ok=True)
    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))

    if screens_to_run:
        winner = rebuild_screens(refiner, pdb_content, state, args.n_screens)
        if winner is None:
            sys.exit(1)

    # A recorded chunk chain descends from one specific seed. If adding screens
    # changed the winner, continuing the old chain under the new winner's name
    # would silently mislabel the provenance -- refuse rather than corrupt it.
    # Skipped under --screens-only, which never advances the chain at all.
    if state["chunks"] and not args.screens_only:
        chain_root = state["chunks"][0].get("resumed_from_state")
        if chain_root and chain_root != winner["state_path"]:
            print(f"\nABORTING: the winning seed changed, but {len(state['chunks'])} production "
                  f"chunk(s) are already recorded.")
            print(f"  existing chain descends from: {chain_root}")
            print(f"  new winner is              : {winner['state_path']}")
            print("\nThose chunks are a continuation of the OLD seed; extending them now would "
                  "attribute\nthem to the new one. Either keep the old winner (drop --n-screens), "
                  "or start the\nchain over from the new winner:")
            print(f"    python3 {Path(__file__).name} --reset-chunks "
                  f"--n-screens {args.n_screens} --rebuild-screens")
            print("(--reset-chunks keeps the discarded chunks in the JSON, it does not delete them.)")
            sys.exit(1)

    if args.screens_only:
        print(f"\n--screens-only: stopping here. Best in pool is screen "
              f"{winner['screen_index']} (occ {winner['screen_occupancy']:.4f}).")
        if state["chunks"]:
            # A production chain is already running off an earlier seed. Leave
            # resumed_from exactly as it is, so relaunching resumes THAT chain
            # rather than silently re-pointing it at whatever just won the pool.
            existing = state.get("resumed_from") or {}
            print(f"\n  {len(state['chunks'])} production chunk(s) already recorded, continuing "
                  f"from screen {existing.get('screen_index')} "
                  f"(occ {existing.get('screen_occupancy', float('nan')):.4f}).")
            print("  The resume pointer was NOT changed -- the screen pool above is recorded as")
            print("  characterisation only. Relaunch without --screens-only and the existing")
            print("  chain picks up where it left off.")
            if winner["state_path"] != (state["chunks"][0].get("resumed_from_state") or ""):
                print(f"\n  NOTE: screen {winner['screen_index']} "
                      f"({winner['screen_occupancy']:.4f}) beat the seed the chain is running on.")
                print("  To abandon the current chain and continue that one instead:")
                print(f"      python3 {Path(__file__).name} --reset-chunks")
        else:
            state["resumed_from"] = dict(winner, source_result=SCREEN_RESULT_PATH.name,
                                         screen_cumulative_ns=SCREEN_CUMULATIVE_NS,
                                         model_id=model_id, conformation=CONFORMATION,
                                         crm1_reference=crm1_path.name)
        save_state(state)
        print(f"\nWrote {CACHE_PATH.name}.")
        return

    state["resumed_from"] = dict(
        winner,
        source_result=SCREEN_RESULT_PATH.name,
        screen_cumulative_ns=SCREEN_CUMULATIVE_NS,
        model_id=model_id,
        conformation=CONFORMATION,
        crm1_reference=crm1_path.name,
        supersedes=("ack1_rank1_continued_trajectory_result.json's 20 ns extension -- "
                    "independent continuation, NOT the same trajectory extended"),
    )
    save_state(state)

    final_pdb = THIS_DIR / f"ack1_rank1_{CONFORMATION}_{args.target_ns:g}ns_final_complex.pdb"
    best_pdb = THIS_DIR / f"ack1_rank1_{CONFORMATION}_{args.target_ns:g}ns_best_anchor_frame_complex.pdb"

    for c in plan:
        if c["done"]:
            print(f"\nchunk {c['index']} already complete, skipping.")
            continue

        if c["index"] == 1:
            resume_path = winner["state_path"]
        else:
            prev = next(d for d in state["chunks"] if d["index"] == c["index"] - 1)
            resume_path = prev["state_path"]

        out_state = STATE_DIR / f"rank1_50ns_chunk{c['index']}_state.xml"
        is_final = c["index"] == plan[-1]["index"]

        print(f"\n{'='*100}")
        print(f"CHUNK {c['index']}/{len(plan)}: {c['cumulative_start_ns']:g} -> "
              f"{c['cumulative_end_ns']:g} ns  ({c['duration_ns']:g} ns of production)")
        print(f"  resuming: {Path(resume_path).name}")
        print(f"{'='*100}")

        kwargs = dict(
            pdb_content=pdb_content,
            candidate={"sequence": CANDIDATE["sequence"], "start": CANDIDATE["start"],
                       "end": CANDIDATE["end"], "full_sequence": None},
            duration_ns=c["duration_ns"],
            starting_conformation=CONFORMATION,
            scramble_registration=False,
            resume_from_state_path=resume_path,
            save_final_state_path=str(out_state),
        )
        if is_final:
            # Only the last chunk's poses are the run's actual deliverable.
            kwargs["save_final_complex_pdb_path"] = str(final_pdb)
            kwargs["save_best_anchor_frame_pdb_path"] = str(best_pdb)

        result = refiner._run_crm1_docking(**kwargs)
        metrics = result.get("md_metrics", {}) or {}

        print(f"\n  chunk {c['index']} done:")
        print(f"    anchor_occupancy_score            : {metrics.get('anchor_occupancy_score')}")
        print(f"    mean_anchor_burial_nm2            : {metrics.get('mean_anchor_burial_nm2')}")
        print(f"    anchor_burial_fraction_well_buried: {metrics.get('anchor_burial_fraction_well_buried')}")
        print(f"    raw_binding_score                 : {metrics.get('raw_binding_score')}")

        state["chunks"].append({
            "index": c["index"],
            "cumulative_start_ns": c["cumulative_start_ns"],
            "cumulative_end_ns": c["cumulative_end_ns"],
            "duration_ns": c["duration_ns"],
            "resumed_from_state": str(resume_path),
            "state_path": str(out_state),
            "metrics": metrics,
            "final_pdb": final_pdb.name if is_final else None,
            "best_anchor_frame_pdb": best_pdb.name if is_final else None,
        })
        save_state(state)
        print(f"    Checkpointed ({len(state['chunks'])}/{len(plan)} chunks) -> {CACHE_PATH.name}")

    print(f"\n{'='*100}\nCOMPLETE -- {args.target_ns:g} ns cumulative\n{'='*100}")
    print(f"{'window (ns)':<18}{'anchor_occ':<14}{'burial_nm2':<14}{'frac_well_buried':<18}")
    screen_label = "0 -> {:g} (screen)".format(SCREEN_CUMULATIVE_NS)
    print(f"{screen_label:<18}{winner['screen_occupancy']:<14.4f}{'-':<14}{'-':<18}")
    for d in sorted(state["chunks"], key=lambda x: x["index"]):
        m = d["metrics"]
        occ, bur = m.get("anchor_occupancy_score"), m.get("mean_anchor_burial_nm2")
        fwb = m.get("anchor_burial_fraction_well_buried")
        window = "{:g} -> {:g}".format(d["cumulative_start_ns"], d["cumulative_end_ns"])
        print(f"{window:<18}"
              f"{(occ if occ is not None else float('nan')):<14.4f}"
              f"{(bur if bur is not None else float('nan')):<14.4f}"
              f"{(fwb if fwb is not None else float('nan')):<18.4f}")

    last = max(state["chunks"], key=lambda x: x["index"])
    print(f"\nHEADLINE (final {last['duration_ns']:g} ns window, "
          f"{last['cumulative_start_ns']:g} -> {last['cumulative_end_ns']:g} ns):")
    print(f"  anchor_occupancy_score = {last['metrics'].get('anchor_occupancy_score')}")
    if final_pdb.exists():
        print(f"  Final-frame pose       : {final_pdb.name}")
    if best_pdb.exists():
        print(f"  Best-anchor-frame pose : {best_pdb.name}")
    print(f"\nWrote {CACHE_PATH}")
    print(f"States in: {STATE_DIR}/  (every chunk boundary is resumable -- "
          f"re-run with a larger --target-ns to go further)")
    print(f"\nPROVENANCE: continued from {winner['origin']} screen {winner['screen_index']} "
          f"(occ {winner['screen_occupancy']:.4f}).")
    print("This trajectory shares only its first 2 ns with the existing 20 ns run and then")
    print("diverges. It supersedes that result; do not report the two as one trajectory")
    print("sampled at two lengths.")


if __name__ == "__main__":
    main()
