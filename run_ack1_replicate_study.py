#!/usr/bin/env python3
"""
run_ack1_replicate_study.py
============================================================
Two-stage replicate study for all 6 ACK1 (Q07912) NES candidates,
answering "which conformation is real" with actual replicate evidence
instead of a single noisy 10 ns trajectory per variant (which already
showed swings as large as 0.538 -> 0.000 for the same candidate/
conformation across independent runs -- MD stochasticity, not signal).

STAGE A -- 3x 2 ns replicates of EVERY variant (all conformations x
correct/scrambled) for all 6 candidates. Seeded from
ack1_replicate_study_seed.json (built locally from the three already-
completed result files: the original 10 ns 6-variant grid for all 6
candidates, plus the 10 ns top-variant-only PDB pull for candidates 1-5)
-- so most (candidate, tag) combos already have 1-2 real data points and
only need 1-2 more @2ns to reach 3 total. 59 new runs total across all 6
candidates. NOTE: replicates are intentionally MIXED DURATION (2 existing
@10ns + 1 new @2ns for most primary-variant tags; 1 existing @10ns + 2 new
@2ns for everything else) -- each run's duration_ns is recorded alongside
its score specifically so this isn't silently treated as apples-to-apples
later.

STAGE B -- once EVERY (candidate, tag) has reached 3 replicates, picks
the winning conformation per candidate by the largest MEAN correct-minus-
scrambled anchor_occupancy_score gap across all its replicates (the same
specificity-gap logic used throughout this project, now backed by real
replicate evidence instead of a single run or the sequence-only
classifier alone). Runs ONE 20 ns production trajectory of that winning
conformation, correct registration only, WITH save_final_complex_pdb_path
(the final complex PDB) -- and since this calls _run_crm1_docking
directly, the returned metrics automatically include
ramachandran_trace, anchor_burial_nm2/anchor_burial_fraction_well_buried,
dssp_helix_fraction_trace, and conformation_prediction, same as every
other run this project.

CHECKPOINTED AT EVERY SINGLE RUN (not per-candidate) in both stages --
safe to lose the pod mid-run at any point; re-running this exact command
resumes from exactly where it left off, both within Stage A and into
Stage B.

USAGE:
    python3 run_ack1_replicate_study.py
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ACK1_ACCESSION = "Q07912"
CANDIDATES = [
    {"rank": 1, "start": 478, "end": 487, "sequence": "FPDRIDELYL"},
    {"rank": 2, "start": 528, "end": 537, "sequence": "LSSDFKRLGL"},
    {"rank": 3, "start": 995, "end": 1007, "sequence": "VEQLFGLGLRPRG"},
    {"rank": 4, "start": 373, "end": 387, "sequence": "EDRPTFVALRDFLLE"},
    {"rank": 5, "start": 11, "end": 21, "sequence": "LELLSEVQLQQ"},
    {"rank": 6, "start": 207, "end": 221, "sequence": "LAPLGSLLDRLRKHQ"},
]
CONFORMATIONS_BY_RANK = {
    1: ['native', 'idealized_helix', 'extended'],
    2: ['native', 'idealized_helix'],
    3: ['native', 'idealized_helix', 'extended'],
    4: ['native', 'idealized_helix', 'extended'],
    5: ['native', 'idealized_helix'],
    6: ['native', 'idealized_helix', 'extended'],
}
TARGET_REPLICATES = 3
STAGE_A_DURATION_NS = 2.0
STAGE_B_DURATION_NS = 20.0

CACHE_PATH = THIS_DIR / 'ack1_replicate_study_result.json'
SEED_PATH = THIS_DIR / 'ack1_replicate_study_seed.json'
POSE_DIR = THIS_DIR / 'ack1_replicate_study_stage_b_poses'


def load_or_seed_state():
    if CACHE_PATH.exists():
        print(f"Resuming from {CACHE_PATH}")
        return json.loads(CACHE_PATH.read_text())
    if not SEED_PATH.exists():
        print(f"No seed file at {SEED_PATH} and no existing checkpoint -- aborting.")
        sys.exit(1)
    print(f"Starting fresh from seed {SEED_PATH}")
    return json.loads(SEED_PATH.read_text())


def save_state(state):
    CACHE_PATH.write_text(json.dumps(state, indent=2, default=str))


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
    model_id = chosen["model_id"]
    print(f"  Using {model_id} ({chosen.get('numResidues')} aa)")
    pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
    dl = requests.get(pdb_url, timeout=30)
    if dl.status_code != 200:
        raise RuntimeError(f"Download failed ({dl.status_code})")
    print(f"  Downloaded {len(dl.text)} bytes\n")
    return dl.text


def run_stage_a(refiner, pdb_content, state):
    print(f"\n{'='*100}\nSTAGE A: replicate every variant to {TARGET_REPLICATES}x (@{STAGE_A_DURATION_NS} ns each)\n{'='*100}")
    all_done = True
    for c in CANDIDATES:
        rank = c['rank']
        rank_key = str(rank)
        state['stage_a'].setdefault(rank_key, {})
        conformations = CONFORMATIONS_BY_RANK[rank]
        tags = [(conf, scr) for conf in conformations for scr in (False, True)]

        for conf, scr in tags:
            tag = conf if not scr else f"{conf}_scrambled"
            runs = state['stage_a'][rank_key].setdefault(tag, [])
            n_needed = TARGET_REPLICATES - len(runs)
            if n_needed <= 0:
                continue
            all_done = False

            for i in range(n_needed):
                print(f"\n-- rank {rank} ({c['sequence']}) / {tag} -- "
                      f"replicate {len(runs)+1}/{TARGET_REPLICATES} @ {STAGE_A_DURATION_NS} ns --")
                candidate = {"sequence": c['sequence'], "start": c['start'], "end": c['end'], "full_sequence": None}
                result = refiner._run_crm1_docking(
                    pdb_content=pdb_content, candidate=candidate,
                    duration_ns=STAGE_A_DURATION_NS, starting_conformation=conf,
                    scramble_registration=scr,
                )
                m = result.get('md_metrics', {}) or {}
                occ = m.get('anchor_occupancy_score')
                rbs = m.get('raw_binding_score')
                print(f"   anchor_occupancy_score={occ}  raw_binding_score={rbs}")

                runs.append({
                    'duration_ns': STAGE_A_DURATION_NS, 'source': 'replicate_study_stage_a',
                    'anchor_occupancy_score': occ, 'raw_binding_score': rbs,
                })
                save_state(state)  # CHECKPOINT after every single run
                print(f"   Checkpointed ({len(runs)}/{TARGET_REPLICATES} replicates for this tag)")

    return all_done


def pick_winners(state):
    winners = {}
    print(f"\n{'='*100}\nSTAGE A COMPLETE -- picking winning conformation per candidate\n{'='*100}")
    for c in CANDIDATES:
        rank = c['rank']
        rank_key = str(rank)
        conformations = CONFORMATIONS_BY_RANK[rank]
        best_conf, best_gap = None, None
        print(f"\nrank {rank} ({c['sequence']}):")
        for conf in conformations:
            correct_runs = state['stage_a'][rank_key].get(conf, [])
            scr_runs = state['stage_a'][rank_key].get(f"{conf}_scrambled", [])
            correct_vals = [r['anchor_occupancy_score'] for r in correct_runs if r.get('anchor_occupancy_score') is not None]
            scr_vals = [r['anchor_occupancy_score'] for r in scr_runs if r.get('anchor_occupancy_score') is not None]
            mean_correct = sum(correct_vals) / len(correct_vals) if correct_vals else 0.0
            mean_scr = sum(scr_vals) / len(scr_vals) if scr_vals else 0.0
            gap = mean_correct - mean_scr
            print(f"    {conf:16s} mean_correct={mean_correct:.3f} (n={len(correct_vals)})  "
                  f"mean_scrambled={mean_scr:.3f} (n={len(scr_vals)})  gap={gap:+.3f}")
            if best_gap is None or gap > best_gap:
                best_conf, best_gap = conf, gap
        print(f"  -> WINNER: {best_conf} (gap={best_gap:+.3f})")
        winners[rank_key] = best_conf
    return winners


def run_stage_b(refiner, pdb_content, state, winners):
    print(f"\n{'='*100}\nSTAGE B: 20 ns production run of the winning conformation per candidate\n{'='*100}")
    POSE_DIR.mkdir(exist_ok=True)
    for c in CANDIDATES:
        rank = c['rank']
        rank_key = str(rank)
        if rank_key in state['stage_b']:
            print(f"rank {rank}: Stage B already done, skipping")
            continue
        conf = winners[rank_key]
        print(f"\n-- rank {rank} ({c['sequence']}) / {conf} -- {STAGE_B_DURATION_NS} ns production --")
        candidate = {"sequence": c['sequence'], "start": c['start'], "end": c['end'], "full_sequence": None}
        pose_path = POSE_DIR / f"ack1_rank{rank}_{conf}_{STAGE_B_DURATION_NS:g}ns_complex.pdb"

        result = refiner._run_crm1_docking(
            pdb_content=pdb_content, candidate=candidate,
            duration_ns=STAGE_B_DURATION_NS, starting_conformation=conf,
            scramble_registration=False,
            save_final_complex_pdb_path=str(pose_path),
        )
        metrics = result.get('md_metrics', {}) or {}
        print(f"   anchor_occupancy_score={metrics.get('anchor_occupancy_score')}  "
              f"raw_binding_score={metrics.get('raw_binding_score')}")
        dssp_trace = metrics.get('dssp_helix_fraction_trace') or []
        if dssp_trace:
            print(f"   DSSP helix fraction (mean): {sum(dssp_trace)/len(dssp_trace):.4f}")
        rama = metrics.get('ramachandran_trace')
        print(f"   ramachandran_trace present: {rama is not None}")
        print(f"   mean_anchor_burial_nm2: {metrics.get('mean_anchor_burial_nm2')}")
        if pose_path.exists():
            print(f"   Saved pose: {pose_path.name}")

        state['stage_b'][rank_key] = {
            'rank': rank, 'sequence': c['sequence'], 'conformation': conf,
            'duration_ns': STAGE_B_DURATION_NS, 'metrics': metrics, 'pose_pdb': str(pose_path.name),
        }
        save_state(state)  # CHECKPOINT after every candidate's stage B run
        print(f"   Checkpointed Stage B ({len(state['stage_b'])}/6 candidates done)")


def main():
    from md_refinement import NESMDRefiner

    crm1_path = THIS_DIR / 'crm1_reference' / 'CRM1_Ran_only.pdb'
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path} -- aborting.")
        return

    state = load_or_seed_state()
    state.setdefault('stage_a', {})
    state.setdefault('stage_b', {})

    pdb_content = download_ack1_structure()
    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))

    stage_a_done = run_stage_a(refiner, pdb_content, state)
    if not stage_a_done:
        # run_stage_a only returns True if NOTHING needed running (i.e. it's
        # already fully done from a prior partial run) -- if it actually did
        # work this call, re-check completeness directly before proceeding.
        stage_a_done = all(
            len(state['stage_a'][str(c['rank'])].get(tag, [])) >= TARGET_REPLICATES
            for c in CANDIDATES
            for conf in CONFORMATIONS_BY_RANK[c['rank']]
            for tag in (conf, f"{conf}_scrambled")
        )
    if not stage_a_done:
        print("\nStage A did not fully complete (unexpected) -- re-run this command to continue.")
        return

    winners = pick_winners(state)
    run_stage_b(refiner, pdb_content, state, winners)

    print(f"\n{'='*100}\nREPLICATE STUDY COMPLETE\n{'='*100}")
    for c in CANDIDATES:
        rank_key = str(c['rank'])
        sb = state['stage_b'].get(rank_key, {})
        print(f"  rank {c['rank']} ({c['sequence']}): winner={sb.get('conformation')}  "
              f"final_anchor_occ={sb.get('metrics', {}).get('anchor_occupancy_score')}  "
              f"pose={sb.get('pose_pdb')}")
    print(f"\nWrote {CACHE_PATH}")
    print(f"Poses in: {POSE_DIR}/")


if __name__ == '__main__':
    main()
