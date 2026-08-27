#!/usr/bin/env python3
"""
pull_ack1_top_variant_pdbs.py
============================================================
Candidates 1-5 of the ACK1 clean re-run (run_ack1_all6_clean_rerun.py)
never had their docked structures saved -- that script goes through
refine_nes_candidates(), which doesn't expose save_final_complex_pdb_path,
so only the JSON metrics survived, not the actual poses (candidate 6 was
fixed separately, via run_ack1_candidate6_checkpointed.py, and already has
its PDBs).

Re-running all 6-24 variants per candidate just to get one structure each
would be wasteful. This script instead runs ONLY each candidate's own
PRIMARY (best-classified) variant, CORRECT registration only (no
scrambled control, no other conformations) -- exactly 5 MD trajectories
total, one per candidate, each with save_final_complex_pdb_path wired in.

Primary variant per candidate, taken directly from
ack1_all6_FINAL_clean_result.json's md_best_starting_conformation:
  rank 1 (478-487,  FPDRIDELYL)      -> extended
  rank 2 (528-537,  LSSDFKRLGL)      -> idealized_helix
  rank 3 (995-1007, VEQLFGLGLRPRG)   -> extended
  rank 4 (373-387,  EDRPTFVALRDFLLE) -> extended
  rank 5 (11-21,    LELLSEVQLQQ)     -> idealized_helix

Same protocol as the original run: 10 ns, clean crm1_reference/CRM1_Ran_only.pdb,
correct (non-scrambled) registration -- since these are for visualizing the
actual reported result, not re-testing specificity (that's already been
established by the full 6-variant grid these numbers came from).

CHECKPOINTED per candidate (writes/updates the result JSON after each one
finishes), same resumable pattern as run_ack1_candidate6_checkpointed.py --
safe to interrupt and re-run.

USAGE:
    python3 pull_ack1_top_variant_pdbs.py
    python3 pull_ack1_top_variant_pdbs.py --duration-ns 10.0
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ACK1_ACCESSION = "Q07912"
CANDIDATES = [
    {"rank": 1, "start": 478, "end": 487, "sequence": "FPDRIDELYL", "conformation": "extended"},
    {"rank": 2, "start": 528, "end": 537, "sequence": "LSSDFKRLGL", "conformation": "idealized_helix"},
    {"rank": 3, "start": 995, "end": 1007, "sequence": "VEQLFGLGLRPRG", "conformation": "extended"},
    {"rank": 4, "start": 373, "end": 387, "sequence": "EDRPTFVALRDFLLE", "conformation": "extended"},
    {"rank": 5, "start": 11, "end": 21, "sequence": "LELLSEVQLQQ", "conformation": "idealized_helix"},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=10.0,
                     help='Matches the original run_ack1_all6_clean_rerun.py protocol these scores came from.')
    ap.add_argument('--crm1-ref', default='crm1_reference/CRM1_Ran_only.pdb')
    ap.add_argument('--cache', default='ack1_top_variant_pdbs_result.json')
    ap.add_argument('--pdb-out-dir', default='ack1_top_variant_poses')
    args = ap.parse_args()

    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app
    import requests

    crm1_path = THIS_DIR / args.crm1_ref
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path} -- aborting.")
        return
    pdb_out_dir = THIS_DIR / args.pdb_out_dir
    pdb_out_dir.mkdir(exist_ok=True)

    print(f"{'='*100}\nACK1 (Q07912) -- top-variant-only PDB pull, 5 candidates\n{'='*100}")
    print(f"Duration: {args.duration_ns} ns per candidate (1 trajectory each, correct registration only)")
    print(f"CRM1 reference: {crm1_path} (clean)\n")

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
    print(f"  Using {model_id} ({chosen.get('numResidues')} aa)\n")

    pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
    print(f"Downloading: {pdb_url}")
    dl = requests.get(pdb_url, timeout=30)
    if dl.status_code != 200:
        print(f"  Download failed ({dl.status_code}) -- aborting.")
        return
    pdb_content = dl.text
    print(f"  {len(pdb_content)} bytes\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))

    cache_path = Path(args.cache)
    results = {}
    if cache_path.exists():
        results = json.loads(cache_path.read_text())
        print(f"Resuming: {len(results)}/5 candidates already done: {list(results.keys())}\n")

    est_min_each = estimate_md_time(1, args.duration_ns)

    for c in CANDIDATES:
        key = str(c['rank'])
        if key in results:
            print(f"[rank {c['rank']}] already done, skipping")
            continue

        print(f"\n{'='*80}\n-- rank {c['rank']}: {c['start']}-{c['end']} {c['sequence']} "
              f"({c['conformation']}) --  (~{est_min_each:.0f} min estimated)\n{'='*80}")

        candidate = {"sequence": c['sequence'], "start": c['start'], "end": c['end'], "full_sequence": None}
        pose_path = pdb_out_dir / f"ack1_rank{c['rank']}_{c['conformation']}_{args.duration_ns:g}ns_complex.pdb"

        result = refiner._run_crm1_docking(
            pdb_content=pdb_content,
            candidate=candidate,
            duration_ns=args.duration_ns,
            starting_conformation=c['conformation'],
            scramble_registration=False,
            save_final_complex_pdb_path=str(pose_path),
        )
        metrics = result.get('md_metrics', {}) or {}
        print(f"  anchor_occupancy_score={metrics.get('anchor_occupancy_score')}  "
              f"raw_binding_score={metrics.get('raw_binding_score')}")
        if pose_path.exists():
            print(f"  Saved pose: {pose_path.name}")
        else:
            print(f"  WARNING: {pose_path.name} was not created -- check the log above for a failure path.")

        # CHECKPOINT after every candidate.
        results[key] = {
            'rank': c['rank'], 'start': c['start'], 'end': c['end'], 'sequence': c['sequence'],
            'conformation': c['conformation'], 'metrics': metrics, 'pose_pdb': str(pose_path.name),
        }
        cache_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Checkpointed ({len(results)}/5 candidates done)")

    print(f"\n{'='*100}")
    if len(results) < len(CANDIDATES):
        print(f"{len(results)}/5 done -- re-run this same command to finish the rest.")
    else:
        print("ALL 5 DONE")
        for c in CANDIDATES:
            r = results[str(c['rank'])]
            print(f"  rank {c['rank']}: {r['pose_pdb']}  anchor_occ={r['metrics'].get('anchor_occupancy_score')}")
    print(f"\nWrote {cache_path}")
    print(f"PDBs in: {pdb_out_dir}/")


if __name__ == '__main__':
    main()
