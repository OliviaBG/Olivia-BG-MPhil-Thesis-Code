#!/usr/bin/env python3
"""
run_ack1_candidate6_checkpointed.py
============================================================
Standalone re-run of ONLY the 6th ACK1 candidate (rank 6, 207-221,
LAPLGSLLDRLRKHQ) -- the one that didn't finish before the pod went down
mid-run. Unlike run_ack1_all6_clean_rerun.py (which calls
refine_nes_candidates() and only checkpoints once ALL variants for a
candidate finish, losing everything if interrupted mid-candidate), this
calls _run_crm1_docking() directly, ONE VARIANT AT A TIME, and writes the
checkpoint file after EVERY SINGLE VARIANT -- so an interruption at any
point loses at most one in-progress variant, not the whole candidate.

Also saves the final docked complex PDB for every variant
(save_final_complex_pdb_path) -- addressing the earlier gap where the
first 5 candidates' actual 3-D poses were never written to disk, only
their metrics.

Same protocol as the other 5 candidates for a clean apples-to-apples
result: 10 ns per variant, clean CRM1_Ran_only.pdb reference, both
starting-conformation testing (native + idealized_helix), PLUS 'extended'
(this candidate classifies as extended_atypical, confirmed before this
run started), PLUS scrambled-registration controls for all three -- 6
variants total: native, native_scrambled, idealized_helix,
idealized_helix_scrambled, extended, extended_scrambled.

RESUMABLE: safe to Ctrl-C or lose the pod at any point and re-run this
same command later -- already-completed variants (per the checkpoint
file) are skipped, not re-run.

Output format matches the other 5 candidates' entries in
ack1_all6_clean_rerun_result.json (same keys: rank, start, end, sequence,
binding_mode_classification, md_best_starting_conformation,
md_primary_variant_selection_method, md_metrics_by_variant) so this can
be appended directly to that file to complete the set of 6.

USAGE:
    python3 run_ack1_candidate6_checkpointed.py
    python3 run_ack1_candidate6_checkpointed.py --duration-ns 10.0
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

ACK1_ACCESSION = "Q07912"
CANDIDATE = {"rank": 6, "start": 207, "end": 221, "sequence": "LAPLGSLLDRLRKHQ"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=10.0)
    ap.add_argument('--crm1-ref', default='crm1_reference/CRM1_Ran_only.pdb')
    ap.add_argument('--cache', default='ack1_candidate6_checkpointed_result.json')
    ap.add_argument('--pdb-out-dir', default='ack1_candidate6_poses')
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

    print(f"{'='*100}\nACK1 (Q07912) candidate 6: {CANDIDATE['start']}-{CANDIDATE['end']} "
          f"{CANDIDATE['sequence']} -- per-variant checkpointed re-run\n{'='*100}")
    print(f"Duration: {args.duration_ns} ns per variant")
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
    print(f"  Using {model_id} ({chosen.get('numResidues')} aa)")

    pdb_url = f"https://alphafold.ebi.ac.uk/files/{model_id}.pdb"
    print(f"Downloading: {pdb_url}")
    dl = requests.get(pdb_url, timeout=30)
    if dl.status_code != 200:
        print(f"  Download failed ({dl.status_code}) -- aborting.")
        return
    pdb_content = dl.text
    print(f"  {len(pdb_content)} bytes\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(crm1_path))
    candidate = {"sequence": CANDIDATE['sequence'], "start": CANDIDATE['start'],
                 "end": CANDIDATE['end'], "full_sequence": None}

    binding_mode = refiner.classify_nes_binding_mode(CANDIDATE['sequence'])
    print(f"binding_mode={binding_mode['binding_mode_class']}  "
          f"recommend={binding_mode['recommended_primary_method']}  "
          f"confidence={binding_mode['confidence']}\n")

    conformations = ['native', 'idealized_helix']
    if binding_mode['recommended_primary_method'] == 'extended' and 'extended' not in conformations:
        conformations.append('extended')
    variants = [(conf, scr) for conf in conformations for scr in (False, True)]
    print(f"Variants to run: {[(c, s) for c, s in variants]}\n")

    cache_path = Path(args.cache)
    results = {}
    if cache_path.exists():
        results = json.loads(cache_path.read_text())
        print(f"Resuming: {len(results)}/{len(variants)} variants already done: {list(results.keys())}\n")

    est_min_each = estimate_md_time(1, args.duration_ns)

    for conf, scr in variants:
        tag = conf if not scr else f"{conf}_scrambled"
        if tag in results:
            print(f"[{tag}] already done, skipping")
            continue

        print(f"\n{'='*80}\n-- {tag} --  (~{est_min_each:.0f} min estimated)\n{'='*80}")
        pose_path = pdb_out_dir / f"ack1_rank6_{tag}_{args.duration_ns:g}ns_complex.pdb"

        result = refiner._run_crm1_docking(
            pdb_content=pdb_content,
            candidate=candidate,
            duration_ns=args.duration_ns,
            starting_conformation=conf,
            scramble_registration=scr,
            save_final_complex_pdb_path=str(pose_path),
        )
        metrics = result.get('md_metrics', {}) or {}
        score = result.get('md_enhanced_score', 0.5)

        print(f"  anchor_occupancy_score={metrics.get('anchor_occupancy_score')}  "
              f"raw_binding_score={metrics.get('raw_binding_score')}")
        if pose_path.exists():
            print(f"  Saved pose: {pose_path.name}")

        # CHECKPOINT IMMEDIATELY -- this is the whole point of this script.
        results[tag] = {'metrics': metrics, 'score': score, 'pose_pdb': str(pose_path.name)}
        cache_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Checkpointed ({len(results)}/{len(variants)} variants done)")

    if len(results) < len(variants):
        print(f"\n{len(results)}/{len(variants)} variants done -- re-run this same command to finish the rest.")
        return

    # All variants done -- assemble the same primary-selection logic
    # refine_nes_candidates() uses, so this slots into ack1_all6_clean_rerun_result.json's format.
    unscrambled_tags = [c for c, s in variants if not s]
    recommended_method = binding_mode.get('recommended_primary_method')
    if binding_mode.get('confidence') != 'low' and recommended_method in unscrambled_tags:
        best_tag = recommended_method
        selection_method = 'sequence_type_classification'
    else:
        best_tag = max(unscrambled_tags, key=lambda t: results[t]['metrics'].get('binding_score', 0.0) or 0.0)
        selection_method = 'max_binding_score_fallback'

    final = {
        'rank': CANDIDATE['rank'], 'start': CANDIDATE['start'], 'end': CANDIDATE['end'],
        'sequence': CANDIDATE['sequence'],
        'binding_mode_classification': binding_mode,
        'md_best_starting_conformation': best_tag,
        'md_primary_variant_selection_method': selection_method,
        'md_metrics_by_variant': {tag: v['metrics'] for tag, v in results.items()},
    }
    out_path = THIS_DIR / 'ack1_candidate6_final.json'
    out_path.write_text(json.dumps(final, indent=2, default=str))

    print(f"\n{'='*100}\nCANDIDATE 6 COMPLETE\n{'='*100}")
    print(f"Primary variant: {best_tag} (method={selection_method})")
    for tag, v in results.items():
        occ = v['metrics'].get('anchor_occupancy_score')
        print(f"  {tag:25s} anchor_occ={occ}")
    print(f"\nWrote {out_path}")
    print(f"To merge into the full 6-candidate set: append this entry to ack1_all6_clean_rerun_result.json")


if __name__ == '__main__':
    main()
