#!/usr/bin/env python3
"""
run_ack1_all6_clean_rerun.py
============================================================
Full clean re-run of ALL 6 ACK1 (Q07912) NES candidates from the report's
Table 1/4 -- replaces the earlier CRM1.pdb-based results (confirmed
contaminated with Snurportin-1 fragments this project) with a run against
crm1_reference/CRM1_Ran_only.pdb (confirmed clean: only CRM1 + Ran, no
Snurportin-1, verified by direct chain inspection).

Also picks up every methodological change made this project automatically,
since this goes through the real refine_nes_candidates() dispatch path
rather than a hand-forced single conformation:
  - Simulated annealing (default-on inside _run_crm1_docking)
  - classify_nes_binding_mode() / recommend_starting_conformation()-driven
    conformation choice -- checked locally before this run: 478-487,
    995-1007, 373-387, and 207-221 all classify as 'extended_atypical'
    under today's fix and will get a THIRD 'extended' trajectory tested
    alongside native + idealized_helix (they only got native+idealized_helix
    before). 528-537 (LSSDFKRLGL) and 11-21 stay 'likely_helical'
    (idealized_helix recommended), same as before.
  - Per-anchor hydrophobic-groove burial verification
  - Ramachandran (phi/psi) trace

SCOPE MATCHES TABLE 4: 10 ns per variant, test_both_conformations=True,
test_specificity_control=True (Table 4 / Figure 30 included scrambled-
registration controls for all 6 candidates x both conformations).

NOTE: this does NOT address the separate (unrelated) subset-re-run finding
that simulated annealing doesn't clearly improve real discrimination
between NES motifs and decoys on the 60-example reference set -- that
caveat is independent of which CRM1 reference is used and still applies
to any new score produced with annealing enabled, including this run.

Estimated cost: 4 candidates x 3 conformations + 2 candidates x 2
conformations = 16 variants x 2 (correct + scrambled) = 32 trajectories
at 10 ns each. Substantial but the shortest duration used for ACK1 work
this project, and checkpointed per-candidate so a partial run can resume.

USAGE:
    python3 run_ack1_all6_clean_rerun.py
    python3 run_ack1_all6_clean_rerun.py --duration-ns 10.0
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=10.0,
                     help='10 ns matches Table 4/Figure 30\'s original protocol exactly.')
    ap.add_argument('--crm1-ref', default='crm1_reference/CRM1_Ran_only.pdb')
    ap.add_argument('--cache', default='ack1_all6_clean_rerun_result.json',
                     help='Checkpointed per-candidate -- safe to Ctrl-C and re-run to resume.')
    args = ap.parse_args()

    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app
    import requests

    crm1_path = THIS_DIR / args.crm1_ref
    if not crm1_path.exists():
        print(f"CRM1 reference not found: {crm1_path} -- aborting.")
        return

    print(f"{'='*100}\nACK1 (Q07912) -- ALL 6 NES candidates, clean re-run\n{'='*100}")
    print(f"Duration: {args.duration_ns} ns per variant")
    print(f"CRM1 reference: {crm1_path} (clean -- verified CRM1+Ran only, no Snurportin-1)\n")

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

    results_path = Path(args.cache)
    results = []
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"Resuming: {len(results)} candidates already done\n")
    done_ranks = {r['rank'] for r in results}

    est_total_min = 0
    for i, c in enumerate(CANDIDATES, 1):
        if c['rank'] in done_ranks:
            continue
        binding_mode = refiner.classify_nes_binding_mode(c['sequence'])
        n_conf = 3 if binding_mode['recommended_primary_method'] == 'extended' else 2
        est_min = estimate_md_time(1, args.duration_ns) * n_conf * 2  # x2 for scrambled controls
        est_total_min += est_min
        print(f"[{i}/6] rank={c['rank']} {c['start']}-{c['end']} {c['sequence']} -- "
              f"binding_mode={binding_mode['binding_mode_class']} "
              f"recommend={binding_mode['recommended_primary_method']} "
              f"(~{n_conf} conformations x2 scrambled, ~{est_min:.0f} min)")
        candidate = {"sequence": c['sequence'], "start": c['start'], "end": c['end'], "full_sequence": None}

        enhanced = refiner.refine_nes_candidates(
            pdb_content, [candidate], args.duration_ns,
            test_both_conformations=True, test_specificity_control=True,
        )
        cand = enhanced[0]
        tested = sorted(cand.get('md_metrics_by_variant', {}).keys())
        best = cand.get('md_best_starting_conformation')
        print(f"    Variants tested: {tested}  Primary: {best} "
              f"(method={cand.get('md_primary_variant_selection_method')})")
        for tag, m in (cand.get('md_metrics_by_variant') or {}).items():
            print(f"      {tag:25s} anchor_occ={m.get('anchor_occupancy_score')}  "
                  f"raw_binding={m.get('raw_binding_score')}")

        results.append({
            'rank': c['rank'], 'start': c['start'], 'end': c['end'], 'sequence': c['sequence'],
            'binding_mode_classification': binding_mode,
            'md_best_starting_conformation': best,
            'md_primary_variant_selection_method': cand.get('md_primary_variant_selection_method'),
            'md_metrics_by_variant': cand.get('md_metrics_by_variant'),
        })
        results_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"    Checkpointed ({len(results)}/6 done)\n")

    print(f"\n{'='*100}\nALL 6 ACK1 CANDIDATES DONE (clean CRM1 reference, new protocol)\n{'='*100}")
    print(f"{'Rank':4s} {'Pos':10s} {'Seq':16s} {'Primary':16s} {'Anchor_occ':10s}")
    for r in sorted(results, key=lambda x: x['rank']):
        primary_metrics = (r.get('md_metrics_by_variant') or {}).get(r.get('md_best_starting_conformation'), {})
        occ = primary_metrics.get('anchor_occupancy_score')
        print(f"{r['rank']:<4d} {r['start']}-{r['end']:<7} {r['sequence']:16s} "
              f"{str(r.get('md_best_starting_conformation')):16s} {str(occ)}")
    print(f"\nWrote {results_path}")


if __name__ == '__main__':
    main()
