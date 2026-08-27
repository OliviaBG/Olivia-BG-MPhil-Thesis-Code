#!/usr/bin/env python3
"""
run_full_pipeline_cli.py
============================================================
Runs the EXACT pipeline the website uses for a single "prototype" protein,
from the terminal -- no frontend, no real HTTP server/port needed.

WHY THIS IS DIFFERENT FROM THE EARLIER ML-ONLY PROTOTYPE SCRIPT: that one
called nes_ml_predictor_improved.py directly, which only ever gives you the
ML classifier's probability -- never the fpocket structural check, the CRM1
pocket-compatibility score (see pocket_detector.py, just improved), the
IUPred2A/ANCHOR2 disorder bonus, or the SASA/pLDDT-based structural
features. That's real information the real server computes but a
standalone ML-only script never touches.

This script instead calls app.py's ACTUAL view functions --
/api/models/<accession> (resolve an AlphaFold model_id, same as the
frontend's search step) then /api/unified_crm1_nes/<model_id> (the same
endpoint the frontend calls on analysis -- see frontend/src/App.jsx,
`axios.get(`${API_BASE}/unified_crm1_nes/${model_id}`)`) -- via Flask's
built-in test_client(). test_client() runs the real view function
in-process (same code, same fpocket call, same combined_score math, same
error handling) without binding a real network port. This is not a
reimplementation; it's the same server code, just invoked directly.

REQUIREMENTS (run locally; these are not available in a
network-restricted environment):
  - fpocket installed and on PATH (pocket_detector.py's _find_fpocket
    checks: fpocket, /usr/bin/fpocket, /usr/local/bin/fpocket,
    /opt/homebrew/bin/fpocket -- if none found, app.py itself already
    falls back to geometry-based pocket detection, same as the real
    server would, so this script still runs, just with the weaker
    fallback pocket scoring)
  - real internet access to alphafold.ebi.ac.uk (structure download) --
    if the protein has no AlphaFold entry, this script (like the real
    server) can't proceed, since fpocket needs a real 3D structure. It
    will tell you the AlphaFold entry page to check manually.
  - real internet access to iupred2a.elte.hu (disorder/ANCHOR2 -- OPTIONAL,
    degrades gracefully to the neutral default if unreachable, same as
    the server)
  - everything app.py itself needs to start normally: your trained model
    files, sumoylation_predictor.py, quick_helix_analysis.py, and the
    Python deps (flask, flask_cors, biopython, scipy, numpy, scikit-learn,
    requests)

USAGE:
    python3 run_full_pipeline_cli.py P04637
    python3 run_full_pipeline_cli.py P04637 --model-id AF-P04637-F1-model_v4
    python3 run_full_pipeline_cli.py P04637 --out result.json --top 20

Run this from the same directory as app.py (it imports app.py directly --
NOT `python app.py`, which would try to bind a real port and block).
"""

import argparse
import json
import sys


def resolve_model_id(client, accession, residues=None):
    """Same lookup the frontend's search step does: GET /api/models/<accession>.
    That endpoint already sorts results by confidence (desc) then version
    (desc) -- see app.py's get_models(), 'models.sort(key=lambda x:
    (-x["avg_confidence"], -x["version"]))' -- so the first AlphaFold entry
    is the best one to use by default, same as picking the top result in
    the UI. Returns None if nothing was found.

    If `residues` is given, filters to the AlphaFold model(s) whose
    'numResidues' matches exactly -- useful when a protein has multiple
    fragments/versions covering different spans (e.g. an older AlphaFold
    version built against a since-revised, differently-lengthed UniProt
    sequence) and you want a specific one rather than whichever sorts
    first."""
    resp = client.get(f'/api/models/{accession}')
    if resp.status_code != 200:
        print(f"  /api/models/{accession} returned HTTP {resp.status_code}: "
              f"{resp.get_data(as_text=True)[:300]}")
        return None
    models = resp.get_json()  # bare list, not wrapped -- see app.py's get_models()
    if not models:
        return None
    alphafold_models = [m for m in models if m.get('source') == 'alphafold']
    if not alphafold_models:
        print(f"  {accession} has {len(models)} structure(s) but none from AlphaFold "
              f"(fpocket in this pipeline needs a real 3D structure -- experimental "
              f"PDB entries aren't wired into --model-id resolution here).")
        return None

    if residues is not None:
        print(f"  Found {len(alphafold_models)} AlphaFold model(s): "
              + ", ".join(f"{m['model_id']} ({m.get('numResidues')} aa)" for m in alphafold_models))
        matches = [m for m in alphafold_models if m.get('numResidues') == residues]
        if not matches:
            print(f"  No AlphaFold model with exactly {residues} residues found for {accession}.")
            return None
        best = matches[0]
        print(f"  Using {best['model_id']} ({best.get('numResidues')} aa, "
              f"avg pLDDT confidence: {best.get('avg_confidence')})")
        return best['model_id']

    best = alphafold_models[0]
    print(f"  Found {len(alphafold_models)} AlphaFold model(s); using {best['model_id']} "
          f"(avg pLDDT confidence: {best.get('avg_confidence')})")
    return best['model_id']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('accession', help='UniProt accession of the prototype protein, e.g. P04637')
    ap.add_argument('--model-id', help="Skip the AlphaFold lookup and use this exact model_id "
                                        "directly, e.g. AF-P04637-F1-model_v4")
    ap.add_argument('--residues', type=int, help="Pick the AlphaFold model whose numResidues "
                                                  "matches exactly, instead of the top-confidence "
                                                  "one (useful when a protein has multiple "
                                                  "fragments/versions of different lengths)")
    ap.add_argument('--out', help='Also write the full JSON response to this file')
    ap.add_argument('--top', type=int, default=10, help='How many top-scoring NES candidates to print (default 10)')
    ap.add_argument('--compact', action='store_true',
                     help='Print the old one-line-per-candidate table instead of the full '
                          'per-candidate breakdown (ML/pattern/structural/CRM1-pocket detail, default)')
    args = ap.parse_args()

    print("Importing app.py (this runs its normal startup: loading trained models, "
          "checking fpocket availability, etc. -- same as a real server start) ...\n")
    import app as flask_app_module  # noqa: E402 (import after argparse so --help doesn't pay this cost)

    client = flask_app_module.app.test_client()

    model_id = args.model_id
    if not model_id:
        print(f"\nResolving AlphaFold model_id for {args.accession} ...")
        model_id = resolve_model_id(client, args.accession, residues=args.residues)
        if not model_id:
            print(f"\nNo usable AlphaFold structure found for {args.accession}.")
            print(f"Check https://alphafold.ebi.ac.uk/entry/{args.accession} manually -- "
                  f"if it's genuinely not there, this protein can't go through the real "
                  f"pipeline (fpocket needs a real structure) without supplying your own "
                  f"predicted structure and using --model-id to point at it.")
            sys.exit(1)
    print(f"\nUsing model_id: {model_id}")

    print(f"\nRunning /api/unified_crm1_nes/{model_id} -- the exact endpoint the "
          f"website calls (fpocket detection + ML NES prediction + pattern matching + "
          f"structural features + CRM1 pocket-compatibility scoring) ...\n")
    resp = client.get(f'/api/unified_crm1_nes/{model_id}')

    if resp.status_code != 200:
        print(f"\nPipeline returned HTTP {resp.status_code}:")
        body = resp.get_json() or {}
        print(body.get('error', resp.get_data(as_text=True)[:2000]))
        if body.get('traceback'):
            print("\n--- server-side traceback ---")
            print(body['traceback'])
        sys.exit(1)

    result = resp.get_json()

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Full JSON response written to {args.out}")

    # ---- concise summary (real field names, see app.py's
    # unified_crm1_nes_analysis() jsonify() call: 'nes_motifs' is the list
    # of predictions, each with 'sequence'/'start'/'end'/'combined_score'/
    # 'components' -- 'crm1_binding_regions' is the identical list under a
    # second key, not separate data) ----
    summary = result.get('summary', {})
    motifs = result.get('nes_motifs', [])

    print(f"\n{'=' * 70}")
    print(f"RESULTS for {args.accession} ({model_id})")
    print(f"{'=' * 70}")
    method = summary.get('pocket_detection_method', 'unknown')
    method_note = {
        'fpocket': '(real alpha-sphere geometry)',
        'geometry_fallback': '(fpocket unavailable/failed -- WEAKER sliding-window heuristic used instead)',
        'no_fpocket_binary': '(fpocket not installed on this machine -- WEAKER fallback used for everything)',
    }.get(method, '')
    print(f"Pockets detected (fpocket + CRM1-compatibility filter): {summary.get('pockets_detected')}  "
          f"[detection method: {method} {method_note}]")
    print(f"Candidate NES motifs before filtering: {summary.get('total_candidates')}")
    print(f"Kept after filtering: {summary.get('filtered_predictions')} "
          f"(high: {summary.get('high_confidence')}, medium: {summary.get('medium_confidence')}, "
          f"low: {summary.get('low_confidence')})")
    print(f"Analysis time: {summary.get('analysis_time')}s   "
          f"(real per-residue SASA computed: {summary.get('sasa_computed')})")

    ranked = sorted(motifs, key=lambda m: -m.get('combined_score', 0))[:args.top]

    if args.compact:
        print(f"\nTop {len(ranked)} candidate NES region(s), by combined_score:")
        print(f"{'sequence':<22} {'pos':>10} {'combined':>9} {'ML prob':>8} {'CRM1 affinity':>14}")
        for m in ranked:
            comp = m.get('components', {})
            pos = f"{m.get('start')}-{m.get('end')}"
            print(f"{m.get('sequence', '?'):<22} {pos:>10} {m.get('combined_score'):>9} "
                  f"{comp.get('ml_probability'):>8} {comp.get('crm1_binding_affinity'):>14}")
    else:
        print(f"\nTop {len(ranked)} candidate NES region(s), full breakdown "
              f"(pass --compact for the one-line-per-candidate table instead):")
        for i, m in enumerate(ranked, 1):
            print_candidate_detail(i, m)

    if not motifs:
        print("\nNo NES candidates passed filtering for this protein/structure.")


def print_candidate_detail(rank, m):
    """Full per-candidate breakdown -- every field app.py's
    unified_crm1_nes_analysis() actually computes (see its 'components'
    dict and calculate_improved_nes_score()'s 'details' dict, both spread
    into each prediction), not just the single combined_score the compact
    table shows. This is the same data the website's UI panel is built
    from -- nothing here is re-derived or approximated by this script."""
    comp = m.get('components', {})
    dd = comp.get('disorder_details') or {}
    flank = comp.get('flanking_analysis') or {}

    print(f"\n{'-' * 70}")
    print(f"#{rank}  {m.get('sequence', '?')}   (residues {m.get('start')}-{m.get('end')}, "
          f"length {m.get('length')})")
    print(f"{'-' * 70}")
    print(f"  combined_score:        {m.get('combined_score')}")
    print(f"  status:                {comp.get('status')}")

    print(f"\n  ML classifier:")
    print(f"    ml_probability:      {comp.get('ml_probability')}   "
          f"(confidence: {comp.get('ml_confidence')})")
    print(f"    nes_classes:         {comp.get('nes_classes')}")
    print(f"    pssm_score:          {comp.get('pssm_score')}")
    print(f"    spacer_hydrophobicity: {comp.get('spacer_hydrophobicity')}")

    print(f"\n  Sequence-pattern checks:")
    print(f"    consensus_pattern:   {comp.get('consensus_pattern')}  "
          f"(strength: {comp.get('pattern_strength')}, pattern_score: {comp.get('pattern_score')})")
    print(f"    anchor_score:        {comp.get('anchor_score')}  (leucine-position based)")
    print(f"    leucine_filter:      {comp.get('leucine_filter')}")
    print(f"    hydrophobicity:      {comp.get('hydrophobicity')}")

    print(f"\n  Structural features:")
    # NOTE: despite the dict key name, this is already the combined/consensus
    # RSA score -- app.py's calculate_sasa(pdb_text=..., return_stats=True)
    # averages 3 methods (FreeSASA Lee-Richards, FreeSASA Shrake-Rupley,
    # Biopython Shrake-Rupley), each Tien-normalized to relative (0-1)
    # accessibility, not raw SASA in Å². "SASA" here was a stale/misleading
    # label (fixed -- the underlying number was already correct.
    print(f"    surface_accessibility (consensus RSA, 0-1 relative, not raw Å² SASA): {comp.get('surface_accessibility')}")
    print(f"    structural_confidence (pLDDT-derived): {comp.get('structural_confidence')}")
    print(f"    flexibility:         {comp.get('flexibility')}")
    print(f"    disorder (combined): {comp.get('disorder')}")
    if dd:
        print(f"      - sequence_disorder (IUPred2A or heuristic fallback): {dd.get('sequence_disorder')} "
              f"[source: {dd.get('disorder_source')}]")
        print(f"      - plddt_flexibility: {dd.get('plddt_flexibility')}")
        print(f"      - in_uniprot_disorder_region: {dd.get('in_uniprot_disorder_region')}")
        print(f"      - anchor2_binding (disordered-binding-region probability): {dd.get('anchor2_binding')}")
    if flank:
        print(f"    flanking_analysis:   {flank}")

    print(f"\n  CRM1 pocket compatibility:")
    print(f"    pocket_compatibility (pocket_score): {comp.get('pocket_compatibility')}")
    print(f"    effective_pocket_score (after SASA-exposure discount): {comp.get('effective_pocket_score')}")
    print(f"    exposure_factor:     {comp.get('exposure_factor')}")
    print(f"    crm1_binding_affinity: {comp.get('crm1_binding_affinity')}  "
          f"(70% fpocket-based + 30% raw hydrophobic burial, see below)")
    burial = comp.get('raw_hydrophobic_burial') or {}
    print(f"      raw_hydrophobic_burial: {burial.get('anchor_raw_area_A2')} Å² of anchor residues "
          f"(L/M/F/I/W/V) actually exposed, vs a {burial.get('reference_A2')} Å² reference "
          f"-> score {burial.get('score')}")
    print(f"    has_crm1_pocket:     {m.get('has_crm1_pocket')}")
    for p in (m.get('pocket_details') or []):
        method = p.get('detection_method', 'unknown')
        flag = '' if method == 'fpocket' else '  <-- fallback, NOT real fpocket geometry'
        print(f"      pocket {p.get('pocket_id')}: score={p.get('score')}  "
              f"volume={p.get('volume')}  druggability={p.get('druggability_score')}  "
              f"hydrophobic_ratio={p.get('hydrophobic_ratio')}  detection={method}{flag}")
        for reason in (p.get('crm1_compatibility_reasons') or []):
            print(f"        - {reason}")


if __name__ == '__main__':
    main()
