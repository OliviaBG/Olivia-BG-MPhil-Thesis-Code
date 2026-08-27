#!/usr/bin/env python3
"""
run_ack1_md_refinement.py
============================================================
Full CRM1-docking MD refinement (md_refinement.NESMDRefiner) for the 6 ACK1
(Q07912) NES candidates in Q07912_full_pipeline_scan.json -- the real
production MD code path, not a reimplementation, run against the real
AlphaFold structure and the project's real CRM1.pdb reference.

WHICH STARTING CONFORMATION EACH CANDIDATE WILL LIKELY USE
------------------------------------------------------------
Before running any MD, classify_nes_binding_mode(sequence) (sequence-only,
no MD needed) is used to predict which starting pose is more likely
trustworthy for each candidate -- 'native' (AlphaFold's own, usually
non-helical local conformation) for atypical/extended-looking sequences
(a Proline or 2+ helix-breaking Pro/Gly residues alongside a matched
Phi-anchor register, the HIV-1 Rev pattern), or 'idealized_helix' (a
built, canonical alpha helix) for sequences that look like the classic
PKI-type canonical helical NES. As of , run against this batch:

  1. FPDRIDELYL      (478-487)   extended_atypical   -> native          (medium confidence)
  2. LSSDFKRLGL      (528-537)   likely_helical       -> idealized_helix (medium confidence)
  3. VEQLFGLGLRPRG   (995-1007)  extended_atypical    -> native          (medium confidence)
  4. EDRPTFVALRDFLLE (373-387)   extended_atypical    -> native          (medium confidence)
  5. LELLSEVQLQQ     (11-21)     likely_helical       -> idealized_helix (medium confidence)
  6. LAPLGSLLDRLRKHQ (207-221)   extended_atypical    -> native          (medium confidence)

By design, this script runs BOTH starting conformations for
ALL 6 candidates regardless of the recommendation above (test_both_
conformations=True) -- the recommended one is still recorded per candidate
(candidate['nes_binding_mode_classification']) and used to pick the
'primary' result, but the non-primary conformation's full result is never
discarded (candidate['md_metrics_by_variant']).

REQUIREMENTS (run locally or on a compute node -- not available in a restricted
environment):
  - OpenMM + PDBFixer installed (conda install -c conda-forge openmm pdbfixer)
  - mdtraj installed (pip install mdtraj) for secondary-structure/SASA metrics
    (optional -- refinement still runs without it, just with fewer metrics)
  - real internet access to alphafold.ebi.ac.uk (structure download)
  - CRM1.pdb present in this directory (already in your AlphaFold folder)
  - a GPU is strongly recommended -- at the default duration_ns=10.0 and
    test_both_conformations=True (12 total dockings: 6 candidates x 2
    conformations), estimate_md_time() below gives a rough sense of the
    wall-clock cost before you kick this off.

USAGE:
    python3 run_ack1_md_refinement.py
    python3 run_ack1_md_refinement.py --duration-ns 5 --out ack1_md_results
    python3 run_ack1_md_refinement.py --candidates 478-487,11-21   # subset, for a quick test
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def resolve_model_id(client, accession, min_residues):
    """GET /api/models/<accession> and pick the entry the candidates were
    actually generated against -- NOT simply the highest-confidence entry. bugfix: Q07912 (ACK1/TNK2) has THREE separate AlphaFold
    entries -- the 1038aa canonical (AF-Q07912-F1, confidence 61.27), a
    1047aa alternate (AF-Q07912-3-F1), and a 528aa fragment
    (AF-Q07912-2-F1) that happens to have the HIGHEST avg_confidence
    (79.23, since it's a shorter, more ordered kinase-domain-only piece).
    A naive "pick whichever sorts first by confidence" pick (what this
    function used to do, copying run_full_pipeline_cli.py's single-
    fragment-assuming resolve_model_id) silently grabbed that 528-residue
    fragment -- candidates like 995-1007 don't even exist in a
    528-residue structure, and 528-537 sits right on its edge, so MD
    against it is either an outright crash or silently wrong.

    Fix: always prefer the exact model_id run_ack1_full_pipeline_scan.py
    used to generate these candidates in the first place
    (f"AF-{accession}-F1", the canonical entry, same convention that
    script's own docstring documents), and hard-fail with a clear error
    -- rather than silently proceeding -- if the resolved structure's
    numResidues is too short for the highest candidate position this run
    needs to cover.
    """
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
        # Canonical model_id not present under that exact name -- fall back
        # to the largest (by numResidues) entry that actually covers every
        # candidate, never the highest-confidence one.
        covering = [m for m in alphafold_models if (m.get('numResidues') or 0) >= min_residues]
        if not covering:
            print(f"  No AlphaFold entry for {accession} covers residue {min_residues} -- "
                  f"entries found: " + ", ".join(f"{m['model_id']} ({m.get('numResidues')} aa)" for m in alphafold_models))
            return None
        chosen = max(covering, key=lambda m: m.get('numResidues') or 0)
        print(f"  NOTE: canonical model_id {canonical_id} not found in the API response; "
              f"using {chosen['model_id']} instead (largest entry that covers residue {min_residues}).")

    n_res = chosen.get('numResidues') or 0
    if n_res < min_residues:
        print(f"  ERROR: chosen entry {chosen['model_id']} only has {n_res} residues, but this "
              f"run needs residue {min_residues} (the highest candidate end position). Aborting "
              f"rather than running MD against a structure that doesn't cover all candidates.")
        return None

    print(f"  Using {chosen['model_id']} ({n_res} aa, avg pLDDT confidence: {chosen.get('avg_confidence')})")
    return chosen['model_id']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accession", default="Q07912", help="UniProt accession (default: Q07912, ACK1/TNK2)")
    ap.add_argument("--scan-json", default=None,
                     help="Path to the full_pipeline_scan.json with the NES candidates to refine "
                          "(default: <accession>_full_pipeline_scan.json in this directory)")
    ap.add_argument("--duration-ns", type=float, default=10.0, help="MD duration per docking run, in ns (default 10.0)")
    ap.add_argument("--candidates", default=None,
                     help="Comma-separated start-end positions to restrict to a subset, e.g. "
                          "'478-487,11-21' (default: all candidates in the scan file)")
    ap.add_argument("--out", default=None, help="Output prefix (default: '<accession>_md_refinement', or "
                                                  "'<accession>_md_refinement_specificity' with --specificity-control)")
    ap.add_argument("--specificity-control", action="store_true",
                     help="Also run the scrambled-registration negative control for each conformation "
                          "(test_specificity_control=True) -- 4 total docking runs per candidate instead "
                          "of 2 (native, native+scrambled, idealized_helix, idealized_helix+scrambled). "
                          "Roughly DOUBLES total runtime on top of --duration-ns. This is what actually "
                          "tells you whether a candidate's anchor_occupancy_score/binding_score means "
                          "anything, per this project's own prior finding that a single 'how tight is the "
                          "final pose' score can run backwards without a scrambled comparison.")
    args = ap.parse_args()

    scan_path = Path(args.scan_json) if args.scan_json else THIS_DIR / f"{args.accession}_full_pipeline_scan.json"
    if not scan_path.exists():
        print(f"Scan file not found: {scan_path}\n"
              f"Run run_ack1_full_pipeline_scan.py first to produce the NES candidates to refine.")
        sys.exit(1)

    scan = json.loads(scan_path.read_text())
    candidates = scan.get("nes_motifs", [])
    if args.candidates:
        wanted = set(args.candidates.split(","))
        candidates = [c for c in candidates if f"{c['start']}-{c['end']}" in wanted]
        if not candidates:
            print(f"No candidates in {scan_path} matched --candidates {args.candidates}")
            sys.exit(1)

    print("Importing app.py and md_refinement.py (this checks for OpenMM/PDBFixer/mdtraj "
          "and loads the trained models) ...\n")
    sys.path.insert(0, str(THIS_DIR))
    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app

    variants_per_candidate = 4 if args.specificity_control else 2
    total_runs = len(candidates) * variants_per_candidate
    est_minutes = estimate_md_time(len(candidates), args.duration_ns) * variants_per_candidate
    print(f"\n{len(candidates)} candidate(s), duration_ns={args.duration_ns}, "
          f"test_both_conformations=True, test_specificity_control={args.specificity_control} "
          f"-> {total_runs} total docking runs")
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h) at typical GPU performance "
          f"(0.5 ns/min) -- treat as a ballpark, real time depends heavily on your hardware.\n")

    max_end = max(c['end'] for c in candidates)
    client = flask_app.test_client()
    print(f"Resolving AlphaFold model_id for {args.accession} (must cover residue {max_end}) ...")
    model_id = resolve_model_id(client, args.accession, min_residues=max_end)
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
    refiner = NESMDRefiner(crm1_pdb_path=crm1_pdb_path if Path(crm1_pdb_path).exists() else None)

    print(f"\n{'=' * 100}")
    print(f"MD REFINEMENT -- {args.accession} ({model_id}), {len(candidates)} candidate(s), "
          f"both starting conformations" + (", + scrambled specificity control" if args.specificity_control else ""))
    print(f"{'=' * 100}")
    enhanced = refiner.refine_nes_candidates(
        pdb_content, candidates, duration_ns=args.duration_ns,
        test_both_conformations=True,
        test_specificity_control=args.specificity_control,
    )

    default_prefix = f"{args.accession}_md_refinement" + ("_specificity" if args.specificity_control else "")
    out_prefix = args.out or default_prefix
    json_path = THIS_DIR / f"{out_prefix}.json"
    json_path.write_text(json.dumps({
        "accession": args.accession,
        "model_id": model_id,
        "duration_ns": args.duration_ns,
        "test_both_conformations": True,
        "candidates": sorted(enhanced, key=lambda c: -(c.get("md_enhanced_score") or 0)),
    }, indent=2))

    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    def occ(by_variant, tag):
        m = by_variant.get(tag) or {}
        return m.get("anchor_occupancy_score")

    if args.specificity_control:
        print(f"{'seq':<18}{'pos':<10}{'conf':<16}{'occ(correct)':<13}{'occ(scrambled)':<15}{'gap':<8}")
        for c in sorted(enhanced, key=lambda c: -(c.get("md_enhanced_score") or 0)):
            by_variant = c.get("md_metrics_by_variant", {})
            pos = f"{c['start']}-{c['end']}"
            for conf in ("native", "idealized_helix"):
                correct = occ(by_variant, conf)
                scrambled = occ(by_variant, f"{conf}_scrambled")
                gap = (correct - scrambled) if (correct is not None and scrambled is not None) else None
                print(f"{c['sequence']:<18}{pos:<10}{conf:<16}"
                      f"{(correct if correct is not None else float('nan')):<13.3f}"
                      f"{(scrambled if scrambled is not None else float('nan')):<15.3f}"
                      f"{(f'{gap:+.3f}' if gap is not None else 'n/a'):<8}")
    else:
        print(f"{'seq':<20}{'pos':<12}{'binding_mode':<20}{'primary_conf':<16}{'md_score':<10}{'native_occ':<11}{'idealized_occ':<13}")
        for c in sorted(enhanced, key=lambda c: -(c.get("md_enhanced_score") or 0)):
            bm = c.get("nes_binding_mode_classification", {})
            by_variant = c.get("md_metrics_by_variant", {})
            native_occ = occ(by_variant, "native")
            idealized_occ = occ(by_variant, "idealized_helix")
            pos = f"{c['start']}-{c['end']}"
            print(f"{c['sequence']:<20}{pos:<12}{bm.get('binding_mode_class', '?'):<20}"
                  f"{c.get('md_best_starting_conformation', '?'):<16}"
                  f"{c.get('md_enhanced_score', 0):<10.3f}"
                  f"{(native_occ if native_occ is not None else float('nan')):<11.3f}"
                  f"{(idealized_occ if idealized_occ is not None else float('nan')):<13.3f}")

    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
