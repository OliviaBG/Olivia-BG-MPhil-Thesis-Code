#!/usr/bin/env python3
"""
run_ack1_md_specificity_control.py
============================================================
Adds the scrambled-registration negative control (specificity check) to an
ALREADY-COMPLETED run_ack1_md_refinement.py result, without redoing the 12
native/idealized_helix runs that result already contains.

WHY THIS EXISTS: md_refinement.NESMDRefiner.refine_nes_candidates()'s
test_specificity_control=True option is all-or-nothing -- it always reruns
ALL 4 variants (native, native+scrambled, idealized_helix,
idealized_helix+scrambled) from scratch, with no way to say "I already have
native and idealized_helix, just add the scrambled ones." But the
lower-level function it calls per variant, NESMDRefiner._run_crm1_docking(),
can be called directly for exactly one (conformation, scramble) combination
at a time. This script does that: it loads an existing
<accession>_md_refinement.json (from run_ack1_md_refinement.py), and for
each candidate calls _run_crm1_docking() ONLY for the 2 missing scrambled
variants (native_scrambled, idealized_helix_scrambled) -- 12 new docking
runs for 6 candidates, instead of 24, roughly halving the remaining
wall-clock time versus rerunning everything with --specificity-control.

The existing native/idealized_helix results are valid, independent MD runs
in their own right (each docking run uses its own random initial
velocities/thermal noise -- there's no reason a "correct-registration" run
needs to be redone in the same batch as its scrambled comparison to be a
valid comparison point).

_run_crm1_docking() MUTATES the candidate dict it's given (setting
candidate['md_metrics']/['md_enhanced_score'] to whatever that ONE call
just produced) -- this script explicitly saves and restores the original
'primary' md_metrics/md_enhanced_score (picked by classify_nes_binding_mode,
same logic refine_nes_candidates itself uses) around each scrambled call, so
adding the specificity control never overwrites which result was already
chosen as primary.

REQUIREMENTS: same as run_ack1_md_refinement.py (OpenMM, PDBFixer, real
internet access, CRM1.pdb, GPU recommended).

USAGE:
    python3 run_ack1_md_specificity_control.py
    python3 run_ack1_md_specificity_control.py --result-json Q07912_md_refinement.json
"""
import argparse
import copy
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def resolve_model_id(client, accession, min_residues):
    """Same fixed logic as run_ack1_md_refinement.py -- always prefer the
    exact canonical model_id these candidates were generated against,
    never simply the highest-confidence AlphaFold entry (Q07912 has a
    528-residue fragment entry that sorts first by confidence but doesn't
    cover most of these candidates -- see run_ack1_md_refinement.py's
    resolve_model_id docstring for the full story)."""
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
        covering = [m for m in alphafold_models if (m.get('numResidues') or 0) >= min_residues]
        if not covering:
            print(f"  No AlphaFold entry for {accession} covers residue {min_residues}")
            return None
        chosen = max(covering, key=lambda m: m.get('numResidues') or 0)
        print(f"  NOTE: canonical model_id {canonical_id} not found; using {chosen['model_id']} instead.")

    n_res = chosen.get('numResidues') or 0
    if n_res < min_residues:
        print(f"  ERROR: {chosen['model_id']} only has {n_res} residues, need {min_residues}. Aborting.")
        return None

    print(f"  Using {chosen['model_id']} ({n_res} aa, avg pLDDT confidence: {chosen.get('avg_confidence')})")
    return chosen['model_id']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accession", default="Q07912")
    ap.add_argument("--result-json", default=None,
                     help="Path to the existing run_ack1_md_refinement.py output to extend "
                          "(default: <accession>_md_refinement.json in this directory)")
    ap.add_argument("--out", default=None,
                     help="Output path (default: '<accession>_md_refinement_specificity.json')")
    args = ap.parse_args()

    result_path = Path(args.result_json) if args.result_json else THIS_DIR / f"{args.accession}_md_refinement.json"
    if not result_path.exists():
        print(f"Existing result file not found: {result_path}\n"
              f"Run run_ack1_md_refinement.py first (without --specificity-control) to produce it.")
        sys.exit(1)

    existing = json.loads(result_path.read_text())
    candidates = existing["candidates"]
    duration_ns = existing["duration_ns"]  # match the original run exactly, for a fair comparison
    print(f"Loaded {len(candidates)} candidate(s) from {result_path}, duration_ns={duration_ns} "
          f"(matching the original run)")

    for c in candidates:
        by_variant = c.get("md_metrics_by_variant", {})
        have = [t for t in ("native", "idealized_helix") if by_variant.get(t)]
        print(f"  {c['sequence']} ({c['start']}-{c['end']}): already have {have}")

    print("\nImporting app.py and md_refinement.py ...\n")
    sys.path.insert(0, str(THIS_DIR))
    from md_refinement import NESMDRefiner, estimate_md_time
    from app import app as flask_app

    est_minutes = estimate_md_time(len(candidates), duration_ns) * 2  # 2 NEW variants/candidate
    print(f"{len(candidates)} candidate(s) x 2 new scrambled variants each = "
          f"{len(candidates) * 2} new docking runs (native/idealized_helix reused, not rerun)")
    print(f"Rough estimate: ~{est_minutes:.0f} min (~{est_minutes/60:.1f} h) -- ballpark only.\n")

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
    print(f"SPECIFICITY CONTROL ONLY -- {args.accession} ({model_id}), {len(candidates)} candidate(s), "
          f"2 new scrambled variants each")
    print(f"{'=' * 100}")

    for idx, candidate in enumerate(candidates):
        print(f"\n  Candidate {idx + 1}/{len(candidates)}: {candidate['sequence']} "
              f"({candidate['start']}-{candidate['end']})")

        # Save the primary pick from the ORIGINAL (non-scrambled) run --
        # _run_crm1_docking mutates candidate['md_metrics']/['md_enhanced_score']
        # in place on every call, so without this the last scrambled call
        # below would silently become the new "primary" result.
        original_md_metrics = copy.deepcopy(candidate.get("md_metrics"))
        original_md_enhanced_score = candidate.get("md_enhanced_score")

        by_variant = candidate.setdefault("md_metrics_by_variant", {})

        for conf in ("native", "idealized_helix"):
            if not by_variant.get(conf):
                print(f"    Warning: No existing '{conf}' result for this candidate -- skipping "
                      f"'{conf}_scrambled' too (nothing to compare it against).")
                continue
            tag = f"{conf}_scrambled"
            print(f"    -- {tag} --")
            result = refiner._run_crm1_docking(
                pdb_content, candidate, duration_ns,
                starting_conformation=conf, scramble_registration=True,
            )
            by_variant[tag] = result.get("md_metrics", {}) or {}

            correct_occ = (by_variant.get(conf) or {}).get("anchor_occupancy_score")
            scrambled_occ = by_variant[tag].get("anchor_occupancy_score")
            if correct_occ is not None and scrambled_occ is not None:
                print(f"    Specificity gap ({conf}): correct anchor_occupancy={correct_occ:.3f}  "
                      f"scrambled={scrambled_occ:.3f}  gap={correct_occ - scrambled_occ:+.3f}")

        # Restore the primary pick -- unaffected by adding scrambled data,
        # same as refine_nes_candidates' own logic (primary is only ever
        # chosen from the unscrambled variants).
        candidate["md_metrics"] = original_md_metrics
        candidate["md_enhanced_score"] = original_md_enhanced_score

    out_path = Path(args.out) if args.out else THIS_DIR / f"{args.accession}_md_refinement_specificity.json"
    out_path.write_text(json.dumps({
        "accession": args.accession,
        "model_id": model_id,
        "duration_ns": duration_ns,
        "test_both_conformations": True,
        "test_specificity_control": True,
        "note": "native/idealized_helix reused from the original run; only the "
                "*_scrambled variants were computed by this script.",
        "candidates": sorted(candidates, key=lambda c: -(c.get("md_enhanced_score") or 0)),
    }, indent=2))

    print(f"\n{'=' * 100}")
    print("SUMMARY -- correct vs. scrambled anchor_occupancy_score")
    print(f"{'=' * 100}")
    print(f"{'seq':<18}{'pos':<10}{'conf':<16}{'occ(correct)':<13}{'occ(scrambled)':<15}{'gap':<8}")
    for c in sorted(candidates, key=lambda c: -(c.get("md_enhanced_score") or 0)):
        by_variant = c.get("md_metrics_by_variant", {})
        pos = f"{c['start']}-{c['end']}"
        for conf in ("native", "idealized_helix"):
            correct = (by_variant.get(conf) or {}).get("anchor_occupancy_score")
            scrambled = (by_variant.get(f"{conf}_scrambled") or {}).get("anchor_occupancy_score")
            gap = (correct - scrambled) if (correct is not None and scrambled is not None) else None
            print(f"{c['sequence']:<18}{pos:<10}{conf:<16}"
                  f"{(correct if correct is not None else float('nan')):<13.3f}"
                  f"{(scrambled if scrambled is not None else float('nan')):<15.3f}"
                  f"{(f'{gap:+.3f}' if gap is not None else 'n/a'):<8}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
