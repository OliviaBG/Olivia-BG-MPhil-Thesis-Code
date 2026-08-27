#!/usr/bin/env python3
"""
rerun_subset_annealing_check.py
============================================================
Partial re-run of a hand-picked SUBSET of the 60-example reference set
(anchor_occupancy_eval_v4_clean.json) under the NEW MD protocol -- i.e. with
use_simulated_annealing=True now baked into md_refinement.py's
_run_crm1_docking as the default (see the Rev-NES/3NBZ
helix-trapping finding this project).

WHY THIS EXISTS: the 60-example reference set's anchor_occupancy_score
values (and any percentile grounding computed from them, e.g. in the
thesis report's MD Refinement section) were all produced WITHOUT
annealing. Since annealing just nearly tripled Rev-NES's
anchor_occupancy_score (0.121 -> 0.343) and broke a kinetic helix-trap,
any NEW result produced under the new (annealing-on) protocol is not
strictly apples-to-apples against that OLD reference distribution anymore.
A full 60-example re-run would settle this cleanly but costs a lot of GPU
time. This script instead re-runs a SMALL, DELIBERATELY CHOSEN subset --
the examples whose old anchor_occupancy_score was most diagnostic -- as a
cheap sanity check for whether the ranking/percentile structure actually
shifts enough to matter before committing to a full re-run.

SUBSET SELECTION RATIONALE (10 examples, all previously won by the
idealized_helix starting conformation -- i.e. exactly the variant
use_simulated_annealing now changes):
  - 6 of the highest-scoring REAL POSITIVES under the old protocol
    (P42566, O35973, Q14872, Q00987, Q14653, P14635) -- if annealing is
    neutral-to-helpful for real NES motifs (as it was for Rev-NES), these
    should hold up or improve.
  - 4 of the highest-scoring HARD NEGATIVES under the old protocol
    (Q96CS2 269-278, Q9BZF9 320-329, Q13439, Q08DM8) -- these are exactly
    the examples the specificity-control finding flagged as
    packing suspiciously TIGHT (some scoring higher than any real
    positive). If annealing helps real NES motifs specifically fit their
    own registration (rather than just generically loosening every
    trajectory), these should NOT improve as much as the positives do --
    if they improve by just as much or more, that's evidence annealing
    doesn't fix the specificity problem and a full re-run's conclusions
    would look different.

Reuses compute_features()/get_structure_bundle()/build_candidate() from
evaluate_anchor_occupancy_signal.py unchanged -- same duration (2 ns, same
as that script's default and what actually produced the reference file),
same test_both_conformations=True, same test_specificity_control=True.
The ONLY thing different from the original run is that md_refinement.py's
_run_crm1_docking now defaults use_simulated_annealing=True.

USAGE (on the pod, same environment as the other MD scripts):
    python3 rerun_subset_annealing_check.py
    python3 rerun_subset_annealing_check.py --duration-ns 2.0

Writes rerun_subset_annealing_check.json (list of dicts, same shape as
anchor_occupancy_eval_v4_clean.json entries) and prints an old-vs-new
comparison table using the matching entries already in
anchor_occupancy_eval_v4_clean.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / 'nes_data_pipeline'))

from evaluate_anchor_occupancy_signal import compute_features  # noqa: E402
from evaluate_crm1_pocket_signal import CRM1_REF_CANDIDATES, write_text_atomic_with_retry  # noqa: E402
from md_refinement import NESMDRefiner  # noqa: E402

# (accession, start, end, label, old_occ) -- old_occ is just for the
# printed comparison table, pulled from anchor_occupancy_eval_v4_clean.json.
SUBSET = [
    ("P42566", 766, 800, 1, 0.592),
    ("O35973", 485, 498, 1, 0.462),
    ("Q14872", 336, 344, 1, 0.462),
    ("Q00987", 188, 202, 1, 0.443),
    ("Q14653", 136, 150, 1, 0.292),
    ("P14635", 141, 153, 1, 0.278),
    ("Q96CS2", 269, 278, 0, 0.481),
    ("Q9BZF9", 320, 329, 0, 0.354),
    ("Q13439", 2175, 2183, 0, 0.334),
    ("Q08DM8", 173, 183, 0, 0.230),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration-ns', type=float, default=2.0,
                     help='Matches the original reference-set run (evaluate_anchor_occupancy_signal.py default)')
    ap.add_argument('--cache', default='rerun_subset_annealing_check.json')
    ap.add_argument('--pdb-cache-dir', default='crm1_eval_pdb_cache',
                     help='Reuses the on-disk PDB cache from the original reference-set run if present')
    args = ap.parse_args()

    pdb_cache_dir = Path(args.pdb_cache_dir)
    pdb_cache_dir.mkdir(exist_ok=True)

    crm1_ref = next((p for p in CRM1_REF_CANDIDATES if (THIS_DIR / p).exists()), None)
    if not crm1_ref:
        print(f"No CRM1 reference structure found (checked: {CRM1_REF_CANDIDATES}) -- aborting.")
        return
    print(f"Using CRM1 reference: {crm1_ref}")
    print(f"MD duration per candidate: {args.duration_ns} ns")
    print(f"Re-running {len(SUBSET)} examples ({sum(1 for s in SUBSET if s[3]==1)} positive, "
          f"{sum(1 for s in SUBSET if s[3]==0)} negative) with use_simulated_annealing=True (new default)\n")

    refiner = NESMDRefiner(crm1_pdb_path=str(THIS_DIR / crm1_ref))

    results_path = Path(args.cache)
    results = []
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"Resuming: {len(results)} examples already done in {results_path}")
    done_keys = {(r['accession'], r['start'], r['end']) for r in results}

    struct_cache = {}
    for i, (accession, start, end, label, old_occ) in enumerate(SUBSET, 1):
        key = (accession, start, end)
        if key in done_keys:
            continue
        print(f"[{i}/{len(SUBSET)}] {accession} {start}-{end} (label={label}, old anchor_occupancy_score={old_occ:.3f})")
        t0 = time.time()
        try:
            feats = compute_features(refiner, accession, start, end, args.duration_ns,
                                      struct_cache, pdb_cache_dir,
                                      test_both_conformations=True, test_specificity_control=True)
        except Exception as e:
            print(f"    MD run failed, skipping: {e}")
            feats = None
        elapsed = time.time() - t0

        if feats is None:
            print(f"    (skipped -- no structure / gap / MD failure, {elapsed:.0f}s)")
            continue

        new_occ = feats.get('anchor_occupancy_score')
        print(f"    NEW anchor_occupancy_score={new_occ}  (old={old_occ:.3f}, "
              f"delta={'n/a' if new_occ is None else f'{new_occ - old_occ:+.3f}'})  ({elapsed:.0f}s)")

        results.append({'accession': accession, 'start': start, 'end': end, 'label': label,
                         'old_anchor_occupancy_score': old_occ, **feats})
        write_text_atomic_with_retry(results_path, json.dumps(results, indent=2, default=str))

    print(f"\n{'='*90}\nOLD (no annealing) vs NEW (annealing default-on) comparison\n{'='*90}")
    print(f"{'accession':10s} {'range':12s} {'label':6s} {'old_occ':>9s} {'new_occ':>9s} {'delta':>9s}")
    pos_deltas, neg_deltas = [], []
    for r in results:
        new_occ = r.get('anchor_occupancy_score')
        old_occ = r['old_anchor_occupancy_score']
        delta = None if new_occ is None else new_occ - old_occ
        if delta is not None:
            (pos_deltas if r['label'] == 1 else neg_deltas).append(delta)
        print(f"{r['accession']:10s} {r['start']}-{r['end']:<7} {r['label']:<6} {old_occ:>9.3f} "
              f"{'n/a' if new_occ is None else f'{new_occ:>9.3f}'} "
              f"{'n/a' if delta is None else f'{delta:>+9.3f}'}")

    if pos_deltas:
        print(f"\nMean delta, positives: {sum(pos_deltas)/len(pos_deltas):+.3f}  (n={len(pos_deltas)})")
    if neg_deltas:
        print(f"Mean delta, negatives: {sum(neg_deltas)/len(neg_deltas):+.3f}  (n={len(neg_deltas)})")
    if pos_deltas and neg_deltas:
        gap = (sum(pos_deltas)/len(pos_deltas)) - (sum(neg_deltas)/len(neg_deltas))
        print(f"\nGap (mean positive delta - mean negative delta): {gap:+.3f}")
        print("  Positive gap => annealing helps real positives MORE than it helps hard negatives")
        print("  (good -- ranking/percentile structure likely holds, full re-run probably not urgent).")
        print("  Negative or near-zero gap => annealing moves both classes similarly or favors")
        print("  negatives (bad -- the old reference percentiles are no longer trustworthy for new")
        print("  results, and a full 60-example re-run is worth doing before reporting new percentiles).")

    print(f"\nWrote {results_path}")


if __name__ == '__main__':
    main()
