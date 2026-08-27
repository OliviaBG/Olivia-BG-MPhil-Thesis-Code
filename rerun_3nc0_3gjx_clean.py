#!/usr/bin/env python3
"""
rerun_3nc0_3gjx_clean.py
============================================================
Re-runs 3NC0 and 3GJX through the EXACT same grid crystal_sanity_check.py
uses (correct vs. scrambled registration x relaxed vs. unrelaxed
side-chains, starting_conformation='crystal', same 2 ns default) -- the
only change is the CRM1 reference file, swapped from the original
CRM1_Ran_3NC0.pdb / CRM1_Ran_3GJX.pdb (confirmed this project to contain
a duplicate CRM1 copy and a likely second Snurportin1 copy respectively,
alongside the intended single CRM1+Ran pair) to the newly-built
CRM1_Ran_3NC0_v2clean.pdb / CRM1_Ran_3GJX_v2clean.pdb (verified via
build_clean_crystal_references.py: exactly 2 chains each, CRM1 + Ran
only, no extras).

Directly reuses crystal_sanity_check.py's own run_one()/CRYSTAL_STRUCTURES
rather than reimplementing the grid, so this is a true apples-to-apples
comparison against the existing report figures for these two structures
-- only the reference file differs.

USAGE:
    python3 rerun_3nc0_3gjx_clean.py
    python3 rerun_3nc0_3gjx_clean.py --duration-ns 2.0
"""
import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from crystal_sanity_check import CRYSTAL_STRUCTURES, run_one, _summarize, REF_DIR  # noqa: E402
from md_refinement import NESMDRefiner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration-ns', type=float, default=2.0,
                     help='Matches crystal_sanity_check.py\'s own default exactly, for a clean comparison.')
    ap.add_argument('--out', default='crystal_sanity_check_3nc0_3gjx_clean_result.json')
    args = ap.parse_args()

    targets = {
        '3NC0': REF_DIR / 'CRM1_Ran_3NC0_v2clean.pdb',
        '3GJX': REF_DIR / 'CRM1_Ran_3GJX_v2clean.pdb',
    }

    missing = [sid for sid, p in targets.items() if not p.exists()]
    if missing:
        print(f"Missing clean reference file(s) for: {missing} -- run build_clean_crystal_references.py first. Aborting.")
        return

    all_results = {}
    for sid, clean_ref in targets.items():
        cfg = dict(CRYSTAL_STRUCTURES[sid])
        cfg['crm1_pdb'] = clean_ref  # the only change from the original protocol

        print("\n" + "#" * 70)
        print(f"# {sid} (CLEAN reference): {cfg['label']}")
        print("#" * 70)
        print(f"CRM1+Ran reference: {clean_ref.name} (verified clean -- CRM1+Ran only)")
        print(f"Crystal peptide: {cfg['peptide_pdb'].name}  sequence={cfg['sequence']}")

        refiner = NESMDRefiner(crm1_pdb_path=str(clean_ref))
        all_results[sid] = {}

        for relax in (False, True):
            for scramble in (False, True):
                run_label = f"{'scrambled' if scramble else 'correct'}_{'relaxed' if relax else 'norelax'}"
                print(f"\n{sid} / {run_label} (registration={'scrambled' if scramble else 'correct'}, "
                      f"relax_sidechains={relax})")
                metrics = run_one(refiner, cfg, args.duration_ns, scramble, relax)
                _summarize(run_label, metrics)
                all_results[sid][run_label] = metrics

        Path(args.out).write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n(checkpoint saved to {args.out} after {sid})")

    print(f"\n{'='*70}\nDONE -- wrote {args.out}")
    print("Compare anchor_occupancy_score / correct-vs-scrambled gap here against the existing")
    print("crystal_sanity_check_results*.json entries for 3NC0/3GJX (contaminated-reference results)")
    print("to see whether the fix changes anything for these two structures.")


if __name__ == '__main__':
    main()
