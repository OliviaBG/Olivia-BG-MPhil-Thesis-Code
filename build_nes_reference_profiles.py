#!/usr/bin/env python3
"""
build_nes_reference_profiles.py
============================================================
Builds a REAL database-average NES profile (CIDER + RSA) from this
project's own real NES training data -- NOT an average across whichever
candidates happen to come out of analyzing one protein (that's a
different, much smaller and less meaningful sample).

CIDER reference (hydropathy / NCPR / FCR / complexity): computed from
nes_data_pipeline/nes_dataset.json's real positive NES entries (NESdb +
NESbase, 499 total). Each entry has full_sequence + nes_start/nes_end, so
a +/-20 residue window can be sliced around the real NES exactly like
app.py does for a live candidate. CIDER is sequence-only (localCIDER),
so no 3D structure is needed -- every entry with real flanking context
is usable (entries missing full_sequence/nes_start/nes_end, e.g. isolated
motif-only records with no protein context, are skipped and counted).
Uses app.py's own compute_linear_cider_profiles(), not a reimplementation.

RSA reference: genuinely needs real 3D structure, which only exists for
a subset of the training data. Uses
nes_data_pipeline/nes30_structural_cider_data.json -- 26 real NES entries
with real per-residue SASA from real AlphaFold structures (Shrake-Rupley,
computed over the FULL protein). Converts raw SASA (Å²) to RSA using the
exact same Tien et al. 2013 per-residue-type normalization app.py's live
pipeline uses (MAX_ASA_TIEN2013), so this is on the same 0-1 scale as a
live candidate's rsa_profile. N=26 is real, not padded -- most training
entries don't have a fetched structure yet.

Both are aligned by each NES's own motif center (same scheme
plot_candidate_profiles.py uses for live candidates) over a +/-20 residue
window, then averaged per relative offset.

Output: nes_reference_profiles.json, consumed by
plot_candidate_profiles.py.

USAGE (run once, or again after the training data changes):
    python3 build_nes_reference_profiles.py
"""

import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

print("Importing app.py to reuse its real compute_linear_cider_profiles() "
      "and MAX_ASA_TIEN2013 constant (same startup cost as run_full_pipeline_cli.py "
      "-- one-time, this script is meant to be re-run occasionally, not per-analysis) ...\n")
from app import compute_linear_cider_profiles, MAX_ASA_TIEN2013, DEFAULT_MAX_ASA  # noqa: E402

FLANK = 20

ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def _accumulate(offsets, values, by_offset):
    for off, val in zip(offsets, values):
        if val is None:
            continue
        by_offset.setdefault(round(off), []).append(float(val))


def build_cider_reference(dataset_path):
    with open(dataset_path, encoding='utf-8') as f:
        entries = json.load(f)

    keys = ['linear_hydropathy', 'linear_ncpr', 'linear_fcr', 'linear_complexity']
    by_offset = {k: {} for k in keys}
    n_used = 0
    n_skipped = 0

    for rec in entries:
        full_seq = rec.get('full_sequence')
        nes_start = rec.get('nes_start')
        nes_end = rec.get('nes_end')
        if not full_seq or not nes_start or not nes_end:
            n_skipped += 1
            continue
        start_idx = nes_start - 1  # 1-indexed -> 0-indexed
        end_idx = nes_end - 1
        if start_idx < 0 or end_idx >= len(full_seq) or start_idx > end_idx:
            n_skipped += 1
            continue

        ctx_start = max(0, start_idx - FLANK)
        ctx_end = min(len(full_seq), end_idx + 1 + FLANK)
        ctx_seq = full_seq[ctx_start:ctx_end]

        cider = compute_linear_cider_profiles(ctx_seq)
        if not cider['cider_computed']:
            n_skipped += 1
            continue

        center = ((start_idx - ctx_start) + (end_idx - ctx_start)) / 2.0
        offsets = [i - center for i in range(len(ctx_seq))]
        for key in keys:
            _accumulate(offsets, cider[key], by_offset[key])
        n_used += 1

    result = {}
    for key in keys:
        offs = sorted(by_offset[key].keys())
        result[key] = {
            'offsets': offs,
            'values': [float(np.mean(by_offset[key][o])) for o in offs],
            'n_per_offset': {str(o): len(by_offset[key][o]) for o in offs},
        }
    result['n_entries_used'] = n_used
    result['n_entries_skipped'] = n_skipped
    return result


def build_rsa_reference(structural_cider_path):
    with open(structural_cider_path, encoding='utf-8') as f:
        entries = json.load(f)

    by_offset = {}
    n_used = 0
    n_skipped = 0

    for rec in entries:
        full_seq = rec.get('full_sequence')
        sasa_raw = rec.get('sasa_per_residue')
        nes_start = rec.get('nes_start')
        nes_end = rec.get('nes_end')
        if not full_seq or not sasa_raw or not nes_start or not nes_end:
            n_skipped += 1
            continue
        if len(sasa_raw) != len(full_seq):
            n_skipped += 1  # index alignment isn't safe to assume otherwise
            continue

        start_idx = nes_start - 1
        end_idx = nes_end - 1
        if start_idx < 0 or end_idx >= len(full_seq) or start_idx > end_idx:
            n_skipped += 1
            continue

        ctx_start = max(0, start_idx - FLANK)
        ctx_end = min(len(full_seq), end_idx + 1 + FLANK)
        ctx_seq = full_seq[ctx_start:ctx_end]
        ctx_raw = sasa_raw[ctx_start:ctx_end]

        rsa = []
        for aa, raw in zip(ctx_seq, ctx_raw):
            resname = ONE_TO_THREE.get(aa.upper())
            max_asa = MAX_ASA_TIEN2013.get(resname, DEFAULT_MAX_ASA)
            rsa.append(min(raw / max_asa, 1.5))

        center = ((start_idx - ctx_start) + (end_idx - ctx_start)) / 2.0
        offsets = [i - center for i in range(len(ctx_seq))]
        _accumulate(offsets, rsa, by_offset)
        n_used += 1

    offs = sorted(by_offset.keys())
    return {
        'consensus_rsa': {
            'offsets': offs,
            'values': [float(np.mean(by_offset[o])) for o in offs],
            'n_per_offset': {str(o): len(by_offset[o]) for o in offs},
        },
        'n_entries_used': n_used,
        'n_entries_skipped': n_skipped,
    }


def main():
    cider_ref = build_cider_reference(THIS_DIR / 'nes_data_pipeline' / 'nes_dataset.json')
    rsa_ref = build_rsa_reference(THIS_DIR / 'nes_data_pipeline' / 'nes30_structural_cider_data.json')

    print(f"\nCIDER reference: {cider_ref['n_entries_used']} real positive NES entries used "
          f"(NESdb + NESbase), {cider_ref['n_entries_skipped']} skipped (no full_sequence/"
          f"nes_start/nes_end -- isolated motif records with no protein context)")
    print(f"RSA reference: {rsa_ref['n_entries_used']} real positive NES entries used -- "
          f"limited to entries with a real fetched AlphaFold structure already computed "
          f"(most training entries don't have one yet), {rsa_ref['n_entries_skipped']} skipped")

    out = {'cider': cider_ref, 'rsa': rsa_ref}
    out_path = THIS_DIR / 'nes_reference_profiles.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
