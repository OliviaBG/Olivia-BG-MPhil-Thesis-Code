#!/usr/bin/env python3
"""
expand_leucine_zipper_negatives.py
============================================================
Pulls MORE real leucine-zipper hard negatives from UniProt than
nes_negatives.csv currently has (68 rows from 37 unique proteins), reusing
negative_dataset_builder.py's real pipeline (same UniProt query mechanics,
same regex/PSSM scoring, same known-NES-overlap filter, same match_start indexing fix) rather than reimplementing any of it.

WHY THIS EXISTS: evaluate_crm1_pocket_signal.py found leucine_zipper
negatives trend closer to real NES than coiled_coil negatives do on
charge_score (AUC=0.563 vs 0.615) -- i.e. they're the HARDER, more
NES-like case. UniProt has 129 real reviewed human proteins with a
Leucine-zipper annotation (500+ across all organisms), but the existing
dataset only ever sourced 37 of them, so there's real room to grow this
specific hard-negative category.

DELIBERATELY WRITES TO A SEPARATE FILE, not merged into nes_negatives.csv:
that file is used both by this evaluation harness AND by the actual NES
predictor's training data. Oversampling the hard (leucine-zipper) case is
straightforwardly good for evaluation power, but silently inflating its
share of the TRAINING negative class shifts the model's calibration away
from the real ~90% coiled_coil / ~10% leucine_zipper population without
that being a deliberate choice. Merge deliberately, not by accident.

Also explicitly excludes any accession already present in
nes_negatives.csv, so this is purely the NEW, additional pool -- no
duplicated proteins between the two files.

REQUIREMENTS: real internet access to rest.uniprot.org, since this runs a
live cursor-paginated query loop. Run it locally or on a compute node, the
same as evaluate_crm1_pocket_signal.py.

USAGE:
    pip install requests
    python3 expand_leucine_zipper_negatives.py
    python3 expand_leucine_zipper_negatives.py --taxon all --max-entries 600
"""
import argparse
import csv
from pathlib import Path

from negative_dataset_builder import (
    build_negative_dataset, write_outputs, fetch_uniprot_entries,
)

THIS_DIR = Path(__file__).resolve().parent
EXISTING_NEGATIVES_CSV = THIS_DIR / 'nes_negatives' / 'nes_negatives.csv'
OUTDIR = THIS_DIR / 'nes_negatives_leucine_zipper_expansion'


def existing_accessions():
    if not EXISTING_NEGATIVES_CSV.exists():
        return set()
    with open(EXISTING_NEGATIVES_CSV, newline='', encoding='utf-8') as f:
        return {row['accession'] for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--taxon', default='9606', help='NCBI taxon ID, or "all" (default: 9606 = human, '
                                                       'matching the existing dataset -- 129 reviewed entries)')
    ap.add_argument('--max-entries', type=int, default=200,
                     help='Max UniProt entries to pull (default 200, comfortably above the '
                          '~129 human leucine-zipper entries that actually exist)')
    ap.add_argument('--min-score', type=float, default=0.4,
                     help='Same PSSM score threshold negative_dataset_builder.py uses by default')
    args = ap.parse_args()

    already = existing_accessions()
    print(f"{len(already)} accessions already in {EXISTING_NEGATIVES_CSV.relative_to(THIS_DIR)} "
          f"-- these will be excluded from the new pull\n")

    print("Querying UniProt for leucine-zipper-annotated entries...")
    hits = build_negative_dataset(
        taxon=args.taxon,
        max_entries=args.max_entries,
        min_score=args.min_score,
        want_coiled_coil=False,       # already well-represented; this run is leucine_zipper-only
        want_leucine_zipper=True,
    )
    print(f"{len(hits)} raw regex hits across all fetched entries (before dedup)")

    new_hits = [h for h in hits if h.accession not in already]
    n_new_proteins = len({h.accession for h in new_hits})
    print(f"{len(new_hits)} hits from {n_new_proteins} NEW proteins not already in nes_negatives.csv "
          f"({len(hits) - len(new_hits)} hits dropped as duplicates)")

    if not new_hits:
        print("Nothing new to write -- either the taxon/max-entries didn't find anything beyond "
              "what's already in nes_negatives.csv, or something upstream failed silently.")
        return

    csv_path, fasta_path = write_outputs(new_hits, str(OUTDIR))
    print(f"\nWrote {len(new_hits)} NEW leucine-zipper negatives to:\n  {csv_path}\n  {fasta_path}")
    print(f"\nThis is a SEPARATE file from nes_negatives.csv on purpose -- see this script's "
          f"docstring for why. evaluate_crm1_pocket_signal.py's load_negative_examples() now "
          f"reads both files automatically for evaluation purposes. Folding any of this into "
          f"actual model training (nes_ml_predictor_improved.py) is a separate, deliberate "
          f"decision this does NOT make for you.")


if __name__ == '__main__':
    main()
