#!/usr/bin/env python3
"""
decontaminate_holdout_leakage.py
============================================================
fetch_bipartite_and_nonclassical_candidates.py's collision
check only excluded accessions already present in nls_dataset.csv -- it
never checked against nls_holdout_data/candidates.json, the file that's
supposed to be permanently held out of training. The result: the broad
multi-taxa UniProt pull re-added 23 of the 25 holdout POSITIVE accessions
back into nls_dataset.csv as new "real" training positives (27 rows total,
some accessions contributing >1 row), plus 46 auto-generated
protein_matched_random negative rows for those same 23 proteins. Confirmed
via nls_holdout_test_results.json's accession_overlap_with_training_pool
field after the resulting retrain, and independently verified against
nls_data_pipeline/nls_dataset_backup_before_viral_expansion_2026-08-05_1337.csv
and nls_negatives_backup_before_viral_expansion_2026-08-05_1337.csv that
NONE of these 23 accessions were present before this project's scraping --
this is new contamination, not a pre-existing issue.

This script removes exactly those rows (by accession) from both CSVs --
nothing else is touched. The 2 holdout positives that were NOT re-scraped
(P01106, P09651) were never contaminated and aren't affected. All ~4500
other newly-scraped rows (real bipartite/non-classical/monopartite
positives for accessions that are NOT in the holdout set) are left alone.

Usage: python3 decontaminate_holdout_leakage.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_CSV = HERE / "nls_dataset.csv"
NEGATIVES_CSV = HERE / "nls_negatives.csv"
HOLDOUT_CANDIDATES = HERE.parent / "nls_holdout_data" / "candidates.json"


def main():
    holdout = json.loads(HOLDOUT_CANDIDATES.read_text())
    holdout_pos_accs = {p[1] for p in holdout["positives"]}
    holdout_neg_accs = {n[1] for n in holdout["negatives"]}
    all_holdout_accs = holdout_pos_accs | holdout_neg_accs

    pos_rows = list(csv.DictReader(open(DATASET_CSV, encoding="utf-8")))
    neg_rows = list(csv.DictReader(open(NEGATIVES_CSV, encoding="utf-8")))

    contaminated_pos = sorted({r["accession"] for r in pos_rows if r["accession"] in all_holdout_accs})
    print(f"Found {len(contaminated_pos)} contaminated accession(s) in nls_dataset.csv "
          f"(overlap with nls_holdout_data/candidates.json): {contaminated_pos}")

    kept_pos = [r for r in pos_rows if r["accession"] not in all_holdout_accs]
    kept_neg = [r for r in neg_rows if r["accession"] not in all_holdout_accs]

    n_pos_removed = len(pos_rows) - len(kept_pos)
    n_neg_removed = len(neg_rows) - len(kept_neg)

    with open(DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pos_rows[0].keys()))
        w.writeheader()
        w.writerows(kept_pos)

    with open(NEGATIVES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(neg_rows[0].keys()))
        w.writeheader()
        w.writerows(kept_neg)

    print(f"\nnls_dataset.csv: removed {n_pos_removed} rows ({len(pos_rows)} -> {len(kept_pos)})")
    print(f"nls_negatives.csv: removed {n_neg_removed} rows ({len(neg_rows)} -> {len(kept_neg)})")
    print(f"\nStill-clean holdout positives untouched: {sorted(holdout_pos_accs - all_holdout_accs.intersection(contaminated_pos))}")
    print("\nRetrain after this (`python3 nls_ml_predictor.py train`), then re-run "
          "run_nls_holdout_pipeline_test.py for a trustworthy holdout number.")


if __name__ == "__main__":
    main()
