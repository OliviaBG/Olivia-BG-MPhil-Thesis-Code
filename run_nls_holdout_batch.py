"""Batched runner for the NLS holdout benchmark.

run_nls_holdout_pipeline_test.py scores all 50 candidates in a single
process, which is impractical wherever a per-invocation time limit
applies. This script reuses run_one() from that script UNCHANGED, but
processes positives and negatives in separate invocations and caches
partial results to disk, so an interrupted run can be resumed and the two
halves merged afterwards.

Usage: python3 run_nls_holdout_batch.py positives
       python3 run_nls_holdout_batch.py negatives
       python3 run_nls_holdout_batch.py merge
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_nls_holdout_pipeline_test import run_one, CANDIDATES_PATH, format_md_row

CACHE_DIR = THIS_DIR / "nls_holdout_batch_cache"
CACHE_DIR.mkdir(exist_ok=True)


def run_batch(which):
    candidates = json.loads(CANDIDATES_PATH.read_text())
    items = candidates[which]

    sys.path.insert(0, str(THIS_DIR))
    from app import app as flask_app
    client = flask_app.test_client()

    results = []
    for name, accession, full_sequence, start, end, note in items:
        r = run_one(client, name, accession, full_sequence, start, end, note)
        results.append(r)
        print(f"{accession}: matched={r.get('matched')} "
              f"struct={r.get('structural_fetch_status', '')[:30]}")

    (CACHE_DIR / f"{which}.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} {which} results")


def merge():
    import csv
    pos_results = json.loads((CACHE_DIR / "positives.json").read_text())
    neg_results = json.loads((CACHE_DIR / "negatives.json").read_text())

    pool = set()
    for fname in ["nls_dataset.csv", "nls_negatives.csv"]:
        path = THIS_DIR / "nls_data_pipeline" / fname
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                pool.update(row["accession"] for row in csv.DictReader(f))
    candidates = json.loads(CANDIDATES_PATH.read_text())
    test_accessions = {a for _, a, *_ in candidates["positives"]} | {a for _, a, *_ in candidates["negatives"]}
    overlap = test_accessions & pool

    n_pos_matched = sum(1 for r in pos_results if r.get("matched"))
    n_neg_matched = sum(1 for r in neg_results if r.get("matched"))
    n_pos_errors = sum(1 for r in pos_results if "error" in r)
    n_neg_errors = sum(1 for r in neg_results if "error" in r)
    sensitivity = n_pos_matched / max(1, len(pos_results) - n_pos_errors)
    specificity = 1 - (n_neg_matched / max(1, len(neg_results) - n_neg_errors))

    all_results = pos_results + neg_results
    n_real_structure = sum(1 for r in all_results if r.get("structural_data_used"))
    fetch_status_counts = {}
    for r in all_results:
        s = r.get("structural_fetch_status", "unknown").split(":")[0]
        fetch_status_counts[s] = fetch_status_counts.get(s, 0) + 1

    out_json = {
        "accession_overlap_with_training_pool": sorted(overlap),
        "structural_data_summary": {
            "n_candidates": len(all_results),
            "n_used_real_structural_data": n_real_structure,
            "n_fell_back_to_neutral": len(all_results) - n_real_structure,
            "fetch_status_breakdown": fetch_status_counts,
        },
        "summary": {
            "n_positives": len(pos_results), "n_positives_matched": n_pos_matched,
            "n_positives_errors": n_pos_errors,
            "n_negatives": len(neg_results), "n_negatives_matched_incorrectly": n_neg_matched,
            "n_negatives_errors": n_neg_errors,
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
        },
        "positives": pos_results,
        "negatives": neg_results,
    }
    (THIS_DIR / "nls_holdout_test_results.json").write_text(json.dumps(out_json, indent=2))

    md_lines = [
        "# NLS holdout pipeline test results",
        "",
        "Full app.py NLS pipeline (`NLSPredictor.scan_sequence()` + real `/api/nls_scan` route: "
        "consensus regex pre-filter, trained classifier, bipartite/tripartite detection, "
        "greedy overlap removal, `_nls_exposure_factor()` accessibility gate, CIDER/RSA profiling), "
        "run via Flask's test_client -- the exact production code path. For each candidate the real "
        "`/api/structure/<model_id>` route was also called first, exactly as the live app does on "
        "structure load. Run in two batches (positives/negatives) and merged, so that the "
        "full Flask + app.py overhead for all 50 candidates does not have to fit in a single "
        "invocation.",
        "",
        f"**Structural data coverage:** {n_real_structure}/{len(all_results)} candidates got real "
        f"structural data; the rest fell back to `/api/nls_scan`'s documented neutral default "
        f"(RSA=0.4, exposure_factor=1.0x). Fetch status breakdown: {fetch_status_counts}.",
        "",
        f"**Summary:** sensitivity {sensitivity:.1%} ({n_pos_matched}/{len(pos_results) - n_pos_errors} "
        f"positives matched), specificity {specificity:.1%} "
        f"({len(neg_results) - n_neg_errors - n_neg_matched}/{len(neg_results) - n_neg_errors} "
        f"negatives correctly rejected).",
        "",
        f"Accession overlap with nls_dataset.csv/nls_negatives.csv training pool: {sorted(overlap) or 'none'}",
        "",
        "## Positives",
        "",
        "| Name | Accession | Structural data | nls_probability | raw_nls_probability | predicted_class | is_bipartite | pssm_score | accessibility_rsa |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in pos_results:
        md_lines.append(format_md_row(r))
    md_lines += [
        "",
        "## Negatives",
        "",
        "| Name | Accession | Structural data | nls_probability | raw_nls_probability | predicted_class | is_bipartite | pssm_score | accessibility_rsa |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in neg_results:
        md_lines.append(format_md_row(r))

    (THIS_DIR / "nls_holdout_test_results.md").write_text("\n".join(md_lines))

    print(f"Sensitivity: {sensitivity:.1%}  |  Specificity: {specificity:.1%}")
    print(f"Structural data used: {n_real_structure}/{len(all_results)}  {fetch_status_counts}")
    print(f"Accession overlap: {overlap or 'none'}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("positives", "negatives"):
        run_batch(arg)
    elif arg == "merge":
        merge()
    else:
        print("Usage: positives | negatives | merge")
