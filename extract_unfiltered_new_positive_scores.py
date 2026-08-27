#!/usr/bin/env python3
"""
extract_unfiltered_new_positive_scores.py
============================================================
Same technique as extract_unfiltered_negative_scores.py (sys.settrace on the
live /api/unified_crm1_nes/<model_id> route, no app.py edits), applied to
the three NEW positives that came back "no surviving candidate overlapped
target window" in the latest run_holdout_pipeline_test.py run:

    P27958 (HCV core protein)      target 109-133
    Q99153 (Chicken Anemia VP1)    target 375-388
    Q9Y4Z2 (Neurogenin-3, human)   target 131-142

These are real, experimentally-validated positives, but this project's
pipeline scored them <=0.45 (or never proposed a window at all) and they
got filtered out of holdout_test_results.json's nes_motifs -- exactly the
same "matched: false" situation the original 5 negatives were in before
extract_unfiltered_negative_scores.py was written. This script recovers the
same pre-filter detail for these 3 positives so the comparison figures can
show your model's real (sub-threshold) score for them instead of leaving
the cell blank.

REQUIREMENTS: identical to run_holdout_pipeline_test.py -- real internet
(AlphaFold structure fetches) and fpocket installed. Run on the same
pod/machine.

Usage (from the AlphaFold directory):
    python3 extract_unfiltered_new_positive_scores.py

Outputs:
    holdout_positives_unfiltered.json
    holdout_positives_unfiltered.md
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

POSITIVES = [
    ("HCV core protein (CRM1/LMB-validated, Cerutti et al. 2011)", "P27958", 109, 133),
    ("Chicken Anemia Virus VP1 (CRM1/LMB-validated, 2019)", "Q99153", 375, 388),
    ("Neurogenin-3, human (CRM1/LMB-validated, Simon-Areces et al. 2013)", "Q9Y4Z2", 131, 142),
]

TARGET_FUNC_NAME = "unified_crm1_nes_analysis"
_captured = []


def _local_tracer(frame, event, arg):
    if event == "return" and frame.f_code.co_name == TARGET_FUNC_NAME:
        _captured.append({
            "model_id": frame.f_locals.get("model_id"),
            "unified_predictions": frame.f_locals.get("unified_predictions"),
        })
    return _local_tracer


def _global_tracer(frame, event, arg):
    if event == "call" and frame.f_code.co_name == TARGET_FUNC_NAME:
        return _local_tracer
    return None


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def main():
    print("Loading app.py (this triggers full ML/CRM1 initialization -- may take a moment)...")
    sys.path.insert(0, str(THIS_DIR))
    from app import app as flask_app, pocket_detector
    if pocket_detector is not None:
        pocket_detector.fpocket_timeout = 300
        # See pocket_detector.py -- the CRM1-compatibility
        # scoring step after fpocket returns had no timeout of its own.
        pocket_detector.pocket_filter_timeout = 300
    client = flask_app.test_client()

    sys.settrace(_global_tracer)
    try:
        for name, accession, start, end in POSITIVES:
            model_id = f"AF-{accession}-F1"
            print(f"Requesting {name} ({accession})...")
            client.get(f"/api/unified_crm1_nes/{model_id}?uniprot_id={accession}")
    finally:
        sys.settrace(None)

    if len(_captured) != len(POSITIVES):
        print(f"\nWARNING: expected {len(POSITIVES)} captured calls, got {len(_captured)}.\n")

    results = []
    for i, (name, accession, start, end) in enumerate(POSITIVES):
        entry = {"name": name, "accession": accession, "target_window": f"{start}-{end}"}
        call = _captured[i] if i < len(_captured) else {}
        unified = call.get("unified_predictions")

        if unified is None:
            entry["note"] = "unified_predictions wasn't set -- request errored before scoring."
            entry["overlapping_candidates"] = []
        else:
            overlapping = [m for m in unified if overlaps(m["start"], m["end"], start, end)]
            overlapping.sort(key=lambda m: m["combined_score"], reverse=True)
            entry["n_total_candidates_in_protein"] = len(unified)
            entry["n_overlapping_target_window"] = len(overlapping)
            entry["overlapping_candidates"] = [
                {
                    "start": m["start"], "end": m["end"], "sequence": m["sequence"],
                    "combined_score": m["combined_score"],
                    "cleared_0.45_threshold": m["combined_score"] > 0.45,
                    "components": m["components"],
                }
                for m in overlapping
            ]
            if not overlapping:
                entry["note"] = "No candidate window was even proposed over this span."
        results.append(entry)
        print(json.dumps(entry, indent=2, default=str))
        print("-" * 100)

    json_path = THIS_DIR / "holdout_positives_unfiltered.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    md_lines = ["# Unfiltered new-positive candidate scores", ""]
    for entry in results:
        md_lines.append(f"## {entry['name']} ({entry['accession']}), target {entry['target_window']}")
        md_lines.append("")
        if not entry["overlapping_candidates"]:
            md_lines.append(f"*{entry.get('note', 'no overlapping candidates')}*")
        else:
            md_lines.append("| Window | Sequence | combined_score | cleared 0.45? |")
            md_lines.append("|---|---|---|---|")
            for c in entry["overlapping_candidates"]:
                md_lines.append(
                    f"| {c['start']}-{c['end']} | {c['sequence']} | {c['combined_score']:.3f} | "
                    f"{'yes' if c['cleared_0.45_threshold'] else 'no'} |"
                )
        md_lines.append("")
    md_path = THIS_DIR / "holdout_positives_unfiltered.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
