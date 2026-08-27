#!/usr/bin/env python3
"""
extract_unfiltered_negative_scores.py
============================================================
run_holdout_pipeline_test.py's negatives all came back "no surviving
candidate overlapped target window" -- which only tells you a candidate's
combined_score was either (a) never proposed as a window at all, or
(b) proposed and scored <=0.45, the route's post-overlap-filter cutoff
(app.py, unified_crm1_nes_analysis(), around the "LOWERED threshold from
0.55 to 0.45" comment). Neither run_holdout_pipeline_test.py nor
holdout_test_results.json ever look at the actual number, because
/api/unified_crm1_nes/<model_id> only returns `filtered_predictions` in
its `nes_motifs` field -- the pre-filter `unified_predictions` list (every
candidate window in the protein, every real combined_score, before the
>0.45 cutoff and before overlap-deduplication) is a local variable inside
unified_crm1_nes_analysis() and is thrown away the moment the function
returns.

This script does NOT edit app.py -- not even temporarily. It uses Python's
built-in sys.settrace (the same mechanism pdb/debuggers use) to read that
local variable at the instant the function returns. That's a read-only
observation of the exact live code path app.py already runs, not a
reimplementation and not a modification of the file on disk.

REQUIREMENTS: identical to run_holdout_pipeline_test.py -- real internet
(AlphaFold structure fetches) and fpocket installed. Will NOT work in a
network-isolated environment. Run this on the same pod/machine you ran
run_holdout_pipeline_test.py on.

Usage (from the AlphaFold directory):
    python3 extract_unfiltered_negative_scores.py

Outputs:
    holdout_negatives_unfiltered.json -- every candidate window overlapping
        each negative's target span, with its real combined_score and full
        component breakdown, whether or not it cleared 0.45.
    holdout_negatives_unfiltered.md   -- same thing, as a readable table.

After this runs, generate_holdout_comparison_figures.py will automatically
pick up holdout_negatives_unfiltered.json (if present in the same
directory) and plot the negatives' real scores instead of leaving them
blank.
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Kept in sync by hand with the NEGATIVES list in run_holdout_pipeline_test.py.
# If you've changed that list, update this one too.
NEGATIVES = [
    ("Jun dimerization protein 2 (mouse, bZIP leucine zipper TF)", "P97875", 114, 123),
    ("Caveolae-associated protein 1 (mouse)",                      "O54724", 63, 72),
    ("Apoptosis inhibitor 5 (mouse)",                              "O35841", 384, 393),
    ("Protein AF-10 / MLLT10 (mouse)",                             "O54826", 773, 782),
    ("Spermatogenic leucine zipper protein 1 (bovine)",            "Q32L17", 114, 124),
    ("Lamin A/C, human (coiled-coil rod domain)", "P02545", 362, 371),
    ("Tropomyosin beta chain / TPM2, human (coiled-coil)", "P07951", 4, 13),
    ("Myosin-9, human (coiled-coil rod domain)", "P35579", 1067, 1076),
]

# The exact function name behind @app.route('/api/unified_crm1_nes/<model_id>')
# in app.py. If this script starts printing the "expected N captured calls"
# warning below, check that this name still matches -- it means app.py's
# route function got renamed since this script was written.
TARGET_FUNC_NAME = "unified_crm1_nes_analysis"

_captured = []  # one entry appended per call to the target function, in call order


def _local_tracer(frame, event, arg):
    if event == "return" and frame.f_code.co_name == TARGET_FUNC_NAME:
        _captured.append({
            "model_id": frame.f_locals.get("model_id"),
            "unified_predictions": frame.f_locals.get("unified_predictions"),
        })
    return _local_tracer


def _global_tracer(frame, event, arg):
    # Only ask Python to keep tracing frames belonging to the one function we
    # care about -- every other function call in the app (structure parsing,
    # fpocket, ML scoring,...) gets skipped (returning None means "don't
    # trace this frame's lines"), so this has negligible overhead.
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
        pocket_detector.fpocket_timeout = 300  # 5 min, enough for large (~2000-residue) proteins
        # The CRM1-compatibility scoring step that runs after
        # fpocket returns had no timeout of its own (see pocket_detector.py)
        # -- this NEGATIVES list includes Myosin-9 (1960 residues), which
        # hung on exactly this step before the fix.
        pocket_detector.pocket_filter_timeout = 300
    client = flask_app.test_client()

    sys.settrace(_global_tracer)
    try:
        for name, accession, start, end in NEGATIVES:
            model_id = f"AF-{accession}-F1"
            print(f"Requesting {name} ({accession})...")
            client.get(f"/api/unified_crm1_nes/{model_id}?uniprot_id={accession}")
    finally:
        sys.settrace(None)

    if len(_captured) != len(NEGATIVES):
        print(f"\nWARNING: expected {len(NEGATIVES)} captured calls, got {len(_captured)}. "
              f"Results below are matched to NEGATIVES by call order, so a mismatch means some "
              f"entries may be misaligned or missing -- check TARGET_FUNC_NAME above still matches "
              f"app.py's route function.\n")

    results = []
    for i, (name, accession, start, end) in enumerate(NEGATIVES):
        entry = {"name": name, "accession": accession, "target_window": f"{start}-{end}"}
        call = _captured[i] if i < len(_captured) else {}
        unified = call.get("unified_predictions")

        if unified is None:
            entry["note"] = ("unified_predictions wasn't set when the function returned -- the "
                              "request errored out before scoring even started (e.g. structure "
                              "fetch failure), same situation as the ERROR rows in "
                              "holdout_test_results.json.")
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
                entry["note"] = ("No candidate window was even proposed over this span -- the "
                                  "window-generation step never got this far. Not a scoring issue: "
                                  "there was simply nothing to score here.")
        results.append(entry)
        print(json.dumps(entry, indent=2, default=str))
        print("-" * 100)

    json_path = THIS_DIR / "holdout_negatives_unfiltered.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    md_lines = [
        "# Unfiltered negative candidate scores",
        "",
        "Every candidate window overlapping each negative's target window, read directly out of "
        "app.py's unified_predictions (before the >0.45 filter) via sys.settrace. app.py was not "
        "modified to produce this.",
        "",
    ]
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
    md_path = THIS_DIR / "holdout_negatives_unfiltered.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
