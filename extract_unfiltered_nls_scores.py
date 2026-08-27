#!/usr/bin/env python3
"""
extract_unfiltered_nls_scores.py
============================================================
NLS-side analog of extract_unfiltered_negative_scores.py /
extract_unfiltered_new_positive_scores.py (both NES-only). Extracts the
scores for every NLS candidate, including those excluded by the
accessibility gate, so the full score distribution is available for the
tool-comparison plots.

run_nls_holdout_pipeline_test.py's misses all come back "no surviving
candidate overlapped target window", which conflates two different real
situations: (a) scan_sequence() never proposed a window over that span at
all, or (b) it did, and the real nls_probability (after every gate/veto)
was simply <= 0.5, the app.py route's survival cutoff (see nls_scan(),
"regions = [r for r in regions if r['nls_probability'] > 0.5 ...]"). Neither
run_nls_holdout_pipeline_test.py nor nls_holdout_test_results.json ever look
at the real number for case (b), because /api/nls_scan only returns the
POST-filter list in its `nls_binding_regions` field.

Same read-only sys.settrace technique as the NES scripts (does NOT edit
app.py, not even temporarily) -- but with one real difference worth noting:
NES's unified_predictions is never reassigned inside its route function, so
capturing frame.f_locals at the 'return' event alone was enough. NLS's
`regions` local IS reassigned partway through nls_scan() (the exact
`regions = [r for r in regions if ...]` line above) -- by the time the
function returns, the pre-filter list is already gone. So this script traces
every 'line' event inside nls_scan() (not just 'return') and keeps whichever
`regions` snapshot was LONGEST for that call, which is always the pre-filter
one (filtering can only shrink the list, never grow it) -- robust to exactly
where in the function body that reassignment happens, so it won't silently
break if app.py's internals get refactored later. BUGFIX: this script previously computed the struct_seq-vs-
full_sequence offset per accession (for fallback-structure cases using a
partial PDB fragment or ESMFold model instead of native AlphaFold) but then
discarded it before the overlap check, always comparing candidate windows
(in struct_seq's own, possibly-shifted numbering) against the raw,
un-remapped UniProt target window. This was caught and hand-fixed for one
case before (P03269), but never applied generally or checked for any other
accession -- confirmed by hand this project that P03466, P03255, and
P05777 (three of the five "no candidate proposed" positives) all have a
genuine consensus-regex match at their real target sequence, directly
contradicting the old output's "pre-filter never generated a window here"
claim. Now fixed to always apply the same offset remap
run_nls_holdout_pipeline_test.py already used correctly (see
offsets_by_accession / remap_applied below) -- re-run this script to get
corrected results; any accession using a substitute structure should be
double-checked in the new output's `remap_applied` / `remapped_target_window`
fields.

REQUIREMENTS: identical to run_nls_holdout_pipeline_test.py -- real internet
(AlphaFold + UniProt fetches). Will NOT work in a network-isolated environment.
Run this on the same machine you run the holdout tests on.

Usage (from the AlphaFold directory):
    python3 extract_unfiltered_nls_scores.py

Outputs:
    nls_holdout_unfiltered.json -- every candidate window overlapping each
        holdout accession's target span, with its real nls_probability and
        full veto/factor breakdown, whether or not it cleared 0.5. Covers
        BOTH positives and negatives (one file, unlike the two separate NES
        scripts) since both come from the same nls_scan() code path and the
        same nls_holdout_data/candidates.json source.
    nls_holdout_unfiltered.md   -- same thing, as a readable table.

After this runs, plot_nls_candidate_count_parsimony.py and
generate_nls_comparison_figures.py can both be pointed at
nls_holdout_unfiltered.json for real pre-filter scores instead of leaving
misses blank.
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CANDIDATES_PATH = THIS_DIR / "nls_holdout_data" / "candidates.json"

TARGET_FUNC_NAME = "nls_scan"

_captured = []   # one entry appended per call, in call order
_current = None  # {'regions': [...], 'model_id':...} for the call in progress


def _local_tracer(frame, event, arg):
    global _current
    if frame.f_code.co_name != TARGET_FUNC_NAME:
        return _local_tracer
    if event in ("line", "return"):
        regions = frame.f_locals.get("regions")
        if isinstance(regions, list) and (_current is None or len(regions) > len(_current["regions"])):
            _current = {"regions": regions, "model_id": frame.f_locals.get("model_id")}
    if event == "return":
        _captured.append(_current or {"regions": [], "model_id": None})
        _current = None
        return None
    return _local_tracer


def _global_tracer(frame, event, arg):
    global _current
    if event == "call" and frame.f_code.co_name == TARGET_FUNC_NAME:
        _current = {"regions": [], "model_id": None}
        return _local_tracer
    return None


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def main():
    if not CANDIDATES_PATH.exists():
        print(f"ERROR: {CANDIDATES_PATH} not found.")
        sys.exit(1)

    print("Loading app.py (this triggers full ML predictor initialization -- may take a moment)...")
    sys.path.insert(0, str(THIS_DIR))
    from app import app as flask_app
    import run_nls_holdout_pipeline_test as holdout_mod

    client = flask_app.test_client()
    candidates = json.loads(CANDIDATES_PATH.read_text())

    entries = []  # (label, name, accession, full_sequence, start, end, note)
    for name, accession, seq, start, end, note in candidates.get("positives", []):
        entries.append(("positive", name, accession, seq, start, end, note))
    for name, accession, seq, start, end, note in candidates.get("negatives", []):
        entries.append(("negative", name, accession, seq, start, end, note))

    # BUGFIX. This dict is the actual fix -- previously `offset`
    # was computed per-entry inside this loop (the `found = full_sequence.
    # find(struct_seq)` line below) but then discarded the moment the loop
    # moved to the next accession, so the results-building loop further
    # down always compared candidate windows (in struct_seq's own, possibly
    # shifted numbering) against the RAW, un-remapped UniProt target window.
    # For any accession using a partial/fragment structure (a PDB chain
    # that doesn't start at residue 1, or similarly offset ESMFold output)
    # this silently produced false "no candidate overlapped" results even
    # when scan_sequence() genuinely did propose a real, matching candidate
    # -- caught by hand for P03269 previously (see run_nls_holdout_pipeline_
    # test.py's remapped_target_start/end, which already does this
    # correctly), but never applied here, and never checked for any other
    # fallback-structure accession. Mirrors run_nls_holdout_pipeline_test.
    # py's own remap logic exactly (same "clean contiguous substring or
    # leave unremapped" rule) so the two scripts agree.
    offsets_by_accession = {}

    sys.settrace(_global_tracer)
    try:
        for label, name, accession, full_sequence, target_start, target_end, note in entries:
            print(f"Requesting {name} ({accession}, {label})...")
            struct_data, struct_status = holdout_mod.fetch_structural_data(client, accession)
            used_sequence = full_sequence
            plddt, sasa, consensus_z, agreement_sd = None, None, None, None
            offset = 0
            if struct_status == "ok":
                struct_seq = struct_data.get("sequence", "")
                if struct_seq:
                    used_sequence = struct_seq
                    if struct_seq != full_sequence:
                        found = full_sequence.find(struct_seq)
                        if found >= 0:
                            offset = found
                        else:
                            offset = None  # not a clean substring -- leave target unremapped, same as run_nls_holdout_pipeline_test.py
                plddt = struct_data.get("plddt")
                sasa = struct_data.get("sasa")
                consensus_z = struct_data.get("consensus_z")
                agreement_sd = struct_data.get("agreement_sd")
            offsets_by_accession[accession] = offset

            payload = {"sequence": used_sequence, "model_id": accession}
            if plddt:
                payload["plddt"] = plddt
            if sasa:
                payload["sasa"] = sasa
            if consensus_z:
                payload["consensus_z"] = consensus_z
            if agreement_sd:
                payload["agreement_sd"] = agreement_sd
            client.post("/api/nls_scan", json=payload)
    finally:
        sys.settrace(None)

    if len(_captured) != len(entries):
        print(f"\nWARNING: expected {len(entries)} captured calls, got {len(_captured)}. "
              f"Results below are matched by call order -- check TARGET_FUNC_NAME still matches "
              f"app.py's route function name if this looks wrong.\n")

    results = []
    for i, (label, name, accession, full_sequence, target_start, target_end, note) in enumerate(entries):
        entry = {"label": label, "name": name, "accession": accession,
                  "target_window": f"{target_start}-{target_end}", "note": note}
        call = _captured[i] if i < len(_captured) else {}
        regions = call.get("regions") or []

        # BUGFIX: actually apply the offset captured during the
        # request loop above (offsets_by_accession), same remap
        # run_nls_holdout_pipeline_test.py already does correctly -- this
        # replaces the old version of this block, which always compared
        # against the raw, un-remapped target window regardless of offset
        # (see git history / the comment above offsets_by_accession
        # for the full story). offset is None when struct_seq existed but
        # wasn't a clean contiguous substring of full_sequence (real
        # indels/mutations, not just a truncation) -- left unremapped in
        # that case too, same "wrong-but-honest" choice as the reference
        # script, and flagged via remap_applied=False below.
        offset = offsets_by_accession.get(accession, 0)
        if offset:
            remapped_start, remapped_end = target_start - offset, target_end - offset
        else:
            remapped_start, remapped_end = target_start, target_end
        entry["remap_applied"] = bool(offset)
        if offset:
            entry["remapped_target_window"] = f"{remapped_start}-{remapped_end}"

        overlapping = [r for r in regions if overlaps(r["start"], r["end"], remapped_start - 1, remapped_end - 1)]
        overlapping.sort(key=lambda r: r["nls_probability"], reverse=True)
        entry["n_total_candidates_in_protein"] = len(regions)
        entry["n_overlapping_target_window"] = len(overlapping)
        entry["overlapping_candidates"] = [
            {
                "start": r["start"], "end": r["end"],
                "nls_probability": r["nls_probability"],
                "raw_nls_probability": r.get("raw_nls_probability"),
                "cleared_0.5_threshold": r["nls_probability"] > 0.5,
                "predicted_class": r.get("predicted_class"),
                "basic_background_factor": r.get("basic_background_factor"),
                "anchor_factor": r.get("anchor_factor"),
                "dna_binding_domain_factor": r.get("dna_binding_domain_factor"),
            }
            for r in overlapping
        ]
        if not overlapping:
            if entry["remap_applied"]:
                entry["note_extraction"] = (
                    f"No candidate window overlapped the target after remapping it into the "
                    f"structure's own numbering ({entry['remapped_target_window']}, offset={offset} "
                    f"from full_sequence). scan_sequence() found {len(regions)} candidate(s) somewhere "
                    f"in this protein but none over this specific (remapped) span -- a real miss, not "
                    f"an unremapped-coordinate artifact."
                )
            else:
                entry["note_extraction"] = ("No candidate window was proposed over this span at all -- "
                                              "scan_sequence()'s consensus regex pre-filter never generated "
                                              "a window here. Not a scoring/veto issue.")
        results.append(entry)
        print(json.dumps(entry, indent=2, default=str))
        print("-" * 100)

    json_path = THIS_DIR / "nls_holdout_unfiltered.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    md_lines = [
        "# Unfiltered NLS holdout candidate scores",
        "",
        "Every candidate window overlapping each holdout accession's target window, read directly "
        "out of app.py's nls_scan() pre-filter `regions` list via sys.settrace. app.py was not "
        "modified to produce this. As of , any accession whose structure is a partial/"
        "fragment model (not starting at UniProt residue 1) gets its target window remapped into "
        "the structure's own numbering before the overlap check -- see each entry's `remap_applied` "
        "/ `remapped_target_window` fields below for which ones this affected.",
        "",
    ]
    for entry in results:
        md_lines.append(f"## {entry['name']} ({entry['accession']}, {entry['label']}), "
                         f"target {entry['target_window']}")
        md_lines.append("")
        if not entry["overlapping_candidates"]:
            md_lines.append(f"*{entry.get('note_extraction', 'no overlapping candidates')}*")
        else:
            md_lines.append("| Window | nls_probability | cleared 0.5? | class |")
            md_lines.append("|---|---|---|---|")
            for c in entry["overlapping_candidates"]:
                md_lines.append(
                    f"| {c['start']}-{c['end']} | {c['nls_probability']:.3f} | "
                    f"{'yes' if c['cleared_0.5_threshold'] else 'no'} | {c['predicted_class']} |"
                )
        md_lines.append("")
    md_path = THIS_DIR / "nls_holdout_unfiltered.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
