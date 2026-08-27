#!/usr/bin/env python3
"""
run_nls_holdout_pipeline_test.py
============================================================
NLS-side analog of run_holdout_pipeline_test.py (the NES holdout script).
Runs a genuinely held-out set of 25 positives + 25 negatives (see
nls_holdout_candidates.md / nls_holdout_data/candidates.json for full
provenance and UniProt verification) through THIS project's real, live
/api/nls_scan Flask route -- i.e. NOT a bare call to NLSPredictor.predict()
in isolation, but the exact same whole-sequence scan pipeline the running
app uses: NLSPredictor.scan_sequence() (monopartite/bipartite consensus
regex pre-filter -> trained classifier -> bipartite/tripartite detection ->
greedy overlap removal), then the real _nls_exposure_factor() accessibility
gate and CIDER/RSA profile generation exactly as app.py performs them.

Uses Flask's test_client() against the imported app module directly, same
approach and same reasoning as the NES script: this is the exact code path
production traffic hits, not a reimplementation.

STRUCTURAL DATA: for each candidate this script first calls the real,
unmodified /api/structure/<model_id> route (same one the frontend calls on
initial load) to fetch a genuine AlphaFold model and compute real per-residue
pLDDT + consensus SASA/RSA (Shrake-Rupley/Lee-Richards/etc, see
calculate_sasa()/consensus_accessibility.py) exactly as app.py does it --
real outbound requests.get() calls to alphafold.ebi.ac.uk and
rest.uniprot.org, not a mock. If that succeeds, those real arrays are passed
into /api/nls_scan so the real _nls_exposure_factor() accessibility gate is
exercised for real, not at its neutral default.

This will NOT succeed for every candidate, and that's expected, not a bug:
(a) several entries here are viral proteins or short secreted peptides
unlikely to have an AlphaFold DB entry at all (AlphaFold DB is
UniProt-proteome-based; most single viral strain proteins and cleaved mature
peptides aren't separately modeled), and (b) a domain-allowlisted network
returns HTTP 403 from its proxy for alphafold.ebi.ac.uk and
rest.uniprot.org, so every fetch fails in that case no matter what -- this is exactly the same constraint the NES side's
run_holdout_pipeline_test.py already documents ("Will NOT work in an
environment with no general network access"). Run in an environment with
real internet access, this script will
get real structural gating for whichever candidates AlphaFold DB actually
covers; the ones it doesn't cover, or that fail here, fall back to the same
honest neutral default (RSA=0.4, exposure_factor=1.0x) that
/api/nls_scan/_nls_exposure_factor already use when sasa isn't available --
an existing, intentional, honestly-labelled fallback in app.py itself, not a
workaround invented for this script. Each result records
'structural_data_used': true/false so this is never silently ambiguous.

WHY THIS SET: none of these 50 examples are anywhere in this project's NLS
training data (nls_data_pipeline/nls_dataset.csv, nls_negatives.csv --
320-accession pool). See nls_holdout_candidates.md for full sourcing,
UniProt motif/region citations, and the exclusion-check methodology.

Usage: python3 run_nls_holdout_pipeline_test.py
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CANDIDATES_PATH = THIS_DIR / "nls_holdout_data" / "candidates.json"


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def fetch_structural_data(client, accession):
    """Calls the real, unmodified /api/structure/<model_id> route -- exactly
    what the frontend calls on initial load. Real outbound network calls to
    AlphaFold DB + UniProt happen inside app.py's get_structure(), not here.

    Returns (data_dict, status) where status is one of:
      'ok'            -- real structure fetched, plddt/sasa arrays present
      'not_found'     -- AlphaFold DB has no entry for this accession (HTTP 404
                          from get_structure, or its own 'error' key)
      'network_error' -- the request itself failed (connection error/timeout/
                          proxy block) -- can't tell not_found from network
                          trouble from inside a Flask test_client response, so
                          this is inferred from an exception during the call
      'no_plddt'      -- structure returned but with an empty/missing plddt
                          array (shouldn't normally happen, defensive case)
    """
    model_id = f"AF-{accession}-F1"
    try:
        resp = client.get(f"/api/structure/{model_id}?uniprot_id={accession}")
    except Exception as e:
        return None, f"network_error: {e}"

    try:
        data = resp.get_json()
    except Exception as e:
        return None, f"network_error: bad JSON ({e})"

    if data is None:
        return None, "network_error: empty response"
    if "error" in data:
        err = data["error"]
        if "not found" in err.lower() or "404" in err or "Failed to download" in err:
            return None, f"not_found: {err}"
        return None, f"network_error: {err}"
    if not data.get("plddt"):
        return None, "no_plddt: structure returned but plddt array empty"

    return data, "ok"


def run_one(client, name, accession, full_sequence, target_start, target_end, note):
    struct_data, struct_status = fetch_structural_data(client, accession)

    used_sequence = full_sequence
    plddt, sasa, consensus_z, agreement_sd = None, None, None, None
    structural_data_used = False
    sequence_mismatch_warning = None

    # Remapped into struct_seq's own numbering below if the two sequences
    # differ by a clean substring offset; otherwise stay as-is.
    remapped_target_start, remapped_target_end = target_start, target_end

    if struct_status == "ok":
        struct_seq = struct_data.get("sequence", "")
        # AlphaFold models the canonical UniProt sequence, so struct_seq
        # should equal our own hardcoded full_sequence exactly. If it
        # doesn't (isoform drift, an AlphaFold fragment model, etc.), the
        # plddt/sasa arrays are aligned to struct_seq, not to our copy --
        # so struct_seq (and its own numbering) takes priority, and we
        # just record the mismatch rather than silently misaligning arrays.
        if struct_seq and struct_seq != full_sequence:
            sequence_mismatch_warning = (
                f"AlphaFold structure sequence ({len(struct_seq)} aa) differs from "
                f"the UniProt sequence used to define the target window "
                f"({len(full_sequence)} aa) -- using the structure's own sequence/"
                f"numbering for this run, since plddt/sasa are aligned to it.")
            # Found via nls_debug_p03269.py -- Adenovirus 2 pTP
            # (P03269) scores its real target NLS at raw_nls_probability
            # 0.982 (candidate 367-372 on the 653-aa AlphaFold structure
            # sequence) but was reported as a miss, because target_start/
            # target_end (380-389) are defined against the 671-aa UniProt
            # full_sequence while candidate coordinates come back in
            # struct_seq's own numbering -- an 18-residue N-terminal
            # truncation between the two (AlphaFold modeled a shorter
            # fragment/mature-chain form), silently comparing two different
            # coordinate systems as if they were the same one. Confirmed
            # this is the ONLY candidate in the 50-protein set with a
            # sequence_mismatch_warning at all, so a full alignment library
            # is overkill: if struct_seq is a clean contiguous substring of
            # full_sequence (the common case for a modeled fragment/mature
            # chain of a larger UniProt precursor), remap the target window
            # by that substring's offset. If it's not a clean substring
            # (real indels/mutations, not just a truncation), leave the
            # target window unremapped and keep the pre-existing behaviour
            # -- a wrong-but-honest "no overlap" is safer than a wrong
            # guessed remap in that case.
            offset = full_sequence.find(struct_seq) if struct_seq else -1
            if offset >= 0:
                remapped_target_start = target_start - offset
                remapped_target_end = target_end - offset
                sequence_mismatch_warning += (
                    f" struct_seq is a clean substring of full_sequence at offset "
                    f"{offset} -- target window remapped to "
                    f"{remapped_target_start}-{remapped_target_end} in struct_seq's "
                    f"own numbering for the overlap check below.")
            else:
                sequence_mismatch_warning += (
                    " struct_seq is NOT a clean contiguous substring of full_sequence "
                    "(more than a simple truncation) -- target window left "
                    "unremapped; overlap results for this candidate may be unreliable.")
        if struct_seq:
            used_sequence = struct_seq
        plddt = struct_data.get("plddt")
        sasa = struct_data.get("sasa")
        consensus_z = struct_data.get("consensus_z")
        agreement_sd = struct_data.get("agreement_sd")
        structural_data_used = bool(struct_data.get("sasa_computed"))

    payload = {"sequence": used_sequence, "model_id": accession}
    if plddt:
        payload["plddt"] = plddt
    if sasa:
        payload["sasa"] = sasa
    if consensus_z:
        payload["consensus_z"] = consensus_z
    if agreement_sd:
        payload["agreement_sd"] = agreement_sd

    resp = client.post("/api/nls_scan", json=payload)
    try:
        data = resp.get_json()
    except Exception as e:
        return {"name": name, "accession": accession, "error": f"bad JSON: {e}"}

    if data is None or "error" in data:
        return {"name": name, "accession": accession,
                "error": (data or {}).get("error", f"HTTP {resp.status_code}, no body")}

    regions = data.get("nls_binding_regions", [])
    best = None
    for r in regions:
        if overlaps(r["start"], r["end"], remapped_target_start, remapped_target_end):
            if best is None or r["nls_probability"] > best["nls_probability"]:
                best = r

    result = {
        "name": name, "accession": accession,
        "target_window": f"{target_start}-{target_end}",
        "note": note,
        "structural_fetch_status": struct_status,
        "structural_data_used": structural_data_used,
        "n_candidate_regions_in_protein": len(regions),
    }
    if (remapped_target_start, remapped_target_end) != (target_start, target_end):
        result["remapped_target_window"] = f"{remapped_target_start}-{remapped_target_end}"
    if sequence_mismatch_warning:
        result["sequence_mismatch_warning"] = sequence_mismatch_warning
    if best is None:
        result["matched"] = False
        result["detail"] = ("No surviving nls_scan candidate (post-consensus-filter, "
                             "post-classifier, post-accessibility-gate, nls_probability > 0.5) "
                             "overlapped the target window. Either scan_sequence() never "
                             "proposed this span as a candidate, or it did and scored <=0.5. "
                             "Both are real, honest results, not errors.")
    else:
        result["matched"] = True
        result["matched_window"] = f"{best['start']}-{best['end']}"
        result["nls_probability"] = best["nls_probability"]
        result["raw_nls_probability"] = best["raw_nls_probability"]
        result["accessibility_rsa"] = best["accessibility_rsa"]
        result["exposure_factor"] = best["exposure_factor"]
        result["predicted_class"] = best["predicted_class"]
        result["is_bipartite"] = best["is_bipartite"]
        result["pssm_score"] = best["pssm_score"]
        # Previously dropped on the floor -- app.py's nls_scan()
        # route sets these on every region dict (uniprot_nonnuclear_anchored
        # and dna_binding_domain_factor are set even when the veto DIDN'T
        # fire, e.g. False / None), but this script only ever copied the
        # 7 fields above, so there was no way to tell from a holdout run
        # alone whether a veto was checked-and-declined-to-fire vs never
        # checked at all. Caught debugging why CXCL12 (P48061, a real
        # 'Secreted' heparin-binding chemokine) stayed a 0.532 false
        # positive with raw_nls_probability identical to nls_probability --
        # meaning neither veto discounted it -- but with no way to see WHY
        # the anchor veto (which explicitly lists 'secreted' as a trigger
        # keyword) didn't engage. Saving these now so the next run answers
        # that directly instead of needing another live round-trip.
        result["uniprot_nonnuclear_anchored"] = best.get("uniprot_nonnuclear_anchored")
        result["uniprot_location_raw_debug"] = best.get("uniprot_location_raw_debug")
        result["anchor_factor"] = best.get("anchor_factor")
        result["uniprot_location"] = best.get("uniprot_location")
        result["anchor_caveat"] = best.get("anchor_caveat")
        result["dna_binding_domain_factor"] = best.get("dna_binding_domain_factor")
        result["dna_binding_caveat"] = best.get("dna_binding_caveat")
    return result


def format_md_row(r):
    struct = "real" if r.get("structural_data_used") else "neutral(fallback)"
    if "error" in r:
        return f"| {r['name']} | {r['accession']} | {struct} | ERROR: {r['error']} | | | | | |"
    if not r.get("matched"):
        return (f"| {r['name']} | {r['accession']} | {struct} | "
                f"no surviving candidate overlapped target window | | | | | |")
    return (f"| {r['name']} | {r['accession']} | {struct} | {r['nls_probability']:.3f} | "
            f"{r['raw_nls_probability']:.3f} | {r['predicted_class']} | "
            f"{'yes' if r['is_bipartite'] else 'no'} | {r['pssm_score']:.3f} | "
            f"{r['accessibility_rsa']} |")


def main():
    if not CANDIDATES_PATH.exists():
        print(f"ERROR: {CANDIDATES_PATH} not found. Run "
              f"nls_holdout_data/build_candidates.py first (or restore the file).")
        sys.exit(1)

    candidates = json.loads(CANDIDATES_PATH.read_text())
    positives = candidates["positives"]
    negatives = candidates["negatives"]

    print("Loading app.py (this triggers full NES + NLS + SUMO predictor "
          "initialization -- may take a moment)...")
    sys.path.insert(0, str(THIS_DIR))
    from app import app as flask_app
    client = flask_app.test_client()

    # Spot-check accession overlap against the NLS training/eval pool, same
    # transparency principle as the NES script's source_pool/weight_fit_pool
    # checks -- confirmed by direct set-intersection, not assumed.
    import csv
    pool = set()
    for fname in ["nls_dataset.csv", "nls_negatives.csv"]:
        path = THIS_DIR / "nls_data_pipeline" / fname
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                pool.update(row["accession"] for row in csv.DictReader(f))

    test_accessions = {a for _, a, *_ in positives} | {a for _, a, *_ in negatives}
    overlap = test_accessions & pool
    print(f"Accession overlap with nls_dataset.csv/nls_negatives.csv "
          f"(320-accession training/eval pool) -- should be EMPTY: {overlap or 'none'}\n")

    pos_results, neg_results = [], []

    print("=" * 100)
    print("POSITIVES -- real, UniProt/literature-verified NLS motifs (not in this project's training data)")
    print("=" * 100)
    for name, accession, full_sequence, start, end, note in positives:
        r = run_one(client, name, accession, full_sequence, start, end, note)
        pos_results.append(r)
        print(json.dumps({k: v for k, v in r.items()}, indent=2))
        print("-" * 100)

    print("\n" + "=" * 100)
    print("NEGATIVES -- real, hard basic-residue decoys (not real NLS)")
    print("=" * 100)
    for name, accession, full_sequence, start, end, note in negatives:
        r = run_one(client, name, accession, full_sequence, start, end, note)
        neg_results.append(r)
        print(json.dumps({k: v for k, v in r.items()}, indent=2))
        print("-" * 100)

    # --------------------------------------------------------------
    # Summary stats
    # --------------------------------------------------------------
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
    json_path = THIS_DIR / "nls_holdout_test_results.json"
    json_path.write_text(json.dumps(out_json, indent=2))

    md_lines = [
        "# NLS holdout pipeline test results",
        "",
        "Full app.py NLS pipeline (`NLSPredictor.scan_sequence()` + real `/api/nls_scan` route: "
        "consensus regex pre-filter, trained classifier, bipartite/tripartite detection, "
        "greedy overlap removal, `_nls_exposure_factor()` accessibility gate, CIDER/RSA profiling), "
        "run via Flask's test_client -- the exact production code path, not a reimplementation "
        "and not a bare call to the raw ML classifier. For each candidate the real "
        "`/api/structure/<model_id>` route was also called first (real AlphaFold DB + UniProt "
        "network requests) to try to get genuine per-residue pLDDT/SASA for the accessibility "
        "gate, exactly as the live app does on structure load.",
        "",
        f"**Structural data coverage:** {n_real_structure}/{len(all_results)} candidates got real "
        f"structural data; the rest fell back to `/api/nls_scan`'s documented neutral default "
        f"(RSA=0.4, exposure_factor=1.0x -- a no-op, not a bug). Fetch status breakdown: "
        f"{fetch_status_counts}. `not_found` means AlphaFold DB has no entry for that accession "
        f"(expected for most viral proteins and cleaved mature peptides); `network_error` means "
        f"the request itself failed (e.g. no network egress to alphafold.ebi.ac.uk from wherever "
        f"this ran).",
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

    md_path = THIS_DIR / "nls_holdout_test_results.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"\nSensitivity: {sensitivity:.1%}  |  Specificity: {specificity:.1%}")


if __name__ == "__main__":
    main()
