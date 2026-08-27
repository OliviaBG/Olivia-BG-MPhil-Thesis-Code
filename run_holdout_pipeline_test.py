#!/usr/bin/env python3
"""
run_holdout_pipeline_test.py
============================================================
Runs a small, genuinely held-out test set through THIS project's real,
live scoring pipeline exactly as the running app would -- full ML NES
predictor + the reweighted CRM1AwarePocketDetector -- via app.py's actual
Flask route (/api/unified_crm1_nes/<model_id>), using Flask's test_client
so no separate server process is needed. Not a reimplementation: this
imports app.py directly, so it's the exact same code path production
traffic hits.

WHY THIS SET: none of these examples are in this project's training/eval
data (nes_negatives.csv, nesdb_resolved_accessions.json, NESbase-derived
positives) -- they were sourced independently, after the fact, from:

POSITIVES (real, experimentally-validated NES motifs, not sequence
predictions): Sendino, Omaetxebarria & Rodriguez 2020 (bioRxiv
2020.10.06.328138) -- coronavirus nucleocapsid NES motifs tested with a
real Rev(1.4)-GFP nuclear export assay (not an NES predictor) and each
assigned an experimental export-activity "strength" score (1+ weakest to
9+ strongest). Verified against real UniProt sequences below (exact
substring match, not approximate). Being a 2020 bioRxiv preprint focused
on a then-novel in-silico-predicted motif (Gussow et al. 2020 PNAS), these
specific NES windows are very unlikely to be present in LocNES's or
NESmapper's training sets, both published years earlier (2014-2015) --
and SARS-CoV wasn't even tested (only SARS-CoV-2, due to near-identity),
so it can't have leaked in via that route either.

NEGATIVES (hard, real leucine-zipper decoys -- trip the same NES-
consensus regex without being real NESs): drawn from
nes_negatives_leucine_zipper_expansion/nes_negatives.csv, picking the
highest-scoring (most NES-consensus-like, i.e. hardest) real human
examples, distinct from anything sampled into crm1_eval_results.json so
far (spot-check accession overlap printed below).

REQUIREMENTS: must run in an environment where app.py's full dependency
stack works (real internet for AlphaFold structure fetches, real fpocket
installed), same as evaluate_crm1_pocket_signal.py. Will NOT work in an
environment with no general network access.

Usage: python3 run_holdout_pipeline_test.py
"""
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Candidate set
# ----------------------------------------------------------------------
# (name, accession, start, end, label, note)
# start/end are 1-indexed inclusive, matching this project's convention
# elsewhere (evaluate_crm1_pocket_signal.py, negative_dataset_builder.py).
POSITIVES = [
    # accession, window, real experimental export-assay strength score (Sendino et al. 2020)
    ("SARS-CoV-2 N (Betacoronavirus)", "P0DTC9", 218, 236, "+3"),
    # REMOVED: MERS-CoV N (K9N4V7) and HCoV-OC43 N (P33469, below)
    # both 404 from the AlphaFold API ("Could not resolve a current version",
    # confirmed on two separate runs including one after the app.py-side
    # ?uniprot_id= resolution fix referenced above) -- no predicted structure
    # is available for either accession, not a transient fetch issue. This
    # project's own pipeline and SPSignal both never produced a score for
    # them (see generate_holdout_comparison_figures.py, which already
    # excludes these two entirely rather than faking placeholder values).
    # Dropped here too so this script doesn't spend time on structure
    # fetches that can never succeed; POSITIVES is now the same 8-protein
    # coronavirus/viral set generate_holdout_comparison_figures.py uses.
    ("HCoV-HKU1 N (Betacoronavirus)",  "Q5MQC6", 231, 249, "+1"),
    ("HCoV-NL63 N (Alphacoronavirus)", "Q6Q1R8", 181, 199, "+8"),
    ("HCoV-229E N (Alphacoronavirus)", "P15130", 178, 196, "+4"),
    # SARS-CoV wasn't independently tested in the assay (near-identical to
    # SARS-CoV-2's motif) but is included as a bonus, unscored data point.
    ("SARS-CoV N (not independently assayed)", "P59595", 219, 237, "n/a"),
    ("SARS-CoV ORF9b (CRM1/LMB-validated, Sharma et al. 2011)", "P59636", 47, 55, "n/a"),
    ("HCV core protein (CRM1/LMB-validated, Cerutti et al. 2011)", "P27958", 109, 133, "n/a"),
    ("Chicken Anemia Virus VP1 (CRM1/LMB-validated, 2019)", "Q99153", 375, 388, "n/a"),
    ("Neurogenin-3, human (CRM1/LMB-validated, Simon-Areces et al. 2013)", "Q9Y4Z2", 131, 142, "n/a"),
]

NEGATIVES = [
    # REPLACED. The original 5 (P13010/Ku80, Q13323/BIK,
    # Q16584/MAP3K12, Q15631/Translin, Q96IZ0/PAWR) were all human entries
    # pulled from nes_negatives/nes_negatives.csv -- the EXACT file
    # nes_ml_predictor_improved.py's HARD_NEGATIVE_CANDIDATES trains its
    # hard-negative class on (confirmed by direct set-intersection check).
    # That's not a held-out test of the ML predictor at all -- it's asking
    # the model to recognize its own training examples, so the high scores
    # it gave them don't tell us whether it generalizes to *new* leucine
    # zippers. This docstring's own header claimed these came from the
    # expansion file; they didn't -- that was wrong and is fixed now.
    #
    # New set: highest-PSSM-regex-score (hardest / most NES-consensus-like)
    # entries from nes_negatives_leucine_zipper_expansion/nes_negatives.csv,
    # filtered to exclude anything in nes_negatives/nes_negatives.csv (the
    # ML predictor's training source) AND anything in
    # crm1_eval_results.json (the CRM1 pocket weight-fitting pool) --
    # verified via set intersection, not assumed. No human entries exist in
    # the expansion file at all (it's mouse/rat/bovine/orangutan/etc.), so
    # these are mammalian orthologs of real leucine-zipper/coiled-coil
    # proteins instead -- still real, hard, NES-consensus-matching
    # sequences, just genuinely unseen by both the ML model and the CRM1
    # weight fit. Picked to avoid family overlap with each other and with
    # the old negative set (no repeated BIK/Par-4/MAP3K/Ku80/Translin
    # relatives).
    ("Jun dimerization protein 2 (mouse, bZIP leucine zipper TF)", "P97875", 114, 123),
    ("Caveolae-associated protein 1 (mouse)",                      "O54724", 63, 72),
    ("Apoptosis inhibitor 5 (mouse)",                              "O35841", 384, 393),
    ("Protein AF-10 / MLLT10 (mouse)",                             "O54826", 773, 782),
    ("Spermatogenic leucine zipper protein 1 (bovine)",            "Q32L17", 114, 124),
    ("Lamin A/C, human (coiled-coil rod domain)", "P02545", 362, 371),
    ("Tropomyosin beta chain / TPM2, human (coiled-coil)", "P07951", 4, 13),
    ("Myosin-9, human (coiled-coil rod domain)", "P35579", 1067, 1076),
]


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def run_one(client, name, accession, start, end):
    # Stopped trying to pre-resolve AlphaFold's real entryId/
    # version here -- that assumed entryId always looks like
    # "AF-{accession}-F1", which is FALSE for several of the coronavirus N
    # proteins in this exact test set (their real entryId is a bare
    # internal numeric ID with no accession embedded). app.py's endpoint
    # now accepts an explicit ?uniprot_id= query param (same precedent as
    # get_structure()) and resolves the current version/pdbUrl itself via
    # the AlphaFold API using that accession directly -- more robust than
    # duplicating that resolution here. model_id itself is now just a
    # human-readable label for logging; only uniprot_id matters.
    model_id = f"AF-{accession}-F1"
    resp = client.get(f"/api/unified_crm1_nes/{model_id}?uniprot_id={accession}")
    try:
        data = resp.get_json()
    except Exception as e:
        return {"name": name, "accession": accession, "error": f"bad JSON: {e}"}

    if data is None or "error" in data:
        return {"name": name, "accession": accession,
                "error": (data or {}).get("error", f"HTTP {resp.status_code}, no body")}

    motifs = data.get("nes_motifs", [])
    best = None
    for m in motifs:
        if overlaps(m["start"], m["end"], start, end):
            if best is None or m["combined_score"] > best["combined_score"]:
                best = m

    result = {
        "name": name, "accession": accession, "target_window": f"{start}-{end}",
        "n_surviving_candidates_in_protein": len(motifs),
    }
    if best is None:
        result["matched"] = False
        result["note"] = ("No surviving candidate (combined_score > 0.45, post-overlap-filter) "
                           "overlapped the target window -- either the app's window-generation "
                           "step never proposed this span as a candidate, or it did and scored "
                           "<=0.45 and got filtered out. Both are real, honest results, not errors.")
    else:
        result["matched"] = True
        result["matched_window"] = f"{best['start']}-{best['end']}"
        result["combined_score"] = best["combined_score"]
        comps = best["components"]
        result["ml_probability"] = comps.get("ml_probability")
        result["crm1_binding_affinity"] = comps.get("crm1_binding_affinity")
        result["pocket_compatibility"] = comps.get("pocket_compatibility")
        result["hydrophobicity"] = comps.get("hydrophobicity")
        result["disorder"] = comps.get("disorder")
        result["surface_accessibility"] = comps.get("surface_accessibility")
    return result


def format_md_row(r, extra_cols):
    if "error" in r:
        base = f"| {r['name']} | {r['accession']} | ERROR: {r['error']} |"
        return base + " |" * len(extra_cols)
    if not r.get("matched"):
        base = f"| {r['name']} | {r['accession']} | no surviving candidate overlapped target window |"
        return base + " |" * len(extra_cols)
    cells = [
        r["name"], r["accession"],
        f"{r['combined_score']:.3f}",
        f"{r['ml_probability']:.3f}" if r.get("ml_probability") is not None else "n/a",
        f"{r['crm1_binding_affinity']:.3f}" if r.get("crm1_binding_affinity") is not None else "n/a",
        f"{r['pocket_compatibility']:.3f}" if r.get("pocket_compatibility") is not None else "n/a",
    ]
    for col in extra_cols:
        cells.append(str(r.get(col, "")))
    return "| " + " | ".join(cells) + " |"


def main():
    print("Loading app.py (this triggers full ML/CRM1 initialization -- may take a moment)...")
    sys.path.insert(0, str(THIS_DIR))
    from app import app as flask_app, pocket_detector
    if pocket_detector is not None:
        pocket_detector.fpocket_timeout = 300  # 5 min, enough for large (~2000-residue) proteins
        # Separate cap on the CRM1-compatibility scoring step
        # that runs after fpocket returns (see pocket_detector.py's
        # pocket_filter_timeout docstring) -- was previously unbounded and
        # could hang on large/elongated structures (e.g. Myosin-9, 1960
        # residues) even with fpocket_timeout raised. Same generous-for-
        # offline-batch-use reasoning as fpocket_timeout above.
        pocket_detector.pocket_filter_timeout = 300
    client = flask_app.test_client()

    # Spot-check accession overlap at TWO levels, not just one:
    #  (a) the raw source negative-dataset CSVs (the whole candidate pool
    #      negative_dataset_builder.py / expand_leucine_zipper_negatives.py
    #      produced) -- being IN this pool is expected/fine, it's where these
    #      5 negatives were picked FROM.
    #  (b) crm1_eval_results.json -- the specific subset that was actually
    #      scored and fed into compute_crm1_joint_weights.py to derive the
    #      CURRENT pocket-scoring weights. Being in THIS file means the
    #      example already influenced the weights being tested, so a
    #      low/high score on it is not an independent check.
    import csv
    source_pool = set()
    for path in [THIS_DIR / "nes_negatives" / "nes_negatives.csv",
                 THIS_DIR / "nes_negatives_leucine_zipper_expansion" / "nes_negatives.csv"]:
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                source_pool.update(row["accession"] for row in csv.DictReader(f))

    weight_fit_pool = set()
    eval_results_path = THIS_DIR / "crm1_eval_results.json"
    if eval_results_path.exists():
        eval_data = json.loads(eval_results_path.read_text())
        weight_fit_pool.update(r.get("accession") for r in eval_data if r.get("accession"))

    test_accessions = {a for _, a, *_ in POSITIVES} | {a for _, a, *_ in NEGATIVES}
    overlap_source = test_accessions & source_pool
    overlap_weightfit = test_accessions & weight_fit_pool
    print(f"Accession overlap with source negative-dataset CSVs (expected for the 5 negatives, "
          f"picked from this pool): {overlap_source or 'none'}")
    print(f"Accession overlap with crm1_eval_results.json (the subset that ACTUALLY fit the "
          f"current weights -- overlap here means NOT an independent check for those examples): "
          f"{overlap_weightfit or 'none'}\n")

    pos_results, neg_results = [], []

    print("=" * 100)
    print("POSITIVES -- real, experimentally-validated viral NES motifs (not in this project's training data)")
    print("=" * 100)
    for name, accession, start, end, strength in POSITIVES:
        r = run_one(client, name, accession, start, end)
        r["experimental_strength"] = strength
        r["in_weight_fitting_pool"] = accession in weight_fit_pool
        pos_results.append(r)
        print(json.dumps(r, indent=2))
        print("-" * 100)

    print("\n" + "=" * 100)
    print("NEGATIVES -- real, hard leucine-zipper decoys (close to positive, not real NES)")
    print("=" * 100)
    for name, accession, start, end in NEGATIVES:
        r = run_one(client, name, accession, start, end)
        r["in_weight_fitting_pool"] = accession in weight_fit_pool
        neg_results.append(r)
        print(json.dumps(r, indent=2))
        print("-" * 100)

    # --------------------------------------------------------------
    # Write real file outputs, not just terminal text.
    # --------------------------------------------------------------
    out_json = {
        "accession_overlap_with_source_pool": sorted(overlap_source),
        "accession_overlap_with_weight_fitting_pool": sorted(overlap_weightfit),
        "positives": pos_results,
        "negatives": neg_results,
    }
    json_path = THIS_DIR / "holdout_test_results.json"
    json_path.write_text(json.dumps(out_json, indent=2))

    md_lines = [
        "# Holdout pipeline test results",
        "",
        "Full ML NES predictor + reweighted CRM1AwarePocketDetector, run via app.py's real "
        "/api/unified_crm1_nes/<model_id> endpoint (Flask test_client, exact production code path).",
        "",
        "## Positives (real, experimentally-validated viral NES motifs)",
        "",
        "| Name | Accession | combined_score | ml_probability | crm1_binding_affinity | "
        "pocket_compatibility | experimental_strength | in_weight_fitting_pool |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in pos_results:
        md_lines.append(format_md_row(r, ["experimental_strength", "in_weight_fitting_pool"]))

    md_lines += [
        "",
        "## Negatives (real, hard leucine-zipper decoys)",
        "",
        "| Name | Accession | combined_score | ml_probability | crm1_binding_affinity | "
        "pocket_compatibility | in_weight_fitting_pool |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in neg_results:
        md_lines.append(format_md_row(r, ["in_weight_fitting_pool"]))

    md_lines += [
        "",
        f"Accession overlap with source negative-dataset CSVs: {sorted(overlap_source) or 'none'}",
        "",
        f"Accession overlap with crm1_eval_results.json (already influenced current weights, "
        f"not an independent check for these): {sorted(overlap_weightfit) or 'none'}",
    ]
    md_path = THIS_DIR / "holdout_test_results.md"
    md_path.write_text("\n".join(md_lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
