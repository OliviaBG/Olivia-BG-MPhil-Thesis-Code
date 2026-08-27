#!/usr/bin/env python3
"""
test_flanking_multiplier_ablation.py
============================================================
A/B test, NOT a pipeline change: does dropping the post-hoc
`prob = min(1.0, prob * flanking['combined_likelihood'])` line in
ImprovedNESPredictor.predict() (nes_ml_predictor_improved.py) help or hurt
performance on the same genuinely-held-out set run_holdout_pipeline_test.py
uses?

Rationale for testing this at all: hpr_likelihood/nc_likelihood (the two
factors combined_likelihood is built from) are ALREADY trained-in input
features of the classifier (_extract_features / _feature_names, "flank_hpr_
likelihood"/"flank_nc_likelihood"). The post-hoc multiply re-applies that
same NESmapper evidence a second time on top of a model that already learned
its own weight for it from real training data -- possible double-counting,
not just an arithmetic quirk in how the multiplier is applied. This script
tests the effect empirically instead of assuming either fix direction.

HOW: does NOT edit nes_ml_predictor_improved.py or app.py on disk. Imports
app.py exactly as run_holdout_pipeline_test.py does (so pocket detection /
routing / overlap-filtering logic is identical, real production code), then
monkeypatches ImprovedNESPredictor.predict at the class level for the
"ablated" pass only -- an exact copy of the real method with that one line
removed, everything else byte-identical. The original method is restored
after, so nothing about the live app/class is left modified once this script
exits.

Runs the EXACT SAME held-out positives/negatives as run_holdout_pipeline_
test.py (copied verbatim, not resampled) under both conditions, then reports
confusion matrix (threshold=0.45, matching this project's documented
decision threshold), AUC/Mann-Whitney U, and per-example score deltas so you
can see exactly which candidates moved and by how much.

REQUIREMENTS: same as run_holdout_pipeline_test.py -- must run somewhere
app.py's full stack works (real internet for AlphaFold fetches, real fpocket
installed). It will not produce real results without both.

Usage: python3 test_flanking_multiplier_ablation.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import numpy as np
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent

# Per-example wall-clock ceiling for this SCRIPT only -- does not touch
# pocket_detector.py's fpocket_timeout (which only bounds the fpocket
# binary itself). _filter_for_crm1_compatibility (pocket_detector.py),
# which scores every detected pocket against the CRM1 reference groove
# residue-by-residue, runs AFTER fpocket returns and has no timeout of its
# own -- on an unusually large/elongated structure (e.g. Myosin-9, 1960
# residues) that can find many candidate pockets, this step alone can run
# long past fpocket_timeout. Rather than edit production pocket_detector.py
# to add a timeout there, this wraps each example in a thread with its own
# deadline so one pathological structure can't block the whole comparison;
# a timed-out example is logged and excluded from stats exactly like a
# fetch/API error. NOTE: Python can't forcibly kill a running thread, so a
# timed-out call keeps consuming CPU in the background for the rest of this
# script's run -- a known, accepted tradeoff for not touching app.py/
# pocket_detector.py.
PER_EXAMPLE_TIMEOUT_S = 600  # 10 min

# ----------------------------------------------------------------------
# Candidate set -- copied verbatim from run_holdout_pipeline_test.py so
# this is directly comparable to that script's published results, not a
# different/easier test set.
# ----------------------------------------------------------------------
POSITIVES = [
    ("SARS-CoV-2 N (Betacoronavirus)", "P0DTC9", 218, 236, "+3"),
    # REMOVED (matches run_holdout_pipeline_test.py): MERS-CoV N
    # (K9N4V7) and HCoV-OC43 N (P33469) both 404 from the AlphaFold API --
    # no predicted structure exists for either, not a transient issue.
    ("HCoV-HKU1 N (Betacoronavirus)",  "Q5MQC6", 231, 249, "+1"),
    ("HCoV-NL63 N (Alphacoronavirus)", "Q6Q1R8", 181, 199, "+8"),
    ("HCoV-229E N (Alphacoronavirus)", "P15130", 178, 196, "+4"),
    ("SARS-CoV N (not independently assayed)", "P59595", 219, 237, "n/a"),
    ("SARS-CoV ORF9b (CRM1/LMB-validated, Sharma et al. 2011)", "P59636", 47, 55, "n/a"),
    ("HCV core protein (CRM1/LMB-validated, Cerutti et al. 2011)", "P27958", 109, 133, "n/a"),
    ("Chicken Anemia Virus VP1 (CRM1/LMB-validated, 2019)", "Q99153", 375, 388, "n/a"),
    ("Neurogenin-3, human (CRM1/LMB-validated, Simon-Areces et al. 2013)", "Q9Y4Z2", 131, 142, "n/a"),
]

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

THRESHOLD = 0.45  # this project's documented "your model" decision threshold


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def run_one(client, name, accession, start, end):
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

    result = {"name": name, "accession": accession, "target_window": f"{start}-{end}"}
    if best is None:
        result["matched"] = False
        result["combined_score"] = 0.0  # unmatched -> treated as below threshold, same
                                          # convention generate_holdout_comparison_figures.py uses
        result["ml_probability"] = None
    else:
        result["matched"] = True
        result["combined_score"] = best["combined_score"]
        result["ml_probability"] = best["components"].get("ml_probability")
        result["flanking_analysis"] = best["components"].get("flanking_analysis")
    return result


def _log_result(tag, name, r):
    if "error" in r:
        print(f"  [{tag}] {name[:50]:50s} ERROR (excluded from stats): {r['error']}")
    else:
        print(f"  [{tag}] {name[:50]:50s} combined_score={r['combined_score']:.3f}"
              f"  ml_prob={r.get('ml_probability')}")


def run_one_with_timeout(client, name, accession, start, end):
    # A FRESH, disposable single-thread executor per call, not one shared
    # executor reused across the whole run. With a shared max_workers=1
    # executor, an abandoned/hung call keeps occupying its one worker thread
    # forever, so every SUBSEQUENT submission queues behind it and would
    # immediately report "timed out" too, even fast ones, without ever
    # actually running -- verified this empirically before shipping it
    # (a shared-executor version made every example after the first hang
    # falsely time out). A per-call executor means an abandoned call's
    # thread leaks in the background on its own, but never blocks the next
    # example from getting an immediate, fresh thread.
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(run_one, client, name, accession, start, end)
    try:
        result = future.result(timeout=PER_EXAMPLE_TIMEOUT_S)
        executor.shutdown(wait=False)
        return result
    except FutureTimeoutError:
        executor.shutdown(wait=False)
        return {"name": name, "accession": accession,
                "error": f"client-side timeout after {PER_EXAMPLE_TIMEOUT_S}s "
                         f"(likely unbounded CRM1-pocket-compatibility scoring on a large "
                         f"structure -- see PER_EXAMPLE_TIMEOUT_S comment at top of file)"}


def run_pass(client, label):
    print(f"\n{'=' * 100}\nRunning pass: {label}\n{'=' * 100}")
    pos, neg = [], []
    for name, accession, start, end, strength in POSITIVES:
        r = run_one_with_timeout(client, name, accession, start, end)
        r["experimental_strength"] = strength
        pos.append(r)
        _log_result("pos", name, r)
    for name, accession, start, end in NEGATIVES:
        r = run_one_with_timeout(client, name, accession, start, end)
        neg.append(r)
        _log_result("neg", name, r)
    return pos, neg


def compute_stats(pos, neg, label):
    # Fetch/API failures (e.g. no AlphaFold structure published for an
    # accession) are a data-availability problem, not a model scoring
    # outcome -- excluded from stats rather than counted as combined_score=0,
    # which would silently treat "couldn't check" as "confirmed negative".
    # Reported explicitly here so small-N exclusions are never invisible.
    pos_ok = [r for r in pos if "error" not in r]
    neg_ok = [r for r in neg if "error" not in r]
    pos_errors = [r for r in pos if "error" in r]
    neg_errors = [r for r in neg if "error" in r]
    if pos_errors or neg_errors:
        print(f"  [{label}] excluded from stats due to fetch/API errors: "
              f"{len(pos_errors)} positive(s), {len(neg_errors)} negative(s) "
              f"-- {[r['accession'] for r in pos_errors + neg_errors]}")

    pos_scores = np.array([r["combined_score"] for r in pos_ok])
    neg_scores = np.array([r["combined_score"] for r in neg_ok])

    tp = int((pos_scores > THRESHOLD).sum())
    fn = int((pos_scores <= THRESHOLD).sum())
    tn = int((neg_scores <= THRESHOLD).sum())
    fp = int((neg_scores > THRESHOLD).sum())

    # Mann-Whitney U / rank-biserial AUC: probability a random positive
    # outscores a random negative -- same statistic generate_holdout_
    # comparison_figures.py's docstring references for this project.
    try:
        u_stat, p_mw = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        auc = u_stat / (len(pos_scores) * len(neg_scores))
    except ValueError:
        u_stat, p_mw, auc = float("nan"), float("nan"), float("nan")

    try:
        odds_ratio, p_fisher = stats.fisher_exact([[tp, fp], [fn, tn]])
    except ValueError:
        odds_ratio, p_fisher = float("nan"), float("nan")

    return {
        "label": label, "n_pos_used": len(pos_ok), "n_neg_used": len(neg_ok),
        "n_pos_excluded_error": len(pos_errors), "n_neg_excluded_error": len(neg_errors),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "auc_vs_negatives": round(float(auc), 4),
        "mannwhitney_p": round(float(p_mw), 4),
        "fisher_odds_ratio": (round(float(odds_ratio), 3) if np.isfinite(odds_ratio) else "inf"),
        "fisher_p": round(float(p_fisher), 4),
        "pos_mean_score": (round(float(pos_scores.mean()), 3) if len(pos_scores) else float("nan")),
        "neg_mean_score": (round(float(neg_scores.mean()), 3) if len(neg_scores) else float("nan")),
    }


def main():
    print("Loading app.py (full ML/CRM1 initialization -- may take a moment)...")
    sys.path.insert(0, str(THIS_DIR))
    from app import app as flask_app, pocket_detector
    if pocket_detector is not None:
        pocket_detector.fpocket_timeout = 300
        # Now redundant with the fix in pocket_detector.py (the previously-
        # unbounded CRM1-compatibility scoring loop now has its own cap),
        # but harmless to keep as an explicit override for this batch run --
        # and PER_EXAMPLE_TIMEOUT_S above is still a real safety net on top
        # of both.
        pocket_detector.pocket_filter_timeout = 300
    client = flask_app.test_client()

    from nes_ml_predictor_improved import ImprovedNESPredictor
    original_predict = ImprovedNESPredictor.predict

    # -------------------------------------------------------------
    # PASS 1: baseline, unmodified class method (multiplier ON)
    # -------------------------------------------------------------
    pos_baseline, neg_baseline = run_pass(client, "BASELINE (flanking multiplier ON, current pipeline)")

    # -------------------------------------------------------------
    # PASS 2: monkeypatched class method, multiplier line removed.
    # Exact copy of ImprovedNESPredictor.predict from nes_ml_predictor_
    # improved.py (as of with ONLY the
    # `prob = min(1.0, max(0.0, prob * flanking['combined_likelihood']))`
    # line deleted -- flanking_analysis is still computed and still
    # attached to `details` for comparison, it's just no longer applied
    # to prob. Nothing else changed; if the real method's body drifts
    # from this copy in the future, diff against nes_ml_predictor_
    # improved.py before trusting this pass's numbers.
    # -------------------------------------------------------------
    def predict_no_flanking_multiplier(self, sequence, full_sequence=None, nes_start=0,
                                        plddt=None, sasa=None, max_helix_run=None):
        if self.model is None:
            return 0.5, 'unknown', {}

        features = self._extract_features(
            sequence, full_sequence, nes_start, plddt, sasa,
            max_helix_run=max_helix_run,
        )
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.model.predict_proba(features_scaled)[0][1]

        details = {
            'pssm_score': self._calculate_pssm_score(sequence),
            'nes_classes': self._classify_nes_pattern(sequence),
            'spacer_hydrophobicity': self._calculate_spacer_hydrophobicity(sequence),
            'ncpr_local': self._calculate_ncpr(sequence),
            'cider_linear_features': self._calculate_cider_linear_features(sequence, full_sequence, nes_start),
            'model_name': self.model_name,
        }

        if full_sequence and len(full_sequence) > len(sequence):
            nes_end = nes_start + len(sequence)
            flanking = self._analyze_flanking_regions(full_sequence, nes_start, nes_end)
            details['flanking_analysis'] = flanking
            # <-- multiplier line intentionally removed for this test

        if prob > 0.85 or prob < 0.15:
            confidence = 'very_high'
        elif prob > 0.7 or prob < 0.3:
            confidence = 'high'
        elif prob > 0.55 or prob < 0.45:
            confidence = 'medium'
        else:
            confidence = 'low'

        return prob, confidence, details

    ImprovedNESPredictor.predict = predict_no_flanking_multiplier
    try:
        pos_ablated, neg_ablated = run_pass(client, "ABLATED (flanking multiplier OFF, test only)")
    finally:
        # Always restore, even if the pass errors -- leaves the live
        # class exactly as it was before this script ran.
        ImprovedNESPredictor.predict = original_predict

    # -------------------------------------------------------------
    # Compare
    # -------------------------------------------------------------
    stats_baseline = compute_stats(pos_baseline, neg_baseline, "baseline (multiplier ON)")
    stats_ablated = compute_stats(pos_ablated, neg_ablated, "ablated (multiplier OFF)")

    print(f"\n{'=' * 100}\nCOMPARISON\n{'=' * 100}")
    for key in ["n_pos_used", "n_neg_used", "n_pos_excluded_error", "n_neg_excluded_error",
                "tp", "fp", "fn", "tn", "auc_vs_negatives", "mannwhitney_p",
                "fisher_odds_ratio", "fisher_p", "pos_mean_score", "neg_mean_score"]:
        print(f"  {key:22s} baseline={stats_baseline[key]!s:>10}   ablated={stats_ablated[key]!s:>10}")

    print("\nPer-example combined_score, baseline -> ablated (skipping any example that errored "
          "in either pass -- not comparable):")
    for b, a in zip(pos_baseline, pos_ablated):
        if "error" in b or "error" in a:
            print(f"  [pos] {b['name'][:45]:45s} SKIPPED (error in baseline and/or ablated pass)")
            continue
        print(f"  [pos] {b['name'][:45]:45s} {b['combined_score']:.3f} -> {a['combined_score']:.3f}"
              f"  (delta {a['combined_score'] - b['combined_score']:+.3f})")
    for b, a in zip(neg_baseline, neg_ablated):
        if "error" in b or "error" in a:
            print(f"  [neg] {b['name'][:45]:45s} SKIPPED (error in baseline and/or ablated pass)")
            continue
        print(f"  [neg] {b['name'][:45]:45s} {b['combined_score']:.3f} -> {a['combined_score']:.3f}"
              f"  (delta {a['combined_score'] - b['combined_score']:+.3f})")

    out = {
        "threshold": THRESHOLD,
        "stats": {"baseline": stats_baseline, "ablated": stats_ablated},
        "positives": {"baseline": pos_baseline, "ablated": pos_ablated},
        "negatives": {"baseline": neg_baseline, "ablated": neg_ablated},
    }
    out_path = THIS_DIR / "flanking_multiplier_ablation_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
    # Force-exit rather than a normal return: concurrent.futures registers an
    # atexit hook that joins ALL worker threads (across every disposable
    # executor run_one_with_timeout created) before the interpreter exits,
    # which would silently re-introduce the exact hang this script exists to
    # avoid if any example timed out above. os._exit() skips atexit/cleanup
    # entirely -- safe here since results are already written to disk by
    # this point.
    import os
    sys.stdout.flush()
    os._exit(0)
