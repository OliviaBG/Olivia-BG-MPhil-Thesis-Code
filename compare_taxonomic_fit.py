#!/usr/bin/env python3
"""
compare_taxonomic_fit.py
============================================================
Standalone diagnostic: how well does each predictor's OUT-OF-FOLD
prediction (never the shipped model's own in-sample score -- see below)
separate real positives from real negatives, broken out by the training
example's taxonomic group (Human / Rodent / Yeast-Fungi / Plant / Viral /
Other vertebrate / Invertebrate-Other)?

WHY OUT-OF-FOLD, NOT THE SHIPPED MODEL'S OWN PREDICTIONS:
The shipped model (nes_ml_predictor_improved.py / nls_ml_predictor.py's
_train_model()) is, by design, refit on 100% of the data as its very last
step (see that method's "Now that model selection + honest evaluation are
done, refit on ALL data" comment). Asking that final model to score its own
training examples would be circular -- of course it scores them well, it's
seen the answers. Instead this script redoes 5-fold cross-validation from
scratch and uses sklearn's cross_val_predict(method='predict_proba') so
EVERY example gets a probability from a fold that never trained on it --
exactly the same honest standard already used everywhere else in this
project's diagnostics (diagnose_feature_importance.py's held-out split,
compare_split_methodology.py's kfold_cv arm).

WHAT THIS DOES NOT CHANGE:
This script does not touch, import from a training-writing path, or alter
_train_model() or any shipped model artifact in any way. The production
training methodology stays exactly k-fold-for-selection +
100%-data-refit-for-shipping, as decided in the split-methodology
comparison. This is purely a read-only diagnostic that reuses
build_training_dataset() (the same read-only data assembly every other
diagnostic script in this project already uses).

WHAT IT DOES, per target (nes / nls):
  1. Loads {target}_data_pipeline/organism_data.json (from
     fetch_organism_data.py -- run that first).
  2. Calls the real predictor's build_training_dataset() to get the exact
     same X, y used for real training, plus each example's seq (positives/
     negatives lists preserve the exact row order X/y were built in -- see
     build_training_dataset()'s own docstring).
  3. Independently rebuilds a seq -> accession map from
     structural_data_v2.json / structural_data.json (both have 'seq' and
     'accession' per record) and joins each training row to its taxonomic
     group via organism_data.json. Examples with no structural-data match
     (e.g. curated-seed positives, synthetic decoys) are labeled "No
     accession match" rather than silently dropped -- transparency over a
     tidier chart.
  4. Picks the same classifier type the real training run would pick (mean
     CV F1 across the same 7-8 candidates), then gets honest per-example
     probabilities via cross_val_predict(..., method='predict_proba',
     cv=StratifiedKFold(n_splits=5, ...)).
  5. Writes {target}_taxonomic_fit_data.csv (one row per training example:
     seq, accession, organism, taxonomic_group, true_label,
     oof_predicted_proba) and {target}_taxonomic_composition.json (what
     fraction of TRAINING EXAMPLES, not unique accessions, came from each
     group -- for the pie chart).

Usage:
    python3 compare_taxonomic_fit.py --target nes
    python3 compare_taxonomic_fit.py --target nls
    python3 compare_taxonomic_fit.py --target both
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from resolve_taxonomic_provenance import resolve as resolve_provenance

THIS_DIR = Path(__file__).resolve().parent
UNKNOWN_GROUP = "No accession match"  # still used by the now-unused _seq_to_accession_and_group() below


def _nes_candidates():
    """Exact same 7(-8) classifier definitions as
    nes_ml_predictor_improved.py's _train_model() (redefined locally, not
    imported -- same zero-coupling reasoning as compare_split_methodology.py)."""
    c = {
        'svm_linear': lambda: SVC(kernel='linear', C=0.01, probability=True, random_state=42, class_weight='balanced'),
        'svm_rbf': lambda: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced'),
        'random_forest': lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
        'extra_trees': lambda: ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
        'gradient_boosting': lambda: GradientBoostingClassifier(random_state=42),
        'hist_gradient_boosting': lambda: HistGradientBoostingClassifier(random_state=42),
        'mlp': lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42, early_stopping=True),
    }
    if XGBOOST_AVAILABLE:
        c['xgboost'] = lambda: XGBClassifier(random_state=42, eval_metric='logloss')
    return c


def _nls_candidates():
    """Exact same candidates as nls_ml_predictor.py's _train_model() (only
    difference from NES: svm_linear's C=0.1 not 0.01, matching that file)."""
    c = _nes_candidates()
    c['svm_linear'] = lambda: SVC(kernel='linear', C=0.1, probability=True, random_state=42, class_weight='balanced')
    return c


TARGETS = {
    "nes": {
        "structural_data": Path("nes_data_pipeline/structural_data_v2.json"),
        "organism_data": Path("nes_data_pipeline/organism_data.json"),
        "candidates": _nes_candidates,
        "out_csv": Path("nes_data_pipeline/nes_taxonomic_fit_data.csv"),
        "out_composition": Path("nes_data_pipeline/nes_taxonomic_composition.json"),
    },
    "nls": {
        "structural_data": Path("nls_data_pipeline/structural_data.json"),
        "organism_data": Path("nls_data_pipeline/organism_data.json"),
        "candidates": _nls_candidates,
        "out_csv": Path("nls_data_pipeline/nls_taxonomic_fit_data.csv"),
        "out_composition": Path("nls_data_pipeline/nls_taxonomic_composition.json"),
    },
}


def _load_predictor(target):
    if target == "nes":
        from nes_ml_predictor_improved import ImprovedNESPredictor
        return ImprovedNESPredictor()
    from nls_ml_predictor import NLSPredictor
    return NLSPredictor()


def _seq_to_accession_and_group(structural_data_path, organism_data_path):
    records = json.loads(structural_data_path.read_text(encoding="utf-8"))
    seq_to_acc = {r["seq"].upper(): r["accession"] for r in records if r.get("seq") and r.get("accession")}

    organism_data = {}
    if organism_data_path.exists():
        organism_data = json.loads(organism_data_path.read_text(encoding="utf-8"))
    else:
        print(f"  WARNING: {organism_data_path} not found -- every example will be "
              f"'{UNKNOWN_GROUP}'. Run fetch_organism_data.py first.")
    return seq_to_acc, organism_data


def _pick_best_classifier(X, y, candidates, cv_folds):
    best_name, best_score = None, -1.0
    for name, factory in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", factory())])
        try:
            scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring="f1")
        except Exception as e:
            print(f"    {name}: CV failed ({e})")
            continue
        print(f"    {name}: CV F1 = {scores.mean():.3f} +/- {scores.std():.3f}")
        if scores.mean() > best_score:
            best_score, best_name = scores.mean(), name
    return best_name, best_score


def run_target(target, cfg):
    print(f"\n{'=' * 70}\n{target.upper()}: taxonomic fit (out-of-fold)\n{'=' * 70}")

    if not cfg["structural_data"].exists():
        print(f"  SKIP: {cfg['structural_data']} not found.")
        return

    predictor = _load_predictor(target)
    dataset = predictor.build_training_dataset()
    X, y = np.asarray(dataset["X"], dtype=float), np.asarray(dataset["y"], dtype=int)
    seqs = [p["seq"].upper() for p in dataset["positives"]] + [n["seq"].upper() for n in dataset["negatives"]]
    if len(seqs) != len(y):
        raise RuntimeError(f"{target}: sequence count ({len(seqs)}) != label count ({len(y)}) -- "
                            "build_training_dataset() row-order assumption broke.")

    # Was seq -> structural_data.json -> UniProt-accession join
    # (_seq_to_accession_and_group, still defined below for reference/
    # backward compatibility but no longer called here). That join relied
    # on a side-file most training examples never made it into, so ~59% of
    # NES rows and ~12% of NLS rows fell into a catch-all "no accession"
    # bucket that actually mixed real-but-untracked sequences (curated
    # literature motifs, protein-matched negative windows generated at
    # train time, structural hard negatives) in with the handful of
    # genuinely fabricated ones. resolve_taxonomic_provenance.py reads
    # organism/accession straight out of the project's own raw source files
    # instead (nes_dataset.csv, nes_negatives.csv, nls_dataset.csv,
    # nls_negatives.csv all already carry this -- see that module's
    # docstring), which is both more complete and, for the fabricated cases,
    # more honest (they get their own explicit "Synthetic (fabricated)"
    # group instead of being invisible inside "no accession match").
    resolved = resolve_provenance(target, dataset, THIS_DIR)
    organisms = [r[0] or "" for r in resolved]
    groups = [r[1] for r in resolved]
    sources = [r[2] for r in resolved]

    print(f"  {len(y)} examples ({int(y.sum())} pos / {int((1 - y).sum())} neg)")
    group_counts = Counter(groups)
    print("  Taxonomic-group breakdown (all training examples, incl. negatives):")
    for g, n in group_counts.most_common():
        print(f"    {g}: {n} ({n / len(y) * 100:.1f}%)")

    cv_folds = max(2, min(5, int(min(np.bincount(y)))))
    candidates = cfg["candidates"]()
    print(f"\n  Picking classifier via {cv_folds}-fold CV on the FULL dataset "
          f"(same selection logic as production _train_model(), just run on "
          f"all data since there's no held-out split needed for this "
          f"diagnostic -- cross_val_predict below re-does proper CV anyway):")
    best_name, best_score = _pick_best_classifier(X, y, candidates, cv_folds)
    print(f"  Chosen for this diagnostic: {best_name} (CV F1={best_score:.3f})")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", candidates[best_name]())])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    print(f"  Computing out-of-fold predicted probabilities ({cv_folds}-fold, "
          f"cross_val_predict) -- every example scored by a fold that never "
          f"trained on it ...")
    oof_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    import csv
    with open(cfg["out_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "organism", "taxonomic_group", "provenance_source",
                          "true_label", "oof_predicted_proba"])
        for seq, org, grp, src, label, proba in zip(seqs, organisms, groups, sources, y, oof_proba):
            writer.writerow([seq, org, grp, src, int(label), float(proba)])
    print(f"  Wrote {len(y)} rows to {cfg['out_csv']}")

    composition = {g: {"n": n, "fraction": n / len(y)} for g, n in group_counts.items()}
    cfg["out_composition"].write_text(json.dumps({
        "target": target, "n_total": len(y), "chosen_classifier": best_name,
        "cv_folds": cv_folds, "composition": composition,
    }, indent=2))
    print(f"  Wrote composition summary to {cfg['out_composition']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["nes", "nls", "both"], default="both")
    args = ap.parse_args()

    targets = TARGETS.keys() if args.target == "both" else [args.target]
    for name in targets:
        run_target(name, TARGETS[name])

    print("\nDone. Next step: python3 generate_thesis_figures.py "
          "(picks up *_taxonomic_fit_data.csv / *_taxonomic_composition.json "
          "automatically if present) to render the violin + pie chart figures.")


if __name__ == "__main__":
    main()
