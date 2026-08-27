#!/usr/bin/env python3
"""
compare_split_methodology.py
============================================================
Standalone Monte Carlo comparison of THREE model-selection/evaluation
methodologies for the NES and NLS predictors:

  A) "train_val_test" -- the ORIGINAL NLS design, before the switch to k-fold CV (see nls_ml_predictor_before_kfold.py for the
     untouched original): 60/20/20 split, classifier chosen by F1 on a
     single held-out validation slice, final model refit on train+val
     (80% of the data), test (20%) used only for the one reported score.

  B) "kfold_cv" -- the (now former) production design, 80/20 split, classifier chosen by mean F1 across 5-fold
     CV on the training 80%, final model refit on ALL data (100%) after
     the held-out evaluation is done.

  C) "nested_cv" -- the CURRENT production design currently (see
     nes_ml_predictor_improved.py / nls_ml_predictor.py's _train_model()):
     outer k-fold CV for the reported performance estimate, inner k-fold
     CV re-run fresh inside each outer training fold for classifier
     selection -- the outer test fold can never leak into a selection
     decision, unlike kfold_cv's single 80/20 split. A separate final
     selection+fit pass on 100% of the data (after the numbers above are
     locked in) decides what actually ships.

This script does NOT import, touch, or overwrite anything belonging to
either production predictor's training/saving path -- it only calls
ImprovedNESPredictor.build_training_dataset() / NLSPredictor.build_training_dataset()
(the same read-only data-assembly method every diagnostic script in this
project already uses) to get the same X, y, then runs its own completely
separate train/eval loop purely for this comparison. Nothing here is wired
into predict()/scan_sequence()/_train_model() or any shipped model
artifact. Classifier candidates are redefined locally (not imported from
either predictor file) so this script has zero coupling in either
direction.

Why repeat N times rather than run each methodology once: a single 80/20
or 60/20/20 split (or, for nested_cv, a single outer-fold partition) is
itself one random draw, so a single run's F1 could look better or worse
purely by chance -- especially with datasets in the hundreds of examples
(NES ~1194, NLS ~740). Repeating with N different random seeds and
looking at the SPREAD of the reported score (not just its mean) is the
actual evidence for whether a methodology is trustworthy: a method that
reports a higher mean F1 but with much higher run-to-run variance, or
that picks a different "best" classifier almost every time depending on
the luck of the split, is LESS reliable, not more -- even though any
single run of it might look good.

The test fraction is held at 20% (or, for nested_cv, ~20% per outer fold)
for all three methodologies on purpose, so the comparison isolates the
thing actually being compared (how the "best" classifier is chosen -- one
held-out val slice vs 5-fold CV vs nested 5-fold CV) and isn't confounded
by also giving one method a bigger test set than another.

Usage:
    python3 compare_split_methodology.py --target nes --n-repeats 30
    python3 compare_split_methodology.py --target nls --n-repeats 30
    python3 compare_split_methodology.py --target both --n-repeats 30

Runtime note: each repeat fits all 7-8 candidates once for train_val_test,
once (x cv_folds internally) for kfold_cv, and once PER OUTER FOLD
(x inner cv_folds internally, ~5x kfold_cv's work) for nested_cv -- 30
repeats is noticeably slower per target than before this method was
added. Lower --n-repeats if you just want a quick look; the more repeats,
the tighter the reported confidence intervals.

Outputs (written to --out, default 'split_methodology_comparison/'):
    {target}_repeats.csv    one row per (methodology, repeat): chosen
                             classifier, held-out test F1/precision/
                             recall/ROC-AUC/accuracy, the selection
                             criterion's own score, how many examples
                             fed the selection decision, and what
                             fraction of the full dataset ends up in the
                             final shipped-equivalent model
    {target}_summary.json   aggregated mean/std/95%-CI per methodology,
                             plus a classifier-selection-stability tally
                             (how often each classifier "won", and how
                             many DISTINCT classifiers won at least once
                             across the repeats)
    {target}_nested_cv_folds.csv  one row per (repeat seed, outer fold):
                             chosen classifier, held-out fold F1/precision/
                             recall/ROC-AUC/accuracy, inner-selection score,
                             train/test sizes -- the un-averaged detail
                             behind nested_cv's single averaged row in
                             {target}_repeats.csv, for a per-fold spread panel
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

THIS_DIR = Path(__file__).resolve().parent


def _candidates(target):
    """Same 7(-8) classifier definitions currently used by
    nes_ml_predictor_improved.py / nls_ml_predictor.py's _train_model() --
    redefined locally (not imported) so this script can never be affected
    by, or accidentally affect, either predictor file. Factories (not
    fitted instances) so each repeat/method gets a genuinely fresh,
    unfitted estimator.

    fix: this used to hardcode svm_linear's C=0.1 for BOTH
    targets -- correct for nls_ml_predictor.py, but nes_ml_predictor_improved.py
    actually uses C=0.01. svm_linear has never been the winning classifier
    in practice (extra_trees/svm_rbf dominate), so this likely didn't
    change any past conclusions, but it means every NES run before this
    fix was comparing against a classifier with different hyperparameters
    than the one actually deployed. Now takes target so it can never
    silently drift from either predictor's real candidates again."""
    svm_linear_c = 0.01 if target == 'nes' else 0.1
    c = {
        'svm_linear': lambda: SVC(kernel='linear', C=svm_linear_c, probability=True, random_state=42, class_weight='balanced'),
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


def _load_data(target):
    """Read-only: calls the SAME build_training_dataset() every other
    diagnostic script in this project uses, so this comparison is against
    the real, current data -- but never calls _train_model() and never
    writes to either predictor's model_dir."""
    if target == 'nes':
        from nes_ml_predictor_improved import ImprovedNESPredictor
        predictor = ImprovedNESPredictor()
    else:
        from nls_ml_predictor import NLSPredictor
        predictor = NLSPredictor()
    dataset = predictor.build_training_dataset()
    X = np.asarray(dataset['X'], dtype=float)
    y = np.asarray(dataset['y'], dtype=int)
    return X, y


def _eval(model, scaler, X_test, y_test):
    Xs = scaler.transform(X_test)
    pred = model.predict(Xs)
    proba = model.predict_proba(Xs)[:, 1]
    return {
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, proba)) if len(set(y_test)) == 2 else None,
        'n_test': int(len(y_test)), 'n_test_pos': int(y_test.sum()),
    }


def run_train_val_test(X, y, seed, target):
    """Method A -- the ORIGINAL NLS design: 60/20/20, val-F1-based
    selection, final model refit on train+val (80% of the data). Test
    (20%) is never used for fitting anything."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    candidates = _candidates(target)
    best_name, best_val_f1 = None, -1.0
    for name, factory in candidates.items():
        try:
            mdl = factory().fit(X_train_s, y_train)
        except Exception:
            continue
        val_f1 = f1_score(y_val, mdl.predict(X_val_s), zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, best_name = val_f1, name

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    scaler_final = StandardScaler().fit(X_trainval)
    model = candidates[best_name]().fit(scaler_final.transform(X_trainval), y_trainval)

    metrics = _eval(model, scaler_final, X_test, y_test)
    metrics.update({
        'methodology': 'train_val_test', 'seed': int(seed), 'chosen_classifier': best_name,
        'selection_score': float(best_val_f1), 'n_used_for_selection_decision': int(len(y_val)),
        'n_total': int(len(y)), 'frac_data_in_final_model': float(len(y_trainval) / len(y)),
    })
    return metrics


def run_kfold(X, y, seed, target):
    """Method B -- the (now former) production design: 80/20, CV-mean-F1-
    based selection. Held-out test metrics are measured on a model fit on
    the train 80% ONLY (matching exactly what production code evaluated
    before its own separate final 100%-data refit) -- this keeps the test
    fraction identical to Method A (20%) so the comparison isolates the
    SELECTION methodology rather than also changing how much test data
    each method gets to report its score against.

    production itself moved on to nested_cv (see run_nested_cv
    below) -- this arm is kept as the middle point of the 3-way comparison
    (single held-out split, but CV-based selection instead of a single
    validation slice)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    candidates = _candidates(target)
    cv_folds = max(2, min(5, int(min(np.bincount(y_train)))))
    best_name, best_cv_f1 = None, -1.0
    for name, factory in candidates.items():
        try:
            scores = cross_val_score(factory(), X_train_s, y_train, cv=cv_folds, scoring='f1')
        except Exception:
            continue
        if scores.mean() > best_cv_f1:
            best_cv_f1, best_name = scores.mean(), name

    model = candidates[best_name]().fit(X_train_s, y_train)
    metrics = _eval(model, scaler, X_test, y_test)
    metrics.update({
        'methodology': 'kfold_cv', 'seed': int(seed), 'chosen_classifier': best_name,
        'selection_score': float(best_cv_f1), 'n_used_for_selection_decision': int(len(y_train)),
        'n_total': int(len(y)),
        # this arm's shipped-equivalent model would refit on 100% of the
        # data AFTER this held-out evaluation (as production used to, pre-
        # nested-CV) -- this run doesn't repeat that refit (there'd be
        # nothing left to evaluate it against), but the fraction below
        # reflects what would end up shipped.
        'frac_data_in_final_model': 1.0,
    })
    return metrics


def run_nested_cv(X, y, seed, target, fold_sink=None):
    """Method C -- the CURRENT production design currently (see
    nes_ml_predictor_improved.py / nls_ml_predictor.py's _train_model()):
    outer k-fold CV for an unbiased performance estimate, inner k-fold CV
    (re-run fresh inside each outer training fold) for classifier
    selection, so the outer test fold never leaks into a selection
    decision.

    To keep this comparable to the other two arms' one-row-per-repeat CSV
    shape, this whole nested procedure counts as ONE "repeat": the row
    below is the outer-fold-averaged test performance (mirroring what
    you'd actually quote as "the" nested CV estimate for this repeat's
    random seed), the modal outer-fold winner as chosen_classifier, and
    the mean inner-selection F1 as selection_score. n_used_for_selection_decision
    is the mean outer-training-fold size (~80% of the data, same ballpark
    as the other two arms' selection-time data) -- NOT the same thing as
    "how much data the final shipped model sees" (that's still 100%,
    reflected in frac_data_in_final_model, since nested CV's own final
    step is a separate selection+fit pass on all the data, same as
    kfold_cv's shipped-equivalent).

    fold_sink: optional list -- if given, one dict per OUTER fold is
    appended to it (seed, fold_idx, chosen classifier, held-out test
    metrics, inner-selection score, train/test sizes). This is the raw,
    un-averaged data the {target}_repeats.csv row above collapses into a
    single mean -- kept separately so a per-fold tracking panel can show
    the real fold-to-fold spread within a single nested_cv repeat, not
    just repeat-to-repeat spread of the already-averaged number."""
    outer_folds = max(2, min(5, int(y.sum()), int((y == 0).sum())))
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    candidates_template = _candidates(target)

    fold_metrics, fold_winners, fold_selection_scores, fold_n_train = [], [], [], []
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        inner_folds = max(2, min(5, int(min(np.bincount(y_tr)))))
        candidates = _candidates(target)
        best_name, best_score = None, -1.0
        for name, factory in candidates.items():
            try:
                scores = cross_val_score(factory(), X_tr_s, y_tr, cv=inner_folds, scoring='f1')
            except Exception:
                continue
            if scores.mean() > best_score:
                best_score, best_name = scores.mean(), name

        model = candidates[best_name]().fit(X_tr_s, y_tr)
        fold_eval = _eval(model, scaler, X_te, y_te)
        fold_metrics.append(fold_eval)
        fold_winners.append(best_name)
        fold_selection_scores.append(best_score)
        fold_n_train.append(len(y_tr))

        if fold_sink is not None:
            fold_sink.append({
                'seed': int(seed), 'fold_idx': int(fold_idx), 'n_outer_folds': int(outer_folds),
                'chosen_classifier': best_name, 'selection_score': float(best_score),
                'n_train': int(len(y_tr)), 'n_test': int(fold_eval['n_test']),
                'n_test_pos': int(fold_eval['n_test_pos']),
                'accuracy': fold_eval['accuracy'], 'precision': fold_eval['precision'],
                'recall': fold_eval['recall'], 'f1': fold_eval['f1'], 'roc_auc': fold_eval['roc_auc'],
            })

    mode_name = Counter(fold_winners).most_common(1)[0][0]
    metrics = {
        'accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'precision': float(np.mean([m['precision'] for m in fold_metrics])),
        'recall': float(np.mean([m['recall'] for m in fold_metrics])),
        'f1': float(np.mean([m['f1'] for m in fold_metrics])),
        'roc_auc': float(np.mean([m['roc_auc'] for m in fold_metrics if m['roc_auc'] is not None])),
        'n_test': int(np.mean([m['n_test'] for m in fold_metrics])),
        'n_test_pos': int(np.mean([m['n_test_pos'] for m in fold_metrics])),
        'methodology': 'nested_cv', 'seed': int(seed), 'chosen_classifier': mode_name,
        'selection_score': float(np.mean(fold_selection_scores)),
        'n_used_for_selection_decision': int(np.mean(fold_n_train)),
        'n_total': int(len(y)),
        # same as kfold_cv: nested CV's own final step is a separate
        # selection+fit pass on 100% of the data, after these outer-fold
        # numbers are already locked in.
        'frac_data_in_final_model': 1.0,
    }
    return metrics


def summarize(df):
    summary = {}
    for meth in ('train_val_test', 'kfold_cv', 'nested_cv'):
        sub = df[df.methodology == meth]
        n = len(sub)
        clf_counts = Counter(sub.chosen_classifier)
        mode_clf, mode_n = clf_counts.most_common(1)[0]
        summary[meth] = {
            'n_repeats': int(n),
            'test_f1_mean': float(sub.f1.mean()), 'test_f1_std': float(sub.f1.std()),
            'test_f1_95ci_halfwidth': float(1.96 * sub.f1.std() / np.sqrt(n)) if n > 1 else None,
            'test_roc_auc_mean': float(sub.roc_auc.mean()), 'test_roc_auc_std': float(sub.roc_auc.std()),
            'test_precision_mean': float(sub.precision.mean()), 'test_precision_std': float(sub.precision.std()),
            'test_recall_mean': float(sub.recall.mean()), 'test_recall_std': float(sub.recall.std()),
            'test_accuracy_mean': float(sub.accuracy.mean()), 'test_accuracy_std': float(sub.accuracy.std()),
            'selection_score_mean': float(sub.selection_score.mean()),
            'selection_score_std': float(sub.selection_score.std()),
            'frac_data_in_final_model_mean': float(sub.frac_data_in_final_model.mean()),
            'n_test_mean': float(sub.n_test.mean()),
            'classifier_selection_counts': {k: int(v) for k, v in clf_counts.items()},
            'classifier_selection_mode': mode_clf,
            'classifier_selection_mode_frequency': float(mode_n / n),
            'n_distinct_classifiers_chosen': len(clf_counts),
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', choices=['nes', 'nls', 'both'], default='both')
    ap.add_argument('--n-repeats', type=int, default=30)
    ap.add_argument('--out', default='split_methodology_comparison')
    args = ap.parse_args()

    out_dir = THIS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = ['nes', 'nls'] if args.target == 'both' else [args.target]
    for target in targets:
        print(f"\n{'=' * 70}\n{target.upper()}: split-methodology comparison "
              f"({args.n_repeats} repeats each)\n{'=' * 70}")
        X, y = _load_data(target)
        print(f"  n={len(y)} ({int(y.sum())} pos / {int((y == 0).sum())} neg)")

        rows = []
        fold_rows = []
        for seed in range(args.n_repeats):
            rows.append(run_train_val_test(X, y, seed, target))
            rows.append(run_kfold(X, y, seed, target))
            # nested_cv does a full inner classifier sweep inside EACH of its
            # ~5 outer folds (~5x the work of kfold_cv's single sweep per
            # repeat) -- noticeably slower, that's expected.
            rows.append(run_nested_cv(X, y, seed, target, fold_sink=fold_rows))
            if (seed + 1) % 5 == 0 or seed == args.n_repeats - 1:
                print(f"  {seed + 1}/{args.n_repeats} repeats done")

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f'{target}_repeats.csv', index=False)

        # Per-outer-fold nested_cv detail (un-averaged) -- feeds the
        # per-fold tracking panel in generate_thesis_figures.py's
        # fig_split_methodology_comparison, which the {target}_repeats.csv
        # row above (one averaged number per repeat) can't show on its own.
        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(out_dir / f'{target}_nested_cv_folds.csv', index=False)

        summary = summarize(df)
        with open(out_dir / f'{target}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        a, b, c = summary['train_val_test'], summary['kfold_cv'], summary['nested_cv']
        print(f"\n  train_val_test: test F1 = {a['test_f1_mean']:.3f} +/- {a['test_f1_std']:.3f}  "
              f"(mode classifier: {a['classifier_selection_mode']}, chosen in "
              f"{a['classifier_selection_mode_frequency'] * 100:.0f}% of runs, "
              f"{a['n_distinct_classifiers_chosen']} distinct winners across {a['n_repeats']} repeats)")
        print(f"  kfold_cv:       test F1 = {b['test_f1_mean']:.3f} +/- {b['test_f1_std']:.3f}  "
              f"(mode classifier: {b['classifier_selection_mode']}, chosen in "
              f"{b['classifier_selection_mode_frequency'] * 100:.0f}% of runs, "
              f"{b['n_distinct_classifiers_chosen']} distinct winners across {b['n_repeats']} repeats)")
        print(f"  nested_cv:      test F1 = {c['test_f1_mean']:.3f} +/- {c['test_f1_std']:.3f}  "
              f"(mode classifier: {c['classifier_selection_mode']}, chosen in "
              f"{c['classifier_selection_mode_frequency'] * 100:.0f}% of runs, "
              f"{c['n_distinct_classifiers_chosen']} distinct winners across {c['n_repeats']} repeats)")
        print(f"\n  Saved {out_dir / f'{target}_repeats.csv'}")
        print(f"  Saved {out_dir / f'{target}_summary.json'}")
        print(f"  Saved {out_dir / f'{target}_nested_cv_folds.csv'}")

    print(f"\nDone. Feed {out_dir}/ into generate_thesis_figures.py "
          f"(fig_split_methodology_comparison) for the comparison figure.")


if __name__ == '__main__':
    main()
