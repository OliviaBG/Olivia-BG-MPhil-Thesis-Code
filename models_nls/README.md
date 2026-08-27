# Trained NLS model

Artefacts written by `nls_ml_predictor.py train`. They correspond to the
datasets committed in `nls_data_pipeline/`, so predictions reproduce without
retraining.

| File | Contents |
| --- | --- |
| `nls_classifier.pkl` | the fitted classifier |
| `nls_scaler.pkl` | feature scaler fitted on the training split |
| `nls_pssm.pkl` | position-specific scoring matrix, anchored on the basic-cluster register |
| `nls_metrics.json` | dataset composition, per-model CV scores, nested-CV estimate and feature importances |
| `nls_feature_importance.json` | impurity and permutation importance per feature |
| `feature_diagnosis/` | output of `diagnose_feature_importance_nls.py` |

## Current model

Selected by nested cross-validation over the same eight model families used
on the NES side: **XGBoost**, chosen in 5 of 5 outer folds.

| Metric | Value |
| --- | --- |
| Outer-fold F1 | 0.887 +/- 0.005 |
| Outer-fold ROC-AUC | 0.983 +/- 0.002 |
| Features | 29 |
| Positives | 2,447 (218 experimentally annotated, 673 bipartite) |
| Negatives | 11,966 |

Negatives break down as 8,175 protein-matched random windows, 3,691
UniProt-annotated DNA-binding regions, 80 synthetic polybasic decoys and a
small hand-curated set of specific false-positive classes: extreme
arginine-rich condensin subunits, calmodulin and PIP2 effectors, linker
histone paralogues, heparin-binding chemokines and CAAX membrane anchors.

The DNA-binding negatives are the load-bearing ones. They are genuinely
basic and genuinely functional but are not import signals, which is exactly
the failure mode that NLS predictors relying on basic-residue density fall
into.

By permutation importance the top features are basic fraction (0.233) and
PSSM score (0.184), which is the expected chemistry; the bipartite spacer
length carries independent signal (0.019) while the bipartite pattern flag
itself is nearly redundant with it (0.0005).

Retrain with `python nls_ml_predictor.py train`, or use
`run_full_nls_retrain_pipeline.sh` to refresh the structural and disorder
features first.
