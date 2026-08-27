# Trained NES model

Artefacts written by `ImprovedNESPredictor._train_model()` in
`nes_ml_predictor_improved.py`. They correspond to the datasets committed in
`nes_data_pipeline/`, `nes_negatives/` and
`nes_negatives_leucine_zipper_expansion/`, so predictions reproduce without
retraining.

| File | Contents |
| --- | --- |
| `nes_svm_v2.pkl` | the fitted classifier (the filename predates the model search; the selected model is recorded in the metadata below) |
| `nes_scaler_v2.pkl` | feature scaler fitted on the training split |
| `nes_pssm_v2.pkl` | position-specific scoring matrix, anchored on the hydrophobic register |
| `nes_model_meta_v2.json` | selected model family and the ordered feature names |
| `nes_metrics_v2.json` | dataset composition, per-model CV scores and the nested-CV estimate |
| `nes_permutation_importance_v2.json` | permutation importance per feature |
| `feature_diagnosis/` | output of `diagnose_feature_importance.py`: impurity, permutation, univariate ROC-AUC and correlation, cross-checked |

## Current model

Selected by nested cross-validation over eight model families:
**extra trees**, chosen in 5 of 5 outer folds.

| Metric | Value |
| --- | --- |
| Outer-fold F1 | 0.898 +/- 0.014 |
| Outer-fold ROC-AUC | 0.984 +/- 0.001 |
| Features | 46 |
| Positives | 305 (271 database-derived, 34 curated seed) |
| Negatives | 889 (542 protein-matched, 300 structural hard negatives, 47 synthetic) |

The performance estimate comes from the outer folds, not a single held-out
split, so it does not depend on one lucky partition. Note that the negatives
are deliberately adversarial -- coiled-coil and leucine-zipper regions that
match the NES consensus by sequence -- so these numbers are not comparable
to studies that evaluate against random decoys.

Retrain with:

```python
from nes_ml_predictor_improved import ImprovedNESPredictor
ImprovedNESPredictor()._train_model()
```
