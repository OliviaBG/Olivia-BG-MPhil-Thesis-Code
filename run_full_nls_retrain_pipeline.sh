#!/usr/bin/env bash
# run_full_nls_retrain_pipeline.sh
#
# Run this on a machine with real internet access: alphafold.ebi.ac.uk,
# rest.uniprot.org, files.rcsb.org and iupred2a.elte.hu all need to be
# reachable. From the project root:
#
#     bash run_full_nls_retrain_pipeline.sh
#
# What it does, in order:
#   1. Backfill real per-residue pLDDT/SASA for every nls_dataset.csv /
#      nls_negatives.csv row that's missing it -- in particular the 13 new
#      hard negatives added (protamine paralogs, MARCKS-family,
#      extra histone paralogs, chemokines), which currently have NONE and
#      are training on neutral defaults.
#   2. Backfill real IUPred2A disorder / ANCHOR2 scores for the same gap.
#   3. Run the FULL nested-CV training --
#      regenerates models_nls/nls_classifier.pkl + nls_scaler.pkl +
#      nls_pssm.pkl + nls_metrics.json + nls_feature_importance.json
#      consistently, with a real outer-fold performance estimate and
#      permutation importance report.
#   4. Re-run the 25+25 held-out benchmark against the real trained model,
#      this time with real structural data actually reaching
#      /api/structure/<model_id> instead of falling back to neutral
#      RSA=0.4.
#
# Every step below is resumable/checkpointed on its own (steps 1 and 2 cache
# per-accession, so re-running after an interruption only fetches what's
# still missing) -- safe to Ctrl-C and restart.
set -e

cd "$(dirname "$0")"

echo "=================================================================="
echo "[1/4] Backfilling real pLDDT/SASA (nls_data_pipeline/structural_dataset_pipeline.py)"
echo "=================================================================="
pip install freesasa requests --break-system-packages 2>/dev/null || pip install freesasa requests
python3 nls_data_pipeline/structural_dataset_pipeline.py

echo
echo "=================================================================="
echo "[2/4] Backfilling real IUPred2A/ANCHOR2 (fetch_iupred_training_data.py)"
echo "=================================================================="
python3 fetch_iupred_training_data.py --pipeline nls

echo
echo "=================================================================="
echo "[3/4] Full nested-CV retrain (nls_ml_predictor.py train)"
echo "=================================================================="
echo "This is the slow step (outer 5-fold CV + inner model selection over"
echo "~750 examples, 8 classifier candidates) -- expect a few minutes, not"
echo "seconds. Do not interrupt once it starts writing models_nls/."
python3 nls_ml_predictor.py train

echo
echo "=================================================================="
echo "[4/4] Re-running the 25+25 held-out benchmark with real structural data"
echo "=================================================================="
python3 run_nls_holdout_pipeline_test.py

echo
echo "=================================================================="
echo "Done. Compare nls_holdout_test_results.md's new sensitivity/specificity"
echo "and 'Structural data coverage' line against the 56.0%/68.0%,"
echo "0/50-real-structural-data run from earlier today."
echo "Also check models_nls/nls_metrics.json's nested_cv block and"
echo "nls_feature_importance.json -- both are now consistent with the"
echo "current 29-feature model instead of the pre- numbers."
echo "=================================================================="
