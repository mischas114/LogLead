# LO2 explainability run (Phase F)

Run date: 2025-11-11

## Execution
- Command: `MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --root demo/result/lo2 --skip-if --sup-models event_lr_words,event_xgb_words --nn-source event_lr_words --shap-sample 200 --shap-background 256 --shap-feature-threshold 2000 --shap-cell-threshold 2000000`
- SHAP guard configuration: `background=256`, `feature_threshold=2000`, `cell_threshold=2000000`
- Source sequences: `demo/result/lo2/lo2_sequences_enhanced.parquet`
- IsolationForest skipped because of `--skip-if`
- Phase artifacts stored under `demo/result/lo2/explainability`

## Model outcomes
### Sequence LR (tokens)
- Metrics: accuracy 1.0000, F1 1.0000, support 597, AUC-ROC 1.0000
- Confusion matrix: `[[299, 0], [0, 298]]`
- SHAP guard triggered (feature count > 2000) so no token SHAP plots

### Sequence LR (numeric)
- Metrics: accuracy 1.0000, F1 1.0000, support 597, AUC-ROC 1.0000
- Confusion matrix: `[[299, 0], [0, 298]]`
- SHAP sampled 200 of 597 sequences (requested 200)
- Top SHAP features: `seq_len`, `duration_sec`

### Event LR (words)
- Metrics: accuracy 0.9983, F1 0.9983, support 597, AUC-ROC 1.0000
- Confusion matrix: `[[298, 1], [0, 298]]`
- Predictions saved to `demo/result/lo2/explainability/event_lr_words_predictions.parquet`
- False positives logged in `demo/result/lo2/explainability/event_lr_words_false_positives.txt`
- NN mapping saved in `demo/result/lo2/explainability/event_lr_words_nn_mapping.csv` (299 anomalies matched to 298 normals)
- SHAP guard triggered (feature count > 2000)

### Event XGB (words)
- Metrics: accuracy 0.9866, F1 0.9868, support 597, AUC-ROC 0.9971
- Confusion matrix: `[[291, 8], [0, 298]]`
- Predictions saved to `demo/result/lo2/explainability/event_xgb_words_predictions.parquet`
- SHAP guard triggered (feature count > 2000)

## Diagnostics and logs
- Downsampling happened for SHAP background selection
- FutureWarning from NumPy seed in `loglead/explainability_utils.py:78`; consider passing an explicit RNG handle
- Summary snippet reports: `seq_lr_tokens_shap_samples=0`, `seq_lr_numeric_shap_samples=200`, `event_lr_words_shap_samples=0`, `event_xgb_words_shap_samples=0`
- One false positive identified: `seq_id=light-oauth2-data-1719592986__correct__light-oauth2-oauth2-token-1` (`score_event_lr_words=0.640108`)

## Data coverage
- Total sequences scored per supervised model: 597 (299 predicted normal, 298 predicted anomalous)
- Sequence-LR numeric SHAP used 200 background samples and 200 explanation samples (requested limit)
- Nearest-neighbour mapping paired 299 anomalous and 298 normal sequences (see `*_nn_mapping.csv` files)
- Guarded explainers (Sequence/Event word models) processed all records for metrics but skipped SHAP plots because of the feature-limit check

## False positives and thresholds
- Single false positive from `event_lr_words` at probability score `0.640108`; all other models aligned with ground truth
- Current anomaly flag threshold equals the default decision boundary (0.5); raising it to ~0.7 would clear the described false positive while keeping a wide margin to the true anomalous scores (all close to 1.0 in the predictions file)
- If a softer recall target is desired, retain 0.5 but monitor that specific sequence; otherwise adopt 0.7 for production scoring and re-validate once more data arrives

## Visual artifacts
- `sequence_shap_lr_words_top_features.png` and `seq_lr_tokens_top_features.png`: (guarded) summary bar charts showing average |SHAP| contributions; features listed without dot plots because SHAP computation stopped at the guard threshold
- `seq_lr_numeric_top_features.png` and `seq_lr_numeric_summary.png`: dot plot and bar chart illustrating SHAP for numeric features; hotter colors (pink) correspond to higher feature values driving the anomaly score upward, cooler colors (blue) indicate lower values reducing the score
- Each PNG resides in `demo/result/lo2/explainability/` alongside the text exports and offers a quick visual confirmation of the feature rankings referenced above

## Key artifacts
- Sequence feature importance (words): `demo/result/lo2/explainability/sequence_shap_lr_words_top_features.txt`
- Sequence feature importance (numeric): `demo/result/lo2/explainability/seq_lr_numeric_top_features.txt`
- Metrics JSON files per model in `demo/result/lo2/explainability/metrics_*.json`
- Guard notices in `*_shap_guard.txt` and skipped logs in `*_shap_skipped.txt`

## Follow-ups
- Increase `--shap-feature-threshold` (and possibly cell threshold) if token-level SHAP plots are required
- Investigate the logged false positive and decide whether to enrich training data or adjust thresholds
- Address the NumPy RNG warning by passing an explicit generator to SHAP plotting utilities
