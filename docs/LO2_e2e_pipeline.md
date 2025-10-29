# LO2 End-to-End Pipeline Guide

This document outlines how to run the LO2 MVP pipeline from raw log folders to machine-learning based anomaly detection and explainability artefacts. The flow reuses the demo scripts shipped with this repository and produces intermediate Parquet exports that can be inspected or fed into downstream tooling.

## Prerequisites
- A checkout of this repository with the Python dependencies installed (see `pyproject.toml`).
- Access to the LO2 run folders laid out as `<root>/run_<id>/<test_case>/*.log` with optional `metrics/*.json` subfolders.
- Sufficient disk space under `demo/result/lo2/` (default output location) or a custom path supplied via CLI options.

## Step 1 – Export LO2 logs to Parquet (`demo/run_lo2_loader.py`)
The loader script wraps `loglead.loaders.LO2Loader` and converts raw log files into event and sequence level tables.

```bash
python demo/run_lo2_loader.py \
  --root /path/to/lo2_runs \
  --runs 5 \                # optional: limit runs processed
  --errors-per-run 1 \       # optional: number of error cases per run
  --single-service client \  # optional: restrict to oauth2-oauth2-client logs
  --save-parquet              # required for later steps
```

```bash
python demo/run_lo2_loader.py \
  --root /Users/MTETTEN/Documents/Bachelorarbeit/lo2/lo2-analysis/data/lo2-sample/logs \
  --runs 9999 \
  --errors-per-run 1 \
  --single-service client \
  --allow-duplicates \
  --save-parquet \
  --output-dir demo/result/lo2
```

Outputs (defaults under `demo/result/lo2/`):
- `lo2_events.parquet` – event-level table; every row represents a single log line enriched with its run id, test case, service name, derived `seq_id`, parsed timestamp, and a `normal/anomaly` label.
- `lo2_sequences.parquet` – sequence-level aggregates; each row groups all events that share the same `seq_id` (i.e. run + test case + service), storing the concatenated message block plus start/end timestamps and the normal/anomaly flag for that entire sequence. Storing both granular events and aggregated sequences lets downstream steps choose between fine-grained log analysis and faster sequence-level modelling.
- `lo2_metrics.parquet` – only when `--load-metrics` is set and metric files exist.

## Step 2 – Run the enhancement + anomaly detection demo (`demo/LO2_samples.py`)
This script reads the Parquet exports, enriches them with textual features, and executes multiple anomaly detectors.

```bash
python demo/LO2_samples.py \
  --phase full \                   # enhancers → IsolationForest → LR/DT + SHAP setup
  --sample-seed 42 \               # controls down-sampling and random logging
  --if-contamination 0.1 \         # IsolationForest hyper-parameter
  --if-item e_words \              # token column to use
  --if-numeric e_chars_len \       # optional comma-separated numeric extras
  --save-enhancers \               # persist enhanced Parquet tables
  --enhancers-output-dir result/lo2/enhanced
```
```bash
# Headless variant to keep SHAP from opening interactive windows
MPLBACKEND=Agg python demo/LO2_samples.py \
  --phase full \
  --sample-seed 42 \
  --if-contamination 0.1 \
  --if-item e_words \
  --if-numeric e_chars_len \
  --save-enhancers \
  --enhancers-output-dir result/lo2/enhanced
```

Key phases:
- **Enhancers** – `EventLogEnhancer` normalises messages, tokenises (words, trigrams), runs Drain parsing, derives length features; `SequenceEnhancer` aggregates sequences when available.
- **Phase D (IF baseline)** – IsolationForest trained on `test_case == "correct"` events, scores full dataset, writes `lo2_if_predictions.parquet` (path via `--save-if`).
- **Phase E (Supervised models)** – Logistic Regression on words, Decision Tree on trigrams, optional sequence-level LR when sequence table exists. Creates prediction tables and prints top-ranked anomalies.

Artifacts:
- Enhanced event/sequence Parquet files when `--save-enhancers` is passed.
- IsolationForest predictions stored at `--save-if` (default `result/lo2/lo2_if_predictions.parquet`).
- Console output summarising top anomalies and sample feature rows.

## Step 3 – Generate explainability artefacts (`demo/lo2_phase_f_explainability.py`)
Consumes the enhanced event table (and optional sequence table) to rebuild the IsolationForest baseline, compute SHAP values for supervised models, and produce NNExplainer mappings.

What happens internally:
- The script re-runs the Isolation Forest from Phase D (using Drain event IDs plus length features) so that the explainability outputs line up with the latest feature engineering tweaks.
- `NNExplainer` selects the top-`nn-top-k` anomalous rows, finds their closest normal neighbours in feature space, and emits a mapping that helps analysts compare offending vs. healthy sequences.
- For the supervised Phase-E models, the script trains Logistic Regression on word tokens and Decision Trees on trigrams, then uses SHAP to rank the most influential tokens and create summary/bar plots.
- When `lo2_sequences.parquet` exists, it also retrains the sequence-level Logistic Regression model, logs its metrics, and (when possible) generates SHAP artefacts for the sequence features.
- Metrics for each model (accuracy, F1, AUC when available) are written alongside the plots so you can judge model quality next to the explanations.

```bash
python demo/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.45 \      # tuned contamination from Phase D experiments
  --nn-top-k 50 \                # anomalous rows exported for NN mapping
  --shap-sample 200               # sample size for SHAP calculations
```
```bash
MPLBACKEND=Agg python demo/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.45 \
  --nn-top-k 50 \
  --shap-sample 200
```


Parameter tips:
- `--if-contamination` sets the anomaly share for the rebuilt Isolation Forest. Keep it aligned with Phase D; higher values mark more rows as anomalous and enlarge the NN/SHAP focus set, lower values make the model pickier.
- `--if-n-estimators` (defaults to 200) behaves like in Step 3: more trees stabilise scores at the cost of runtime. Increase to 300+ only if the ranking is unstable across runs.
- `--nn-top-k` controls how many anomalous rows get paired with nearest normals. Use 20–50 for manual review, increase when you need richer context for dashboards.
- `--nn-normal-sample` caps how many “healthy” rows accompany the anomalies. Raising it gives more comparison examples but also larger files.
- `--shap-sample` limits how many test rows SHAP processes. Larger samples produce smoother feature rankings but can grow computation time rapidly; 100–200 works well for exploratory work, while >500 is only advisable if you have plenty of compute.

Outputs (under `demo/result/lo2/explainability/`):
- `lo2_if_predictions.parquet` – rebuilt IsolationForest prediction table with scores/ranks.
  - `pred_ano` is the model’s binary decision (1 = predicted anomalous, 0 = predicted normal).
  - `score_if` is the raw IsolationForest anomaly score (higher means more unusual).
  - `rank_if` is the dense rank of `score_if`, so 1 marks the most suspicious row.
- `if_nn_mapping.csv` – nearest-normal mapping for anomalous events; `if_false_positives.txt` lists FP cases when present.
- `lr_shap_*`, `dt_shap_*` images and text files summarising SHAP top features for LR and DecisionTree models.
- `metrics_*.json` – accuracy/F1/AUC metrics logged per model.
- Sequence-level SHAP artefacts when `lo2_sequences.parquet` exists and contains data.

## Putting It Together
1. **Load & export** – `run_lo2_loader.py --save-parquet` on the raw LO2 runs.
2. **Enhance & detect** – `LO2_samples.py --phase full` (or `--phase enhancers` / `--phase if` for partial runs).
3. **Explainability** – `lo2_phase_f_explainability.py` to create NNExplainer outputs and SHAP plots.

After step 4 you have a complete chain from raw logs to ranked anomalies with explainability artefacts suitable for reports or downstream integrations. Adjust CLI flags to point at custom input/output directories when embedding the scripts into other environments.

## After the MVP Run
- Inspect `demo/result/lo2/explainability/` artefacts: SHAP plots, `if_nn_mapping.csv`, and `if_false_positives.txt` surface the sequences worth manual review.
- `python tools/lo2_result_scan.py --dry-run` previews the same checks and metrics; drop `--dry-run` to append the summary to `summary-result.md` (use `--summary-file` / `--ticket-template` for custom targets).
- Check IsolationForest, Logistic Regression, and DecisionTree metrics; if AUC or precision are weak, revisit contamination, sampling, or feature selection before increasing data volume.
- Keep `result/lo2/enhanced/` Parquets for notebook experiments or dashboard prototyping; the production scripts always rebuild features from the raw exports.
- When scaling to larger datasets, follow the workflow tips in `docs/NEXT_STEPS.md` to size loaders, tune parameters, and avoid slow trial-and-error reruns.
