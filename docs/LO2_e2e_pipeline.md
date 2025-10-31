# LO2 End-to-End Pipeline Guide

This single reference explains how to execute, persist, and iterate on the LO2 MVP pipeline. It consolidates the earlier quick-start, prototype, and persistence notes so new contributors have one place to check.

## Prerequisites
- Repository checkout with dependencies installed (`pip install -e .` or poetry).  
- LO2 run folders organised as `<root>/run_<id>/<test_case>/*.log` plus optional `metrics/*.json`.  
- Working directory with enough space under `demo/result/lo2/` (or a custom `--output-dir`).

## Quick Start Commands
```bash
python demo/lo2_e2e/run_lo2_loader.py \
  --root /path/to/lo2_runs \
  --runs 5 \
  --errors-per-run 1 \
  --single-service client \
  --save-parquet \
  --output-dir demo/result/lo2

python demo/lo2_e2e/run_lo2_loader.py --root --errors-per-run 1 /Users/MTETTEN/Documents/Bachelorarbeit/lo2/lo2-analysis/data/lo2-sample/logs --service-types code token refresh-token --save-parquet --output-dir demo/result/lo2

# 2) Enhance + detection (phases A–E)
MPLBACKEND=Agg python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --sample-seed 42 \
  --if-contamination 0.1 \
  --if-item e_words \
  --if-numeric e_chars_len \
  --save-enhancers \
  --enhancers-output-dir result/lo2/enhanced

  MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --root demo/result/lo2 --if-contamination 0.5 --if-n-estimators 400 --nn-top-k 50 --shap-sample 200

```
- --if-contamination 0.10: let Isolation Forest assume roughly 10 % of events may be anomalous.
- --if-n-estimators 400: build the Isolation Forest with 400 trees for a steadier score distribution.
- --nn-top-k 50: keep the 50 highest-scoring anomalies for nearest-normal comparisons in the explainability pass.
- --shap-sample 200: limit SHAP calculations to 200 sampled events/sequences so plots stay fast.
```
# 3) Explainability pass (phase F)
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.45 \
  --nn-top-k 50 \
  --shap-sample 200
```

### Expected Outputs (`demo/result/lo2/`)
- `lo2_events.parquet` / `lo2_sequences.parquet`: loader exports with labels, timestamps, `seq_id`.  
- `lo2_if_predictions.parquet`: Isolation Forest scores and ranks (phases D/F).  
- `result/lo2/enhanced/*.parquet`: optional cache with engineered features when `--save-enhancers` is set.  
- `result/lo2/explainability/`: NN mappings, SHAP plots, metrics JSON files.

## Pipeline Phases at a Glance
- **Phase A – Setup:** activate virtualenv, ensure `polars`, `scikit-learn`, `joblib`, `loglead` import without errors.  
- **Phase B – Loader:** `run_lo2_loader.py --save-parquet`; confirm `lo2_events.parquet` (and sequences/metrics).  
- **Phase C – Enhancers:** `LO2_samples.py --phase enhancers`; expect columns such as `e_words`, `e_trigrams`, `e_chars_len`, optional Drain IDs and sequence metrics.  
- **Phase D – Isolation Forest:** `--phase if` with tuned `--if-*` flags; store predictions and optionally `joblib.dump((sad.model, sad.vec), ...)`.  
- **Phase E – Supervised Benchmarks:** `--phase full` trains LR/DT (event) and optional sequence LR for comparison.  
- **Phase F – Explainability:** `lo2_phase_f_explainability.py` rebuilds IF, runs NNExplainer, calculates SHAP plots, and republishes metrics.  
- **Phase G/H – Iteration & Reuse:** reload Parquets, adjust contamination/features, or apply a persisted model to fresh runs.

Refer back to the demo scripts when you need finer control (e.g., `--phase if` only, or different feature columns). They mirror the methods in `loglead.loaders`, `loglead.enhancers`, and `loglead.AnomalyDetector`.

## CLI Flag Reference
- **Loader:** `--root`, `--runs`, `--errors-per-run`, `--single-service`, `--save-parquet`, `--allow-duplicates`, `--load-metrics`, `--output-dir`.  
- **Detector:** `--phase {enhancers,if,full}`, `--if-contamination`, `--if-n-estimators`, `--if-item`, `--if-numeric`, `--if-max-samples`, `--save-if`, `--save-enhancers`.  
- **Explainability:** `--if-contamination`, `--if-n-estimators`, `--nn-top-k`, `--nn-normal-sample`, `--shap-sample`, `--root`.

## Persistence & Reuse
1. **Loader artefacts** – always run the loader with `--save-parquet` so `lo2_events.parquet` and `lo2_sequences.parquet` survive across sessions.  
2. **Enhanced caches (optional)** – after `EventLogEnhancer`, keep `result/lo2/enhanced/*.parquet` if notebooks should start with engineered features.  
3. **Models + vectorisers** – pass `--save-model models/lo2_if.joblib` (and optionally `--overwrite-model`) to `LO2_samples.py` so the IsolationForest + vectoriser bundle is persisted automatically.  
   Later reuse via `model, vec = joblib.load("models/lo2_if.joblib")`, `sad.prepare_train_test_data(vectorizer_class=vec)`, then `sad.predict()`.
4. **Predictions & reports** – persist `pred_df.write_parquet(...)` and the explainability artefacts under `result/lo2/explainability/`.  
5. **After restart** – load Parquets with `pl.read_parquet(...)`, reload the joblib bundle, and rerun detection against new runs without retraining.

## Iterative Tuning Tips
- Start with small subsets (`--runs 3`) to iterate quickly; scale up once parameters stabilise.  
- Sweep `--if-contamination` (0.1–0.45) and token columns (`e_words`, `e_trigrams`, `e_event_drain_id`) to understand precision/recall trade-offs.  
- Add numeric helpers (`e_chars_len`, `e_event_id_len`, sequence durations) via `--if-numeric`.  
- Inspect `result/lo2/lo2_if_predictions.parquet` sorted by `score_if`; false-positive hotspots (e.g., `light-oauth2-oauth2-client-1`) indicate services to filter from the training split or feature candidates to extend.  
- Record settings and metrics in `summary-result.md` or the tuning table in `docs/NEXT_STEPS.md` for reproducibility.

## Explainability Outputs
- `if_nn_mapping.csv` / `if_false_positives.txt`: nearest-normal comparisons for the top-k IF anomalies.  
- `lr_shap_*`, `dt_shap_*`, `seq_lr_*`: SHAP summary and bar plots plus token rankings.  
- `metrics_*.json`: Accuracy, F1, AUC snapshots for LR/DT/Sequence-LR.  
- Use `python tools/lo2_result_scan.py --dry-run` to summarise whether the expected files are present and append findings to `summary-result.md` when required.

## Artifact Index
| Artifact | Created by | Location | Notes |
| --- | --- | --- | --- |
| `lo2_events.parquet` | LO2Loader | `demo/result/lo2` | Event-level dataset with labels, seq IDs, timestamps |
| `lo2_sequences.parquet` | LO2Loader | `demo/result/lo2` | Sequence aggregates per run/test/service |
| `lo2_if_predictions.parquet` | AnomalyDetector | `demo/result/lo2` | IF scores, ranks, predictions |
| `result/lo2/enhanced/*.parquet` | LO2_samples.py (`--save-enhancers`) | `result/lo2/enhanced` | Optional cache of engineered features |
| `models/lo2_if.joblib` | `LO2_samples.py --save-model` | `models/` | Isolation Forest + vectoriser bundle |
| `result/lo2/explainability/*` | lo2_phase_f_explainability.py | `demo/result/lo2/explainability` | NN mapping, SHAP plots, metrics |

## Known Limitations & Next Steps
- Isolation Forest can over-score legitimate client runs; mitigate by filtering training services, enriching features, or lowering contamination.  
- Decision Tree benchmarks on small samples may report perfect metrics—treat as overfitting until validated on larger corpora.  
- Sequence-level models remain weaker without extra engineered features; see `docs/NEXT_STEPS.md` for data expansion and feature ideas.  
- TensorFlow-dependent enhancers (e.g., BERT embeddings) are optional; warnings can be ignored when unavailable.

## Related References
- `docs/LO2_MVP_Classes.md` – class-level design notes for the loader and demos.  
- `docs/NEXT_STEPS.md` – scaling plan, dataset roadmap, and tuning backlog.  
- `tools/lo2_result_scan.py` – automated artefact audit.  
- `summary-result.md` – running log of experiment outcomes.
