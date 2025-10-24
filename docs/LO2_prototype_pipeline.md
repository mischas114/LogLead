# LO2 Prototype Pipeline – Detailed Specification

## 1. High-Level Flow
1. **Input Acquisition**: Collect raw LO2 log bundles (events, sequences, optional metrics) from the sample repository.
2. **Loader Execution**: Use `loglead.loaders.lo2.LO2Loader` (invoked via `demo/run_lo2_loader.py`) to normalize basic fields and persist Parquet artifacts.
3. **Feature Enrichment**: Chain `EventLogEnhancer` and, optionally, `SequenceEnhancer` in `demo/LO2_samples.py` to create token, grammar and length representations.
4. **Model Preparation**: Configure `loglead.AnomalyDetector` with selected item/numeric columns; split training vs. evaluation frames.
5. **Unsupervised Baseline**: Train Isolation Forest (`train_IsolationForest`) on the curated "correct" subset; score the full dataset and rank sequences.
6. **Optional Supervised Comparison**: Fit Logistic Regression / Decision Tree (`train_LR`, `train_DT`) on labeled data to benchmark against the unsupervised baseline.
7. **Explainability Hooks**: Apply `loglead.explainer.NNExplainer` (for IF) or `ShapExplainer` (for LR/Tree) to surface reasons behind top-ranked anomalies.
8. **Result Persistence & Reporting**: Store predictions, metrics, and explanation assets; summarize findings in Markdown / dashboards.

## 2. Components & Responsibilities
### 2.1 Loader Layer
- **Class**: `loglead.loaders.lo2.LO2Loader`
- **Entry Point**: `python demo/run_lo2_loader.py ...`
- **Tasks**:
  - Traverse the chosen LO2 sample folder and collect raw event/sequnce logs.
  - Normalize timestamps, services, and anomaly flags.
  - Emit `lo2_events.parquet`, `lo2_sequences.parquet`, and optionally `lo2_metrics.parquet` under `demo/result/lo2/`.
- **Key Arguments**:
  - `--root`: source directory.
  - `--runs`, `--errors-per-run`, `--single-service`: control sample size and anomaly density.
  - `--save-parquet`: enforce artifact persistence.
  - `--output-dir`: target directory for generated Parquet files.

### 2.2 Feature Enhancement Layer
- **Classes**: `loglead.enhancers.EventLogEnhancer`, `loglead.enhancers.SequenceEnhancer`
- **Entry Point**: `python demo/LO2_samples.py --phase enhancers`
- **Task Breakdown**:
  1. `normalize()`: cleans raw messages and harmonizes whitespace/punctuation.
  2. `words()`: tokenizes messages (`e_words`, `e_words_len`).
  3. `trigrams()`: constructs sliding 3-character tokens (`e_trigrams`, `e_trigrams_len`).
  4. `parse_drain()`: runs Drain parser to assign template IDs (`e_event_drain_id`).
  5. `length()`: adds character/line counts (`e_chars_len`, `e_lines_len`).
  6. (Optional) `SequenceEnhancer.seq_len()`, `.duration()`, `.tokens()` for run-level features.
- **Artifacts**: Enriched Parquet files containing the new columns; reused directly by the detector phase.

### 2.3 Modeling Layer
- **Class**: `loglead.anomaly_detection.AnomalyDetector`
- **Primary Fields**:
  - `item_list_col`: feature column containing tokens or IDs.
  - `numeric_cols`: list of additional scalar features.
  - `train_df`, `test_df`: Polars DataFrames assigned externally.
- **Key Methods**:
  1. `prepare_train_test_data(vectorizer_class=CountVectorizer)`: builds sparse matrices / numpy arrays and label vectors.
  2. `train_IsolationForest(filter_anos=True, n_estimators=200, contamination=0.1, max_samples='auto')`: fits unsupervised baseline.
  3. `train_LR(...)`, `train_DT()`: alternative supervised baselines.
  4. `predict()`: outputs predictions (`pred_ano`, optional `pred_ano_proba`), prints metrics (Accuracy, F1, Confusion Matrix, AUCROC) and returns a Polars DataFrame aligned with `test_df`.
- **Pipeline Hooks**:
  - `sad.train_df = df_events.filter(pl.col("test_case") == "correct")`
  - `sad.test_df = df_events`
  - `sad.prepare_train_test_data()`
  - `sad.train_IsolationForest(filter_anos=True, contamination=<tuned value>, ...)`
  - `pred_df = sad.predict()`
  - `pred_df = pred_df.with_columns(pl.Series("score_if", -sad.model.score_samples(sad.X_test_no_anos)))`
  - `pred_df.write_parquet("demo/result/lo2/lo2_if_predictions.parquet")`

### 2.4 Explainability Layer
- **Isolation Forest**: `loglead.explainer.NNExplainer`
  - `ex = NNExplainer(sad)`
  - `report = ex.explain_top_k(k=5)` → returns nearest-neighbour style explanations for high scores.
- **Supervised Models**: `loglead.explainer.ShapExplainer`
  - `ex = ShapExplainer(sad, ignore_warning=True)`
  - `ex.calc_shapvalues()` followed by `ex.plot(plottype="summary")` or `ex.explain_sample(sample_id)`.
- **Outputs**: Visual or tabular explanation artifacts suitable for inclusion in the MVP report.

## 3. Execution Order & Command Sequence
1. **Loader** – `python demo/run_lo2_loader.py --root <path> --runs 5 --errors-per-run 1 --single-service client --save-parquet --output-dir demo/result/lo2`
2. **Enhancer** – `python demo/LO2_samples.py --phase enhancers`
3. **Isolation Forest Baseline** – `python demo/LO2_samples.py --phase if --if-contamination <value> --if-item <column> --if-numeric <cols> --save-if demo/result/lo2/lo2_if_predictions.parquet`
4. **Optional Supervised Benchmark** – `python demo/LO2_samples.py --phase full --if-contamination <value> ...` or interactively call `train_LR()` / `train_DT()`
5. **Explainability** – run dedicated notebook/cell invoking `NNExplainer` / `ShapExplainer`, storing charts and CSV explanations.
6. **Documentation** – update `docs/LO2_minimal_IF_XAI_workflow.md` (metrics, limitations, tuning log).

## 4. Data Artifacts & Versions
| Artifact | Producer | Location | Notes |
| --- | --- | --- | --- |
| `lo2_events.parquet` | Loader | `demo/result/lo2` | Pre-processed events with labels |
| `lo2_sequences.parquet` | Loader | `demo/result/lo2` | Sequence-level aggregation |
| `lo2_if_predictions.parquet` | AnomalyDetector | `demo/result/lo2` | Scores + ranks |
| `models/lo2_if.joblib` | AnomalyDetector | `models/` (manual) | Serialized Isolation Forest + vectorizer |
| SHAP/NN reports | Explainer | `docs/result/…` (manual) | Visual explanations |

## 5. Configuration Surface
- CLI flags in `demo/LO2_samples.py`:
  - `--phase {enhancers,if,full}`
  - `--if-contamination`, `--if-n-estimators`, `--if-max-samples`
  - `--if-item`, `--if-numeric`
  - `--save-if`
- Loader flags (see Section 2.1)
- Environment variables: none required; virtual environment manages dependencies.

## 6. Known Limitations (as of 23.10.2025)
- Isolation Forest currently produces high scores for certain legitimate runs (`light-oauth2-oauth2-client-1`), leading to false positives at the top ranks.
- TensorFlow is optional; absence triggers a warning for BERT embeddings but has no functional impact.
- Drift between training and evaluation requires manual filtering (service-based or time-based) to avoid skewed ranking.

## 7. Improvement Roadmap
1. **Training Filters**: Exclude noisy services or apply time-based windows when populating `train_df`.
2. **Feature Expansion**: Incorporate sequence-level durations, unique token counts, or HTTP status aggregates.
3. **Score Calibration**: Experiment with `score_if = -sad.model.score_samples(...)` or alternative ranking heuristics.
4. **Automated Reporting**: Generate Top-K anomaly tables and tuning summaries programmatically.
5. **Model Persistence**: Standardize joblib dumps and version metadata for reproducibility.
6. **Explainability UX**: Bundle NN/SHAP outputs into a compact HTML or PDF report for stakeholders.

This document clarifies the roles, responsibilities, and interactions within the prototype pipeline so that new contributors can reproduce and extend the MVP without revisiting the exploratory conversation.
