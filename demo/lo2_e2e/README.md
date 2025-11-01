# LO2 E2E Demo Flow

This folder groups the runnable scripts for the LO2 pipeline, covering data loading, enhancement, anomaly detection, and explainability artefacts.

## Quickstart

1. **Load raw runs to Parquet**
   ```bash
   python demo/lo2_e2e/run_lo2_loader.py --root /path/to/lo2_data --runs 5 --save-parquet --output-dir demo/result/lo2
   ```
2. **Generate enhancements and anomaly predictions**
   ```bash
   python demo/lo2_e2e/LO2_samples.py --phase full --save-enhancers --save-model models/lo2_if.joblib
   ```
   The `--save-model` argument stores the IsolationForest + vectorizer bundle for reuse; add `--overwrite-model` when replacing an existing dump.
   Additional opt-in benchmarking aids: `--if-holdout-fraction 0.1 --if-threshold-percentile 99.5 --report-precision-at 100 --report-fp-alpha 0.005 --report-psi --metrics-dir result/lo2/metrics --dump-metadata`.
3. **Create explainability artefacts**
   ```bash
   MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --root demo/result/lo2 --shap-sample 200
   ```

All outputs are written beneath `demo/result/lo2` by default. Adjust CLI options to mirror your dataset size and desired sampling behaviour.

## Configurable anomaly detectors

`LO2_samples.py` now behaves like a small testbed: you can mix and match anomaly detectors without touching the code.

- Show every available key with `python demo/lo2_e2e/LO2_samples.py --list-models`.
- The default set (`event_lr_words,event_dt_trigrams,sequence_lr_numeric,sequence_shap_lr_words`) mirrors the original Phase E/F baseline.
- Pass `--models key1,key2,...` to run a custom line-up. Example:
  ```bash
  python demo/lo2_e2e/LO2_samples.py --models event_lr_words,event_rf_words,event_lof_words,sequence_shap_lr_words
  ```
- Sequence-level detectors are skipped automatically when no `lo2_sequences.parquet` file is available.

Selected keys (use `--list-models` to inspect the full list):

| Key | Level | Kurzbeschreibung |
| --- | --- | --- |
| `event_lr_words` | Events | LogisticRegression auf Worttokens (BOW) |
| `event_dt_trigrams` | Events | DecisionTree auf Trigram-Features |
| `event_lsvm_words` | Events | LinearSVM für Worttokens |
| `event_rf_words` | Events | RandomForest für Worttokens |
| `event_lof_words` | Events | LocalOutlierFactor (trainiert nur auf `test_case=correct`) |
| `sequence_lr_numeric` | Sequenzen | LogisticRegression auf `seq_len` + `duration_sec` |
| `sequence_shap_lr_words` | Sequenzen | LogisticRegression auf Worttokens + SHAP-Plot |

Combine the CLI switches like `--if-*` or `--report-*` with `--models` to benchmark how explainability artefacts evolve across different detectors.
