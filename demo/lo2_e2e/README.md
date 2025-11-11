# LO2 E2E Demo Flow

This folder groups the runnable scripts for the LO2 pipeline, covering data loading, enhancement, anomaly detection, and explainability artefacts.

## Quickstart

1. **Load raw runs to Parquet (sequences only by default)**
   ```bash
   python demo/lo2_e2e/run_lo2_loader.py --root /path/to/lo2_data --runs 5 --save-parquet --output-dir demo/result/lo2
   ```
   Der Loader schreibt `lo2_sequences_enhanced.parquet` (und optional mit `--save-base-sequences` auch `lo2_sequences.parquet`). Falls du zusätzlich die Event-Tabelle brauchst, ergänze `--save-events`.
2. **Generate enhancements and anomaly predictions**
   ```bash
   python demo/lo2_e2e/LO2_samples.py --phase full --save-enhancers --save-model models/lo2_if.joblib
   ```
   The `--save-model` argument stores the IsolationForest + vectorizer bundle for reuse; add `--overwrite-model` when replacing an existing dump. To reuse an existing bundle and skip retraining, pass `--load-model models/lo2_if.joblib`.
   Additional opt-in benchmarking aids: `--if-holdout-fraction 0.1 --if-threshold-percentile 99.5 --report-precision-at 100 --report-fp-alpha 0.005 --report-psi --metrics-dir result/lo2/metrics --dump-metadata`.
3. **Create explainability artefacts**
   ```bash
   MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --root demo/result/lo2 --shap-sample 200
   ```

## Sequence hold-out

Alle Modelle (IsolationForest wie auch die supervised Varianten) laufen auf sequenzbasierten Features und reservieren automatisch einen run-basierten Hold-out von 20 % der Runs pro Service/Test-Case. Die wichtigsten Schalter:

- `--sup-holdout-fraction`: Anteil der Run-Gruppen im Hold-out (Standard 0.2, 0 deaktiviert den Split).
- `--sup-holdout-min-groups`: Mindestanzahl an Gruppen pro Bucket, die reserviert werden (Standard 1).
- `--sup-holdout-shuffle`: Zufällige statt zeitlicher Auswahl (nutzt `--sample-seed`).

Beispiel:

```bash
python demo/lo2_e2e/LO2_samples.py --phase full --models event_dt_trigrams --sup-holdout-fraction 0.2
```

### Fast scoring with an existing model

Once you saved a bundle, you can score new data without retraining:

```bash
python demo/lo2_e2e/LO2_samples.py --phase if --load-model models/lo2_if.joblib
```

Phase F can also reuse the model for its artefacts:

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
   --root demo/result/lo2 \
   --load-model models/lo2_if.joblib \
   --shap-sample 200
```

### Saving and reusing supervised models

To save training time, you can persist supervised models trained in LO2_samples.py and reuse them in Phase F:

```bash
# Step 1: Train and save supervised models
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_xgb_words,event_dt_trigrams \
  --sup-holdout-fraction 0.2 \
  --save-sup-models models/supervised

# Step 2: Load saved models in Phase F for explainability (no retraining)
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words,event_xgb_words,event_dt_trigrams \
  --load-sup-models models/supervised \
  --shap-sample 200
```

Each supervised model is saved as `<model_key>.joblib` containing the trained model + vectorizer bundle. This workflow is useful for:
- Running multiple explainability experiments with different SHAP parameters
- Generating explanations for different sample sizes without retraining
- Ensuring consistent models across different analysis runs

All outputs are written beneath `demo/result/lo2` by default. Adjust CLI options to mirror your dataset size and desired sampling behaviour.

## Configurable anomaly detectors

`LO2_samples.py` now behaves like a small testbed: you can mix and match sequence-based anomaly detectors without touching the code.

- Show every available key with `python demo/lo2_e2e/LO2_samples.py --list-models`.
- The default set (`event_lr_words,event_dt_trigrams,sequence_lr_numeric,sequence_shap_lr_words`) mirrors the original Phase E/F baseline – alle Schlüssel arbeiten jetzt auf Sequenzrepräsentationen.
- Pass `--models key1,key2,...` to run a custom line-up. Example:
  ```bash
  python demo/lo2_e2e/LO2_samples.py --models event_lr_words,event_rf_words,event_lof_words,sequence_shap_lr_words
  ```
- Die Pipeline erwartet `lo2_sequences_enhanced.parquet` (oder erzeugt es aus `lo2_sequences.parquet` + optionalen Events).

Selected keys (use `--list-models` to inspect the full list):

| Key | Level | Kurzbeschreibung |
| --- | --- | --- |
| `event_lr_words` | Sequenzen | LogisticRegression auf Worttokens (BOW) |
| `event_dt_trigrams` | Sequenzen | DecisionTree auf Trigram-Features |
| `event_lsvm_words` | Sequenzen | LinearSVM für Worttokens |
| `event_rf_words` | Sequenzen | RandomForest für Worttokens |
| `event_lof_words` | Sequenzen | LocalOutlierFactor (trainiert nur auf `test_case=correct`) |
| `sequence_lr_numeric` | Sequenzen | LogisticRegression auf `seq_len` + `duration_sec` |
| `sequence_shap_lr_words` | Sequenzen | LogisticRegression auf Worttokens + SHAP-Plot |

Combine the CLI switches like `--if-*` or `--report-*` with `--models` to benchmark how explainability artefacts evolve across different detectors.
