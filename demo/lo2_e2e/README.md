# LO2 E2E Demo Flow

This folder provides both legacy scripts and a new **declarative pipeline architecture** for LO2 data processing, anomaly detection, and explainability.

## New: Declarative Pipeline (Recommended)

The refactored pipeline eliminates IF-based control flow and provides unified explainability for Decision Trees, Random Forests, and XGBoost.

### Quick Start with Declarative Pipeline

1. **Run a complete pipeline from configuration:**
   ```bash
   python -m demo.lo2_e2e.cli run --pipeline demo/lo2_e2e/config/pipeline.yaml
   ```

2. **List available components:**
   ```bash
   python -m demo.lo2_e2e.cli list
   ```

### Key Features

- ✅ **No IF Branching**: Model and step selection via registry-based dispatch
- ✅ **Glass-Box Decision Trees**: Complete decision paths with thresholds and node details
- ✅ **Unified Explainability**: Consistent interface for DT, RF, and XGBoost
- ✅ **Configuration-Driven**: Add new models by editing YAML, not code
- ✅ **SHAP Support**: Optional SHAP with graceful degradation

### Architecture

```
demo/lo2_e2e/
├── core/
│   ├── registry.py      # Component registration system
│   ├── adapters.py      # Model adapters (DT, RF, XGB)
│   ├── explainers.py    # Unified explainer interface
│   └── runner.py        # Config-driven pipeline executor
├── steps/
│   ├── load_data.py     # Data loading step
│   ├── preprocess.py    # Preprocessing step
│   ├── predict.py       # Prediction step
│   └── explain.py       # Explanation step
├── config/
│   ├── pipeline.yaml    # Pipeline configuration
│   └── models.yaml      # Model adapter configuration
├── cli.py               # Command-line interface
└── docs/
    └── EXPLANATIONS.md  # Explanation format documentation
```

### Example Pipeline Configuration

```yaml
pipeline:
  - step: load_data
    with:
      sequences_path: "demo/result/lo2/lo2_sequences_enhanced.parquet"
  
  - step: preprocess
    with:
      feature_columns: ["seq_len", "duration_sec"]
  
  - step: predict
    with:
      model: "dt_v1"
      models_config: "demo/lo2_e2e/config/models.yaml"
  
  - step: explain
    with:
      model: "dt_v1"
      max_samples: 10
      output_file: "demo/result/lo2/explanations.jsonl"
```

### Adding a New Model

No code changes needed - just add to `config/models.yaml`:

```yaml
models:
  my_new_rf:
    adapter: random_forest
    path: "path/to/model.joblib"
    description: "My custom RF model"
```

### Explanation Output

Explanations are saved as JSONL with complete decision paths, feature contributions, and human-readable summaries. See [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md) for format details.

## Legacy Scripts (Original Pipeline)

### Quickstart

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
