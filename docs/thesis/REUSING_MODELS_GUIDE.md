# Guide: Reusing Trained Models for Explainability

## Overview

This guide explains how to **save trained supervised models** from Phase E (LO2_samples.py) and **reuse them** in Phase F (lo2_phase_f_explainability.py) for explainability experiments. This workflow significantly reduces training time and ensures consistency across experiments.

## Why Reuse Models?

### Time Savings
- Training a supervised model takes ~30 seconds per model
- SHAP explanation generation takes ~10-60 seconds depending on sample size
- With model reuse, you can iterate on explainability parameters without retraining

### Consistency
- Same model weights across multiple explainability runs
- Reproducible results when comparing different SHAP configurations
- Eliminate training variance between runs

### Flexibility
- Experiment with different `--shap-sample` sizes
- Adjust `--shap-background`, `--shap-feature-threshold`, `--shap-cell-threshold`
- Generate multiple visualizations from the same model

## Complete Workflow

### Phase E: Train and Save Models

Use `LO2_samples.py` to train supervised models and save them to disk:

```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_xgb_words,event_dt_trigrams \
  --sup-holdout-fraction 0.2 \
  --sup-holdout-shuffle \
  --save-sup-models models/supervised \
  --save-model models/lo2_if.joblib
```

**What this does:**
- Trains 3 supervised models: LogisticRegression, XGBoost, DecisionTree
- Uses 20% run-based holdout for validation
- Saves each model to `models/supervised/<model_key>.joblib`
- Also saves IsolationForest to `models/lo2_if.joblib`

**Output files:**
```
models/
├── supervised/
│   ├── event_lr_words.joblib
│   ├── event_xgb_words.joblib
│   └── event_dt_trigrams.joblib
└── lo2_if.joblib
```

Each `.joblib` file contains:
- Trained model (sklearn classifier)
- Fitted vectorizer (CountVectorizer or TfidfVectorizer)

### Phase F: Load Saved Models for Explainability

Use `lo2_phase_f_explainability.py` to load pre-trained models and generate explanations:

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words,event_xgb_words,event_dt_trigrams \
  --load-sup-models models/supervised \
  --shap-sample 200 \
  --sup-holdout-fraction 0.2
```

**What this does:**
- Loads pre-trained models from `models/supervised/`
- Skips retraining (saves ~90 seconds)
- Generates SHAP explanations for 200 samples per model
- Uses the same 20% holdout split for evaluation

**Output files:**
```
demo/result/lo2/explainability/
├── event_lr_words_predictions.parquet
├── event_lr_words_shap.png
├── event_lr_words_top_features.txt
├── event_xgb_words_predictions.parquet
├── event_xgb_words_shap.png
├── event_xgb_words_top_features.txt
├── event_dt_trigrams_predictions.parquet
├── event_dt_trigrams_shap.png
└── event_dt_trigrams_top_features.txt
```

## Advanced Use Cases

### Experiment 1: Small SHAP Sample (Fast)
```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --load-sup-models models/supervised \
  --shap-sample 100
```
Time saved: ~30 seconds (no retraining)

### Experiment 2: Large SHAP Sample (Comprehensive)
```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --load-sup-models models/supervised \
  --shap-sample 500
```
Time saved: ~30 seconds (no retraining)

### Experiment 3: Different SHAP Background Size
```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --load-sup-models models/supervised \
  --shap-sample 200 \
  --shap-background 512
```
Time saved: ~30 seconds (no retraining)

### Experiment 4: Multiple Models in Parallel
```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words,event_xgb_words,event_dt_trigrams \
  --load-sup-models models/supervised \
  --shap-sample 200
```
Time saved: ~90 seconds (no retraining for 3 models)

## Model Compatibility

### When Models Can Be Reused ✅
- Same dataset (`lo2_sequences_enhanced.parquet`)
- Same features (e.g., `e_words`, `e_trigrams`)
- Same holdout configuration (`--sup-holdout-fraction`, `--sup-holdout-shuffle`, `--sample-seed`)
- Same model hyperparameters

### When You Must Retrain ❌
- Dataset changed (new runs loaded)
- Features changed (different enhancement pipeline)
- Different holdout settings (e.g., changed from 0.2 to 0.3)
- Different random seed
- Model hyperparameters changed

## Command Reference

### LO2_samples.py (Phase E)

| Argument | Description | Example |
|----------|-------------|---------|
| `--save-sup-models` | Directory to save supervised models | `models/supervised` |
| `--models` | Comma-separated model keys to train | `event_lr_words,event_xgb_words` |
| `--sup-holdout-fraction` | Fraction for holdout (0.0-0.5) | `0.2` |
| `--sup-holdout-shuffle` | Shuffle runs instead of temporal split | Flag |
| `--sample-seed` | Random seed for reproducibility | `42` |

### lo2_phase_f_explainability.py (Phase F)

| Argument | Description | Example |
|----------|-------------|---------|
| `--load-sup-models` | Directory with saved supervised models | `models/supervised` |
| `--sup-models` | Comma-separated model keys to load | `event_lr_words,event_xgb_words` |
| `--shap-sample` | Max samples for SHAP (0 = all) | `200` |
| `--shap-background` | Background samples for SHAP | `256` |
| `--skip-if` | Skip IsolationForest (supervised only) | Flag |

## Troubleshooting

### Problem: "Kein gespeichertes Modell gefunden"
**Cause:** Model file doesn't exist at expected path

**Solution:**
```bash
# Check if models directory exists
ls -la models/supervised/

# Re-run Phase E to save models
python demo/lo2_e2e/LO2_samples.py --phase full --save-sup-models models/supervised
```

### Problem: "Bundle-Format ungültig"
**Cause:** Corrupted or incompatible model file

**Solution:**
```bash
# Delete corrupted files
rm models/supervised/*.joblib

# Retrain and save fresh models
python demo/lo2_e2e/LO2_samples.py --phase full --save-sup-models models/supervised
```

### Problem: Model loaded but predictions are different
**Cause:** Dataset or features changed

**Solution:**
```bash
# Retrain models with current dataset
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --save-sup-models models/supervised \
  --sup-holdout-fraction 0.2
```

### Problem: "Spalte 'test_case' fehlt"
**Cause:** Model requires train_selector='correct_only' but data lacks test_case column

**Solution:**
```bash
# Ensure data has test_case column from loader
python demo/lo2_e2e/run_lo2_loader.py --root /path/to/lo2 --save-parquet
```

## Best Practices

### 1. Version Your Models
```bash
# Use descriptive directories with dates
--save-sup-models models/supervised_2025-11-11
```

### 2. Document Training Parameters
```bash
# Save metadata alongside models
echo "Trained with: --sup-holdout-fraction 0.2 --sample-seed 42" > models/supervised/README.txt
```

### 3. Test Model Loading
```bash
# Quick test with small sample
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --load-sup-models models/supervised \
  --shap-sample 10
```

### 4. Clean Up Old Models
```bash
# Remove outdated models
rm -rf models/supervised_old/
```

## Performance Comparison

### Without Model Reuse (Traditional)
```bash
# Phase F trains models fresh every time
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words,event_xgb_words,event_dt_trigrams \
  --shap-sample 200
```
**Time:** ~120 seconds (90s training + 30s SHAP)

### With Model Reuse (Optimized)
```bash
# Step 1: Train once (Phase E)
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_xgb_words,event_dt_trigrams \
  --save-sup-models models/supervised

# Step 2: Reuse many times (Phase F)
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words,event_xgb_words,event_dt_trigrams \
  --load-sup-models models/supervised \
  --shap-sample 200
```
**Time:** ~120 seconds (first run) + ~30 seconds (subsequent runs)

**Speedup for 5 explainability experiments:**
- Traditional: 5 × 120s = **600 seconds** (10 minutes)
- Optimized: 120s + 4 × 30s = **240 seconds** (4 minutes)
- **Time saved: 360 seconds (6 minutes, 60% reduction)**

## Integration with Thesis Workflow

This feature is particularly useful for thesis experiments:

### Experiment E02: LR Supervised Baseline
```bash
# Train and save
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words \
  --sup-holdout-fraction 0.2 \
  --save-sup-models models/supervised

# Generate explainability (can run multiple times)
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --load-sup-models models/supervised \
  --shap-sample 200 \
  --sup-holdout-fraction 0.2
```

### Experiment E03: XGBoost Comparison
```bash
# Load existing model, no retraining
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_xgb_words \
  --load-sup-models models/supervised \
  --shap-sample 200 \
  --sup-holdout-fraction 0.2
```

### Experiment E05: SHAP Sample Size Sensitivity
```bash
# Same model, different SHAP samples
for size in 50 100 200 500; do
  MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
    --root demo/result/lo2 \
    --skip-if \
    --sup-models event_lr_words \
    --load-sup-models models/supervised \
    --shap-sample $size \
    --sup-holdout-fraction 0.2
  
  # Rename outputs
  mv demo/result/lo2/explainability/event_lr_words_shap.png \
     demo/result/lo2/explainability/event_lr_words_shap_n${size}.png
done
```

## Summary

The model reuse feature enables:

1. ⚡ **Faster experimentation:** Skip retraining (save ~30s per model)
2. 🔄 **Reproducibility:** Same model across multiple runs
3. 🧪 **Flexibility:** Iterate on explainability parameters
4. 📊 **Consistency:** Compare SHAP plots with identical models
5. 💾 **Resource efficiency:** Train once, explain many times

**Recommended workflow:**
1. Train and save models in Phase E with `--save-sup-models`
2. Iterate on explainability in Phase F with `--load-sup-models`
3. Document model training parameters for reproducibility
4. Retrain only when dataset or configuration changes

For questions or issues, see:
- `docs/thesis/01-quick-start-guide.md` - Quick start examples
- `docs/thesis/03-todo-checklist.md` - Experiment checklist
- `demo/lo2_e2e/README.md` - Pipeline documentation
