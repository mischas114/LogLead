# Quick Start Guide
## Your First Thesis Experiment in 30 Minutes

**Goal:** Complete experiment with interpretable results

---

## Prerequisites (5 minute check)

```bash
# 1. Check Python dependencies
python -c "import loglead, shap, xgboost, sklearn, polars; print('✅ All dependencies OK')"

# 2. Data available?
ls -lh demo/result/lo2/lo2_sequences_enhanced.parquet
# Should be several MB

# 3. IF model available?
ls -lh models/lo2_if.joblib models/model.yml
# Both should exist
```

**If errors occur:** See setup guide in `docs/lo2-e2e-setup.md`

---

## Option 1: Supervised Baseline (Recommended - 30 minutes)

**What you'll get:**
- Accuracy >90% (typically ~97%)
- Interpretable SHAP plots
- Top-20 OAuth features identified
- Nearest-normal mapping for anomalies
- False-positive analysis

### Step 1: Training (5 minutes)

```bash
cd /Users/MTETTEN/Projects/LogLead

python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --skip-if \
  --models event_lr_words \
  --sup-holdout-fraction 0.2 \
  --dump-metadata \
  --save-model experiments/exp02_lr_supervised/model.joblib
```

**Expected output:**
```
[event_lr_words] LogisticRegression auf Sequenz-Worttokens (Bag-of-Words).
  -> Hold-out aktiv: 1 Gruppen, 123 Zeilen.
[Resource] event_lr_words: time=2.34s, features=1234, vocab=5678
[Evaluate] event_lr_words
  Accuracy: 0.9756 | AUC-ROC: 0.9912 | F1: 0.9623
```

### Step 2: Explainability (10 minutes)

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --shap-sample 200 \
  --nn-top-k 50 \
  --nn-normal-sample 100
```

**Expected output:**
```
[INFO] Trainiere event_lr_words ...
[INFO] Erkläre event_lr_words mit SHAP (200 Samples) ...
[INFO] SHAP-Plots gespeichert: event_lr_words_shap_summary.png
[INFO] Top-Features: event_lr_words_top_features.txt
[INFO] NN-Mapping: 50 Anomalien → Nearest Normal
```

### Step 3: Inspect Results (5 minutes)

```bash
# View metrics
cat demo/result/lo2/explainability/metrics_event_lr_words.json

# Top-10 features
head -10 demo/result/lo2/explainability/event_lr_words_top_features.txt

# Inspect predictions
python -c "
import polars as pl
df = pl.read_parquet('demo/result/lo2/explainability/event_lr_words_predictions.parquet')
print('Total sequences:', df.height)
print('Anomalies detected:', df['pred_ano'].sum())
print('Accuracy:', ((df['pred_ano'] == df['anomaly']).sum() / df.height).round(4))
"

# View SHAP plots (opens image)
open demo/result/lo2/explainability/event_lr_words_shap_summary.png
```

### Step 4: Document (10 minutes)

```bash
# Open tracking file
open docs/THESIS_EXPERIMENT_TRACKING.md
```

**Fill in (Experiment E02):**
1. Status: 🔴 → 🟢 Done
2. Transfer metrics from `metrics_event_lr_words.json`
3. Note top-5 features from `event_lr_words_top_features.txt`
4. Add rating:
   - Performance: ⭐⭐⭐⭐⭐ (>95% accuracy)
   - Interpretability: ⭐⭐⭐⭐⭐ (SHAP + LR coefficients consistent)
   - Practicality: ⭐⭐⭐⭐⭐ (fast, reliable)

**Write interpretation:**
```markdown
#### Interpretation
Logistic Regression achieves excellent performance (97% accuracy) with
full interpretability. Top features correlate directly with OAuth error
codes (e.g., "invalid_grant", "token_expired"). SHAP values and native
LR coefficients align, building trust in explanations. NN-Mapping shows
clear differences between anomalies (contain error tokens) and normals
(contain success tokens).

**Conclusion:** Excellent as baseline and for production environments.
```

---

## Option 2: IF Baseline + Explainability (20 minutes)

**What you'll get:**
- IF performance benchmark (~47% accuracy with 50% anomalies)
- SHAP plots for unsupervised approach
- Demonstration of limitations
- Comparison data for "poor solution"

### Step 1: Generate IF Explainability (15 minutes)

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --load-model models/lo2_if.joblib \
  --shap-sample 200 \
  --nn-top-k 50 \
  --sup-models event_lr_words
```

**Note:** `--sup-models event_lr_words` is included for comparison, can be omitted

### Step 2: Inspect IF Results (5 minutes)

```bash
# IF metadata
cat models/model.yml

# IF predictions (if available)
python -c "
import polars as pl
df = pl.read_parquet('demo/result/lo2/explainability/lo2_if_predictions.parquet')
print('IF Accuracy:', ((df['pred_ano'] == df['anomaly']).sum() / df.height).round(4))
print('Precision@100:', df.sort('score_if', descending=True).head(100)['anomaly'].mean().round(4))
"

# SHAP plots
open demo/result/lo2/explainability/if_shap_summary.png
```

### Step 3: Document (see Option 1)

**Key insights for E01:**
```markdown
#### Interpretation
IsolationForest achieves only ~47% accuracy (close to random guess with 50% anomaly rate).
This is because IF is optimized for outlier detection with <10% anomalies.
For OAuth logs with 50% error sequences, IF is structurally unsuitable.

SHAP explanations are available but hard to interpret, as feature
importance varies widely without clear patterns.

**Conclusion:** Unsuitable as main classifier. Shows necessity of supervised approaches.
```

---

## Comparison of Both Options

| Aspect | Option 1 (LR) | Option 2 (IF) |
|--------|---------------|---------------|
| **Time required** | 30 min | 20 min |
| **Accuracy** | ~97% | ~47% |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Thesis value** | "Good solution" | "Poor solution" |
| **Recommendation** | **Start here** | For comparison |

---

## After Your First Experiment

### What you now have:
- Complete documented experiment
- Interpretable results (SHAP plots, features, metrics)
- Baseline for further comparisons
- Material for thesis chapter 4 (evaluation)

### Next steps:
```bash
# Experiment E03: XGBoost (tree-based)
python demo/lo2_e2e/LO2_samples.py --models event_xgb_words
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --sup-models event_xgb_words

# Experiment E04: Feature comparison
python demo/lo2_e2e/LO2_samples.py --models event_lr_words,event_dt_trigrams,sequence_lr_numeric
```

See complete list in: `docs/THESIS_TODO_CHECKLIST.md`

---

## Troubleshooting

### Problem: "No module named 'shap'"
```bash
pip install shap
```

### Problem: "SHAP skipped due to feature threshold"
```bash
# Disable guards
python ... --shap-feature-threshold 0 --shap-cell-threshold 0
```

### Problem: "Out of memory"
```bash
# Reduce samples
python ... --shap-sample 100 --shap-background 128
```

### Problem: "No file lo2_sequences_enhanced.parquet"
```bash
# Re-run loader
python demo/lo2_e2e/run_lo2_loader.py \
  --root ~/Data/LO2 \
  --runs 5 \
  --save-parquet \
  --output-dir demo/result/lo2
```

---

## Further Resources

- **Complete guide:** `docs/THESIS_DOCUMENTATION_SUMMARY.md`
- **Experiment templates:** `docs/THESIS_EXPERIMENT_TEMPLATES.md`
- **Tracking system:** `docs/THESIS_EXPERIMENT_TRACKING.md`
- **TODO checklist:** `docs/THESIS_TODO_CHECKLIST.md`
- **Feasibility analysis:** `docs/THESIS_MACHBARKEIT_ANALYSIS.md`

---

**Good luck! 🚀**

Questions? See `docs/THESIS_DOCUMENTATION_SUMMARY.md` section "Troubleshooting & FAQ"
