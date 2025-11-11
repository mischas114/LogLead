# Experiment [NUMBER]: [TITLE]

**Date:** [YYYY-MM-DD]  
**Duration:** [X minutes]  
**Status:** 🟡 In Progress / 🟢 Complete

---

## Experiment Metadata

- **Experiment ID:** E0X
- **Model:** [e.g., event_lr_words, event_xgb_words, if_baseline]
- **Dataset:** `demo/result/lo2/lo2_sequences_enhanced.parquet`
- **Hold-out fraction:** [e.g., 0.2]
- **SHAP samples:** [e.g., 200]
- **Random seed:** [e.g., 42]

---

## Commands Executed

```bash
# Phase E: Training
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --skip-if \
  --models [MODEL_KEY] \
  --sup-holdout-fraction 0.2 \
  --dump-metadata

# Phase F: Explainability
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models [MODEL_KEY] \
  --shap-sample 200 \
  --nn-top-k 50
```

---

## Results

### Performance Metrics

| Metric | Value | Comment |
|--------|-------|---------|
| **Accuracy** | 0.XXXX | |
| **F1-Score** | 0.XXXX | |
| **AUC-ROC** | 0.XXXX | |
| **Precision@100** | 0.XXXX | |
| **False Positives** | XXX | |
| **False Negatives** | XXX | |
| **Training Time** | XX.XX s | |

### Top-10 Features (from SHAP)

1. [feature_name] - [SHAP value or description]
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9. 
10. 

**Feature observations:**
- 
- 
- 

---

## SHAP Analysis

### Summary Plot Observations

**File:** `demo/result/lo2/explainability/[model]_shap_summary.png`

**Key insights:**
- 
- 
- 

### Bar Chart Observations

**File:** `demo/result/lo2/explainability/[model]_shap_bar.png`

**Key insights:**
- 
- 
- 

---

## NN-Mapping Analysis

**File:** `demo/result/lo2/explainability/[model]_nn_mapping.csv`

### Example Mappings

**Anomaly 1:**
- **Anomaly ID:** 
- **Normal ID:** 
- **Key difference:** 
- **Interpretation:** 

**Anomaly 2:**
- **Anomaly ID:** 
- **Normal ID:** 
- **Key difference:** 
- **Interpretation:** 

**Anomaly 3:**
- **Anomaly ID:** 
- **Normal ID:** 
- **Key difference:** 
- **Interpretation:** 

---

## False-Positive Analysis

**File:** `demo/result/lo2/explainability/[model]_false_positives.txt`

### Patterns Identified

1. **Pattern 1:** [Description]
   - **Frequency:** [X occurrences]
   - **Root cause:** 
   - **Potential fix:** 

2. **Pattern 2:** [Description]
   - **Frequency:** [X occurrences]
   - **Root cause:** 
   - **Potential fix:** 

---

## Interpretation

### What Worked Well
- 
- 
- 

### Challenges Encountered
- 
- 
- 

### Surprises / Unexpected Results
- 
- 
- 

### OAuth-Specific Insights
- 
- 
- 

---

## Evaluation

**Performance:** ⭐⭐⭐⭐⭐ (1-5 stars)  
**Interpretability:** ⭐⭐⭐⭐⭐ (1-5 stars)  
**Practicality:** ⭐⭐⭐⭐⭐ (1-5 stars)  

**Overall assessment:**
- 

**Suitability for production:**
- 

**Comparison to previous experiments:**
- 

---

## Thesis Contribution

### Which chapter does this support?
- [ ] Chapter 3 - Methodology
- [ ] Chapter 4 - Results
- [ ] Chapter 5 - Discussion
- [ ] Chapter 6 - Conclusion

### Key quotes/findings for thesis:
> 
> 

### Figures to include:
- [ ] SHAP summary plot
- [ ] SHAP bar chart
- [ ] NN-Mapping example
- [ ] Performance comparison table

---

## Next Steps

**Follow-up experiments:**
- 
- 

**Open questions:**
- 
- 

**Improvements to try:**
- 
- 

---

## Artifacts

**Location:** `demo/result/lo2/explainability/`

- [ ] `[model]_predictions.parquet`
- [ ] `metrics_[model].json`
- [ ] `[model]_shap_summary.png`
- [ ] `[model]_shap_bar.png`
- [ ] `[model]_top_features.txt`
- [ ] `[model]_nn_mapping.csv`
- [ ] `[model]_false_positives.txt`

**Backup location (if needed):**
- 

---

**Completed:** [DATE]  
**Documented in tracking sheet:** [ ] Yes / [ ] No
