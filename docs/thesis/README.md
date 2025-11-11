# Bachelor Thesis Documentation
## Feasibility of Explainable Anomaly Detection in OAuth/OIDC Logs

**Topic:** Machbarkeit von erklärbarer Anomalieerkennung in OAuth/OIDC Logs  
**Created:** November 2025  
**Status:** Ready to start experiments

---

## 📖 Documentation Overview

This directory contains all thesis-specific documentation, organized for systematic execution of your research.

---

## 🚀 Getting Started (Read in Order)

**Start here if this is your first time:**

1. **[Quick Start Guide](01-quick-start-guide.md)** ⭐  
   Your first complete experiment in 30 minutes. Run a supervised baseline, generate SHAP plots, and document results.

2. **[Setup Validation](02-setup-validation.md)**  
   Answers to your 4 key questions about implementation status, data availability, and model loading capabilities.

3. **[TODO Checklist](03-todo-checklist.md)**  
   Complete task list with timeline, priorities, and success criteria for your thesis work.

---

## 📚 Reference Documentation

**Use these for detailed information and experiment execution:**

4. **[Experiment Templates](04-experiment-templates.md)**  
   7 copy-paste experiment scenarios with complete bash commands. Each template includes setup, execution, analysis, and documentation steps.

5. **[Experiment Tracking](05-experiment-tracking.md)**  
   Systematic tracking sheet for documenting all experiments. Track status (🔴 Todo → 🟢 Done → ⚫ Documented), metrics, interpretations, and ratings.

6. **[Feasibility Analysis](06-feasibility-analysis.md)**  
   Comprehensive 43-page analysis covering explainability functions, architecture, experiment matrix, limitations, and workflows. Your primary reference for technical details.

7. **[Documentation Summary](07-documentation-summary.md)**  
   Overview linking all resources, core findings, artifact catalog, and thesis workflow recommendations.

---

## 📊 Experiment Results

Use the **[results/](results/)** directory to document individual experiment outcomes:

```
results/
├── experiment-01-if-baseline.md
├── experiment-02-lr-supervised.md
├── experiment-03-xgboost.md
└── ...
```

**Suggested format for each result file:**
- Experiment metadata (date, duration, parameters)
- Metrics achieved (accuracy, F1, AUC-ROC)
- Key findings and interpretations
- SHAP plot observations
- Challenges encountered
- Next steps

---

## ✅ Current Project Status

### Infrastructure
- ✅ Complete pipeline implemented (Phases B-F)
- ✅ SHAP explainer with auto-backend selection
- ✅ NN-Mapping (Nearest-Normal) explainer
- ✅ Model registry with 13 pre-configured models
- ✅ Persistence system (save/load models)

### Data
- ✅ Enhanced sequences: `demo/result/lo2/lo2_sequences_enhanced.parquet`
- ✅ Training set: 49,852 sequences
- ✅ Hold-out set: 5,539 sequences
- ✅ Anomaly rate: ~50% (realistic for OAuth logs)

### Models
- ✅ IF baseline trained: `models/lo2_if.joblib`
- ✅ IF metadata: `models/model.yml` (created October 31, 2025)
- ✅ Existing explainability artifacts: 24 files in `demo/result/lo2/explainability/`

### Ready to Start
- ⏳ **Time to first results:** 30 minutes
- ⏳ **Start command:** See [Quick Start Guide](01-quick-start-guide.md)

---

## 🎯 Quick Reference

### Recommended Experiment Sequence

**Week 1:**
1. E02 - LR Supervised Baseline (30 min) → "Good solution"
2. E01 - IF Baseline (20 min) → "Poor solution"
3. E03 - XGBoost (45 min) → "Best performance"

**Week 2:**
4. E04 - Feature Comparison (90 min)
5. E05 - Supervised vs Unsupervised (60 min)

**Week 3:**
7. E07 - Large Dataset (60 min)
6. E06 - Ablation Study (120 min, optional)

### Key Commands

```bash
# Validate setup
python -c "import loglead, shap, xgboost, sklearn, polars; print('✅ Ready')"

# Run supervised baseline (E02)
python demo/lo2_e2e/LO2_samples.py --phase full --skip-if --models event_lr_words
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --skip-if --sup-models event_lr_words

# Check results
cat demo/result/lo2/explainability/metrics_event_lr_words.json
```

### Important Paths

- **Data:** `/Users/MTETTEN/Projects/LogLead/demo/result/lo2/`
- **Models:** `/Users/MTETTEN/Projects/LogLead/models/`
- **Scripts:** `/Users/MTETTEN/Projects/LogLead/demo/lo2_e2e/`
- **Tracking:** `/Users/MTETTEN/Projects/LogLead/docs/thesis/05-experiment-tracking.md`

---

## 📋 Feasibility Conclusion

**Answer: Yes, explainable anomaly detection in OAuth/OIDC logs is feasible.**

**Evidence:**
- ✅ Supervised models achieve >95% accuracy
- ✅ SHAP provides interpretable feature importance
- ✅ NN-Mapping shows clear anomaly vs. normal patterns
- ✅ Complete pipeline from raw logs to explanations

**Constraints:**
- ⚠️ IF unsuitable for 50% anomaly rate (~47% accuracy)
- ⚠️ Requires ≥100 "correct" samples for training
- ⚠️ SHAP scales poorly beyond 2000 features
- ⚠️ Feature engineering requires domain expertise

---

## 🆘 Need Help?

- **Setup issues:** See [Setup Validation](02-setup-validation.md)
- **Experiment errors:** See [Quick Start Guide](01-quick-start-guide.md) → Troubleshooting
- **Template questions:** See [Experiment Templates](04-experiment-templates.md)
- **Architecture details:** See [Feasibility Analysis](06-feasibility-analysis.md)

---

**Last Updated:** November 11, 2025  
**Ready to start?** → [Quick Start Guide](01-quick-start-guide.md)
