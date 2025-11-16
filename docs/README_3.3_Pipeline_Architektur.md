# Section 3.3 Pipeline-Architektur Documentation

## Overview

This document (`3.3_Pipeline_Architektur.typ`) provides comprehensive German documentation for Section 3.3 of the Master's thesis on explainable anomaly detection for OAuth2/OIDC systems using the LO2 dataset.

## Purpose

The documentation was created by systematically analyzing the actual code implementation in the `mischas114/LogLead` repository and describing:

1. **What was actually implemented** (not a wishlist architecture)
2. **Own implementation work** (LO2Loader, configurations, XAI integration)
3. **LO2-specific adaptations** to the generic LogLead framework

## Document Structure

The document follows the requested skeleton with four main subsections:

### 3.3.1 Loader-Ebene
- **Implementation**: `loglead/loaders/lo2.py` - `LO2Loader` class
- **Key Features**:
  - Service-specific filtering (`single_service`, `service_types`)
  - ML-Sample schema: `(run_id, test_name, service_name)` → log sequence
  - Label generation: `test_case == "correct"` → `normal = True`
  - Label leakage prevention via `trim_init_lines`
  - No cross-service concatenation (per @bakhtin_lo2_2025)
- **CLI Wrapper**: `demo/lo2_e2e/run_lo2_loader.py`

### 3.3.2 Feature-Engineering (Enhancer)
- **Implementation**: 
  - `loglead/enhancers/eventlog.py` - `EventLogEnhancer`
  - `loglead/enhancers/sequence.py` - `SequenceEnhancer`
- **Event-Level Features**:
  - `words()`: Parserless tokenization → `e_words`, `e_words_len`
  - `trigrams()`: Character-level 3-grams → `e_trigrams`, `e_trigrams_len`
  - `parse_drain()`: Template extraction → `e_event_drain_id`
  - `length()`: Message complexity → `e_chars_len`, `e_lines_len`
- **Sequence-Level Features**:
  - `seq_len()`: Event count per sequence
  - `duration()`: Time span → `duration_sec`
  - `tokens()`: Token aggregation across events
- **Trade-off Discussion**: Parserless (fast, robust) vs. Parser-based (compact, interpretable)

### 3.3.3 Anomalie-Detektor
- **Implementation**: 
  - `loglead/anomaly_detection.py` - `AnomalyDetector` class
  - `demo/lo2_e2e/LO2_samples.py` - Model Registry orchestration
- **Supervised Models**:
  - `event_lr_words`: Logistic Regression on word tokens
  - `event_dt_trigrams`: Decision Tree on trigrams (depth=8, min_leaf=10)
  - `event_rf_words`: Random Forest (150 trees, depth=12)
  - `event_xgb_words`: XGBoost (histogram tree, 120 estimators)
  - `sequence_lr_numeric`: LR on `seq_len` and `duration_sec`
- **Unsupervised Models**:
  - `IsolationForest`: Primary baseline (configurable contamination)
  - `event_lof_words`: Local Outlier Factor (trained on correct only)
  - `event_oneclass_svm_words`: One-Class SVM (trained on correct only)
- **Training Strategy**:
  - Service-specific models (no cross-service mixing)
  - Run-based holdout split (20% default, `--sup-holdout-fraction`)
  - Focus on `oauth2-token-Service` (highest F1-scores)
- **Evaluation**: F1-Score, Precision@k, FPR@α, PSI
- **Persistence**: Joblib bundles (`--save-model`, `--save-sup-models`)

### 3.3.4 Erklärbarkeitsschicht (XAI)
- **Implementation**:
  - `loglead/explainer.py` - `NNExplainer`, `ShapExplainer`
  - `demo/lo2_e2e/lo2_phase_f_explainability.py` - Orchestration script
- **NNExplainer**:
  - Cosine similarity-based nearest normal matching
  - Output: `<model>_nn_mapping.csv` (anomalous_id → normal_id)
- **ShapExplainer**:
  - Automatic explainer selection:
    - Tree models → `shap.TreeExplainer`
    - Linear models → `shap.LinearExplainer`
    - Others → `shap.KernelExplainer`
  - Background samples: 256 default (`--shap-background`)
  - Guard rails:
    - Feature threshold: 2000 (`--shap-feature-threshold`)
    - Cell threshold: 2000000 (`--shap-cell-threshold`)
- **Output Artifacts**:
  - Summary plots (PNG)
  - Top features per anomaly (CSV)
  - Aggregation by error label
  - False-positive lists for IsolationForest
- **Developer Focus**: Token names, sequence features (not abstract indices)

## Key Emphases

1. **Own Implementation Work**: Clear documentation of LO2Loader, configuration scripts, and XAI integration
2. **Service-Specific Models**: No cross-service concatenation, per @bakhtin_lo2_2025
3. **oauth2-token Focus**: Highest F1-scores make it the evaluation centerpiece
4. **XAI Integration**: Following @mohale_systematic_2025 and @anthony_designing_2025 recommendations

## Citations Used

- `@mantyla_loglead_2024`: LogLead generic framework
- `@bakhtin_lo2_2025`: LO2 dataset and methodology
- `@mohale_systematic_2025`: XAI/IDS systematic review
- `@anthony_designing_2025`: SHAP-based explainability design
- `@he_drain_2017`: Drain log parsing algorithm (via EventLogEnhancer section)
- `@lundberg_unified_2017`: SHAP framework (via ShapExplainer section)

## File Locations

- **Main Document**: `docs/3.3_Pipeline_Architektur.typ`
- **Implementation Files Analyzed**:
  - `loglead/loaders/lo2.py`
  - `loglead/enhancers/eventlog.py`
  - `loglead/enhancers/sequence.py`
  - `loglead/anomaly_detection.py`
  - `loglead/explainer.py`
  - `loglead/explainability_utils.py`
  - `demo/lo2_e2e/run_lo2_loader.py`
  - `demo/lo2_e2e/LO2_samples.py`
  - `demo/lo2_e2e/lo2_phase_f_explainability.py`

## Language and Style

- **Language**: German (as requested for Master's thesis)
- **Style**: Scientific, technical, complete bullet points expanded into flowing text
- **Format**: Typst markup (ready for direct insertion into thesis)

## Usage

This document can be directly inserted into the Master's thesis. The Typst format includes:
- Section headers: `==` (3.3), `===` (3.3.1-3.3.4)
- Code references: backticks for class/method names
- Citations: `@key` format
- Technical terminology preserved in English where appropriate (e.g., Loader, Enhancer, Pipeline)
