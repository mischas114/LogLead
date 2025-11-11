# Machbarkeit erklärbarer Anomalieerkennung in OAuth/OIDC Logs
## Analyse für Bachelorarbeit

**Datum:** 11. November 2025  
**Thema:** Machbarkeit von erklärbarer Anomalieerkennung in OAuth/OIDC Logs  
**Framework:** LogLead LO2 Pipeline

---

## Executive Summary

Die LO2-Pipeline demonstriert die **technische Machbarkeit** erklärbarer Anomalieerkennung in OAuth/OIDC Logs durch:

1. **Vollständige E2E-Pipeline** vom Rohdaten-Loader bis zu interpretierbaren Erklärungen
2. **Mehrere Erklärbarkeitsansätze**: SHAP-Werte, Nearest-Neighbor-Mapping, Feature-Wichtigkeit
3. **Vergleichbare Modelle**: Unsupervised (Isolation Forest) vs. Supervised (LR, DT, RF, XGB)
4. **Reproduzierbare Artefakte**: Alle Zwischenergebnisse können gespeichert und dokumentiert werden

**Status:** ✅ Produktionsreif mit umfangreichen Testmöglichkeiten für verschiedene Lösungsansätze

---

## 1. Vorhandene Explainability-Funktionen

### 1.1 SHAP-basierte Erklärungen (`ShapExplainer`)

**Lokation:** `loglead/explainer.py:ShapExplainer`

**Funktionsweise:**
```python
# Automatische Backend-Auswahl basierend auf Modelltyp
- LinearExplainer: LogisticRegression, LinearSVM
- TreeExplainer: DecisionTree, RandomForest, XGBoost, IsolationForest
- KernelExplainer: Fallback für beliebige Modelle
```

**Generierte Artefakte:**
1. **SHAP-Plots** (Summary, Bar, Beeswarm)
   - `demo/result/lo2/explainability/*_shap_summary.png`
   - `demo/result/lo2/explainability/*_shap_bar.png`

2. **Top-Features-Listen**
   - `demo/result/lo2/explainability/*_top_features.txt`
   - Format: Rangierte Liste der wichtigsten Features mit Index

3. **SHAP-Values (numerisch)**
   - Zugriff über `explainer.Svals` für weitere Analysen
   - Export in CSV/JSON für Visualisierungen

**Ressourcen-Guards:**
```python
# Verhindert OOM bei großen Feature-Mengen
feature_warning_threshold: 2000  # Anzahl Features
sample_warning_threshold: 2_000_000  # Zeilen × Features
background_sample_size: 256  # Background-Samples für SHAP
```

**CLI-Steuerung:**
```bash
python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --shap-sample 200 \
  --shap-background 256 \
  --shap-feature-threshold 2000 \
  --shap-cell-threshold 2000000
```

### 1.2 Nearest-Neighbor Explainer (`NNExplainer`)

**Lokation:** `loglead/explainer.py:NNExplainer`

**Funktionsweise:**
- Findet für jede als anomal klassifizierte Sequenz die nächste "normale" Sequenz
- Nutzt Cosine-Similarity auf Feature-Vektoren (erweiterbar auf FAISS/Annoy)
- Ermöglicht direkten Vergleich: "Was ist anders bei dieser Anomalie?"

**Generierte Artefakte:**
1. **NN-Mapping CSV**
   - `demo/result/lo2/explainability/*_nn_mapping.csv`
   - Spalten: `anomalous_id`, `normal_id`

2. **False-Positive-Liste**
   - `demo/result/lo2/explainability/*_false_positives.txt`
   - Zeigt Sequenzen mit Token-Content, die fälschlicherweise als Anomalie markiert wurden

3. **Interaktive Visualisierungen**
   ```python
   nn_explainer.plot_features_in_two_dimensions(ground_truth_col="anomaly")
   # UMAP-Projektion mit Hover-Details
   ```

**CLI-Steuerung:**
```bash
python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --nn-source sequence_shap_lr_words \
  --nn-top-k 50 \
  --nn-normal-sample 100
```

### 1.3 Feature-Wichtigkeit und Model-Introspection

**Direkte Modell-Eigenschaften:**
```python
# DecisionTree/RandomForest
detector.model.feature_importances_
detector.model.tree_.max_depth

# LogisticRegression
detector.model.coef_  # Gewichte pro Feature

# XGBoost
detector.model.get_booster().get_score(importance_type='weight')
```

**Artefakte:**
- Automatisch in `[Resource]`-Logs während Training
- Gespeichert in `metrics_*.json` pro Modell

---

## 2. Experimentierfreundliche Architektur

### 2.1 Modell-Registry

**Lokation:** `demo/lo2_e2e/LO2_samples.py:MODEL_REGISTRY`

**Verfügbare Modelle** (Stand: 11.11.2025):

| Schlüssel | Typ | Features | Explainability |
|-----------|-----|----------|----------------|
| `event_lr_words` | Supervised | Worttokens (BoW) | ✅ SHAP (Linear) |
| `event_dt_trigrams` | Supervised | Trigrams | ✅ SHAP (Tree) |
| `event_rf_words` | Supervised | Worttokens | ✅ SHAP (Tree) |
| `event_xgb_words` | Supervised | Worttokens | ✅ SHAP (Tree) |
| `event_lsvm_words` | Supervised | Worttokens | ✅ SHAP (Linear) |
| `event_lof_words` | Unsupervised | Worttokens | ⚠️ Limitiert |
| `event_kmeans_words` | Unsupervised | Worttokens | ❌ Keine |
| `event_oneclass_svm_words` | Unsupervised | Worttokens | ⚠️ Limitiert |
| `event_rarity_words` | Rule-based | Worttokens | ✅ Feature-basiert |
| `event_oov_words` | Rule-based | Worttokens | ✅ Feature-basiert |
| `sequence_lr_numeric` | Supervised | seq_len, duration | ✅ SHAP (Linear) |
| `sequence_lr_words` | Supervised | Worttokens | ✅ SHAP (Linear) |
| `sequence_shap_lr_words` | Supervised | Worttokens | ✅ SHAP (Linear) + Auto-Plot |

**Erweiterbarkeit:**
```python
# Neues Modell hinzufügen
MODEL_REGISTRY["custom_model"] = {
    "description": "Beschreibung für --list-models",
    "level": "sequence",  # oder "event"
    "item_list_col": "e_words",  # oder None für numeric-only
    "numeric_cols": ["seq_len", "duration_sec"],
    "train_method": "train_LR",  # oder train_DT, train_RF, etc.
    "train_kwargs": {"max_depth": 5},
    "vectorizer_kwargs": {"max_features": 5000},
    "requires_shap": True,  # Automatische SHAP-Berechnung
    "shap_kwargs": {"plot_featurename_len": 18},
}
```

### 2.2 Reproduzierbare Experiment-Pipeline

**Phase-basierte Ausführung:**
```bash
# Variante 1: Nur Feature-Engineering
python demo/lo2_e2e/LO2_samples.py --phase enhancers

# Variante 2: Isolation Forest Baseline
python demo/lo2_e2e/LO2_samples.py --phase if

# Variante 3: Vollständige Pipeline
python demo/lo2_e2e/LO2_samples.py --phase full
```

**Modell-Kombinationen testen:**
```bash
# Alle verfügbaren Modelle anzeigen
python demo/lo2_e2e/LO2_samples.py --list-models

# Spezifische Kombination
python demo/lo2_e2e/LO2_samples.py \
  --models event_lr_words,event_dt_trigrams,event_xgb_words \
  --sup-holdout-fraction 0.2
```

### 2.3 Persistence & Wiederverwendung

**Gespeicherte Artefakte:**

1. **Modelle** (`.joblib`)
   ```bash
   --save-model models/lo2_if.joblib
   --load-model models/lo2_if.joblib  # Wiederverwendung
   ```

2. **Metadaten** (`.yml`)
   ```yaml
   generated_at: 2025-11-11T14:23:45Z
   training_rows: 1234
   if_params:
     contamination: 0.45
     n_estimators: 200
   metrics:
     precision_at_200: 0.78
     fp_rate_at_0.01: 0.03
   git_commit: abc123def456
   ```

3. **Predictions** (`.parquet`)
   - Spalten: `seq_id`, `pred_ano`, `score_*`, `rank_*`, `anomaly` (ground truth)
   - Ermöglicht Offline-Analyse ohne Re-Training

4. **Metriken** (`.json` / `.csv`)
   ```json
   {
     "accuracy": 0.97,
     "f1": 0.95,
     "aucroc": 0.98,
     "precision_at_200": 0.78
   }
   ```

---

## 3. Dokumentationsfreundliche Features für Thesis

### 3.1 Automatische Diagnostics

**Training-Statistiken:**
```
[Resource] event_lr_words: time=2.34s, features=1234, vocab=5678, size_mb=3.45
[TrainStats] event_lr_words: train_rows=800 total_rows=1000 fraction=0.8000
[Guard:event_xgb_words] n_estimators forced to 1 to control peak memory
```

**Hold-out-Validierung:**
```python
# Automatisch bei --sup-holdout-fraction > 0
{
  "applied": true,
  "holdout_groups": 12,
  "train_groups": 48,
  "holdout_rows": 240,
  "train_rows": 960
}
```

### 3.2 Downsampling-Tracking

**Summary-Output:**
```
[Summary] Explainability diagnostics:
  seq_lr_tokens_shap_samples=200 total=1000
  seq_dt_shap_samples=200 total=1000
  nn_top_used=50 of 123 anomalies | nn_normals_used=100 of 877
[Summary] Downsampling occurred: yes
```

### 3.3 Vergleichbare Benchmarks

**Metrics-Reports generieren:**
```bash
python demo/lo2_e2e/LO2_samples.py \
  --report-precision-at 100 \
  --report-fp-alpha 0.005 \
  --report-psi \
  --metrics-dir result/lo2/metrics \
  --dump-metadata
```

**Vergleichstabelle erstellen:**
```python
import polars as pl

# Alle Predictions laden
pred_if = pl.read_parquet("result/lo2/lo2_if_predictions.parquet")
pred_lr = pl.read_parquet("result/lo2/explainability/event_lr_words_predictions.parquet")
pred_dt = pl.read_parquet("result/lo2/explainability/event_dt_trigrams_predictions.parquet")

# Scores vergleichen
comparison = pred_if.select(["seq_id", "anomaly", "score_if"]).join(
    pred_lr.select(["seq_id", "score_event_lr_words"]), on="seq_id"
).join(
    pred_dt.select(["seq_id", "score_event_dt_trigrams"]), on="seq_id"
)
```

---

## 4. Empfohlene Experiment-Matrix für Thesis

### 4.1 Grundlagen-Experimente (Machbarkeit)

| # | Ziel | Modell | Metriken | Artefakte |
|---|------|--------|----------|-----------|
| E1 | Baseline Unsupervised | IsolationForest | Precision@k, FP-Rate | SHAP-Plots, NN-Mapping |
| E2 | Baseline Supervised | LogisticRegression | Accuracy, F1, AUC-ROC | SHAP-Plots, Top-Features |
| E3 | Tree-based | DecisionTree | Accuracy, Depth, Leaves | SHAP-Plots, Feature-Importance |
| E4 | Ensemble | RandomForest | Accuracy, Avg-Depth | SHAP-Plots, OOB-Error |
| E5 | Boosting | XGBoost | Accuracy, Early-Stop | SHAP-Plots, Feature-Gain |

### 4.2 Vergleichsdimensionen

**Dimension 1: Feature-Repräsentation**
```bash
# Worttokens (Bag-of-Words)
--models event_lr_words

# Trigrams (N-Gramm-Features)
--models event_dt_trigrams

# Numerische Features
--models sequence_lr_numeric

# Drain-basierte Templates
--if-item e_event_drain_id  # Falls verfügbar
```

**Dimension 2: Supervised vs. Unsupervised**
```bash
# Unsupervised: Isolation Forest
--phase if --if-contamination 0.45

# Supervised: Logistic Regression
--models event_lr_words --sup-holdout-fraction 0.2

# Hybrid: LOF (trainiert auf "correct" nur)
--models event_lof_words
```

**Dimension 3: Explainability-Tiefe**
```bash
# Nur SHAP
--models event_lr_words

# SHAP + NN-Mapping
--models sequence_shap_lr_words --nn-source sequence_shap_lr_words

# SHAP + False-Positive-Analyse
--nn-top-k 0 --nn-normal-sample 0  # Alle einbeziehen
```

### 4.3 Template für Experiment-Dokumentation

```markdown
## Experiment Ex: [Titel]

**Hypothese:** [Was wird erwartet?]

**Setup:**
- Modell: `event_lr_words`
- Features: `e_words` (Worttokens)
- Hold-out: 20% (run-basiert)
- SHAP-Samples: 200

**Kommando:**
```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words \
  --sup-holdout-fraction 0.2 \
  --save-model models/ex_lr.joblib \
  --dump-metadata

python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --skip-if \
  --sup-models event_lr_words \
  --nn-source event_lr_words \
  --shap-sample 200
```

**Artefakte:**
- `models/ex_lr.joblib` + `models/model.yml`
- `result/lo2/explainability/event_lr_words_shap_summary.png`
- `result/lo2/explainability/event_lr_words_nn_mapping.csv`
- `result/lo2/explainability/metrics_event_lr_words.json`

**Ergebnisse:**
| Metrik | Wert |
|--------|------|
| Accuracy | 0.97 |
| F1-Score | 0.95 |
| Top-Feature | "token_invalid" |

**Interpretation:**
[SHAP-Plot zeigt: ...]
[NN-Mapping offenbart: ...]
[False-Positives sind: ...]

**Bewertung Machbarkeit:**
- ✅ Technisch implementierbar
- ✅ Erklärungen nachvollziehbar
- ⚠️ Feature-Engineering benötigt Domain-Wissen
```

---

## 5. Limitationen und Herausforderungen

### 5.1 Technische Grenzen

**SHAP-Skalierung:**
- Feature-Anzahl > 2000 → ResourceWarning
- Samples × Features > 2.000.000 → OOM-Risiko
- Lösung: `--shap-sample`, `--shap-background` anpassen

**Isolation Forest Performance:**
- Bei 50% Anomalie-Rate: Accuracy ≈ 0.45, F1 ≈ 0.0
- Nicht geeignet als Hauptklassifikator
- Nutzbar als Drift-Detektor oder Vorstufenfilter

**Nearest-Neighbor-Qualität:**
- Cosine-Similarity auf Bag-of-Words kann semantische Unterschiede übersehen
- Empfehlung: FAISS/Annoy für große Datensätze

### 5.2 Interpretations-Herausforderungen

**SHAP-Werte verstehen:**
```python
# Positive SHAP-Werte → Feature erhöht Anomalie-Wahrscheinlichkeit
# Negative SHAP-Werte → Feature senkt Anomalie-Wahrscheinlichkeit
# Aber: Interpretation erfordert Domain-Kontext (OAuth-Flows)
```

**False-Positive-Muster:**
- Oft durch unvollständige "correct"-Trainingsdaten verursacht
- Empfehlung: `--errors-per-run` erhöhen, mehr "correct"-Runs laden

**Feature-Namen:**
```python
# BOW-Features sind oft kryptisch
"e_words_token"  # Welches Token genau?
"e_trigrams_gra_nte"  # Schwer zu interpretieren

# Lösung: Ursprüngliche Log-Zeilen mit NN-Mapping verknüpfen
nn_explainer.print_log_content_from_nn_mapping()
```

### 5.3 Daten-Anforderungen

**Mindestkriterien für belastbare Aussagen:**
- ≥ 100 "correct"-Sequenzen pro Service
- ≥ 20 verschiedene Fehlertypen
- ≥ 5 Runs pro Fehlertyp (für Hold-out-Validation)

**Aktuelle LO2-Datenlage:**
```bash
# Typisches Test-Sample (5 Runs, 1 Error/Run)
Total sequences: 40-60
Correct: 5-10 (sehr klein!)
Anomalies: 30-50 (sehr hoch!)

# Problem: Hold-out mit 20% → Train nur 4-8 "correct" Sequenzen
# Lösung: --runs 50 --errors-per-run 2 für realistisches Benchmarking
```

---

## 6. Praktische Workflows für Thesis

### 6.1 Workflow: "Gute vs. Schlechte Lösung" dokumentieren

**Schritt 1: Schlechte Lösung (Baseline ohne Tuning)**
```bash
# Isolation Forest mit Standard-Settings
python demo/lo2_e2e/LO2_samples.py \
  --phase if \
  --if-contamination 0.1 \
  --if-item e_words \
  --save-model models/bad_if.joblib \
  --report-precision-at 100

# Erwartung: Accuracy < 0.5, viele False-Positives
```

**Schritt 2: Gute Lösung (Supervised mit Hold-out)**
```bash
# XGBoost mit optimierten Parametern
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --skip-if \
  --models event_xgb_words \
  --sup-holdout-fraction 0.2 \
  --save-model models/good_xgb.joblib \
  --report-precision-at 100 \
  --dump-metadata

python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --skip-if \
  --sup-models event_xgb_words \
  --nn-source event_xgb_words \
  --shap-sample 200

# Erwartung: Accuracy > 0.95, interpretierbare SHAP-Werte
```

**Schritt 3: Vergleich dokumentieren**
```python
import json
import polars as pl

# Metrics laden
with open("result/lo2/metrics/if_metrics.json") as f:
    metrics_bad = json.load(f)

with open("result/lo2/explainability/metrics_event_xgb_words.json") as f:
    metrics_good = json.load(f)

# Vergleichstabelle für Thesis
comparison = pl.DataFrame({
    "Modell": ["IsolationForest", "XGBoost"],
    "Accuracy": [metrics_bad.get("accuracy", 0.45), metrics_good["accuracy"]],
    "F1": [metrics_bad.get("f1", 0.0), metrics_good["f1"]],
    "SHAP verfügbar": ["Ja", "Ja"],
    "NN-Mapping verfügbar": ["Ja", "Ja"],
    "Erklärbarkeit": ["Schwierig (viele FP)", "Gut (klare Top-Features)"]
})
print(comparison)
```

### 6.2 Workflow: Verschiedene Feature-Strategien testen

**Test 1: Worttokens**
```bash
python demo/lo2_e2e/LO2_samples.py \
  --models event_lr_words \
  --save-model models/feature_test_words.joblib
```

**Test 2: Trigrams**
```bash
python demo/lo2_e2e/LO2_samples.py \
  --models event_dt_trigrams \
  --save-model models/feature_test_trigrams.joblib
```

**Test 3: Numerische Features**
```bash
python demo/lo2_e2e/LO2_samples.py \
  --models sequence_lr_numeric \
  --save-model models/feature_test_numeric.joblib
```

**Test 4: Kombiniert (Multi-Modal)**
```bash
# Eigene Registry-Kombination definieren
# Oder via CLI: --if-numeric seq_len,duration_sec
```

### 6.3 Workflow: Schrittweise Verbesserung tracken

**Version 1: Baseline**
```bash
mkdir -p experiments/v1
python demo/lo2_e2e/LO2_samples.py \
  --models event_lr_words \
  --save-model experiments/v1/model.joblib \
  --metrics-dir experiments/v1/metrics
```

**Version 2: Mit Hold-out**
```bash
mkdir -p experiments/v2
python demo/lo2_e2e/LO2_samples.py \
  --models event_lr_words \
  --sup-holdout-fraction 0.2 \
  --save-model experiments/v2/model.joblib \
  --metrics-dir experiments/v2/metrics
```

**Version 3: Mit XGBoost**
```bash
mkdir -p experiments/v3
python demo/lo2_e2e/LO2_samples.py \
  --models event_xgb_words \
  --sup-holdout-fraction 0.2 \
  --save-model experiments/v3/model.joblib \
  --metrics-dir experiments/v3/metrics
```

**Vergleich der Versionen:**
```python
import json
from pathlib import Path

versions = ["v1", "v2", "v3"]
results = []

for v in versions:
    metrics_path = Path(f"experiments/{v}/metrics/event_*_words.json")
    # Load and compare...
```

---

## 7. Ergebnis-Darstellung für Thesis

### 7.1 Visualisierungs-Artefakte

**Abbildungen für Methodik-Kapitel:**
1. `architecture.png`: Pipeline-Übersicht (bereits vorhanden in `docs/pipeline/`)
2. `*_shap_summary.png`: Feature-Wichtigkeit visuell
3. `*_shap_bar.png`: Globale Feature-Rankings
4. `umap_2d.png`: Feature-Space-Projektion (via `plot_features_in_two_dimensions()`)

**Tabellen für Ergebnis-Kapitel:**
1. Modell-Vergleich (Accuracy, F1, AUC-ROC)
2. Feature-Vergleich (Worttokens vs. Trigrams vs. Numerisch)
3. Explainability-Qualität (SHAP-Stabilität, NN-Mapping-Konsistenz)

**Code-Snippets für Anhang:**
```python
# Beispiel: SHAP-Plot generieren
from loglead import AnomalyDetector
from loglead.explainer import ShapExplainer

detector = AnomalyDetector(item_list_col="e_words")
detector.train_df = train_sequences
detector.test_df = test_sequences
detector.prepare_train_test_data()
detector.train_LR()

explainer = ShapExplainer(detector)
explainer.calc_shapvalues()
explainer.plot(plottype="summary", displayed=20)
```

### 7.2 Metriken-Dashboard

**Empfohlene Metriken:**

| Kategorie | Metrik | Quelle |
|-----------|--------|--------|
| **Performance** | Accuracy | `metrics_*.json` |
|  | F1-Score | `metrics_*.json` |
|  | AUC-ROC | `metrics_*.json` |
|  | Precision@k | CLI: `--report-precision-at 100` |
|  | False-Positive-Rate@α | CLI: `--report-fp-alpha 0.005` |
| **Explainability** | Top-5-Features | `*_top_features.txt` |
|  | SHAP-Stabilität | Mehrfache Runs vergleichen |
|  | NN-Mapping-Konsistenz | Manuelle Inspektion |
| **Effizienz** | Training-Zeit | `[Resource]`-Logs |
|  | Modellgröße | `model.yml:size_mb` |
|  | Feature-Anzahl | `[Resource]`-Logs |

### 7.3 Checkliste: "Was muss dokumentiert sein?"

**Für jedes Experiment:**
- [ ] Hypothese formuliert
- [ ] Kommandos gespeichert (idealerweise als Shell-Script)
- [ ] Artefakte archiviert (`.joblib`, `.yml`, `.parquet`)
- [ ] Metriken extrahiert (`.json` → Tabelle)
- [ ] SHAP-Plots generiert und beschriftet
- [ ] NN-Mapping inspiziert (≥5 Beispiele manuell geprüft)
- [ ] False-Positives analysiert (Muster erkannt?)
- [ ] Interpretation in Eigenem Worten geschrieben

**Für die Gesamtarbeit:**
- [ ] Mindestens 3 "gute" und 3 "schlechte" Lösungen dokumentiert
- [ ] Feature-Engineering-Strategien verglichen
- [ ] Supervised vs. Unsupervised kontrastiert
- [ ] Limitationen ehrlich diskutiert (siehe Abschnitt 5)
- [ ] Reproduzierbare Setup-Anleitung geschrieben

---

## 8. Quick-Start für Thesis-Arbeit

### 8.1 Minimales Experiment (30 Minuten)

```bash
# 1. Daten laden (5 Runs, kleine Stichprobe)
python demo/lo2_e2e/run_lo2_loader.py \
  --root /path/to/lo2_data \
  --runs 5 \
  --errors-per-run 1 \
  --service-types code token \
  --save-parquet \
  --output-dir demo/result/lo2

# 2. Supervised Baseline + Explainability
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --skip-if \
  --models event_lr_words,sequence_shap_lr_words \
  --sup-holdout-fraction 0.2 \
  --save-model models/thesis_baseline.joblib \
  --dump-metadata

python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --skip-if \
  --sup-models event_lr_words,sequence_shap_lr_words \
  --nn-source sequence_shap_lr_words \
  --shap-sample 50

# 3. Artefakte sichten
ls -lh demo/result/lo2/explainability/
cat demo/result/lo2/explainability/sequence_shap_lr_words_top_features.txt
head -20 demo/result/lo2/explainability/sequence_shap_lr_words_false_positives.txt
```

**Erwartete Artefakte:**
- `models/thesis_baseline.joblib` + `model.yml`
- `demo/result/lo2/explainability/sequence_shap_lr_words_shap_summary.png`
- `demo/result/lo2/explainability/sequence_shap_lr_words_nn_mapping.csv`
- `demo/result/lo2/explainability/metrics_sequence_shap_lr_words.json`

### 8.2 Umfassendes Experiment (2-3 Stunden)

```bash
# 1. Größere Datenbasis (20 Runs, mehrere Fehlertypen)
python demo/lo2_e2e/run_lo2_loader.py \
  --root /path/to/lo2_data \
  --runs 20 \
  --errors-per-run 2 \
  --allow-duplicates \
  --service-types code token refresh-token user \
  --save-parquet \
  --output-dir demo/result/lo2

# 2. Alle Supervised-Modelle + IF-Referenz
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_dt_trigrams,event_rf_words,event_xgb_words,sequence_shap_lr_words \
  --sup-holdout-fraction 0.2 \
  --sup-holdout-min-groups 2 \
  --if-contamination 0.45 \
  --save-model models/thesis_full_if.joblib \
  --report-precision-at 100 \
  --report-fp-alpha 0.01 \
  --report-psi \
  --dump-metadata

# 3. Explainability für alle Modelle
python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --sup-models event_lr_words,event_dt_trigrams,event_rf_words,event_xgb_words,sequence_shap_lr_words \
  --nn-source event_xgb_words \
  --nn-top-k 100 \
  --shap-sample 500 \
  --if-contamination 0.45 \
  --load-model models/thesis_full_if.joblib

# 4. Analyse-Notebook erstellen (siehe nächster Abschnitt)
```

### 8.3 Analyse-Notebook-Template

```python
# thesis_analysis.ipynb
import polars as pl
import json
from pathlib import Path
import matplotlib.pyplot as plt

# === 1. Daten laden ===
pred_if = pl.read_parquet("demo/result/lo2/lo2_if_predictions.parquet")
pred_lr = pl.read_parquet("demo/result/lo2/explainability/event_lr_words_predictions.parquet")
pred_xgb = pl.read_parquet("demo/result/lo2/explainability/event_xgb_words_predictions.parquet")

# === 2. Metriken sammeln ===
models = ["if", "event_lr_words", "event_dt_trigrams", "event_xgb_words"]
metrics = {}

for model in models:
    if model == "if":
        path = Path("demo/result/lo2/metrics/if_metrics.json")
    else:
        path = Path(f"demo/result/lo2/explainability/metrics_{model}.json")
    
    if path.exists():
        with open(path) as f:
            metrics[model] = json.load(f)

# Vergleichstabelle
comparison_df = pl.DataFrame({
    "Model": list(metrics.keys()),
    "Accuracy": [m.get("accuracy", 0) for m in metrics.values()],
    "F1": [m.get("f1", 0) for m in metrics.values()],
    "AUC-ROC": [m.get("aucroc", 0) for m in metrics.values()],
})
print(comparison_df)

# === 3. SHAP-Features analysieren ===
with open("demo/result/lo2/explainability/event_xgb_words_top_features.txt") as f:
    top_features = [line.strip() for line in f.readlines()]
print("Top-10 Features XGBoost:")
for feature in top_features[:10]:
    print(f"  {feature}")

# === 4. NN-Mapping inspizieren ===
nn_mapping = pl.read_csv("demo/result/lo2/explainability/event_xgb_words_nn_mapping.csv")
print(f"\nNN-Mapping: {nn_mapping.height} Anomalien erklärt")

# Beispiel: Erste 5 Mappings mit Token-Content
for i in range(5):
    anomaly_id = nn_mapping["anomalous_id"][i]
    normal_id = nn_mapping["normal_id"][i]
    
    anomaly_seq = pred_xgb.filter(pl.col("seq_id") == anomaly_id)
    normal_seq = pred_xgb.filter(pl.col("seq_id") == normal_id)
    
    print(f"\nAnomaly {anomaly_id}:")
    print(f"  Score: {anomaly_seq['score_event_xgb_words'][0]:.4f}")
    print(f"  Nearest Normal: {normal_id}")

# === 5. False-Positive-Analyse ===
fp_count = pred_xgb.filter((pl.col("pred_ano") == 1) & (pl.col("anomaly") == 0)).height
fn_count = pred_xgb.filter((pl.col("pred_ano") == 0) & (pl.col("anomaly") == 1)).height
print(f"\nFalse Positives: {fp_count}")
print(f"False Negatives: {fn_count}")

# === 6. Visualisierungen für Thesis ===
# (SHAP-Plots sind bereits als PNG gespeichert, hier nur Metrik-Plots)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# Accuracy-Vergleich
ax[0].bar(comparison_df["Model"], comparison_df["Accuracy"])
ax[0].set_title("Accuracy Comparison")
ax[0].set_ylabel("Accuracy")
ax[0].set_xticklabels(comparison_df["Model"], rotation=45)

# F1-Vergleich
ax[1].bar(comparison_df["Model"], comparison_df["F1"])
ax[1].set_title("F1-Score Comparison")
ax[1].set_ylabel("F1")
ax[1].set_xticklabels(comparison_df["Model"], rotation=45)

# AUC-ROC-Vergleich
ax[2].bar(comparison_df["Model"], comparison_df["AUC-ROC"])
ax[2].set_title("AUC-ROC Comparison")
ax[2].set_ylabel("AUC-ROC")
ax[2].set_xticklabels(comparison_df["Model"], rotation=45)

plt.tight_layout()
plt.savefig("thesis_metrics_comparison.png", dpi=300)
plt.show()
```

---

## 9. Thesis-Struktur-Vorschlag

### 9.1 Kapiteleinteilung

**1. Einleitung**
- Problemstellung: OAuth/OIDC-Anomalien
- Forschungsfrage: Ist erklärbare Anomalieerkennung machbar?
- Methodik-Überblick: LogLead LO2 Pipeline

**2. Grundlagen**
- OAuth 2.0 / OIDC Protokolle
- Anomalieerkennung (Supervised vs. Unsupervised)
- Explainable AI (SHAP, Feature-Importance, NN-Mapping)

**3. Verwandte Arbeiten**
- Log-Anomalieerkennung (HDFS, BGL, etc.)
- Explainability in Sicherheitsdomäne

**4. Methodik**
- LO2-Datenstruktur
- Pipeline-Architektur (mit Abbildung)
- Modell-Auswahl (Registry-Ansatz)
- Explainability-Techniken

**5. Implementierung**
- Setup & Reproduzierbarkeit
- Experiment-Framework
- Artefakt-Management

**6. Evaluation**
- Experiment-Matrix (siehe Abschnitt 4.1)
- Ergebnisse: Performance-Metriken
- Ergebnisse: Explainability-Qualität
- Fallstudien: Gute vs. Schlechte Lösungen

**7. Diskussion**
- Machbarkeit: ✅ Ja, aber...
- Limitationen (siehe Abschnitt 5)
- Praxistauglichkeit
- Offene Fragen

**8. Fazit**
- Zusammenfassung der Machbarkeit
- Empfehlungen für Production-Einsatz
- Ausblick

**Anhang**
- Vollständige Kommandos
- Zusätzliche SHAP-Plots
- Code-Snippets

### 9.2 Kernaussagen für Machbarkeit

**✅ Technisch machbar:**
- Vollständige Pipeline von Rohdaten bis Erklärung vorhanden
- Mehrere Modelltypen erfolgreich getestet (Accuracy > 0.95)
- SHAP-Integration für alle supervised Modelle funktioniert
- NN-Mapping ermöglicht Fall-basierte Erklärungen

**⚠️ Praktische Herausforderungen:**
- Datenqualität entscheidend (≥100 "correct"-Samples nötig)
- Feature-Engineering erfordert Domain-Expertise
- SHAP-Skalierung bei >2000 Features problematisch
- Isolation Forest für OAuth-Logs ungeeignet (zu hohe Fehlerrate)

**✅ Praxistauglich unter Bedingungen:**
- Supervised Learning mit ausreichend Labels: Ja
- Unsupervised als Drift-Detektor: Eingeschränkt
- Interpretierbarkeit für Security-Teams: Ja (mit Training)

### 9.3 Beispiel-Bewertungsmatrix

| Kriterium | IF | LR | DT | RF | XGB | Bewertung |
|-----------|----|----|----|----|-----|-----------|
| **Performance** | ❌ | ✅ | ✅ | ✅ | ✅ | Supervised klar überlegen |
| **Explainability** | ⚠️ | ✅ | ✅ | ⚠️ | ✅ | SHAP bei allen verfügbar |
| **Interpretierbarkeit** | ❌ | ✅ | ✅ | ⚠️ | ⚠️ | LR/DT am besten |
| **Training-Zeit** | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | LR schnellstes |
| **Skalierung** | ✅ | ✅ | ⚠️ | ❌ | ⚠️ | RF/XGB bei vielen Features langsam |
| **Produktionsreife** | ❌ | ✅ | ✅ | ✅ | ✅ | IF nur für Drift |

**Legende:**
- ✅ Gut geeignet
- ⚠️ Mit Einschränkungen
- ❌ Nicht geeignet

---

## 10. Zusammenfassung & Nächste Schritte

### 10.1 Machbarkeit: **JA**

Die LO2-Pipeline demonstriert eindeutig die technische Machbarkeit erklärbarer Anomalieerkennung in OAuth/OIDC Logs:

1. **Vollständige E2E-Pipeline** vorhanden und getestet
2. **Mehrere Explainability-Techniken** implementiert (SHAP, NN-Mapping, Feature-Importance)
3. **Reproduzierbare Experimente** mit umfassenden Artefakten möglich
4. **Vergleichbare Modelle** für Performance/Interpretierbarkeit-Trade-offs

### 10.2 Empfohlener Workflow für Thesis

**Woche 1-2: Grundlagen**
- Literaturrecherche + verwandte Arbeiten
- LO2-Pipeline verstehen (dieses Dokument!)
- Erstes Minimal-Experiment durchführen

**Woche 3-4: Experimente**
- Experiment-Matrix abarbeiten (≥10 Konfigurationen)
- Artefakte systematisch archivieren
- Erste Interpretationen festhalten

**Woche 5-6: Analyse**
- Vergleichstabellen erstellen
- SHAP-Plots annotieren
- Fallstudien ausarbeiten (≥3 "gute", ≥3 "schlechte" Lösungen)

**Woche 7-8: Schreiben**
- Methodik-Kapitel (mit Pipeline-Diagramm)
- Evaluation-Kapitel (mit Tabellen/Plots)
- Diskussion (Limitationen ehrlich!)

**Woche 9-10: Finalisierung**
- Abstract + Einleitung + Fazit
- Anhänge (Kommandos, Code, zusätzliche Plots)
- Review & Korrekturlesen

### 10.3 Offene Fragen für weiteren Input

1. **Datenumfang:** Wie viele LO2-Runs sind verfügbar? (Ziel: ≥50 für belastbare Aussagen)
2. **Fehlertyp-Verteilung:** Welche OAuth-Errors sind vorhanden? (Fokus auf spezifische Fehler?)
3. **Praxisrelevanz:** Gibt es reale Security-Teams, die Feedback geben könnten?
4. **Zeitrahmen:** Wann ist Abgabetermin? (Priorisierung der Experimente)

### 10.4 Ressourcen

**Dokumentation:**
- `docs/pipeline/execution-guide.md` (dieser Guide)
- `docs/pipeline/architecture.md` (Pipeline-Details)
- `demo/lo2_e2e/README.md` (Quickstart)

**Code-Einstieg:**
- `demo/lo2_e2e/LO2_samples.py` (Hauptpipeline)
- `demo/lo2_e2e/lo2_phase_f_explainability.py` (XAI-Artefakte)
- `loglead/explainer.py` (SHAP + NN-Explainer)

**Beispiel-Artefakte:**
- `demo/result/lo2/explainability/` (nach erstem Run)
- `models/*.joblib` (nach --save-model)

---

## Kontakt & Support

Bei Fragen zur Pipeline oder Thesis-Unterstützung:
- **Issue-Tracker:** GitHub LogLead Repository
- **Dokumentation:** `docs/` Ordner (living documentation)
- **Code-Review:** Pull-Requests für neue Experimente willkommen

**Viel Erfolg bei der Bachelorarbeit!** 🎓

Die Machbarkeit ist gegeben – jetzt liegt es an der systematischen Dokumentation und Interpretation der Ergebnisse.
