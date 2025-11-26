# Kapitel 3: Versuchsaufbau und Methodik
## Stichwortartige Dokumentation basierend auf Code-Analyse

**Stand:** 2025-11-26  
**Quelle:** `demo/lo2_e2e/` Pipeline-Implementierung

---

## 3.1 Zielbild der Pipeline

### Phasenmodell
- **Phase A**: Input – Rohdaten aus `run_*/test_case/*.log`
- **Phase B**: Loader – `LO2Loader` traversiert Ordner, erzeugt Event- und Sequenz-DataFrames
- **Phase C**: Feature-Engineering – `EventLogEnhancer` + `SequenceEnhancer`
- **Phase D**: Unsupervised Baseline – IsolationForest auf `correct`-Sequenzen trainiert
- **Phase E**: Supervised Modelle – Registry-basiert, 13 konfigurierbare Modelle
- **Phase F**: Explainability – SHAP-Erklärungen + NNExplainer-Mappings

### Datenfluss
- Eingang: LO2-Rohdaten → `lo2_events.parquet` (optional) → `lo2_sequences_enhanced.parquet`
- Ausgang: Predictions (`.parquet`), Metriken (`.json`), SHAP-Plots (`.png`), NN-Mappings (`.csv`)

### Kernkomponenten
- `run_lo2_loader.py`: Lädt und persistiert Daten
- `LO2_samples.py`: Orchestriert Enhancement + Modelltraining
- `lo2_phase_f_explainability.py`: Erzeugt XAI-Artefakte

---

## 3.2 Datengrundlage und Auswahl

### Datenquelle
- LO2 OAuth/OIDC-Testdaten
- Struktur: `<root>/run_<id>/<test_case>/*.log`
- Testfälle: `correct` (normal) vs. Fehlertypen (anomal)

### Ladeparameter (`run_lo2_loader.py`)
- `--runs`: Anzahl zu ladender Run-Ordner
- `--errors-per-run`: Fehlerverzeichnisse pro Run
- `--service-types`: Filter auf OAuth-Services (`code`, `token`, `refresh-token`, `user`, etc.)
- `--allow-duplicates` / `--no-duplicates`: Steuerung doppelter Fehlertypen
- `--single-error-type`: Fixierung auf einen Fehlertyp

### Service-Filter (OAuth-relevant)
- Verfügbar: `client`, `code`, `key`, `refresh-token`, `service`, `token`, `user`
- Empfohlen: `code`, `token`, `refresh-token` (OAuth-Flow-Kernkomponenten)

### Label-Definition
- `anomaly = 0`: Sequenzen aus `test_case == "correct"`
- `anomaly = 1`: Sequenzen aus Fehler-Testfällen

### Sequenz-Aggregation
- `seq_id = run__test_case__service`
- Eine Sequenz = alle Log-Zeilen eines Service innerhalb eines Test-Runs

---

## 3.3 Vorverarbeitung

### Event-Level Enhancement (`EventLogEnhancer`)
1. `normalize()` – Standardisierung der Log-Nachrichten
2. `words()` – Tokenisierung in Wortliste → `e_words`
3. `trigrams()` – N-Gram-Extraktion → `e_trigrams`
4. `parse_drain()` – Drain-Template-Erkennung → `e_event_drain_id` (optional)
5. `length()` – Zeichenlänge → `e_chars_len`, `e_lines_len`

### Sequenz-Level Enhancement (`SequenceEnhancer`)
1. `seq_len()` – Anzahl Events pro Sequenz
2. `start_time()` / `duration()` – Zeitstempel und Dauer → `duration_sec`
3. `tokens(token="e_words")` – Aggregierte Worttokens
4. `tokens(token="e_trigrams")` – Aggregierte Trigrams
5. `events("e_event_drain_id")` – Drain-IDs pro Sequenz (optional)

### Output-Schema (Sequenzen)
| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| `seq_id` | String | Eindeutiger Sequenz-Schlüssel |
| `anomaly` | Boolean | Label (0=normal, 1=anomal) |
| `e_words` | List[String] | Aggregierte Wort-Tokens |
| `e_trigrams` | List[String] | Aggregierte Trigrams |
| `seq_len` | Integer | Anzahl Events |
| `duration_sec` | Float | Sequenzdauer in Sekunden |

---

## 3.4 Feature-Engineering

### Token-basierte Features (Vektorisierung)
- **Bag-of-Words (BOW)**: `CountVectorizer` auf `e_words`
- **Trigram-Features**: `CountVectorizer` auf `e_trigrams`
- Standardparameter: `max_features=5000–40000`, `min_df=5`, `binary=True`

### Numerische Features
- `seq_len`: Sequenzlänge (Event-Anzahl)
- `duration_sec`: Zeitdauer der Sequenz

### Feature-Sets pro Modell (aus `MODEL_REGISTRY`)
| Modell-Key | Feature-Typ | Spalte | Dimension |
|------------|-------------|--------|-----------|
| `event_lr_words` | BOW | `e_words` | ~5.000 |
| `event_dt_trigrams` | Trigrams | `e_trigrams` | ~40.000 |
| `event_xgb_words` | BOW | `e_words` | ~40.000 |
| `sequence_lr_numeric` | Numerisch | `seq_len`, `duration_sec` | 2 |

### Vectorizer-Konfiguration
- `DEFAULT_VECTORIZER_KWARGS`: Zentrale Defaults
- Modelspezifisch überschreibbar via `vectorizer_kwargs`

---

## 3.5 Modelle

### Supervised Klassifikatoren
| Key | Algorithmus | Trainingsmethode | Features |
|-----|-------------|------------------|----------|
| `event_lr_words` | Logistic Regression | `train_LR` | BOW |
| `event_dt_trigrams` | Decision Tree | `train_DT` | Trigrams |
| `event_lsvm_words` | Linear SVM | `train_LSVM` | BOW |
| `event_rf_words` | Random Forest | `train_RF` | BOW |
| `event_xgb_words` | XGBoost | `train_XGB` | BOW |
| `sequence_lr_words` | Logistic Regression | `train_LR` | BOW |
| `sequence_lr_numeric` | Logistic Regression | `train_LR` | Numerisch |
| `sequence_shap_lr_words` | Logistic Regression + SHAP | `train_LR` | BOW |

### Unsupervised / Semi-Supervised
| Key | Algorithmus | Trainingsmethode | Besonderheit |
|-----|-------------|------------------|--------------|
| `event_lof_words` | Local Outlier Factor | `train_LOF` | Nur `correct`-Training |
| `event_kmeans_words` | KMeans (2 Cluster) | `train_KMeans` | – |
| `event_oneclass_svm_words` | One-Class SVM | `train_OneClassSVM` | Nur `correct`-Training |
| `event_rarity_words` | Rarity Model | `train_RarityModel` | Token-Seltenheit |
| `event_oov_words` | OOV Detector | `train_OOVDetector` | Nur `correct`-Training |

### IsolationForest (Phase D)
- Training ausschließlich auf `test_case == "correct"`
- Parameter: `contamination`, `n_estimators`, `max_samples`
- Output: `score_if`, `rank_if`, `pred_if`

### Hold-out-Strategie
- Run-basierter Split via `_run_based_holdout_split()`
- Standard: 20% Hold-out (`--sup-holdout-fraction 0.2`)
- Gruppierung nach `(service, run, test_case)`

---

## 3.6 Explainability

### SHAP-Integration (`ShapExplainer`)
- **Backend-Auswahl** (automatisch):
  - `LinearExplainer`: LogisticRegression, LinearSVM
  - `TreeExplainer`: DecisionTree, RandomForest, XGBoost, IsolationForest
  - `KernelExplainer`: Fallback für sonstige Modelle
- **SHAP-fähige Modelle**: `train_LR`, `train_DT`, `train_RF`, `train_XGB`, `train_LSVM`

### SHAP-Artefakte
- `*_shap_summary.png`: Summary-Plot (globale Feature-Wichtigkeit)
- `*_shap_bar.png`: Bar-Chart der Wichtigkeiten
- `*_top_features.txt`: Top-20 Features mit Ranking

### SHAP-Guards (Ressourcenschutz)
- `feature_warning_threshold`: Default 2.000 Features
- `sample_warning_threshold`: Default 2.000.000 Zellen (Rows × Features)
- `background_sample_size`: Default 256–512 Samples

### NNExplainer (Nearest-Neighbour)
- Findet nächste Normalsequenz für jede Anomalie
- Backend: Cosine-Similarity (erweiterbar für FAISS/Annoy)
- Output: `*_nn_mapping.csv` mit `anomalous_id` → `normal_id`

### NNExplainer-Methoden
- `build_mapping()`: Berechnet Anomaly→Normal-Zuordnung
- `print_log_content_from_nn_mapping()`: Zeigt Token-Vergleich
- `print_false_positive_content()`: Listet FP-Sequenzen
- `plot_features_in_two_dimensions()`: UMAP-Visualisierung

### False-Positive-Analyse
- `*_false_positives.txt`: Sequenzen mit `pred_ano=1` aber `anomaly=0`
- Enthält: `row_id`, `seq_id`, `service`, Score, Token-Content

---

## 3.7 Evaluationsstrategie

### Performance-Metriken
| Metrik | Funktion | Beschreibung |
|--------|----------|--------------|
| Accuracy | `accuracy_score()` | Anteil korrekt klassifizierter Sequenzen |
| F1-Score | `f1_score()` | Harmonisches Mittel von Precision/Recall |
| AUC-ROC | `roc_auc_score()` | Fläche unter ROC-Kurve |

### Ranking-Metriken (`metrics_utils.py`)
| Metrik | Funktion | Beschreibung |
|--------|----------|--------------|
| Precision@k | `precision_at_k()` | Precision in Top-k Scores |
| FP-Rate@α | `false_positive_rate_at_alpha()` | FP-Rate im Top-α-Anteil |
| PSI | `population_stability_index()` | Verteilungsstabilität Train vs. Hold-out |

### Splitting-Strategie
- **Run-basiert**: Vermeidet Leakage zwischen Sequenzen desselben Runs
- **Temporal**: Neueste Runs im Hold-out (via Timestamp-Sortierung)
- **Shuffle-Option**: `--sup-holdout-shuffle` für zufällige Auswahl
- **Fallback**: Bei fehlender `run`-Spalte → stratifizierter Zeilensplit

### Hold-out-Parameter
- `--sup-holdout-fraction`: Anteil für Hold-out (Default: 0.2)
- `--sup-holdout-min-groups`: Mindestanzahl Gruppen pro Bucket
- `--if-holdout-fraction`: Separater IF-Hold-out

### Metriken-Output
- `metrics/*.json`: Strukturierte Metriken pro Modell
- `metrics/*.csv`: Tabellarische Zusammenfassung
- Felder: `accuracy`, `f1`, `aucroc`, `precision_at_k`, `fp_rate_at_alpha`, `psi`

---

## Zusammenfassung: Pipeline-Ausführung

### Minimalbeispiel
```bash
# Phase B: Daten laden
python demo/lo2_e2e/run_lo2_loader.py \
  --root /pfad/zu/lo2 \
  --runs 5 \
  --service-types code token refresh-token \
  --save-parquet

# Phase C-E: Enhancement + Modelle
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_xgb_words \
  --sup-holdout-fraction 0.2

# Phase F: Explainability
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --sup-models event_lr_words,event_xgb_words \
  --shap-sample 200 \
  --nn-top-k 50
```

### Artefakte-Übersicht
| Pfad | Beschreibung |
|------|--------------|
| `demo/result/lo2/lo2_sequences_enhanced.parquet` | Angereicherte Sequenzen |
| `demo/result/lo2/lo2_if_predictions.parquet` | IF-Predictions + Scores |
| `demo/result/lo2/metrics/*.json` | Evaluationsmetriken |
| `demo/result/lo2/explainability/*.png` | SHAP-Plots |
| `demo/result/lo2/explainability/*_nn_mapping.csv` | NN-Zuordnungen |
| `demo/result/lo2/explainability/*_false_positives.txt` | FP-Analyse |
| `models/lo2_if.joblib` | Persistiertes IF-Modell |

---

**Erstellt:** 2025-11-26  
**Basis:** Code-Analyse von `demo/lo2_e2e/` und `loglead/explainer.py`
