# LO2 Modellleitfaden – Überblick & Ausführung

Dieses Dokument fasst die verfügbaren Anomalie-Detektoren der LO2-Pipeline zusammen und erklärt, wie du sie zielgerichtet ausführst. Zielgruppe sind Nutzer:innen ohne tiefes ML-Vorwissen, die schnell verstehen möchten, welche Modellfamilie sich für welchen Use-Case eignet und wie der Lauf angestoßen wird.

## 1. Modellkategorien im Überblick

Die Pipeline deckt vier grobe Richtungen ab:

1. **Isolation Forest (baseline)** – Unüberwachter Standard, der nur aus „korrekten“ Runs lernt und Anomalien an hohen Scores erkennt.  
2. **Supervised Event-Modelle** – Klassische Klassifikatoren (LogReg, SVM, DT, RF, XGB), die gelabelte Events brauchen.  
3. **Un-/Semi-Supervised Event-Modelle** – Modelle wie LOF, OneClassSVM, KMeans, OOV, Rarity, die überwiegend Normals sehen.  
4. **Sequence-Modelle** – Arbeiten auf aggregierten Sequenzen statt Einzelevents (LogReg + SHAP).

## 2. Vergleichstabelle (3–5 Kernpunkte je Modell)

| Modellschlüssel | Kategorie | Input-Anforderung | Stärken | Stolpersteine / Aufwand |
| --- | --- | --- | --- | --- |
| `if_baseline` (IsolationForest) | Anomalie (unsupervised) | Nur „correct“-Events + optionale Numeric-Features | - Robust bei wenig Labels<br>- Hold-out/Threshold integrierbar<br>- Explainability über NN/SHAP vorbereitet | - Score-Interpretation nicht trivial<br>- Sensibel auf Feature-Skalierung<br>- Kann viele False Positives haben |
| `event_lr_words` | Supervised | Tokens (Bag-of-Words), Labels | - Reproduzierbare Baseline<br>- Schnelles Training<br>- Gute Interpretierbarkeit (Gewichte) | - Braucht saubere Labels<br>- Overfitting ohne Train/Test-Split |
| `event_lsvm_words` | Supervised | Tokens, Labels | - Gut bei hoher Dimensionalität<br>- Robust gegen Ausreißer | - Keine Wahrscheinlichkeiten off-the-shelf<br>- Ggf. hoher CPU-Bedarf |
| `event_dt_trigrams` | Supervised | Trigram Features, Labels | - Decision Paths leicht erklärbar<br>- Fängt Regeln gut ein | - Neigt zu Overfitting<br>- Braucht Feature-Pruning |
| `event_rf_words` | Supervised Ensemble | Tokens, Labels | - Stabile Performance<br>- Geringe Parametertuning-Notwendigkeit | - Größerer Speicherbedarf<br>- Interpretierbarkeit eingeschränkt |
| `event_xgb_words` | Supervised Boosting | Tokens, Labels | - Starke Performance bei Balancing<br>- Flexible Hyperparameter | - Längere Trainingszeit<br>- Bedarf Feature-Tuning |
| `event_lof_words` | Unsupervised (Local Outlier Factor) | Nur Normaldaten (correct) | - Lokales Anomaliegefühl<br>- Schnell bei kleinen Datensätzen | - Keine Vorhersage für neue Punkte ohne `novelty=True` (hier bereits gesetzt) |
| `event_oneclass_svm_words` | Unsupervised (One-Class) | Normaldaten | - Fokussiert auf Normalregion<br>- Gut bei kleineren Feature-Sets | - Empfindlich gegenüber Feature-Skalierung<br>- Langsam bei vielen Samples |
| `event_kmeans_words` | Clustering | Tokens | - Schnelle Grob-Kategorisierung<br>- Einfach zu interpretieren | - Erwartet Cluster-Anzahl (hier 2)<br>- Nicht probabilistisch |
| `event_rarity_words` | Heuristik | Tokenstatistiken | - Hebt seltene Tokens hervor<br>- Kein Training notwendig | - Keine klassischen Scores<br>- Bedarf manueller Schwellen |
| `event_oov_words` | Heuristik | Tokenlisten + Längen | - Findet „Out-of-Vocabulary“-Events | - Muss auf Vokabulargröße abgestimmt werden |
| `sequence_lr_numeric` | Supervised Sequence | Sequenz-Längen, Dauer | - Schnelle Run-level Analyse<br>- Geringe Dimensionalität | - Verliert Detailinformationen |
| `sequence_lr_words` | Supervised Sequence | Sequenz-Tokens | - Nutzt Textmuster ganzer Runs | - Größere Matrizen, längeres Training |
| `sequence_shap_lr_words` | Supervised + Explainability | Sequenz-Tokens | - SHAP-Plot liefert Feature-Ranking | - SHAP kann rechenintensiv werden |

> **Hinweis:** Die Zusatzmodelle laufen alle innerhalb von `LO2_samples.py` im Registry-Abschnitt. Sie teilen sich die Feature-Pipeline des Skripts – die Übergabe erfolgt über `--models key1,key2,...`.

## 3. Ausführung – empfohlene Befehle

### 3.1 Grundvoraussetzung: Loader einmalig ausführen
```bash
python demo/lo2_e2e/run_lo2_loader.py \
  --root /pfad/zu/lo2_logs \
  --runs 50 \
  --errors-per-run 1 \
  --save-parquet \
  --output-dir demo/result/lo2
```

### 3.2 Isolation Forest Baseline (mit deinem IF-Tuning)
```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/benchmarks/if_baseline_predictions.parquet \
  --metrics-dir demo/result/lo2/benchmarks/if_baseline_metrics \
  --save-model demo/result/lo2/models/if_baseline.joblib \
  --overwrite-model \
  --dump-metadata
```

### 3.3 Supervised Event-Modelle bündeln
```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_lsvm_words,event_dt_trigrams,event_rf_words,event_xgb_words \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/benchmarks/if_supervised_predictions.parquet \
  --metrics-dir demo/result/lo2/benchmarks/if_supervised_metrics \
  --save-model demo/result/lo2/models/if_supervised.joblib \
  --overwrite-model \
  --dump-metadata
```

### 3.4 Un-/Semi-Supervised Paket
```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lof_words,event_oneclass_svm_words,event_kmeans_words,event_rarity_words,event_oov_words \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/benchmarks/if_unsupervised_predictions.parquet \
  --metrics-dir demo/result/lo2/benchmarks/if_unsupervised_metrics \
  --save-model demo/result/lo2/models/if_unsupervised.joblib \
  --overwrite-model \
  --dump-metadata
```

### 3.5 Sequence-Level Modelle
```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models sequence_lr_numeric,sequence_lr_words,sequence_shap_lr_words \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/benchmarks/if_sequence_predictions.parquet \
  --metrics-dir demo/result/lo2/benchmarks/if_sequence_metrics \
  --save-model demo/result/lo2/models/if_sequence.joblib \
  --overwrite-model \
  --dump-metadata
```

### 3.6 Explainability-Pass (optional, auf Basis der erzeugten Artefakte)
```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.25 \
  --if-n-estimators 200 \
  --shap-sample 500 \
  --nn-top-k 100 \
  --nn-normal-sample 200
```

## 4. Empfehlung für die Auswertung

1. **Metrics sammeln:** Die oben genannten Befehle schreiben JSON/CSV unter `demo/result/lo2/benchmarks/*`. Ziehe Precision@200, FP-Rate@0.01, Threshold etc. zum Vergleich heran.  
2. **Konsole nicht überbewerten:** Supervised-Modelle loggen Trainingsmetriken; ohne eigenen Train/Test-Split sind 100 %-Scores zu erwarten. Für seriöse Vergleiche nutze zusätzliche Splits oder ein separates Test-Set.  
3. **Explainability nutzen:** Die SHAP-Plots (`demo/result/lo2/explainability/`) liefern Feature-Rankings, während `if_nn_mapping.csv` auffällige Events beschreibt.  
4. **Notebook/Script schreiben:** Aggregiere die CSV/JSON-Dateien, um ein Ranking zu erstellen und Konfiguration + Ergebnis nachvollziehbar zu dokumentieren.

Damit hast du eine schnelle Entscheidungshilfe, welches Modell sich je nach Datengrundlage und benötigter Erklärungstiefe lohnt – plus die zugehörigen Kommandos, um jeden Kandidaten unmittelbar im LO2-Demo-Setup auszuführen.
