---
title: Modellbefunde & Metriken
summary: Aktueller Stand der Modellbewertungen inklusive bekannter Einschränkungen.
last_updated: 2025-11-05
---

# Modellbefunde & Metriken

Die bisherigen Läufe konzentrieren sich auf drei Konfigurationen: Supervised-Sequenzmodelle auf Token-Features, unsupervised Sequenzmodelle und numerische Sequenzmodelle. Supervised-Varianten nutzen jetzt standardmäßig einen run-basierten Hold-out (20 %), während die unsupervised Modelle weiterhin auf dem Volltraining basieren.

## Aktueller Benchmark (50 % Anomalien, Hold-out 20 %)

| Modell-Key | Algorithmus | Accuracy | F1 | Bemerkung |
| --- | --- | --- | --- | --- |
| event_lr_words | LogisticRegression | 0.6750 | 0.6667 | Solide Baseline; liefert SHAP-Features via `sequence_shap_lr_words`. |
| event_lsvm_words | LinearSVC | 0.9750 | 0.9744 | Sehr hohe Präzision, weiterhin auf größerem Hold-out zu validieren. |
| event_rf_words | RandomForest | 1.0000 | 1.0000 | Perfekte Scores auf 40 Sequenzen → Gefahr von Overfitting, mehr Daten nötig. |
| event_xgb_words | XGBClassifier | 0.9750 | 0.9756 | Benötigt `xgboost`; robustes Ranking auf aktuellem Sample. |
| sequence_shap_lr_words | LogisticRegression | 0.6750 | 0.6667 | Identisch zu `event_lr_words`, zusätzlich automatische SHAP-Plots. |
| isolation_forest | IsolationForest | 0.4500 | 0.0000 | Trotz Kontamination 0.45 keine Treffer; nur noch als Drift-Monitor infrage. |

## Zusammenfassung der letzten Runs

| Paket | Modelle | Stichpunkte |
| --- | --- | --- |
| Supervised (Sequence) | LR, LinearSVM, DecisionTree, RandomForest, XGBoost | Run-basierter Hold-out (20 %) pro Service/Test-Case aktiviert; Trainingsscores bleiben hoch, Evaluation erfolgt auf separaten Runs. |
| Un-/Semi-Supervised (Sequence) | LOF, OneClassSVM, KMeans, Rarity, OOV | Starke Schwankungen: LOF F1≈0.009, OneClassSVM F1≈0.056, KMeans trennt kaum (AUC~0.50), Rarity kollabiert, OOVDetector meldet 100 % Accuracy (vermutlich Leakage). |
| Sequenzmodelle (numeric) | Sequence-LR (numeric/words), Sequence-SHAP | Liefert hohe Scores, basiert aber auf wenigen Sequenzen → Overfitting wahrscheinlich. |

## Detailauszug (unsupervised Paket)

| Modell | Accuracy | F1 | Bemerkung |
| --- | --- | --- | --- |
| LocalOutlierFactor | 0.9207 | 0.0086 | Trainiert auf `correct`-Runs, aber kaum True Positives. |
| OneClassSVM | 0.6019 | 0.0558 | Frühzeitige Konvergenzwarnung (`max_iter=1000`). |
| KMeans | 0.9211 | 0.0000 | Cluster 2 → alle Ereignisse normal, keine Anomalien erkannt. |
| RarityModel | 0.0025 | 0.0041 | Markiert fast alles als anomal; kaum nutzbar. |
| OOVDetector | 1.0000 | 0.9997 | Perfekte Scores → sehr wahrscheinlich Trainings-Leakage. |

Quelle: Lauf vom `demo/lo2_e2e/LO2_samples.py` mit `--models event_lof_words,event_oneclass_svm_words,event_kmeans_words,event_rarity_words,event_oov_words`, Contamination 0.25, Hold-out 0.05.

## Interpretationshilfe

- Supervised-Modelle liefern jetzt belastbare Werte auf einem hold-out von 20 % der Runs (temporal per Service/Test-Case). Aussagekraft verbessert, dennoch auf Verzerrungen durch kleine Testmengen achten.
- IsolationForest fällt auf der aktuellen Stichprobe durch (F1=0.0); Fokus liegt nun auf supervised Sequenzmodellen. Details siehe `pipeline/isolation-forest.md`.
- Sequence-Modelle liefern interpretierbare Signale, Repräsentativität aber unklar (kleine Stichprobe).

## ToDos für belastbare Benchmarks

- TODO: Ergebnisse inkl. Konfigurationen zentral (CSV/Notebook) erfassen statt als Konsolenlog.
- TODO: Leakage-Quellen für OOVDetector und RarityModel identifizieren (z. B. Tokenlisten, Normalisierungen).
- TODO: Hold-out-Größe erhöhen und weitere Runs einspeisen, um RF/XGB/LSVM-Aussagen zu verifizieren.
