# LO2 Isolation Forest E2E Flow

Dieses Dokument fasst den End-to-End-Ablauf der LO2 Anomalieerkennung zusammen und verweist auf die relevanten Quelltextstellen.

## 1. Datenaufbereitung

- **Loader**: `demo/lo2_e2e/run_lo2_loader.py` schreibt alle eingelesenen Runs in `demo/result/lo2/lo2_events.parquet` (Events) sowie optional `lo2_sequences.parquet`. Die Spalten `test_case` und `anomaly` werden dabei gesetzt.
- Ziel: Der Parquet-Speicher ist der einzige Input für spätere Phasen. Neue Logdateien müssen vor jedem Training über den Loader importiert werden.

## 2. Feature Engineering

- **Script**: `demo/lo2_e2e/LO2_samples.py`
- Abschnitte `Enhancing events` und `Aggregating to sequence level` (`demo/lo2_e2e/LO2_samples.py:217-247`) erzeugen Normalisierung, Tokenlisten, Trigramme, Drain-IDs und Längenfeatures.
- Ergebnis: `df_events` enthält die erweiterten Spalten (`e_words`, `e_trigrams`, `e_chars_len` …), `df_seqs` aggregierte Sequenzkennzahlen.

## 3. Trainings-/Testsplit für IF

- Nur Zeilen mit `test_case == "correct"` dienen als Trainingsbasis (`demo/lo2_e2e/LO2_samples.py:280`).
- Optional kann ein zeitlicher Hold-out reserviert werden (`--if-holdout-fraction`, `demo/lo2_e2e/LO2_samples.py:281-296`). Dieser Anteil fließt **nicht** ins Training, sondern in die Schwellenkalibrierung.
- `AnomalyDetector.prepare_train_test_data()` (`loglead/anomaly_detection.py`) vektorisiert das Trainingsset; neue Tokens erweitern dabei automatisch das Vokabular.

## 4. Isolation Forest Training & Score Berechnung

- Training: `sad_if.train_IsolationForest(...)` (`demo/lo2_e2e/LO2_samples.py:308-313`) auf den Feature-Matrizen des Trainingssatzes.
- Scoring: `pred_if = sad_if.predict()` liefert Vorhersagen für das gesamte `df_events`.
- Scores: `score_if = -sad_if.model.score_samples(...)` (`demo/lo2_e2e/LO2_samples.py:316-324`) für Rangreihen.

## 5. Schwellenkalibrierung (optional)

- Flag `--if-threshold-percentile` nutzt Hold-out- oder Trainingsscores, um eine Perzentilschwelle abzuleiten (`demo/lo2_e2e/LO2_samples.py:333-346`).
- Die Schwelle wird als zusätzliche Spalte `pred_if_threshold` im Ergebnis-Datenframe abgelegt, sobald sie gesetzt ist (`demo/lo2_e2e/LO2_samples.py:350-353`).

## 6. Persistenz & Metriken

- Predictions: `lo2_if_predictions.parquet` (`demo/lo2_e2e/LO2_samples.py:355-363`).
- Modelle: `--save-model` speichert `(model, vectorizer)` via joblib (`demo/lo2_e2e/LO2_samples.py:404-423`).
- Optional: `--dump-metadata` erzeugt `models/model.yml` mit Parameter- und Dataset-Infos (`demo/lo2_e2e/LO2_samples.py:426-458`).
- Metriken: `--report-precision-at`, `--report-fp-alpha`, `--report-psi` schreiben JSON/CSV nach `result/lo2/metrics/` (`demo/lo2_e2e/LO2_samples.py:365-402`).

## 7. Benchmarking & Bewertung

- **Precision@k**: Anteil tatsächlich anomaler Zeilen in den Top-k Scores (`demo/lo2_e2e/metrics_utils.py:19-36`).
- **FP-rate@α**: False-Positive-Rate innerhalb der Top-α-Scores (`demo/lo2_e2e/metrics_utils.py:39-60`).
- **PSI**: Vergleich der Scoreverteilungen (Train vs. Hold-out) für Drifterkennung (`demo/lo2_e2e/metrics_utils.py:63-92`).

## 8. Inferenz & Folgeprozesse

- Für neue Logdaten: Loader erneut ausführen, Modell mit aktueller Schwelle anwenden (`joblib.load` + gespeicherte Threshold aus `model.yml`), dann `sad_if.predict()` auf dem neuen Dataset aufrufen.
- Überwachung: `if_metrics.json` vergleichen (Precision↑, FP↓, PSI≤0.2) und bei Drift neu trainieren.
