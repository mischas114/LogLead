# LO2 Isolation Forest E2E Flow

Dieses Dokument fasst den End-to-End-Ablauf der LO2 Anomalieerkennung zusammen und verweist auf die relevanten Quelltextstellen.

## Voraussetzungen & Einstieg

1. **Rohdaten:** Log-Dateien liegen im lokalen Dateisystem (z. B. `~/data/lo2_logs/`).  
2. **Loader ausführen:**  
   ```bash
   python demo/lo2_e2e/run_lo2_loader.py --root ~/data/lo2_logs --save-parquet --output-dir demo/result/lo2
   ```  
   Damit entstehen `demo/result/lo2/lo2_events.parquet` (Events) und optional `lo2_sequences.parquet`.
3. **Labels:** Der Loader setzt `test_case` (`"correct"`/Fehlername) und `anomaly` (0/1); Fixup: Trainingsdaten enthalten ausschließlich `test_case == "correct"`.
4. **Hauptskript:**  
   ```bash
    python demo/lo2_e2e/LO2_samples.py \
       --phase if \
       --save-model models/lo2_if.joblib \
       --if-holdout-fraction 0.1 \
       --if-threshold-percentile 99.5
   ```  
   Mit `--models` lässt sich bestimmen, welche zusätzlichen Detektoren neben dem IsolationForest laufen (Standard: `event_lr_words,event_dt_trigrams,sequence_lr_numeric,sequence_shap_lr_words`). `--list-models` zeigt die verfügbaren Schlüssel.

## Kurzüberblick (Text-Flow)

Loader → Parquet (`lo2_events.parquet`)  
→ Enhancer (`LO2_samples.py`) bereitet Features vor  
→ Trainingssplit (`correct` → IF-Training, alle Events → Evaluation)  
→ Isolation Forest trainiert & scored  
→ Optional: Hold-out-Schwelle + Metriken  
→ Artefakte: `lo2_if_predictions.parquet`, `models/lo2_if.joblib`, `model.yml`, `metrics/*.json`

**Modell-Reuse:** Statt neu zu trainieren, kann ein vorhandenes Bundle geladen werden:

```bash
python demo/lo2_e2e/LO2_samples.py --phase if --load-model models/lo2_if.joblib
```

Auch Phase F unterstützt Reuse:

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
   --root demo/result/lo2 \
   --load-model models/lo2_if.joblib
```

## 1. Datenaufbereitung

- **Loader**: `demo/lo2_e2e/run_lo2_loader.py` schreibt alle eingelesenen Runs in `demo/result/lo2/lo2_events.parquet` (Events) sowie optional `lo2_sequences.parquet`. Die Spalten `test_case` und `anomaly` werden dabei gesetzt.
- Ziel: Der Parquet-Speicher ist der einzige Input für spätere Phasen. Neue Logdateien müssen vor jedem Training über den Loader importiert werden.

## 2. Feature Engineering

- **Script**: `demo/lo2_e2e/LO2_samples.py`
- Abschnitte `Enhancing events` und `Aggregating to sequence level` (`demo/lo2_e2e/LO2_samples.py:217-247`) erzeugen Normalisierung, Tokenlisten, Trigramme, Drain-IDs und Längenfeatures.
- Ergebnis: `df_events` enthält die erweiterten Spalten (`e_words`, `e_trigrams`, `e_chars_len` …), `df_seqs` aggregierte Sequenzkennzahlen.

## 3. Trainings-/Testsplit für IF

- Nur Zeilen mit `test_case == "correct"` dienen als Trainingsbasis (`demo/lo2_e2e/LO2_samples.py:280`).
- Optional kann ein zeitlicher Hold-out reserviert werden (`--if-holdout-fraction`, `demo/lo2_e2e/LO2_samples.py:281-296`). Dieser Anteil fließt **nicht** ins Training, sondern ausschließlich in die Schwellen- bzw. Driftkalibrierung (`--if-threshold-percentile`, PSI-Berechnung).
- `AnomalyDetector.prepare_train_test_data()` (`loglead/anomaly_detection.py`) vektorisiert das Trainingsset; neue Tokens erweitern dabei automatisch das Vokabular.

## 4. Isolation Forest Training & Score Berechnung

- Training: `sad_if.train_IsolationForest(...)` (`demo/lo2_e2e/LO2_samples.py:308-313`) auf den Feature-Matrizen des Trainingssatzes.
- Scoring: `pred_if = sad_if.predict()` liefert Vorhersagen für das gesamte `df_events`.
- Scores: `score_if = -sad_if.model.score_samples(...)` (`demo/lo2_e2e/LO2_samples.py:316-324`) für Rangreihen.

## 5. Schwellenkalibrierung (optional)

- Flag `--if-threshold-percentile` nutzt Hold-out- (falls vorhanden) oder Trainingsscores, um eine Perzentilschwelle abzuleiten (`demo/lo2_e2e/LO2_samples.py:333-346`).
- Die Schwelle wird als zusätzliche Spalte `pred_if_threshold` im Ergebnis-Datenframe abgelegt, sobald sie gesetzt ist (`demo/lo2_e2e/LO2_samples.py:350-353`). Wertebereich: `>1` interpretiert als Prozentangabe, `0-1` als Anteil.

## 6. Persistenz & Metriken

- Predictions: `lo2_if_predictions.parquet` (`demo/lo2_e2e/LO2_samples.py:355-363`).
- Modelle: `--save-model` speichert `(model, vectorizer)` via joblib. Mit `--load-model <pfad>` kann ein bestehendes Bundle geladen und das Training übersprungen werden.
- Optional: `--dump-metadata` erzeugt `models/model.yml` mit Parameter- und Dataset-Infos (`demo/lo2_e2e/LO2_samples.py:426-458`).
- Metriken: `--report-precision-at`, `--report-fp-alpha`, `--report-psi` schreiben JSON/CSV nach `result/lo2/metrics/` (`demo/lo2_e2e/LO2_samples.py:365-402`).

## 7. Modell-Registry & Varianten (Phase E/F)

- `--list-models` listet alle verfügbaren Event- und Sequence-Detektoren (z. B. LogisticRegression, DecisionTree, LocalOutlierFactor, OOVDetector, Sequence-LR mit SHAP).
- `--models key1,key2,...` aktiviert eine Teilmenge; ohne Angabe läuft das Default-Set. Sequence-Modelle werden automatisch übersprungen, wenn **kein** `lo2_sequences.parquet` vorliegt.
- Modelle mit `train_selector=correct_only` (LOF, OOV) trainieren ausschließlich auf `test_case == "correct"`; die Evaluation findet trotzdem auf allen Events/Sequenzen statt.

## 8. Benchmarking & Bewertung

- **Precision@k**: Anteil tatsächlich anomaler Zeilen in den Top-k Scores (`demo/lo2_e2e/metrics_utils.py:19-36`).
- **FP-rate@α**: False-Positive-Rate innerhalb der Top-α-Scores (`demo/lo2_e2e/metrics_utils.py:39-60`).
- **PSI**: Vergleich der Scoreverteilungen (Train vs. Hold-out) für Drifterkennung (`demo/lo2_e2e/metrics_utils.py:63-92`).

## 9. Inferenz & Folgeprozesse

- Für neue Logdaten: Loader erneut ausführen, Modell mit aktueller Schwelle anwenden (`joblib.load` + gespeicherte Threshold aus `model.yml`), dann `sad_if.predict()` auf dem neuen Dataset aufrufen.
- Überwachung: `if_metrics.json` vergleichen (Precision↑, FP↓, PSI≤0.2) und bei Drift neu trainieren.
