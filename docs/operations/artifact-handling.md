---
title: Artefakt- & Persistenzleitfaden
summary: Umgang mit Parquets, Modellbundles und Explainability-Ausgaben der LO2-Pipeline.
last_updated: 2025-11-03
---

# Artefakt- & Persistenzleitfaden

Dieser Leitfaden erklärt, welche Dateien die LO2-Pipeline erzeugt, wann sie benötigt werden und wie sie gepflegt werden sollten.

## Standardausgabe (`demo/result/lo2/`)

- `lo2_sequences_enhanced.parquet` (immer), optional `lo2_sequences.parquet` via `--save-base-sequences`: Loader-Grundlage für alle weiteren Phasen. Niemals manuell verändern. (Event-Parquet nur bei Bedarf via `--save-events`.)
- `lo2_if_predictions.parquet`: Enthält Score, Rang und Threshold-Markierung (`pred_if_threshold`).
- `metrics/*.json|csv`: Kennzahlen zu Precision@k, FP-Rate@α, PSI – Pfad via `--metrics-dir` steuerbar.
- `explainability/`: NN-Mapping (`if_nn_mapping.csv`), False-Positive-Liste, SHAP-Plots (`*.png`), Token-Rankings (`*_top_tokens.txt`).

## Modelle & Metadaten (`models/`)

- `lo2_if.joblib`: Tuple aus IsolationForest und CountVectorizer.
- `model.yml`: Snapshot (Parameter, Trainingsgrößen, Threshold, Git-Commit). Wird von Phase F und externen Anwendungen genutzt.
- Weitere Registry-Modelle können optional eigene Dumps erhalten; Pfad bei Bedarf ergänzen.

## Enhanced-Parquets

Wenn `LO2_samples.py` mit `--save-enhancers` läuft, entsteht:

- `enhanced/lo2_sequences_enhanced.parquet`

**Nutzen:** Beschleunigt Notebook-Analysen, weil Feature-Spalten nicht erneut berechnet werden müssen.

**Einschränkungen:** Die Skripte berechnen Features bei Bedarf neu; nutze den Cache nur für explorative Arbeiten.

## Dateibenennung & Speicherorte

- Alles unterhalb von `demo/result/lo2/` bezieht sich auf einen konkreten Datensatzstand. Für Varianten eigene Unterordner anlegen (z. B. `demo/result/lo2_drain/`).
- Modelle gehören unter `models/`; Vorsicht bei `--overwrite-model`, um versehentliches Überschreiben zu vermeiden.
- Explainability-Ausgaben können groß werden (SHAP-PNGs). Bei Serienläufen alte Artefakte bereinigen oder separat archivieren.

## Wiederverwendung

```python
import joblib
from loglead import AnomalyDetector

model, vectorizer = joblib.load("models/lo2_if.joblib")
sad = AnomalyDetector()
sad.item_list_col = "e_words"
sad.filter_anos = True
sad.prepare_train_test_data(vectorizer_class=vectorizer)
sad.model = model
predictions = sad.predict()
```

- Hold-out-Threshold aus `model.yml` entnehmen und im Score-Dataset anwenden.
- Bei neuen Logs zuerst den Loader erneut ausführen; Modell kann anschließend direkt auf den neuen Events scoren.

## Qualitätssicherung

- `python tools/lo2_result_scan.py --dry-run` listet erwartete Artefakte und meldet fehlende Dateien.
- Metriken regelmäßig versionieren (z. B. `metrics/run_2025-11-03/`), um Vergleiche zwischen Konfigurationen zu erhalten.
- Für SHAP-Läufe `MPLBACKEND=Agg` setzen, damit Plots auch ohne Display entstehen.

## Dateiaustausch

- Rohdaten nie mit Projektrepository teilen – stattdessen Parquet-Exports bereitstellen.
- Modelle/Explainability als ZIP austauschen (`models/`, `metrics/`, `explainability/`).
- Sensible Daten (Tokens, Secrets) anonymisieren; das Skript erzeugt keine, aber bei eigenen Erweiterungen darauf achten.

## Offene Aufgaben

- TODO: Automatisches Aufräumskript für alte Explainability-Ausgaben bereitstellen.
- TODO: Einheitliches Namensschema für Benchmark-Runs definieren (z. B. `<datum>_<feature_set>_<contamination>`).
