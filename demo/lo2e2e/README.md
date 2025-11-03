---
title: LO2 End-to-End Demo
summary: Einstieg in die Ausführung der LO2-Pipeline inkl. Quickstart und Wartungshinweisen.
last_updated: 2025-11-03
---

# LO2 End-to-End Demo

Die LO2-Demo zeigt, wie LogLead OAuth2-Logs lädt, Features erzeugt, den Isolation Forest trainiert und Explainability-Artefakte erstellt. Alle Skripte liegen im Ordner `demo/lo2_e2e/`; dieses README bündelt die wichtigsten Schritte, ohne dass weitere Dokumente nötig sind.

## Quickstart

Voraussetzungen: Python-Umgebung mit `loglead`, LO2-Rohdaten in `<root>/run_<id>/<test_case>` und Schreibrechte unter `demo/result/lo2/`.

```bash
# 1. Loader (Events + Sequenzen persistieren)
python demo/lo2_e2e/run_lo2_loader.py \
  --root /pfad/zu/lo2_runs \
  --runs 5 \
  --errors-per-run 1 \
  --service-types code token refresh-token \
  --save-parquet \
  --output-dir demo/result/lo2

# 2. Pipeline (Features, Isolation Forest, Registry-Modelle)
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --if-contamination 0.15 \
  --if-holdout-fraction 0.05 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-model models/lo2_if.joblib \
  --dump-metadata

# 3. Explainability (SHAP + Nearest Neighbours)
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.15 \
  --nn-top-k 50 \
  --shap-sample 200
```

Dauer bis zu ersten Ergebnissen: < 10 Minuten bei kleinen Datensätzen (≤5 Runs).

## Demo-Workflow

1. `run_lo2_loader.py` extrahiert Events/Sequenzen aus den Logs und speichert sie als Parquet.
2. `LO2_samples.py` erzeugt Features, trainiert IsolationForest und optionale Registry-Modelle, schreibt Scores, Metriken und Modell-Bundles.
3. `lo2_phase_f_explainability.py` wiederholt das IF-Setup, erstellt NN-Mapping, SHAP-Plots und False-Positive-Listen.

Alle Befehle akzeptieren weitere Flags (z. B. `--list-models`, `--if-item`, `--save-enhancers`). Details stehen in `docs/pipeline/execution-guide.md`.

## Verzeichnisstruktur

| Pfad | Inhalt |
| --- | --- |
| `demo/lo2_e2e/run_lo2_loader.py` | Loader-CLI, erzeugt `lo2_events.parquet` und `lo2_sequences.parquet`. |
| `demo/lo2_e2e/LO2_samples.py` | Orchestriert Enhancer, IsolationForest, Modell-Registry, Persistenz. |
| `demo/lo2_e2e/lo2_phase_f_explainability.py` | Explainability-Kit (NNExplainer, SHAP, FP-Listen). |
| `demo/lo2_e2e/metrics_utils.py` | Kennzahlenhelfer für Precision@k, FP-Rate, PSI. |
| `demo/result/lo2/` | Standardausgabeordner für Parquets, Metriken, Explainability. |

## Häufige Aufgaben

- **Andere Modellpakete testen:** `python demo/lo2_e2e/LO2_samples.py --phase full --models event_lof_words,event_oov_words`.
- **Enhanced-Parquets speichern:** `--save-enhancers` setzen (nur für Notebook-Analysen nötig).
- **Vorhandenes Modell wiederverwenden:** `LO2_samples.py --phase if --load-model models/lo2_if.joblib`.
- **Artefakte prüfen:** `python tools/lo2_result_scan.py --dry-run`.

## Wartung & Beitrag

- Neue CLI-Flags oder Modelle in `docs/pipeline/execution-guide.md` bzw. `docs/models/model-catalog.md` dokumentieren.
- Artefakte unter `demo/result/lo2/` regelmäßig bereinigen oder versionieren (z. B. `metrics/2025-11-03/`).
- Änderungen an der Demo hier notieren und zusätzlich in `docs/contributing/change-log.md` festhalten.
- TODO: Automatisierte Tests für Loader/Enhancer ergänzen, sobald repräsentative Testdaten verfügbar sind.

## Weiterführende Links

- `docs/overview/pipeline-overview.md` – Gesamtüberblick.
- `docs/pipeline/execution-guide.md` – Detailanleitung mit Artefaktindex.
- `docs/pipeline/isolation-forest.md` – Parameterempfehlungen und Verbesserungsplan für den IF.
- `docs/models/model-catalog.md` – Registry-Schlüssel & Beschreibung.
