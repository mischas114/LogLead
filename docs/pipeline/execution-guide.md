---
title: Ausführungsleitfaden LO2-Pipeline
summary: Schritt-für-Schritt-Anleitung vom Loader bis zur Explainability.
last_updated: 2025-11-03
---

# Ausführungsleitfaden LO2-Pipeline

Dieses Dokument beschreibt den praktischen Ablauf, um die LO2-Demo innerhalb von zehn Minuten lauffähig zu bekommen.

## Voraussetzungen

- Python-Umgebung mit `loglead` und allen Abhängigkeiten (`pip install -e .` bzw. Poetry).
- LO2-Rohdaten in der Struktur `<root>/run_<id>/<test_case>/*.log`.
- Schreibrechte in `demo/result/lo2/` (standardmäßiger Output der Skripte).

## Quickstart

```bash
# 1) Loader (Phase B)
python demo/lo2_e2e/run_lo2_loader.py \
  --root /pfad/zu/lo2_runs \
  --runs 5 \
  --errors-per-run 1 \
  --service-types code token refresh-token \
  --save-parquet \
  --output-dir demo/result/lo2

# 2) Enhancement + Modelle (Phase C–E)
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --if-contamination 0.15 \
  --if-item e_words \
  --if-numeric e_chars_len \
  --save-model models/lo2_if.joblib \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --dump-metadata

# 3) Explainability (Phase F)
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.15 \
  --nn-top-k 50 \
  --shap-sample 200
```

> Tipp: `python demo/lo2_e2e/LO2_samples.py --list-models` zeigt alle Registry-Schlüssel. Ohne `--models` läuft das Default-Set (`event_lr_words,event_dt_trigrams,sequence_lr_numeric,sequence_shap_lr_words`).

## Phasenüberblick

- **Phase A – Setup:** Umgebung prüfen, Abhängigkeiten installieren.
- **Phase B – Loader:** `run_lo2_loader.py --save-parquet` erzeugt `lo2_events.parquet` und optional `lo2_sequences.parquet`.
- **Phase C – Enhancer:** Wird automatisch in `LO2_samples.py` ausgeführt (Normalisierung, Tokens, Drain, Längen).
- **Phase D – Isolation Forest:** Trainiert immer; Hold-out/Threshold per `--if-holdout-fraction`, `--if-threshold-percentile`.
- **Phase E – Registry-Modelle:** `--models` schaltet zusätzliche Modelle zu (Event/Sequence, supervised/unsupervised).
- **Phase F – Explainability:** `lo2_phase_f_explainability.py` generiert NN-Mapping, SHAP-Plots und False-Positive-Listen.

## Wichtige CLI-Flags

| Ebene | Flags | Zweck |
| --- | --- | --- |
| Loader | `--root`, `--runs`, `--errors-per-run`, `--service-types`, `--save-parquet`, `--output-dir`, `--load-metrics` | Datenumfang und Persistenz |
| Detector | `--phase`, `--if-*`, `--models`, `--list-models`, `--save-if`, `--save-model`, `--save-enhancers`, `--report-*`, `--metrics-dir`, `--dump-metadata` | Feature-Auswahl, Modellkonfiguration, Artefakte |
| Explainability | `--if-*`, `--nn-top-k`, `--nn-normal-sample`, `--shap-sample`, `--root` | Sampling und Ausgabeort für XAI |

## Artefaktindex

| Datei | Quelle | Beschreibung |
| --- | --- | --- |
| `demo/result/lo2/lo2_events.parquet` | Loader | Events mit Labels, Timestamps, `seq_id` |
| `demo/result/lo2/lo2_sequences.parquet` | Loader | Sequenzen (Run × Test × Service) |
| `demo/result/lo2/lo2_if_predictions.parquet` | LO2_samples | Scores, Schwellen, Rankings |
| `demo/result/lo2/metrics/*.json` | LO2_samples (`--report-*`) | Precision@k, FP-Rate, PSI |
| `models/lo2_if.joblib` | LO2_samples (`--save-model`) | IsolationForest + Vectorizer |
| `models/model.yml` | LO2_samples (`--dump-metadata`) | Parameter, Threshold, Git-Commit |
| `demo/result/lo2/explainability/*` | Phase F | NN-Mapping, SHAP-Plots, FP-Liste |

## Persistenz & Wiederverwendung

1. Loader immer mit `--save-parquet` ausführen, damit Events/Sequenzen reproduzierbar vorliegen.
2. Modelle via `--save-model` + `--dump-metadata` sichern. Wiederverwendung mit `--load-model` bzw. `joblib.load(...)`.
3. Enhanced-Parquets (`--save-enhancers`) nur für Notebook-Analysen nutzen; Skripte generieren Features bei jedem Lauf neu.
4. Artefakte regelmäßig mit `tools/lo2_result_scan.py --dry-run` prüfen.

## Fehlerdiagnose

- Fehlen Parquets → Loader erneut mit identischen Flags ausführen.
- Schwache IF-Scores → mehr Normaldaten laden, `--if-item` wechseln, numerische Features ergänzen.
- SHAP-Fehler → `--shap-sample` reduzieren, um RAM zu sparen.
