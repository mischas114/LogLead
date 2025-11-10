---
title: LO2 Change Log
summary: Zeitliche Übersicht wichtiger Dokumentations- und Pipelineänderungen.
last_updated: 2025-11-11
---

# LO2 Change Log

## 2025-11-03

- Dokumentation vollständig restrukturiert: neue Struktur (`overview`, `dataset`, `pipeline`, `models`, `operations`, `roadmap`, `contributing`).
- Historische Dateien nach `docs/_archive/2025-11-03/` verschoben und durch konsolidierte Nachfolger ersetzt.
- `demo/lo2e2e/README.md` als Einstieg (siehe Verzeichnis `demo/lo2e2e/`).
- Neue Leitfäden: Pipeline-Ausführung, Architektur, Isolation-Forest, Artefakt-Handling, Modellkatalog, Roadmap.
- Change Log eingeführt, um künftige Anpassungen nachvollziehbar zu halten.

## 2025-11-10

- Isolation-Forest-Leitfaden um Abschnitt „Runtime-Telemetrie“ erweitert; dokumentiert neue `[Resource] if_baseline`-Ausgabe aus `LO2_samples.py`.
- Modell-Insights aktualisiert: neue Sektion zu Speicher-Guards, deterministischen Seeds und Ressourcentelemetrie der supervised Modelle.
- Ausführungsleitfaden ergänzt um supervisierten Beispiel-Run ohne IsolationForest (`event_lr_words,event_dt_trigrams,event_xgb_words,sequence_shap_lr_words`).

## 2025-11-11

- Phase-F-Skript (`lo2_phase_f_explainability.py`) unterstützt jetzt `--skip-if`, `--sup-models`, `--nn-source`, Hold-out-Splits und `--list-models`; NN-Mapping/Fals-Positive-Listen können damit auf beliebige Registry-Modelle gelegt werden.
- Pipeline-Ausführungsleitfaden aktualisiert: neue Explainability-Beispielbefehle, Prompt-Anpassungen, Tabellen-Update zu Flags sowie Artefaktindex mit den pro Modell erzeugten Prediction-Files.
