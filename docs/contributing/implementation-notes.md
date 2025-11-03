---
title: Implementierungsnotizen LO2
summary: Wichtigste Architekturentscheidungen und Beiträge seit Einführung der LO2-Pipeline.
last_updated: 2025-11-03
---

# Implementierungsnotizen LO2

Dieses Dokument fasst zentrale Beiträge zusammen, die die LO2-Pipeline in ihren aktuellen Zustand versetzt haben. Es dient neuen Beitragenden als Orientierung.

## Schlüsselbeiträge

- **Verzeichnisstruktur:** Demo-Skripte nach `demo/lo2_e2e/` verschoben, konsistente Pfade nach `demo/result/lo2/`.
- **Modell-Registry:** 14 Modelle deklarativ konfigurierbar (`MODEL_REGISTRY`), CLI-Flags `--models` und `--list-models`.
- **Persistenz & Metriken:** `--save-model`, `--save-if`, `--report-*`, `--dump-metadata` ergänzt; `metrics_utils.py` für Precision@k/FPR/PSI.
- **Explainability:** `lo2_phase_f_explainability.py` baut IsolationForest nach, erzeugt NN-Mapping, SHAP-Plots, False-Positive-Liste.
- **Dokumentation:** Alte, verstreute Dateien konsolidiert (Pipeline-Guide, Architekturdetail, Modellleitfaden).
- **Tooling:** `tools/lo2_result_scan.py` für Artefaktprüfung; `model.yml` speichert Parameter + Git-Commit.

## Designprinzipien

- **CLI-first:** Jeder Pipeline-Schritt ist als Skript aufrufbar, Notebook-Einsatz optional.
- **Reproduzierbarkeit:** Standardpfade unter `demo/result/lo2/`, deterministische Artefaktbenennung.
- **Erklärbarkeit:** IsolationForest bleibt Pflichtlauf, weitere Modelle optional für Vergleich/SHAP.
- **Erweiterbarkeit:** Registry erlaubt das Hinzufügen neuer Modelle ohne Core-Änderungen.

## Wartungshinweise

- Änderungen an der Registry dokumentieren (`model-catalog.md` + `change-log.md`).
- Neue CLI-Flags in `execution-guide.md` ergänzen.
- Bei größeren Umbauten (z. B. neuer Loader-Feature-Pipeline) Explainer und Metrics-Skripte auf Kompatibilität prüfen.

## Offene Punkte

- TODO: Einheitliche Tests für Loader/Enhancer schreiben, um Regressionen früh zu erkennen.
- TODO: Guidelines für externe Beiträge (Code-Style, Review-Prozess) ergänzen.
