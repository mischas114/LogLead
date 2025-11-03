---
title: Verbesserungsplan LO2
summary: Offene Aufgaben und Prioritäten für Daten, Modelle und Dokumentation.
last_updated: 2025-11-03
---

# Verbesserungsplan LO2

## Prioritäten (Kurzfrist)

1. **Datenbasis stärken:** Mehr korrekte Runs laden, Sampling-Strategien dokumentieren.
2. **Benchmarking härten:** Einheitlichen Train/Test-Split implementieren, Ergebnisse zentral erfassen.
3. **Isolation Forest schärfen:** Feature-Mix erweitern, Schwellenableitung produktionsnah gestalten.

## Daten & Loader

- Normalanteil erhöhen (`--errors-per-run` variieren, zusätzliche korrekte Runs einbinden).
- Sampling-Strategien für Fehlerfälle dokumentieren (`dup_errors`, `single_error_type`).
- Metrics-/Trace-Daten evaluieren und ggf. in die Pipeline integrieren.
- TODO: Prüfen, ob Service-spezifisches Training (Token vs. Code) False Positives reduziert.

## Features & Modelle

- Numerische Sequenzmerkmale (`seq_len`, `duration_sec`, `unique_tokens`) in den IF integrieren.
- Alternativen zu Word-Features testen (Drain-IDs, n-Gramme, Hybrid-Ansätze).
- Registry-Modelle mit validem Split bewerten (run-basiert oder zeitlich).
- TODO: Automatisierten Hyperparameter-Sweep für `--if-contamination`, `--if-item`, `--if-numeric` aufsetzen.
- TODO: Leakage bei OOVDetector/RarityModel untersuchen und beheben.

## Explainability

- SHAP auch für RandomForest/XGBoost konfigurieren.
- False-Positive-Berichte interpretieren und bei wiederkehrenden Mustern Training anpassen.
- TODO: Kurzen Leitfaden erstellen, wie Explainability-Ergebnisse an Stakeholder kommuniziert werden.

## Operations & Doku

- Einheitliches Naming für Artefakte etablieren (z. B. `<datum>_<setup>`).
- `tools/lo2_result_scan.py` um optionale Auto-Fix-Mechanismen erweitern (z. B. leere Verzeichnisse anlegen).
- Dokumentationspflege regelmäßig (monatlich) prüfen.
- TODO: Beitragende bitten, Änderungen in `contributing/change-log.md` einzutragen.
