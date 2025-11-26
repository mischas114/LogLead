# Repository-Analyse für Kapitel 3 (Versuchsaufbau & Methodik)

**Datum:** 2025-11-26  
**Thema:** Machbarkeit erklärbarer Anomalieerkennung in OAuth/OIDC Logs  
**Basis:** LogLead LO2-Pipeline (`demo/lo2_e2e/`)

---

## Vorbemerkung

Diese Analyse basiert auf dem aktuellen Stand der Repository-Dokumentation und Code-Implementierung. Die referenzierten Typst-Dateien (`02-related-work.typ`, `03-methodik.typ`) wurden im Repository **nicht gefunden**. Die Bewertung erfolgt anhand der vorhandenen Markdown-Dokumentation in `docs/thesis/` und der Pipeline-Implementierung in `demo/lo2_e2e/`.

---

## 3.1 Zielbild der Pipeline

### ✅ Vorhanden
- Mermaid-Diagramm in `docs/pipeline/architecture.md` (Flowchart Phasen A–F)
- Komponenten klar benannt: Loader → Enhancer → Models → Explainability
- Phasenbasierte Struktur (A–F) dokumentiert

### ❌ Fehlt / Verbesserungsbedarf
- Keine **eigenständige Thesis-Abbildung** – Mermaid-Diagramm ist Code-nah, nicht BA-geeignet
- **Unterschied zu LogLead-Vanilla fehlt völlig** (keine Abgrenzung zur Originalbibliothek)
- Implizites "LogLead-Handbuch"-Niveau in `README.md` und `execution-guide.md`
- Begründung für Phasenaufteilung (warum A–F?) nicht konzeptionell erklärt
- Zielgruppe (Security-Team, Entwickler, Wissenschaftler) nicht definiert

### ⚠️ Zu technisch / Implementierungsnah
- CLI-Parameter in `execution-guide.md` dominieren über konzeptionelle Beschreibung
- Mermaid-Diagramm enthält Dateinamen (`lo2_sequences_enhanced.parquet`)
- Kein Überblick über Datenfluss-Mengen (Zeilen, Features) auf konzeptioneller Ebene

### 📌 Empfehlung
- Eigene Abbildung erstellen (PDF/SVG für Thesis)
- 1–2 Absätze: "Warum wurde LogLead gewählt? Was wurde angepasst?"
- Verweise auf Abschnitt 2.3 (LogLead-Grundlagen) einfügen

---

## 3.2 Datengrundlage und Auswahl

### ✅ Vorhanden
- `LO2Loader` in `run_lo2_loader.py` dokumentiert
- Filter: `--runs`, `--errors-per-run`, `--service-types`, `--single-error-type`
- Services beschrieben: `code`, `token`, `refresh-token`, `user`, etc.
- Sequenz-Definition: `seq_id = run + test_case + service`

### ❌ Fehlt / Verbesserungsbedarf
- **Explizite Einschränkungen fehlen**: Welche Services wurden warum gewählt?
- **Keine Referenz auf Kapitel 2.2** (LO2-Architektur) – sollte als Querverweis stehen
- Datenvolumen (Runs, Sequenzen, Events) nicht quantifiziert
- Fehlertypen nicht kategorisiert (OAuth-spezifisch, generisch, etc.)
- Keine Begründung für `init_lines_to_skip = 100` (Label-Leakage-Vermeidung)

### ⚠️ Redundanzen mit Kapitel 2
- LO2-Ordnerstruktur (`run_*/test_case/*.log`) sollte in 2.2 stehen, nicht in 3.2
- Service-Typen-Liste ist Architekturwissen, nicht Methodenwahl

### 📌 Empfehlung
- Satz einfügen: "Die LO2-Datenarchitektur wurde in Abschnitt 2.2 beschrieben."
- Tabelle mit finalen Datenauswahlkriterien (Service-Filter, Run-Anzahl, Fehlerquote)
- Begründung: "Fokus auf `code`, `token`, `refresh-token` wegen OAuth-Flow-Relevanz"

---

## 3.3 Vorverarbeitung

### ✅ Vorhanden
- **Sequenzdefinition**: `seq_id = run__test_case__service` (aus Code ableitbar)
- **Label-Ableitung**: `anomaly = 1` für error-Testfälle, `anomaly = 0` für correct
- Feature-Schritte: `normalize()`, `words()`, `trigrams()`, `parse_drain()`, `length()`
- Sequenz-Aggregation: `seq_len()`, `duration()`, `tokens()`

### ❌ Fehlt / Verbesserungsbedarf
- **Präzise Sequenzdefinition fehlt im Text** – nur aus Code ersichtlich
- Labeling-Logik nicht explizit: "Warum ist `test_case != correct` eine Anomalie?"
- Keine Begründung für Tokenisierung (Interpretierbarkeit, Feature-Reduktion)
- Drain-Parsing als optional erwähnt, aber nicht begründet

### ⚠️ Zu technisch
- Polars-Spalten (`e_words`, `e_trigrams`, `e_event_drain_id`) dominieren
- CLI-Pfade und Parameter (`--save-parquet`, `--output-dir`) gehören nicht hierher
- `EventLogEnhancer` / `SequenceEnhancer` als Klassennamen sind Code-Details

### 📌 Empfehlung
- Definition: "Eine Sequenz umfasst alle Log-Zeilen eines Service innerhalb eines Test-Runs."
- Satz: "Labels werden aus dem Testfall-Typ abgeleitet: `correct` → normal, sonst → Anomalie."
- Kurze Begründung: "Worttokens ermöglichen interpretierbare Feature-Namen."

---

## 3.4 Feature-Engineering

### ✅ Vorhanden
- **Feature-Typen dokumentiert**: Worttokens (BOW), Trigrams, numerisch (`seq_len`, `duration_sec`)
- Drain-Templates als optionales Feature erwähnt
- `MODEL_REGISTRY` definiert Feature-Mapping pro Modell
- Vectorizer-Parameter: `max_features`, `min_df`, `binary`

### ❌ Fehlt / Verbesserungsbedarf
- **Keine kompakte Feature-Set-Tabelle** in Thesis-geeignetem Format
- **Begründung für Auswahl fehlt**: Warum Worttokens statt TF-IDF? Warum Trigrams?
- Keine Diskussion: leichtgewichtig vs. ausdrucksstark
- Feature-Dimensionalität nicht beziffert (typisch: 5.000–60.000 Features)

### ⚠️ Zu technisch
- `CountVectorizer`-Parameter im Text (`max_features=40000`, `min_df=5`)
- `VECTOR_KWARGS`, `VECTORIZER_DEFAULTS` als Code-Referenzen
- Modelspezifische `vectorizer_kwargs` im Registry-Format

### 📌 Empfehlung
- Tabelle erstellen:

| Feature-Set | Eingabe | Typ | Dimension | Begründung |
|-------------|---------|-----|-----------|------------|
| `e_words` | Log-Tokens | BOW | ~5.000 | Interpretierbar, leichtgewichtig |
| `e_trigrams` | Zeichenketten | N-Gram | ~40.000 | Kontexterfassung |
| `seq_len`, `duration_sec` | Sequenz-Metrik | Numerisch | 2–4 | Strukturanomalie-Erkennung |

- Satz: "Worttokens wurden gewählt, da sie direkte Interpretierbarkeit ermöglichen."

---

## 3.5 Modelle

### ✅ Vorhanden
- **13 Modelle in MODEL_REGISTRY** dokumentiert
- Supervised: LR, DT, RF, XGBoost, LinearSVM
- Unsupervised: IsolationForest, LOF, OneClassSVM, KMeans
- Rule-based: RarityModel, OOVDetector
- Hauptidee pro Modell: Beschreibung im Registry (`"description": "..."`)

### ❌ Fehlt / Verbesserungsbedarf
- **Einsatzbegründung pro Modell unklar**: Warum LR? Warum XGBoost?
- Kein Bezug zu Feature-Sets: "LR arbeitet auf Worttokens, weil..."
- **Hinweis auf Kapitel 4 (Setup/HPs) fehlt** – HP-Details sollten dorthin
- Keine Diskussion: interpretierbar vs. performant

### ⚠️ Zu technisch
- `train_kwargs` mit vollständigen Hyperparameter-Listen (`max_depth=8`, `n_estimators=120`)
- `train_method` als String-Referenz (`"train_LR"`, `"train_XGB"`)
- CLI-Parameter (`--models event_lr_words,event_xgb_words`)

### 📌 Empfehlung
- Kurztabelle mit Modellauswahl und Begründung:

| Modell | Typ | Begründung |
|--------|-----|------------|
| LogisticRegression | Supervised | Interpretierbare Koeffizienten, SHAP-kompatibel |
| XGBoost | Supervised | Beste Performance erwartet, TreeSHAP verfügbar |
| IsolationForest | Unsupervised | Referenz für Label-freies Szenario |

- Satz: "Hyperparameter-Optimierung erfolgt in Kapitel 4."

---

## 3.6 Explainability

### ✅ Vorhanden
- **SHAP-Integration**: ShapExplainer für LR, DT, RF, XGBoost
- **NNExplainer**: Cosine-Similarity für Anomaly → Nearest Normal
- Artefakte: `*_shap_summary.png`, `*_top_features.txt`, `*_nn_mapping.csv`
- Guards: `feature_warning_threshold`, `sample_warning_threshold`

### ❌ Fehlt / Verbesserungsbedarf
- **Welche Modelle erhalten SHAP?** – Nicht explizit dokumentiert (aus Code: `SHAP_CAPABLE_METHODS`)
- Artefakte-Typ nicht klar: "Output-Formate" statt "Ergebnisse" als Titel
- **NNExplainer konzeptionell nicht erklärt** – nur Code-Referenz
- Keine Diskussion: global vs. local Erklärungen

### ⚠️ Zu technisch
- `ShapExplainer(detector, **shap_kwargs)` als API-Aufruf
- `--shap-sample 200`, `--shap-background 256` als CLI-Parameter
- `calc_shapvalues()`, `plot(plottype="summary")` als Methodenaufrufe

### 📌 Empfehlung
- Tabelle:

| Modell | SHAP-Backend | Artefakte |
|--------|--------------|-----------|
| LR | LinearExplainer | Global Rankings, Local Waterfall |
| XGBoost | TreeExplainer | Feature-Importance, Summary-Plot |
| IsolationForest | TreeExplainer (begrenzt) | NNExplainer stattdessen |

- NNExplainer-Absatz: "Für nicht SHAP-kompatible Modelle wird der nächste Normalfall als Kontrastbeispiel angezeigt."
- Umbenennen: "Ergebnisse" → "Erklärungsartefakte" oder "Output-Formate"

---

## 3.7 Evaluationsstrategie

### ✅ Vorhanden
- **Metriken**: Accuracy, F1, AUC-ROC (in `compute_metrics()`)
- **Zusätzlich**: Precision@k, FP-Rate@α, PSI (in `metrics_utils.py`)
- **Splitting**: Run-basierter Hold-out (`_run_based_holdout_split()`)
- Hold-out-Parameter: `--sup-holdout-fraction 0.2`

### ❌ Fehlt / Verbesserungsbedarf
- **Explainability-Metriken fehlen**: Keine Fidelity, kein Overhead
- **Splitting-Strategie nicht konzeptionell erklärt**: Warum run-basiert?
- Keine Varianten diskutiert (z. B. stratified, temporal, service-basiert)
- Begründung für 20% Hold-out fehlt

### ⚠️ Zu technisch
- `precision_at_k()`, `false_positive_rate_at_alpha()`, `population_stability_index()` als Funktionsnamen
- CLI-Parameter (`--report-precision-at 100`, `--report-fp-alpha 0.005`)

### 📌 Empfehlung
- Metriken-Tabelle:

| Kategorie | Metrik | Zweck |
|-----------|--------|-------|
| Performance | F1-Score | Balance Precision/Recall |
| Performance | AUC-ROC | Ranking-Qualität |
| Explainability | Fidelity (TODO) | Erklärungstreue |
| Explainability | Overhead (TODO) | Laufzeitkosten |

- Satz: "Der Hold-out erfolgt run-basiert, um Leakage zwischen Trainings- und Test-Sequenzen zu vermeiden."

---

## Zusatzprüfungen (global über Kapitel 3)

### Überschneidungen mit Kapitel 2

| Inhalt | Aktuell in | Sollte in |
|--------|-----------|-----------|
| LO2-Ordnerstruktur | 3.2 / architecture.md | 2.2 (LO2-Architektur) |
| OAuth-Flow-Grundlagen | implicit in 3.2 | 2.1 (OAuth-Grundlagen) |
| LogLead-Features | 3.4 | 2.3 (LogLead-Übersicht) |
| SHAP-Theorie | fehlt | 2.3 oder 2.4 (XAI-Grundlagen) |

### Verrutscht: Methodik → Ergebnisse
- `06-feasibility-analysis.md` enthält erwartete Metriken (Accuracy ~97%) – gehört in Kapitel 4/5
- `03-todo-checklist.md` mischt Planung und Auswertung

### Fehlende Verweise
- Keine `vgl. Abschnitt 2.2` oder `siehe Kapitel 2.3` in der aktuellen Dokumentation
- Pipeline-Architektur referenziert keine Related-Work-Konzepte

### GitHub-README-Niveau
- `demo/lo2_e2e/README.md`: Quickstart-Format, nicht BA-geeignet
- `docs/pipeline/execution-guide.md`: CLI-zentriert, technisches Handbuch
- `docs/thesis/*.md`: Zwischen Thesis-Text und Arbeitsdokumentation vermischt

### Technische Tiefe (nicht BA-geeignet)
- Hyperparameter-Listen in MODEL_REGISTRY (gehören in Anhang oder Kapitel 4)
- Vectorizer-Parameter (`max_features=40000`, `min_df=5`)
- Polars-Spalten als Referenz statt konzeptioneller Beschreibung

### Fehlende Transition-Sätze
- Kein Übergang: "Nach der Datenauswahl (3.2) erfolgt die Vorverarbeitung (3.3)..."
- Kein Abschluss: "Zusammenfassend definiert dieses Kapitel den Versuchsaufbau für..."

---

## Zusammenfassung pro Abschnitt

### 3.1 Zielbild der Pipeline
- Eigene Abbildung fehlt
- Unterschied zu LogLead-Vanilla fehlt
- Zu viele Dateinamen und CLI-Parameter

### 3.2 Datengrundlage und Auswahl
- Filter beschrieben, aber nicht begründet
- Keine Referenz auf Kapitel 2.2
- Datenvolumen nicht quantifiziert

### 3.3 Vorverarbeitung
- Sequenzdefinition nur aus Code erschließbar
- Label-Ableitung implizit
- Polars-Spalten statt konzeptioneller Begriffe

### 3.4 Feature-Engineering
- Feature-Typen vorhanden, Tabelle fehlt
- Keine Begründung für Auswahl
- Vectorizer-Parameter zu technisch

### 3.5 Modelle
- 13 Modelle dokumentiert
- Einsatzbegründung fehlt
- Hyperparameter-Details gehören in Kapitel 4

### 3.6 Explainability
- SHAP und NNExplainer vorhanden
- Modell-SHAP-Zuordnung unklar
- "Ergebnisse" sollte "Output-Formate" heißen

### 3.7 Evaluationsstrategie
- Performance-Metriken vorhanden
- Explainability-Metriken fehlen
- Splitting-Begründung fehlt

---

## Nächste Schritte

1. **Typst-Datei erstellen** (`03-methodik.typ`) mit den oben beschriebenen Inhalten
2. **Abbildung zeichnen** (Pipeline-Übersicht, nicht Mermaid)
3. **Querverweis-Struktur** definieren (vgl. 2.2, siehe Anhang A)
4. **Tabellen vorbereiten** (Feature-Sets, Modelle, Metriken)
5. **Hyperparameter-Details auslagern** in Kapitel 4 oder Anhang

---

**Erstellt:** 2025-11-26  
**Basis:** Repository-Stand nach Commit des GitHub-PR
