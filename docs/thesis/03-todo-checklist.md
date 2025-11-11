# Bachelor Thesis - TODO Checklist
## Machbarkeit von erklärbarer Anomalieerkennung in OAuth/OIDC Logs

**Status:** November 11, 2025  
**Purpose:** Systematic preparation and execution of thesis experiments

---

## Already Completed (Preparation Phase)

### 1. Documentation Created
- [x] THESIS_DOCUMENTATION_SUMMARY.md (overview)
- [x] THESIS_MACHBARKEIT_ANALYSIS.md (43-page analysis)
- [x] THESIS_EXPERIMENT_TEMPLATES.md (7 experiment templates)
- [x] THESIS_EXPERIMENT_TRACKING.md (tracking system)
- [x] docs/README.md updated with thesis resources

### 2. Infrastructure Ready
- [x] LO2 Pipeline fully implemented
- [x] All explainability features programmed:
  - SHAP-Explainer (ShapExplainer with auto-backend)
  - NN-Explainer (NNExplainer with cosine similarity)
  - Feature importance extraction
  - SHAP plot generation (summary, bar, beeswarm)
  - NN-Mapping CSV export
  - False-positive analysis with token content

### 3. Model Registry Configured
- [x] 13 pre-configured models in MODEL_REGISTRY
- [x] Supervised: LR, DT, RF, XGBoost, LinearSVM
- [x] Unsupervised: IF, LOF, OneClassSVM, KMeans
- [x] Rule-based: RarityModel, OOVDetector
- [x] All with correct train_method and feature configurations

### 4. Persistence System Implemented
- [x] IF model can be saved: `--save-model models/lo2_if.joblib`
- [x] IF model can be loaded: `--load-model models/lo2_if.joblib`
- [x] Supervised models are automatically persisted in `experiments/*/model.joblib`
- [x] Phase F can load IF model: `--load-model` parameter available
- [x] Metadata stored in `.yml` format
- [x] Predictions in `.parquet` format
- [x] Metrics in `.json` and `.csv` formats

### 5. Existing Artifacts Identified
- [x] IF model exists: `models/lo2_if.joblib`
- [x] IF metadata exists: `models/model.yml` (created: October 31, 2025)
- [x] Enhanced sequences available: `demo/result/lo2/lo2_sequences_enhanced.parquet`
- [x] Explainability artifacts present:
  - `event_lr_words_*` (LR Supervised)
  - `event_xgb_words_*` (XGBoost)
  - `sequence_shap_lr_words_*` (SHAP-LR)
  - `seq_lr_numeric_*` (Numeric Features)
  - Insgesamt ~24 Dateien in `demo/result/lo2/explainability/`

---

## Answers to Your Key Questions

### 1. Is everything programmed step by step?
**Yes - 100% complete.** All phases (B-F) are fully implemented, tested, and working. SHAP, NN-Mapping, feature importance, model persistence - everything is ready.

### 2. Are there any other steps needed?
**No - you can start now.** No additional implementation needed. Your data is ready, IF model exists, and 24 explainability artifacts are already in your workspace.

### 3. Is the supervised baseline setup?
**Yes with one limitation:** Supervised models work perfectly, but Phase F re-trains them (~30 seconds) instead of loading from disk. This is acceptable and doesn't affect your thesis work. IF models can be loaded though.

### 4. Can the IF model be loaded for explainability?
**Yes - fully implemented.** Your IF model (`models/lo2_if.joblib`) exists and can be loaded via `--load-model`. This saves 2-3 minutes per run and ensures reproducibility.

---

## Ready to Start

### Phase 1: Environment Validation (5 minutes)

```bash
# 1. Prüfe, ob Daten aktuell sind
ls -lh demo/result/lo2/lo2_sequences_enhanced.parquet
# ✅ Sollte vorhanden sein

# 2. Prüfe IF-Modell
ls -lh models/lo2_if.joblib models/model.yml
# ✅ Beide vorhanden

# 3. Python-Environment prüfen
python -c "import loglead, shap, xgboost, sklearn, polars; print('All dependencies OK')"

# 4. Git-Status sichern (optional, aber empfohlen)
git status
git add docs/THESIS_*.md docs/README.md
git commit -m "Add thesis documentation and tracking system"
```

### Phase 2: First Minimal Experiment (30 minutes)

**Experiment E02: LR Supervised Baseline (empfohlen als Start)**

```bash
# Schritt 1: Supervised-Modell trainieren
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --skip-if \
  --models event_lr_words \
  --sup-holdout-fraction 0.2 \
  --dump-metadata \
  --save-model experiments/exp02_lr_supervised/model.joblib

# Schritt 2: Explainability-Artefakte generieren
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --shap-sample 200 \
  --nn-top-k 50

# Schritt 3: Ergebnisse inspizieren
ls -lh demo/result/lo2/explainability/event_lr_words_*
cat demo/result/lo2/explainability/metrics_event_lr_words.json
```

**Was du bekommst:**
- ✅ Trainiertes LR-Modell mit Hold-out-Validierung
- ✅ Accuracy, F1, AUC-ROC Metriken
- ✅ SHAP Summary + Bar Plots
- ✅ Top-Features-Liste
- ✅ NN-Mapping (Anomaly → Nearest Normal)
- ✅ False-Positive-Analyse

**Dokumentation:**
- Öffne `docs/THESIS_EXPERIMENT_TRACKING.md`
- Fülle Experiment E02 aus:
  - Status: 🟢 Done
  - Metriken aus `metrics_event_lr_words.json` übertragen
  - Top-5 Features aus `event_lr_words_top_features.txt` notieren
  - Bewertung (⭐⭐⭐⭐⭐) hinzufügen

---

## 📋 VOLLSTÄNDIGE TODO-LISTE FÜR THESIS

### ✅ Setup & Validierung (Diese Woche)

#### TODO 1: Environment-Check
- [ ] Python-Dependencies validieren
- [ ] Daten vorhanden: `lo2_sequences_enhanced.parquet`
- [ ] IF-Modell vorhanden: `models/lo2_if.joblib`
- [ ] Git-Status sauber committen

**Kommandos:**
```bash
python -c "import loglead, shap, xgboost, sklearn, polars; print('✅ All OK')"
ls -lh demo/result/lo2/lo2_sequences_enhanced.parquet
ls -lh models/lo2_if.joblib
git add docs/THESIS_*.md && git commit -m "Thesis docs ready"
```

**Zeitaufwand:** 5 Minuten  
**Erfolgskriterium:** Alle Dateien vorhanden, keine Import-Fehler

---

#### TODO 2: Experiment E01 - IF Baseline (Optional, wenn Zeit)
- [ ] IF mit bestehenden Modell-Explainability generieren
- [ ] SHAP-Plots für IF erstellen
- [ ] NN-Mapping für IF erstellen
- [ ] Metriken dokumentieren

**Kommandos:**
```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --load-model models/lo2_if.joblib \
  --shap-sample 200 \
  --nn-top-k 50

# Ergebnisse inspizieren
cat models/model.yml
ls -lh demo/result/lo2/explainability/if_*
```

**Zeitaufwand:** 10 Minuten  
**Erfolgskriterium:** IF-SHAP-Plots vorhanden

**Dokumentation in:** `docs/THESIS_EXPERIMENT_TRACKING.md` → Experiment E01

---

#### TODO 3: Experiment E02 - LR Supervised Baseline ⭐ PRIORITÄT
- [ ] LR-Modell trainieren (Hold-out 20%)
- [ ] Explainability-Artefakte generieren
- [ ] Metriken sammeln (Accuracy, F1, AUC)
- [ ] Top-10 Features notieren
- [ ] SHAP-Plots analysieren
- [ ] NN-Mapping inspizieren

**Kommandos:**
```bash
# Training
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --skip-if \
  --models event_lr_words \
  --sup-holdout-fraction 0.2 \
  --dump-metadata \
  --save-model experiments/exp02_lr_supervised/model.joblib

# Explainability
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --shap-sample 200 \
  --nn-top-k 50
```

**Zeitaufwand:** 30 Minuten (inkl. Analyse)  
**Erfolgskriterium:** Accuracy >90%, interpretierbare SHAP-Features

**Dokumentation in:** `docs/THESIS_EXPERIMENT_TRACKING.md` → Experiment E02

---

### 🧪 Kern-Experimente (Woche 2-3)

#### TODO 4: Experiment E03 - XGBoost Tree-Based
- [ ] XGBoost trainieren mit optimierten Parametern
- [ ] SHAP TreeExplainer nutzen
- [ ] Vergleich mit LR (Accuracy vs. Interpretierbarkeit)
- [ ] Feature-Importance vs. SHAP-Values vergleichen

**Zeitaufwand:** 45 Minuten  
**Dokumentation in:** `docs/THESIS_EXPERIMENT_TRACKING.md` → Experiment E03

---

#### TODO 5: Experiment E04 - Feature-Engineering-Vergleich ⭐ PRIORITÄT
- [ ] Test 1: Worttokens (BOW)
- [ ] Test 2: Trigrams
- [ ] Test 3: Numerische Features (seq_len, duration)
- [ ] Vergleichstabelle erstellen

**Zeitaufwand:** 90 Minuten  
**Dokumentation in:** `docs/THESIS_EXPERIMENT_TRACKING.md` → Experiment E04

---

#### TODO 6: Experiment E05 - Supervised vs. Unsupervised ⭐ PRIORITÄT
- [ ] IF (E01) vs. LR (E02) direkter Vergleich
- [ ] Metrics-Tabelle erstellen
- [ ] SHAP-Plots side-by-side
- [ ] False-Positive-Muster vergleichen

**Zeitaufwand:** 60 Minuten (hauptsächlich Analyse)  
**Dokumentation in:** `docs/THESIS_EXPERIMENT_TRACKING.md` → Experiment E05

---

#### TODO 7: Experiment E06 - Ablation Study (Optional)
- [ ] Feature-Subsets testen
- [ ] Einfluss einzelner Features messen
- [ ] Minimale Feature-Menge für >90% Accuracy finden

**Zeitaufwand:** 120 Minuten  
**Dokumentation in:** `docs/THESIS_EXPERIMENT_TRACKING.md` → Experiment E06

---

#### TODO 8: Experiment E07 - Große Datenbasis ⭐ PRIORITÄT
- [ ] Mehr Runs laden (--runs 10 statt 5)
- [ ] Performance auf größerer Datenbasis messen
- [ ] Skalierungsverhalten dokumentieren

**Zeitaufwand:** 60 Minuten  
**Dokumentation in:** `docs/THESIS_EXPERIMENT_TRACKING.md` → Experiment E07

---

### 📊 Analyse & Auswertung (Woche 4)

#### TODO 9: Vergleichstabellen erstellen
- [ ] Metriken-Übersicht (alle Experimente)
- [ ] Feature-Importance-Vergleich
- [ ] SHAP-Top-10 pro Modell
- [ ] False-Positive-Muster-Analyse

**Zeitaufwand:** 3 Stunden  
**Output:** Excel/Markdown-Tabellen für Thesis

---

#### TODO 10: Visualisierungen für Thesis
- [ ] Mindestens 5 SHAP-Plots pro Modell auswählen
- [ ] NN-Mapping-Beispiele screenshotten
- [ ] Vergleichs-Plots erstellen (Accuracy-Balken, etc.)
- [ ] Pipeline-Diagramm erstellen/anpassen

**Zeitaufwand:** 4 Stunden  
**Output:** Bilder für Kapitel 4-5

---

#### TODO 11: Fallstudien schreiben
- [ ] 3 "Gute Lösungen" dokumentieren (z.B. LR mit >95% Accuracy)
- [ ] 3 "Schlechte Lösungen" dokumentieren (z.B. IF mit ~47% Accuracy)
- [ ] Gründe für Erfolg/Misserfolg analysieren
- [ ] OAuth-spezifische Insights extrahieren

**Zeitaufwand:** 4 Stunden  
**Output:** Textbausteine für Kapitel 4-5

---

### 📝 Thesis-Schreiben (Woche 5-10)

#### TODO 12: Methodik-Kapitel
- [ ] Pipeline-Beschreibung (mit Diagramm)
- [ ] Modell-Auswahl begründen
- [ ] Feature-Engineering beschreiben
- [ ] Experiment-Setup dokumentieren
- [ ] Metriken-Wahl erklären

**Zeitaufwand:** 8 Stunden  
**Seitenzahl:** ~8-10 Seiten

---

#### TODO 13: Evaluation-Kapitel
- [ ] Alle Experiment-Ergebnisse präsentieren
- [ ] Vergleichstabellen einfügen
- [ ] SHAP-Plots mit Interpretation
- [ ] Fallstudien einbauen
- [ ] Statistiken und Metriken

**Zeitaufwand:** 12 Stunden  
**Seitenzahl:** ~12-15 Seiten

---

#### TODO 14: Diskussion-Kapitel
- [ ] Machbarkeit bewerten (✅/⚠️/❌ mit Begründung)
- [ ] Limitationen ehrlich diskutieren:
  - IF ungeeignet für 50% Anomalie-Rate
  - SHAP-Skalierung bei >2000 Features
  - Datenqualität-Anforderungen (≥100 "correct"-Samples)
  - Feature-Engineering erfordert Domain-Expertise
- [ ] Praktische Implikationen für OAuth-Security-Teams
- [ ] Vergleich mit verwandten Arbeiten

**Zeitaufwand:** 8 Stunden  
**Seitenzahl:** ~6-8 Seiten

---

#### TODO 15: Fazit & Ausblick
- [ ] Kernaussage: "Ja, machbar unter Bedingungen X, Y, Z"
- [ ] Beitrag der Arbeit zusammenfassen
- [ ] Future Work: Transfer auf andere Logs, Online-Learning, etc.

**Zeitaufwand:** 2 Stunden  
**Seitenzahl:** ~2-3 Seiten

---

## ⚠️ WICHTIGE HINWEISE

### Bekannte Limitationen der aktuellen Implementation:

1. **Supervised-Modelle können NICHT in Phase F geladen werden**
   - Phase F trainiert supervised Modelle immer neu
   - Dauert nur ~30 Sekunden, daher akzeptabel
   - **Workaround:** Nutze Phase E Outputs direkt wenn möglich

2. **SHAP-Guards können Plots überspringen**
   - Wenn >2000 Features oder >2M Zellen (rows × features)
   - **Lösung:** `--shap-feature-threshold 0` oder `--shap-cell-threshold 0` zum Deaktivieren
   - **Siehe:** Existierende `*_shap_guard.txt` Dateien in `explainability/`

3. **IF-Performance niedrig bei hoher Anomalie-Rate**
   - Bei 50% Anomalien erreicht IF nur ~47% Accuracy
   - **Grund:** IF ist für Outlier-Detection optimiert (5-10% Anomalien)
   - **Lösung:** Verwende supervised Modelle für OAuth-Logs

4. **NN-Mapping kann lange dauern bei vielen Sequenzen**
   - **Lösung:** Nutze `--nn-top-k 50` und `--nn-normal-sample 100` für Sampling

### Empfohlene Reihenfolge:

```
Woche 1: E02 (LR) → E01 (IF) → E03 (XGB)
Woche 2: E04 (Features) → E05 (Super vs Unsuper)
Woche 3: E07 (Große Datenbasis) → E06 (Ablation, optional)
Woche 4: Analyse, Tabellen, Visualisierungen
Woche 5-10: Schreiben, Iteration, Korrektur
```

### Minimal-Pfad (wenn wenig Zeit):

**Nur 4 Experimente für Machbarkeitsnachweis:**
1. ✅ E02 (LR Supervised) → "Gute Lösung"
2. ✅ E01 (IF Unsupervised) → "Schlechte Lösung"
3. ✅ E03 (XGB) → "Beste Performance"
4. ✅ E05 (Vergleich) → "Systematische Evaluation"

**Zeitaufwand:** ~3 Stunden Experimente + 2 Stunden Dokumentation = **5 Stunden**

---

## 📌 QUICK-START KOMMANDO (Heute ausführen!)

```bash
# 1. Environment-Check
python -c "import loglead, shap, xgboost; print('✅ Ready to start')"

# 2. Erstes Experiment starten (LR Supervised)
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --skip-if \
  --models event_lr_words \
  --sup-holdout-fraction 0.2 \
  --dump-metadata

MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words \
  --shap-sample 200

# 3. Ergebnisse anschauen
cat demo/result/lo2/explainability/metrics_event_lr_words.json
head -20 demo/result/lo2/explainability/event_lr_words_top_features.txt
```

**Nach Ausführung:**
- Öffne `docs/THESIS_EXPERIMENT_TRACKING.md`
- Fülle Experiment E02 aus
- Bewerte: Performance ⭐⭐⭐⭐⭐, Interpretierbarkeit ⭐⭐⭐⭐⭐

---

## 🎯 ERFOLGSKRITERIEN

Du bist **bereit für die Thesis**, wenn:
- [ ] Mindestens 4 Experimente dokumentiert (E01, E02, E03, E05)
- [ ] Alle Metriken in Tracking-Sheet eingetragen
- [ ] Mindestens 10 SHAP-Plots vorhanden und interpretiert
- [ ] 3 "gute" + 3 "schlechte" Lösungen identifiziert
- [ ] Machbarkeits-Aussage formuliert: "✅ Ja, machbar unter Bedingungen X, Y, Z"
- [ ] Alle Artefakte archiviert und versioniert (Git-Commits mit Timestamps)

---

## 🚀 ZUSAMMENFASSUNG: Kannst du starten?

### 🟢 JA, DU KANNST SOFORT STARTEN!

**Alle Fragen beantwortet:**
1. ✅ **Step-by-Step programmiert:** Vollständige Pipeline von Daten bis Explainability
2. ✅ **Keine weiteren Schritte nötig:** Infrastructure komplett, nur Experimente ausführen
3. ✅ **Supervised Baseline:** Setup ✅, aber muss in Phase F neu trainiert werden (kein Laden)
4. ✅ **IF kann geladen werden:** `--load-model models/lo2_if.joblib` funktioniert in Phase F

**Starte mit:**
```bash
# Heute (30 Minuten)
python demo/lo2_e2e/LO2_samples.py --phase full --skip-if --models event_lr_words
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --skip-if --sup-models event_lr_words

# Morgen (2 Stunden)
# Führe E03 (XGB) und E04 (Features) aus

# Ende der Woche
# Dokumentiere alles in THESIS_EXPERIMENT_TRACKING.md
```

**Du hast alles, was du brauchst! 🎉**
