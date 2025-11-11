# Thesis Experiment Tracking Sheet

**Bachelorarbeit:** Machbarkeit erklärbarer Anomalieerkennung in OAuth/OIDC Logs  
**Autor:** [Dein Name]  
**Datum Start:** [Datum]

---

## Experiment-Übersicht

| ID | Titel | Status | Datum | Priorität | Notizen |
|----|-------|--------|-------|-----------|---------|
| E01 | IF Baseline | 🔴 Todo | - | Hoch | Unsupervised Referenz |
| E02 | LR Supervised | 🔴 Todo | - | Hoch | Supervised Baseline |
| E03 | XGBoost | 🔴 Todo | - | Mittel | Tree-based Performance |
| E04 | Feature-Vergleich | 🔴 Todo | - | Hoch | Words vs Trigrams vs Numeric |
| E05 | Super vs Unsuper | 🔴 Todo | - | Hoch | Direkter Vergleich |
| E06 | Ablation Study | 🔴 Todo | - | Mittel | Feature-Relevanz prüfen |
| E07 | Große Datenbasis | 🔴 Todo | - | Hoch | Realistische Settings |
| E08 | Ensemble | 🔴 Todo | - | Niedrig | Optional |
| E09 | Threshold-Tuning | 🔴 Todo | - | Mittel | Precision optimieren |
| E10 | Production-Sim | 🔴 Todo | - | Mittel | Integration testen |

**Legende:**
- 🔴 Todo
- 🟡 In Progress
- 🟢 Done
- 🔵 Analysiert
- ⚫ Dokumentiert

---

## Experiment E01: Isolation Forest Baseline

### Meta-Informationen
- **Datum:** [Datum]
- **Dauer:** [X Minuten]
- **Status:** 🔴 Todo → 🟡 In Progress → 🟢 Done → 🔵 Analysiert → ⚫ Dokumentiert

### Setup
```bash
# Kommando (ausgefüllt nach Ausführung)
python demo/lo2_e2e/run_lo2_loader.py --root ... --runs ... --save-parquet
python demo/lo2_e2e/LO2_samples.py --phase if --if-contamination ... --save-model ...
python demo/lo2_e2e/lo2_phase_f_explainability.py --load-model ... --shap-sample ...
```

### Parameter
- Runs geladen: ____
- Contamination: ____
- Hold-out: ____
- SHAP-Samples: ____

### Artefakte-Checklist
- [ ] `experiments/exp01_if_baseline/model.joblib`
- [ ] `experiments/exp01_if_baseline/model.yml`
- [ ] `experiments/exp01_if_baseline/metrics/if_metrics.json`
- [ ] `demo/result/lo2/lo2_if_predictions.parquet`
- [ ] `demo/result/lo2/explainability/if_shap_summary.png`
- [ ] `demo/result/lo2/explainability/if_shap_bar.png`
- [ ] `demo/result/lo2/explainability/if_nn_mapping.csv`
- [ ] `demo/result/lo2/explainability/if_false_positives.txt`

### Metriken (aus Auswertung)
| Metrik | Wert | Interpretation |
|--------|------|----------------|
| Accuracy | _____ | |
| F1-Score | _____ | |
| Precision@100 | _____ | |
| FP-Rate@0.01 | _____ | |
| PSI | _____ | |
| False-Positives | _____ | |
| False-Negatives | _____ | |

### Top-5 SHAP Features
1. ________________
2. ________________
3. ________________
4. ________________
5. ________________

### NN-Mapping Beispiele (manuell inspiziert)
**Beispiel 1:**
- Anomaly ID: ________________
- Normal ID: ________________
- Unterschied: ________________________________________________

**Beispiel 2:**
- Anomaly ID: ________________
- Normal ID: ________________
- Unterschied: ________________________________________________

### False-Positive-Muster
- Häufigster FP-Typ: ________________________________________________
- Grund für Fehlklassifikation: ________________________________________________
- Lösungsansatz: ________________________________________________

### Interpretation (eigene Worte)
```
[Hier deine Interpretation einfügen]
- Was funktioniert gut?
- Wo sind Schwächen?
- Wie interpretierbar sind die Erklärungen?
- Praxistauglich für Security-Team?
```

### Bewertung Machbarkeit
- **Performance:** ⭐⭐⭐⭐⭐ (1-5)
- **Interpretierbarkeit:** ⭐⭐⭐⭐⭐
- **Aufwand:** ⭐⭐⭐⭐⭐
- **Praxistauglichkeit:** ⭐⭐⭐⭐⭐

### Nächste Schritte / Offene Fragen
- [ ] ________________________________________________
- [ ] ________________________________________________

---

## Experiment E02: Logistic Regression (Supervised)

### Meta-Informationen
- **Datum:** [Datum]
- **Dauer:** [X Minuten]
- **Status:** 🔴 Todo → 🟡 In Progress → 🟢 Done → 🔵 Analysiert → ⚫ Dokumentiert

### Setup
```bash
# Kommando
python demo/lo2_e2e/LO2_samples.py --phase full --skip-if --models event_lr_words ...
python demo/lo2_e2e/lo2_phase_f_explainability.py --skip-if --sup-models event_lr_words ...
```

### Parameter
- Runs geladen: ____
- Hold-out: ____
- Shuffle: [Ja/Nein]
- SHAP-Samples: ____

### Artefakte-Checklist
- [ ] `experiments/exp02_lr_supervised/model.joblib`
- [ ] `experiments/exp02_lr_supervised/model.yml`
- [ ] `demo/result/lo2/explainability/event_lr_words_predictions.parquet`
- [ ] `demo/result/lo2/explainability/metrics_event_lr_words.json`
- [ ] `demo/result/lo2/explainability/event_lr_words_shap_summary.png`
- [ ] `demo/result/lo2/explainability/event_lr_words_top_features.txt`
- [ ] `demo/result/lo2/explainability/event_lr_words_nn_mapping.csv`

### Metriken
| Metrik | Wert | Vergleich zu E01 |
|--------|------|------------------|
| Accuracy | _____ | [+/- _____] |
| F1-Score | _____ | [+/- _____] |
| AUC-ROC | _____ | [N/A für IF] |
| Training-Zeit | _____ s | [+/- _____ s] |
| False-Positives | _____ | [+/- _____] |

### Top-5 SHAP Features
1. ________________ (SHAP-Wert: _____)
2. ________________ (SHAP-Wert: _____)
3. ________________ (SHAP-Wert: _____)
4. ________________ (SHAP-Wert: _____)
5. ________________ (SHAP-Wert: _____)

### LR-Koeffizienten (direkt aus Modell)
1. ________________ (Coef: _____)
2. ________________ (Coef: _____)
3. ________________ (Coef: _____)
4. ________________ (Coef: _____)
5. ________________ (Coef: _____)

### SHAP vs. Koeffizienten Konsistenz
- Top-3 übereinstimmend: [Ja/Nein]
- Diskrepanzen: ________________________________________________

### Interpretation
```
[Deine Interpretation]
```

### Bewertung
- **Performance:** ⭐⭐⭐⭐⭐
- **Interpretierbarkeit:** ⭐⭐⭐⭐⭐
- **Aufwand:** ⭐⭐⭐⭐⭐
- **Praxistauglichkeit:** ⭐⭐⭐⭐⭐

---

## Experiment E03: XGBoost

[Gleiche Struktur wie E02]

---

## Experiment E04: Feature-Vergleich

### Meta-Informationen
- **Datum:** [Datum]
- **Dauer:** [X Minuten]
- **Status:** 🔴 Todo → 🟡 In Progress → 🟢 Done → 🔵 Analysiert → ⚫ Dokumentiert

### Setup
```bash
# Test 1: Worttokens
python demo/lo2_e2e/LO2_samples.py --models event_lr_words ...

# Test 2: Trigrams
python demo/lo2_e2e/LO2_samples.py --models event_dt_trigrams ...

# Test 3: Numerisch
python demo/lo2_e2e/LO2_samples.py --models sequence_lr_numeric ...
```

### Vergleichstabelle
| Feature-Typ | Modell | Accuracy | F1 | Feature-Anzahl | Training-Zeit | Top-Feature |
|-------------|--------|----------|-----|----------------|---------------|-------------|
| Worttokens | LR | _____ | _____ | _____ | _____ s | ____________ |
| Trigrams | DT | _____ | _____ | _____ | _____ s | ____________ |
| Numerisch | LR | _____ | _____ | _____ | _____ s | ____________ |

### Interpretation
```
[Welche Feature-Repräsentation ist am besten?]
[Trade-offs zwischen Performance und Interpretierbarkeit?]
```

### Empfehlung
**Beste Feature-Wahl:** ________________  
**Begründung:** ________________________________________________

---

## Experiment E05: Supervised vs. Unsupervised

### Vergleichstabelle
| Modell | Typ | Accuracy | F1 | Labeling-Aufwand | Interpretierbarkeit |
|--------|-----|----------|-----|------------------|---------------------|
| IF | Unsupervised | _____ | _____ | Keiner | ⭐⭐ |
| LR | Supervised | _____ | _____ | ~2h | ⭐⭐⭐⭐⭐ |
| XGB | Supervised | _____ | _____ | ~2h | ⭐⭐⭐⭐ |

### Trade-off-Analyse
**Labeling-Aufwand vs. Performance-Gewinn:**
- Aufwand: _____ Stunden für _____ Sequenzen
- Performance-Gewinn: _____ (Accuracy-Differenz)
- **Lohnenswert?** [Ja/Nein]

### Interpretation
```
[Wann ist Supervised sinnvoll?]
[Kann Unsupervised als Pre-Filter dienen?]
```

---

## Experiment E06: Ablation Study

### Top-Features (aus SHAP)
1. ________________
2. ________________
3. ________________
4. ________________
5. ________________

### Ablation-Ergebnisse
| Konfiguration | Accuracy | Performance-Drop |
|---------------|----------|------------------|
| Baseline (alle Features) | _____ | 0.0 |
| Ohne Top-1 Feature | _____ | _____ |
| Ohne Top-5 Features | _____ | _____ |
| Ohne Top-10 Features | _____ | _____ |

### Interpretation
```
[Sind SHAP-Features tatsächlich kritisch?]
[Wie viele Features sind minimal nötig?]
```

---

## Experiment E07: Große Datenbasis

### Datenlage-Vergleich
| Datensatz | Runs | Sequenzen | "Correct" | Anomalien |
|-----------|------|-----------|-----------|-----------|
| Klein (E01-E06) | _____ | _____ | _____ | _____ |
| Groß (E07) | _____ | _____ | _____ | _____ |

### Performance-Vergleich
| Modell | Accuracy (klein) | Accuracy (groß) | Verbesserung |
|--------|------------------|-----------------|--------------|
| LR | _____ | _____ | +_____ |
| DT | _____ | _____ | +_____ |
| RF | _____ | _____ | +_____ |
| XGB | _____ | _____ | +_____ |

### Interpretation
```
[Wie wichtig ist große Datenbasis?]
[Welches Modell profitiert am meisten?]
```

---

## Gesamtauswertung (nach allen Experimenten)

### Beste Modell-Konfiguration
**Modell:** ________________  
**Features:** ________________  
**Accuracy:** _____  
**F1-Score:** _____  
**Interpretierbarkeit:** ⭐⭐⭐⭐⭐  
**Begründung:** ________________________________________________

### Schlechteste Modell-Konfiguration
**Modell:** ________________  
**Features:** ________________  
**Accuracy:** _____  
**F1-Score:** _____  
**Problem:** ________________________________________________

### Machbarkeits-Bewertung (Gesamtfazit)

#### ✅ Was funktioniert gut?
1. ________________________________________________
2. ________________________________________________
3. ________________________________________________

#### ⚠️ Herausforderungen
1. ________________________________________________
2. ________________________________________________
3. ________________________________________________

#### ❌ Was funktioniert nicht?
1. ________________________________________________
2. ________________________________________________

#### 🎯 Empfehlungen für Praxis
1. ________________________________________________
2. ________________________________________________
3. ________________________________________________

### Kernaussagen für Thesis-Fazit

**Machbarkeit:** [Ja/Nein/Eingeschränkt]  
**Begründung (3 Sätze):**
```
[Zusammenfassung in eigenen Worten]
```

**Limitationen (3 wichtigste):**
1. ________________________________________________
2. ________________________________________________
3. ________________________________________________

**Ausblick (3 offene Fragen):**
1. ________________________________________________
2. ________________________________________________
3. ________________________________________________

---

## Artefakt-Archiv

### Verzeichnisstruktur
```
thesis_artifacts/
├── experiments/
│   ├── exp01_if_baseline/
│   ├── exp02_lr_supervised/
│   ├── exp03_xgb/
│   ├── exp04_features/
│   ├── exp05_comparison/
│   ├── exp06_ablation/
│   └── exp07_large/
├── demo/result/lo2/
│   ├── explainability/
│   └── metrics/
├── models/
└── thesis_artifacts_YYYYMMDD_HHMMSS.tar.gz
```

### Backup-Checkliste
- [ ] Alle `.joblib` Modelle gesichert
- [ ] Alle `.yml` Metadaten gesichert
- [ ] Alle `.json` Metriken gesichert
- [ ] Alle `.png` SHAP-Plots gesichert
- [ ] Alle `.csv` NN-Mappings gesichert
- [ ] Alle `.txt` False-Positive-Listen gesichert
- [ ] Alle Kommandos dokumentiert
- [ ] Git-Commit-Hash notiert: ________________

---

## Zeiterfassung

| Aktivität | Geschätzt | Tatsächlich | Notizen |
|-----------|-----------|-------------|---------|
| Literaturrecherche | 10h | _____ h | |
| Pipeline verstehen | 5h | _____ h | |
| Experimente durchführen | 12h | _____ h | |
| Auswertung & Analyse | 15h | _____ h | |
| SHAP-Plots interpretieren | 5h | _____ h | |
| Thesis schreiben | 40h | _____ h | |
| Review & Korrektur | 10h | _____ h | |
| **Gesamt** | **97h** | **_____ h** | |

---

## Meilensteine

- [ ] **M1:** Erste 3 Experimente abgeschlossen (Deadline: ______)
- [ ] **M2:** Alle 10 Experimente abgeschlossen (Deadline: ______)
- [ ] **M3:** Gesamtauswertung fertig (Deadline: ______)
- [ ] **M4:** Methodik-Kapitel geschrieben (Deadline: ______)
- [ ] **M5:** Evaluation-Kapitel geschrieben (Deadline: ______)
- [ ] **M6:** Diskussion geschrieben (Deadline: ______)
- [ ] **M7:** Erste Version komplett (Deadline: ______)
- [ ] **M8:** Review-Feedback eingearbeitet (Deadline: ______)
- [ ] **M9:** Abgabe (Deadline: ______)

---

## Notizen & Ideen

### Spontane Beobachtungen
```
[Platz für Notizen während der Experimente]
```

### Zusätzliche Experimente (optional)
1. ________________________________________________
2. ________________________________________________
3. ________________________________________________

### Fragen an Betreuer
1. ________________________________________________
2. ________________________________________________
3. ________________________________________________

---

## Ressourcen & Links

### Dokumentation
- [THESIS_MACHBARKEIT_ANALYSIS.md](./THESIS_MACHBARKEIT_ANALYSIS.md)
- [THESIS_EXPERIMENT_TEMPLATES.md](./THESIS_EXPERIMENT_TEMPLATES.md)
- [pipeline/execution-guide.md](./pipeline/execution-guide.md)
- [demo/lo2_e2e/README.md](../demo/lo2_e2e/README.md)

### Wichtige Code-Files
- `demo/lo2_e2e/LO2_samples.py` (Hauptpipeline)
- `demo/lo2_e2e/lo2_phase_f_explainability.py` (XAI-Artefakte)
- `loglead/explainer.py` (SHAP + NN-Explainer)

### Nützliche Kommandos
```bash
# Schneller Status-Check
ls -lh demo/result/lo2/explainability/
ls -lh experiments/*/

# Metriken-Übersicht
grep -r "accuracy" demo/result/lo2/explainability/*.json

# Alle SHAP-Plots anzeigen
open demo/result/lo2/explainability/*_shap_summary.png
```

---

**Letzte Aktualisierung:** [Datum]  
**Version:** 1.0
