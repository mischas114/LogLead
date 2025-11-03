---
title: Isolation-Forest Leitfaden
summary: Training, Schwellenkalibrierung und Verbesserungsplan für den LO2 Isolation Forest.
last_updated: 2025-11-03
---

# Isolation-Forest Leitfaden

Der Isolation Forest bildet die zentrale unsupervised Baseline der LO2-Pipeline. Dieses Dokument bündelt Ablauf, Parameterempfehlungen und aktuelle Erkenntnisse.

## Ablauf im Überblick

1. **Loader ausführen** (Events + Sequenzen persistieren).
2. **Features erzeugen** (`LO2_samples.py --phase full` → Normalisierung, Tokens, Drain, Längen).
3. **Training** ausschließlich auf `test_case == "correct"`; optional Hold-out reservieren.
4. **Scoring** auf allen Events, Scores invertiert (`score_if = -model.score_samples(...)`).
5. **Schwellenkalibrierung** via Hold-out (`--if-threshold-percentile`) oder Trainingsscores.
6. **Persistenz** (`--save-if`, `--save-model`, `--dump-metadata`) plus Metriken (`--report-*`).
7. **Explainability** (`lo2_phase_f_explainability.py`) für SHAP und NN-Mapping.

## Kernkommandos

```bash
# Training & Scoring
python demo/lo2_e2e/LO2_samples.py \
  --phase if \
  --if-item e_words \
  --if-contamination 0.15 \
  --if-holdout-fraction 0.05 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --report-psi \
  --save-if demo/result/lo2/lo2_if_predictions.parquet \
  --save-model demo/result/lo2/models/lo2_if.joblib \
  --dump-metadata \
  --metrics-dir demo/result/lo2/metrics

# Explainability
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.15 \
  --nn-top-k 50 \
  --shap-sample 200 \
  --load-model demo/result/lo2/models/lo2_if.joblib
```

## Parameterleitfaden

- `--if-item`: Start mit `e_words`, Alternativen `e_event_drain_id`, `e_trigrams`. Kombinationen via `--if-numeric` ergänzen.
- `--if-contamination`: 0.10–0.25 testen, abhängig vom Fehleranteil im Trainingsset.
- `--if-holdout-fraction`: 0.05–0.10, reserviert frische Normaldaten für Drift/Threshold.
- `--if-threshold-percentile`: 99.0–99.5 liefern kontrollierbare Alert-Raten.
- `--if-max-samples`: Bei großen Datasets limitieren (z. B. 100 000) um Laufzeit zu reduzieren.
- `--report-precision-at`, `--report-fp-alpha`: Konsistente Kennzahlen für Vergleich.

## Artefakte

- `lo2_if_predictions.parquet`: Enthält Scores, Schwellenflag (`pred_if_threshold`), Ranking.
- `models/lo2_if.joblib`: Enthält Modell und Vectorizer (tuple).
- `models/model.yml`: Metadaten (Parameter, Schwelle, Datasetgröße, Git-Commit).
- `metrics/*.json`: Precision@k, FP-Rate, PSI, optional PR/ROC-Kurven (abhängig von Flags).
- Explainability-Ausgaben (`if_nn_mapping.csv`, `if_false_positives.txt`, `lr_shap_*.png`, `dt_shap_*.png`).

## Beobachtungen aus den letzten Runs

- **Baseline (Word-Features, Kontamination 0.15):** Accuracy ~0.80, F1 ~0.09, AUC ~0.50 – hohe False-Positive-Rate, Scores trennen Normal/Anomal kaum.
- **Drain-ID + numerische Längen:** Accuracy ~0.71, F1 ~0.11, AUC ~0.50 – minimale Verbesserung; höhere Kontamination erhöht FP-Anteil.
- **Logistic Regression (Wort-Features):** AUC ~0.86, F1 ~0.17 – zeigt, dass Features prinzipiell tragen; Schwellenproblem statt Feature-Defizit.
- **Decision Tree / Sequence-LR:** Perfekte bzw. sehr hohe Scores, jedoch Overfitting (Trainingsmetriken), Sequenz-LR basiert auf kleiner Stichprobe.
- **NN-Mapping:** Top-Scorings stammen überwiegend aus `correct`-Runs → Indikator für unzureichende Normalrepräsentation.

## Verbesserungsplan

- Mehr korrekte Runs einbeziehen (`--errors-per-run` senken oder dedizierte Normal-Batches).
- Feature-Mix erweitern (`e_event_drain_id` + numerische Sequenzmerkmale), ggf. `--if-numeric` ergänzen (`e_chars_len`, `e_event_id_len`, `seq_len`).
- Unterschiedliche Services getrennt trainieren (z. B. Token-only Trainingssplit) und Ergebnisse vergleichen.
- Hold-out größer wählen und PSI beobachten; bei Drift Re-Training auslösen.
- Bei wiederholten Läufen `--overwrite-enhancers` setzen, falls Feature-Caches aktualisiert werden sollen.

## Troubleshooting

- **SHAP speichert nicht:** `MPLBACKEND=Agg` setzen, `--shap-sample` verringern.
- **Hold-out zu klein:** Bei `--if-holdout-fraction < 0.05` fehlen stabile Kennzahlen → Run abbrechen und Wert erhöhen.
- **Persistenzfehler:** Zielpfade (`--save-if`, `--save-model`, `--metrics-dir`) müssen existieren oder erzeugt werden; Pfade nach Möglichkeit unter `demo/result/lo2/` halten.

## Offene Fragen

- TODO: Systematisch evaluieren, ob separate Modelle pro Service (`token`, `code`, `refresh-token`) bessere Ergebnisse liefern.
- TODO: Automatisches Hyperparameter-Sweep-Skript definieren (Kontamination, Feature-Sätze).
- TODO: Schwellenableitung enger mit Produktionsanforderungen koppeln (z. B. Ziel-Alert-Rate).
