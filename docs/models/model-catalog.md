---
title: Modell-Registry Katalog
summary: Übersicht über alle per CLI zuschaltbaren Modelle der LO2-Pipeline.
last_updated: 2025-11-03
---

# Modell-Registry Katalog

Die LO2-Pipeline bringt eine deklarative Modell-Registry mit 14 Einträgen. Über `--models key1,key2` werden Modelle aktiviert, ohne den Code zu ändern.

## Kategorien

- **Baseline:** IsolationForest (`Phase D`), immer aktiv.
- **Supervised (Sequence):** LogisticRegression, LinearSVM, DecisionTree, RandomForest, XGBoost.
- **Un-/Semi-Supervised (Sequence):** LOF, OneClassSVM, KMeans, RarityModel, OOVDetector.
- **Sequence (numeric/token):** LogisticRegression auf numerischen bzw. tokenbasierten Features, inkl. SHAP-Variante.

## Vergleichstabelle

| Schlüssel | Ebene | Eingangsdaten | Stärken | Stolpersteine |
| --- | --- | --- | --- | --- |
| `event_lr_words` | Sequence | Tokens (`e_words`) | Solide Baseline, interpretierbar | Braucht Labels, kein automatischer Split |
| `event_lsvm_words` | Sequence | Tokens | Robuste Trennschärfe | Keine Wahrscheinlichkeiten, ggf. langsam |
| `event_dt_trigrams` | Sequence | Trigramme | Decision Paths verständlich | Overfitting bei vollem Datensatz |
| `event_rf_words` | Sequence | Tokens | Ensemble-robust | Höherer Speicherbedarf |
| `event_xgb_words` | Sequence | Tokens | Flexible Hyperparameter | Längere Trainingszeit |
| `event_lof_words` | Sequence | Tokens (correct-only) | Lokaler Outlier-Fokus | Sensibel für Skalierung |
| `event_oneclass_svm_words` | Sequence | Tokens (correct-only) | Präziser Normalraum | Konvergenzprobleme möglich |
| `event_kmeans_words` | Sequence | Tokens | Schnelle Clusterbildung | Keine Wahrscheinlichkeiten |
| `event_rarity_words` | Sequence | Tokenstatistik | Ohne Training nutzbar | Kaum Schwellenlogik |
| `event_oov_words` | Sequence | Tokens + Längen | Findet unbekannte Tokens | Vokabulargröße abstimmen |
| `sequence_lr_numeric` | Sequence | `seq_len`, `duration`, numerische Features | Schnell, robust | Verzichtet auf Tokenkontext |
| `sequence_lr_words` | Sequence | Tokenbuckets pro Sequenz | Erfasst Sequenzmuster | Größere Matrizen |
| `sequence_shap_lr_words` | Sequence | Tokenbuckets | SHAP-Erklärungen | Rechenintensiv |

> Hinweis: IsolationForest ist nicht in der Registry gelistet, da er fester Bestandteil des Basisskripts ist.

## Typische Aufrufmuster

```bash
# Supervised-Paket
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_lsvm_words,event_dt_trigrams,event_rf_words,event_xgb_words

# Un-/Semi-Supervised
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lof_words,event_oneclass_svm_words,event_kmeans_words,event_rarity_words,event_oov_words

# Sequenz-Analyse mit SHAP
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models sequence_lr_numeric,sequence_lr_words,sequence_shap_lr_words
```

## Pflege

- Neue Modelle in `MODEL_REGISTRY` ergänzen (Schlüssel in kebab-case).
- Optional `requires_shap=True` setzen, wenn SHAP-Antworten generiert werden sollen.
- Bei Modellen mit eigenen Persistenzanforderungen (z. B. joblib-Dump) Pfade im CLI ergänzen.

## Offene Punkte

- TODO: Train/Test-Reporting vereinheitlichen (unsupervised Modelle laufen weiterhin auf Volltraining).
- TODO: Dokumentieren, welche Modelle zusätzliche Python-Pakete benötigen (z. B. XGBoost).
