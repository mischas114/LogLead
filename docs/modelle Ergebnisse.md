Erster Run - Supervised-Fokus (LogReg, Linear-SVM, DecisionTree, RandomForest, XGBoost):

```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lr_words,event_lsvm_words,event_dt_trigrams,event_rf_words,event_xgb_words \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/benchmarks/if_supervised_predictions.parquet \
  --metrics-dir demo/result/lo2/benchmarks/if_supervised_metrics \
  --save-model demo/result/lo2/models/if_supervised.joblib \
  --overwrite-model \
  --dump-metadata
```

IsolationForest steckt bereits fest im Hauptablauf (Abschnitt “Isolation Forest baseline (Phase D)” in demo/lo2_e2e/LO2_samples.py), wird also immer trainiert und ausgewertet – deshalb taucht er nicht in --list-models. Die Registry deckt nur die zusätzlichen, frei kombinierbaren Modelle der Phasen E/F ab.
Zum Benchmark-Vergleich kannst du nacheinander verschiedene Modellsets durchlaufen lassen, z. B.
python demo/lo2_e2e/LO2_samples.py --models event_lr_words,event_rf_words,event_lof_words
python demo/lo2_e2e/LO2_samples.py --models event_lr_words,event_dt_trigrams
Alle Läufe schreiben (neben dem IF-Ergebnis) die jeweiligen Predictions/Metriken; nutze --report-precision-at, --report-fp-alpha, --report-psi und vergleiche die Dateien unter result/lo2/metrics/ sowie SHAP-/NN-Artefakte in demo/result/lo2/explainability/.
Falls du den IF-Benchmark explizit zusammen mit anderen Modellen variieren möchtest, plane einfach mehrere Runs mit unterschiedlichen --if-* Parametern; die Registry brauchst du dafür nicht anzupassen.
Optional kannst du die Registry erweitern, wenn IF dort ebenfalls erscheinen soll – technisch wäre das nur ein Eintrag im MODEL_REGISTRY, aber funktional ändert sich nichts, weil der IF-Block schon vor dem Registry-Loop ausgeführt wird.

Benchmarks alle

2ter Run - Un-/Semi-Supervised Stapel (LOF, OneClassSVM, KMeans, Rarity, OOV):
```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models event_lof_words,event_oneclass_svm_words,event_kmeans_words,event_rarity_words,event_oov_words \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/benchmarks/if_unsupervised_predictions.parquet \
  --metrics-dir demo/result/lo2/benchmarks/if_unsupervised_metrics \
  --save-model demo/result/lo2/models/if_unsupervised.joblib \
  --overwrite-model \
  --dump-metadata
  ```

[event_lof_words] LocalOutlierFactor (novelty) auf Event-Worttokens (trainiert nur auf korrekten Runs).
Results from model: LocalOutlierFactor
Accuracy: 0.9207
F1 Score: 0.0086
Confusion Matrix:
[[275294    230]
 [ 23488    103]]
[TrainStats] event_lof_words: train_rows=275524 total_rows=299115 fraction=0.9211

[event_oneclass_svm_words] OneClassSVM auf Event-Worttokens (trainiert nur auf korrekten Runs).
/Users/MTETTEN/miniconda3/envs/loglead_env/lib/python3.11/site-packages/sklearn/svm/_base.py:305: ConvergenceWarning: Solver terminated early (max_iter=1000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
  warnings.warn(
Results from model: OneClassSVM
Accuracy: 0.6019
F1 Score: 0.0558
Confusion Matrix:
[[176506  99018]
 [ 20074   3517]]
[TrainStats] event_oneclass_svm_words: train_rows=275524 total_rows=299115 fraction=0.9211

[event_kmeans_words] KMeans Clustering auf Event-Worttokens (2 Cluster).
Results from model: KMeans
Accuracy: 0.9211
F1 Score: 0.0000
Confusion Matrix:
[[275524      0]
 [ 23591      0]]
AUCROC: 0.5031
[TrainStats] event_kmeans_words: train_rows=299115 total_rows=299115 fraction=1.0000

[event_rarity_words] RarityModel auf Event-Worttokens.
Results from model: RarityModel
Accuracy: 0.0025
F1 Score: 0.0041
Confusion Matrix:
[[   119 275405]
 [ 22975    616]]
AUCROC: 0.0019
[TrainStats] event_rarity_words: train_rows=299115 total_rows=299115 fraction=1.0000

[event_oov_words] OOVDetector für seltene Tokens (trainiert nur auf korrekten Runs).
Results from model: OOV_detector
Accuracy: 1.0000
F1 Score: 0.9997
Confusion Matrix:
[[275524      0]
 [    13  23578]]
AUCROC: 1.0000
[TrainStats] event_oov_words: train_rows=275524 total_rows=299115 fraction=0.9211

[Summary] Full-data pipeline diagnostics:
  event_lof_words: train_rows=275524 total_rows=299115 fraction=0.9211
  event_oneclass_svm_words: train_rows=275524 total_rows=299115 fraction=0.9211
  event_kmeans_words: train_rows=299115 total_rows=299115 fraction=1.0000
  event_rarity_words: train_rows=299115 total_rows=299115 fraction=1.0000
  event_oov_words: train_rows=275524 total_rows=299115 fraction=0.9211
[Summary] Downsampling occurred: yes

LO2 sample pipeline complete.

3ter Run - Sequence-Fokus (nur Sequenzmodelle + SHAP):
```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --models sequence_lr_numeric,sequence_lr_words,sequence_shap_lr_words \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/benchmarks/if_sequence_predictions.parquet \
  --metrics-dir demo/result/lo2/benchmarks/if_sequence_metrics \
  --save-model demo/result/lo2/models/if_sequence.joblib \
  --overwrite-model \
  --dump-metadata
  ```
