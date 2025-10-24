# LO2 Minimal Workflow – Isolation Forest + XAI

Ziel: Iterativ von rohen LO2-Sample-Logs zu einem lauffähigen Isolation-Forest-Baseline-Modell (mit optionaler Logistic Regression als Vergleich) inklusive leichter Explainability. Jeder Schritt baut auf dem vorherigen auf. Erst weitergehen, wenn die Checks erfüllt sind.

---
## Ziel-Pipeline (Prototyp)
- **Loader (loglead.loaders.lo2.LO2Loader)**: Lädt rohe LO2-Logs, wendet Grundnormalisierung an und persistiert Events/Sequenzen als Parquet in `demo/result/lo2`. Ohne diese Artefakte startet keine weitere Stufe.
- **Enhancer-Kaskade (loglead.enhancers.EventLogEnhancer & SequenceEnhancer)**: Kapselt Standard-Feature-Schritte. Die Event-Enhancer erzeugen normalisierte Nachrichten (`normalize()`), Token-Listen (`words()`), n-Gramme (`trigrams()`), Parser-IDs (`parse_drain()`), sowie Längenfeatures (`length()`). Optional reichert der SequenceEnhancer Laufzeit-Features (`seq_len()`, `duration()`, `tokens()`) auf Sequenzebene an.
- **Feature-Aggregation (demo/LO2_samples.py)**: Führt Loader-Ausgaben und Enhancer zusammen. Das Skript steuert die Phasen über CLI-Flags, schreibt Checkpoints (Parquet) und bildet die Brücke zum Modelltraining.
- **Anomaly Detection (loglead.AnomalyDetector)**: Initialisiert mit `item_list_col`/`numeric_cols`. Der Ablauf: `train_df` aus korrekten Runs setzen → `prepare_train_test_data()` erstellt Sparse-Matrizen → `train_IsolationForest(...)` fitten → `predict()` erzeugt `pred_ano`, Score-Spalten und Kennzahlen. Optionale Varianten: `train_LR()` oder `train_DT()` für Phase E.
- **Persistenz & Versionierung**: `pred_df.write_parquet(...)` speichert Scores; `joblib.dump((sad.model, sad.vec), ...)` ermöglicht erneutes Laden.
- **Explainability (loglead.explainer.NNExplainer / ShapExplainer)**: In Phase F via `demo/lo2_phase_f_explainability.py`. Das Skript speist `NNExplainer(df_pred, X_test, ...)` und `ShapExplainer(sad)` an, exportiert Mapping-CSV, False-Positive-Logs sowie SHAP-Plots/Toplisten für LR, DecisionTree und optional Sequence-LR. Mit `--nn-top-k` und `--nn-normal-sample` lässt sich die Menge der für das Mapping genutzten Events steuern (schützt vor RAM-Ausfällen).
- **Auswertung & Dokumentation**: Ergebnisse fließen in diese Markdown-Datei sowie optionale Tabellen (z.B. Top-rank-Analysen, Parameter-Logbuch). Die pipelineweite Kommunikation (Loader → Enhancer → Detector → Explainer) bleibt so nachvollziehbar.

### Persistenz-Hinweise
- Loader und Skripte halten Events/Sequenzen zunächst als Polars-DataFrames im Arbeitsspeicher (`loader.df`, `df_events`). Diese leben nur, solange das Python-Objekt existiert; nach dem Beenden der Session oder dem Ausschalten des Rechners sind sie weg.
- Dauerhafte Artefakte entstehen nur durch explizites Schreiben, z.B. `python demo/run_lo2_loader.py ... --save-parquet` oder `pred_df.write_parquet(...)`. Die so erzeugten Parquet-Dateien lassen sich später per `pl.read_parquet(...)` erneut als DataFrame laden.
- Trainierte Modelle und Vectorizer gehen ebenfalls verloren, wenn sie nicht gesichert werden. Direkt nach dem Fit `joblib.dump((sad.model, sad.vec), "models/lo2_if.joblib")` aufrufen; für Re-Runs mit neuen Logs `model, vec = joblib.load(...)` laden, `sad.model = model`, `sad.prepare_train_test_data(vectorizer_class=vec)` ausführen und anschließend `sad.predict()` verwenden.

---
## Phase A – Setup & Sanity
- **Zweck**: Umgebung startklar machen.
- **Aktionen**:
  1. Virtuelle Umgebung aktivieren (LO2-spezifisch).
  2. Dependencies installieren: `polars`, `scikit-learn`, `joblib`, `loglead` (lokal).
  3. Prüfen, dass das LO2-Sample-Verzeichnis verfügbar ist (`/Users/.../lo2-sample/logs`).
- **Check**: `python -c "import polars, sklearn"` läuft fehlerfrei; TensorFlow für BERT-Embeddings ist optional.
## Phase B – Loader Smoke Test & Artefakte
- **Zweck**: Mit Bordmitteln den LO2Loader ausführen und die für spätere Schritte nötigen Parquet-Dateien erzeugen.
- **Aktionen**:
  1. `python demo/run_lo2_loader.py --root <pfad-zum-lo2-sample> --runs 5 --errors-per-run 1 --single-service client --save-parquet --output-dir demo/result/lo2`
  2. Optional `--load-metrics`, falls `metrics/*.json` vorliegen (damit entsteht `lo2_metrics.parquet`).
  3. Prüfe Terminalausgabe (`Rows`, `Services present`) und kontrolliere, dass `demo/result/lo2/lo2_events.parquet` (+ optional `lo2_sequences.parquet`, `lo2_metrics.parquet`) existieren.
- **Check**: Events/Sequenzen im Zielordner vorhanden, Loader lief ohne Fehler (z.B. `demo/result/lo2/lo2_events.parquet`).

---
## Phase C – Features mit vorhandenen Enhancern
- **Zweck**: Die in `loglead.enhancers` mitgelieferten Schritte (Normalisierung, Tokens, Trigrams, Drain, Längen) anwenden, ohne eigene Preprocessing-Logik zu bauen.
- **Aktionen**:
  1. `python demo/LO2_samples.py --phase enhancers` (oder Notebook-Variante) ausführen; das Skript bricht nach den `EventLogEnhancer`-Schritten ab.
  2. Sicherstellen, dass `normalize()`, `words()`, `trigrams()`, optional `parse_drain()`, `length()` laufen.
  3. Bei Bedarf Sequenz-Artefakte aktivieren: `SequenceEnhancer` (`seq_len()`, `duration()`, `tokens()`).
  4. Kontrollausgabe prüfen (`Sample enhanced record`, Anomaliezählung).
- **Check**: DataFrame enthält Spalten wie `e_words`, `e_trigrams`, `e_chars_len`, ggf. `e_event_drain_id`; TensorFlow-Warnung bzgl. BERT kann ignoriert werden; keine Exceptions beim Durchlauf.

---
## Phase D – Isolation Forest Baseline
  1. Nach Phase C `python demo/LO2_samples.py --phase if --save-if demo/result/lo2/lo2_if_predictions.parquet` ausführen **oder** interaktiv `sad = AnomalyDetector(item_list_col="e_words")` setzen.
  2. Trainings-/Eval-Split bewusst wählen: `sad.train_df = df_events.filter(pl.col("test_case") == "correct")`; `sad.test_df = df_events` oder ein Eval-Subset.
  3. `sad.prepare_train_test_data(vectorizer_class=sad.vec)` beim Wiederverwenden eines gespeicherten Vektorisierers, sonst Standardaufruf.
  4. `sad.train_IsolationForest(filter_anos=True, n_estimators=..., contamination=..., max_samples=...)` nutzen; im Skript erledigen das die Flags `--if-*`.
  5. `pred_df = sad.predict()` → Ergebnis enthält `pred_ano`; mit `sad.auc_roc=True` zusätzlich `pred_ano_proba`. Optional `score_if = -sad.model.score_samples(...)` ergänzen und ranken.
  6. Scores persistieren (`pred_df.write_parquet("demo/result/lo2/lo2_if_predictions.parquet")`) und Modell sichern (`joblib.dump((sad.model, sad.vec), "models/lo2_if.joblib")`).

## Phase D.2 – IF-Tuning & Plausibilitätscheck
- **Zweck**: Isolation-Forest-Ergebnisse stabilisieren und plausibel machen, bevor supervised Modelle einsteigen.
- **Aktionen**:
  1. Mit kleinen Loader-Subsets iterieren (`python demo/run_lo2_loader.py --runs 3 --single-service client --save-parquet --output-dir demo/result/lo2`) und so schnelle Feedback-Loops erhalten.
  2. `--if-contamination` schrittweise variieren (z.B. `0.20`, `0.25`, `0.30`, `0.35`, `0.40`, `0.45`) und jedes Mal `python demo/LO2_samples.py --phase if --save-if ...` ausführen.
  3. Item-Repräsentationen per Flag tauschen: `--if-item e_trigrams`, `--if-item e_event_drain_id` oder andere vorhandene Enhancer-Spalten testen (für `e_event_id` muss der Enhancer explizit eingeschaltet werden).
  4. Numerische Zusatzfeatures via `--if-numeric e_chars_len,e_event_id_len` aktivieren; weitere Kandidaten sind `e_words_len`, `e_unique_tokens` oder Sequenzlängen (`seq_len,duration_sec`).
  5. Nach jedem Lauf `demo/result/lo2/lo2_if_predictions.parquet` prüfen (`pl.read_parquet(...)`), Top-K nach `score_if` sortieren und `test_case != "correct"` zählen; Accuracy/F1/AUCROC dokumentieren.
  6. Parameter + Feature-Set im Markdown protokollieren, sobald ein Lauf plausibel wirkt.
- **Zusätzliche Hebel**: `--if-n-estimators`, `--if-max-samples`, gezielte Trainingsfilter (`sad_if.train_df` auf Services mit bekannten Anomalien) oder manuelle Score-Anpassungen (`score_if = -sad_if.model.score_samples(...)`).
- **Check**: a) Kennzahlen klar > Zufall. b) Top-N (`rank_if <= 10`) enthält überwiegend echte Anomalien. c) Festgelegtes Setting (Loader-Subset, `--if-*`, Features) dokumentiert. Danach weiter zu Phase E.

## Iterativer Stand – Phase D (23.10.2025)
- **Durchgeführte Läufe**

| Versuch | `--if-contamination` | `--if-item` | `--if-numeric` | F1 | AUCROC | Beobachtung Top-Ränge | Phase E |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.30 | e_words | – | 0.2887 | 0.4699 | Top-5 ausschließlich "correct" Runs | - |
| T1 | 0.35 | e_trigrams | e_chars_len,e_event_id_len | 0.3832 | 0.4783 | Erste Anomalien sichtbar, aber Platz 1 bleibt "correct" | - |
| T2 | 0.38 | e_trigrams | e_chars_len,e_event_id_len | 0.3928 | 0.4802 | Mehrfach dieselbe "correct" Run-ID auf den Top-Plätzen | - |
| T3 | 0.32 | e_event_drain_id | e_chars_len,e_event_id_len | 0.3730 | 0.5060 | Drain-IDs heben AUC, Top-5 weiterhin sauber | - |
| T4 | 0.45 | e_event_drain_id | e_chars_len,e_event_id_len,e_words_len | 0.4732 | 0.5091 | Beste Metriken, Top-5 = Duplikate eines "correct" Runs | LR Acc/F1 0.7583/0.7268; DT Acc/F1 1.0000/1.0000; Seq-LR Acc/F1 0.5556/0.6667; 0.4444/0.4776 |

- **Interpretation**
  - Höhere Kontamination + Längenfeatures steigern Recall, doch der IF hebt legitime Muster (Service `light-oauth2-oauth2-client-1`) hervor – die Top-Ränge bleiben False Positives.
  - Drain-basierte Tokens verbessern AUC leicht (>0.5), lösen aber das Ranking-Problem nicht; Score konzentriert sich auf seltene Normalfälle.
  - Wiederholungen derselben Run-ID deuten darauf hin, dass das Modell ein strukturell auffälliges, aber korrektes Template gelernt hat. Ohne gezielte Filterung wird dieser FP hartnäckig bleiben.

- **Konsequenzen für das MVP**
  1. Der MVP kann mit dokumentiertem False-Positive-Hotspot weitergehen, solange Explainability (Phase F) ihn sichtbar macht und das Team die Einschränkung akzeptiert.
  2. Für produktive Einsätze oder präzisere Benchmarks ist zusätzliche Arbeit nötig, bevor Phase E/F als "erfolgreich" gilt.

- **Verbesserungspfad**
  1. Trainingssplit anpassen (z.B. `sad_if.train_df = df_events.filter((pl.col("test_case") == "correct") & (pl.col("service") != "light-oauth2-oauth2-client-1"))`).
  2. Sequenzmetriken (`seq_len`, `duration_sec`) bzw. aggregierte Run-Features ergänzen, um echte Fehler deutlicher abzusetzen.
  3. Score-Transformation testen (`score_if = -sad_if.model.score_samples(...)`) und Top-K erneut validieren.
  4. Falls der Aufwand aktuell zu hoch ist: mit bestehender Konfiguration weiterarbeiten, False-Positive-Verhalten dokumentieren und Phase E (LR/DT) starten, um Vergleichswerte zu gewinnen.

## Phase E – Supervised Baseline & Vergleich
- **Zweck**: Isolation Forest mit überwachten Modellen kontrastieren und Referenzscores sichern.
- **Aktionen**:
  1. Mit den aktuell besten IF-Parametern `python demo/LO2_samples.py --phase full --if-contamination <wert> --if-item <spalte> --if-numeric <features> --save-if demo/result/lo2/lo2_if_predictions.parquet` ausführen. Der `full`-Pfad trainiert nach dem IF automatisch eine LR (Wörter) und einen DecisionTree (Trigrams).
  2. Nach dem Lauf Terminalmetriken für LR/DT notieren (Accuracy, F1, Confusion Matrix, AUC). Ergänze in der Tuning-Tabelle eine neue Spalte "Phase E" mit den Vergleichswerten.
  3. Für die Modelle im Skript wird der ursprüngliche `sad` wiederverwendet. Stelle sicher, dass `df_events` unverändert bleibt, damit Vectorizer/Features identisch zur IF-Phase sind.
  4. Wenn Sequenzdaten (`lo2_sequences.parquet`) vorhanden sind, bestätigt der `full`-Modus zusätzlich einen Sequence-LR inkl. SHAP (siehe Script-Output). Halte fest, ob Laufzeit-/Längenfeatures zusätzliche Signale liefern.
- **Check**: a) Mindestens ein überwacht trainiertes Modell schlägt Zufallsniveau. b) Vergleichswerte sind dokumentiert (Tabelle oder Markdown-Block). c) Artefakte (`lo2_if_predictions.parquet`, ggf. zusätzliche CSV/Plots) liegen im `demo/result/lo2`-Ordner.

### Phase E – Ergebnisse (23.10.2025)
- **Ausgeführter Befehl**: `python demo/LO2_samples.py --phase full --if-contamination 0.45 --if-item e_event_drain_id --if-numeric e_chars_len,e_event_id_len,e_words_len --save-if demo/result/lo2/lo2_if_predictions.parquet`
- **IsolationForest**: Accuracy 0.5156, F1 0.4673, AUCROC 0.5018. Top-Ränge bleiben vom Service `light-oauth2-oauth2-client-1` dominiert und markieren damit weiterhin den bekannten False-Positive-Hotspot.
- **Logistic Regression (Bag-of-Words)**: Accuracy 0.7583, F1 0.7268, AUCROC 0.8284. Liefert klar bessere Balance zwischen Precision und Recall und fungiert als neuer Referenzwert.
- **DecisionTree (Trigrams)**: Accuracy 1.0000, F1 1.0000. Perfekte Scores deuten auf Overfitting auf dem kleinen Sample hin; im Einsatz mit mehr Daten unbedingt erneut validieren.
- **Sequence Logistic Regression**: Accuracy 0.5556, F1 0.6667 beim ersten Lauf; nach erneuter Vektorisierung im SHAP-Pfad Accuracy 0.4444, F1 0.4776. Sequenzfeatures liefern also nur schwache Signale und profitieren von zusätzlicher Kuratierung.
- **Artefakte**: `demo/result/lo2/lo2_if_predictions.parquet` aktualisiert. SHAP- und NN-Explainability-Plots sind noch nicht gespeichert (Phase F offen).

### Chronologisches Befehlsprotokoll (Phase A–F, 23.10.2025)
| Schritt | Zweck | Befehl | Ergebnis/Notizen |
| --- | --- | --- | --- |
| 1 | Loader-Subset erzeugen | `python demo/run_lo2_loader.py --root <pfad-zum-lo2-sample> --runs 5 --errors-per-run 1 --single-service client --save-parquet --output-dir demo/result/lo2` | Persistiert `lo2_events.parquet` (und Sequenzen, falls vorhanden) als Grundlage aller weiteren Phasen. |
| 2 | Sanity-Check Dependencies | `python -c "import polars, sklearn"` | Sicherstellt, dass Kernabhängigkeiten aktiv sind. |
| 3 | Enhancer vorbereiten | `python demo/LO2_samples.py --phase enhancers` | Schreibt angereicherte Events in-memory; Logausgabe dokumentiert neue Spalten (`e_words`, `e_trigrams`, Längen). |
| 4 | IF Baseline (T0) | `python demo/LO2_samples.py --phase if --if-contamination 0.30 --if-item e_words --save-if demo/result/lo2/lo2_if_predictions.parquet` | Kennzahlen F1 0.2887 / AUC 0.4699; Top-5 bleiben korrekt. |
| 5 | IF Tuning T1 | `python demo/LO2_samples.py --phase if --if-contamination 0.35 --if-item e_trigrams --if-numeric e_chars_len,e_event_id_len --save-if demo/result/lo2/lo2_if_predictions.parquet` | F1 0.3832; erste Anomalien sichtbar. |
| 6 | IF Tuning T2 | `python demo/LO2_samples.py --phase if --if-contamination 0.38 --if-item e_trigrams --if-numeric e_chars_len,e_event_id_len --save-if demo/result/lo2/lo2_if_predictions.parquet` | F1 0.3928; Ranking weiterhin von "correct" Runs dominiert. |
| 7 | IF Tuning T3 | `python demo/LO2_samples.py --phase if --if-contamination 0.32 --if-item e_event_drain_id --if-numeric e_chars_len,e_event_id_len --save-if demo/result/lo2/lo2_if_predictions.parquet` | F1 0.3730; Drain-IDs erhöhen AUC auf 0.5060. |
| 8 | IF Tuning T4 (Bestes Setting) | `python demo/LO2_samples.py --phase if --if-contamination 0.45 --if-item e_event_drain_id --if-numeric e_chars_len,e_event_id_len,e_words_len --save-if demo/result/lo2/lo2_if_predictions.parquet` | F1 0.4732 / AUC 0.5091; False-Positive-Hotspot bleibt bestehen. |
| 9 | Phase E Full Run | `python demo/LO2_samples.py --phase full --if-contamination 0.45 --if-item e_event_drain_id --if-numeric e_chars_len,e_event_id_len,e_words_len --save-if demo/result/lo2/lo2_if_predictions.parquet` | Vergleicht IF, LogisticRegression, DecisionTree, Sequence-LR; erzeugt Kennzahlen laut Abschnitt oben. |
| 10 | Phase F Explainability | `python demo/lo2_phase_f_explainability.py --root demo/result/lo2 --if-contamination 0.45 --if-n-estimators 200 --shap-sample 200 --nn-top-k 50 --nn-normal-sample 200` | Erstellt NN-Mapping (`if_nn_mapping.csv`) und SHAP-Artefakte (`lr_*/dt_*`, optional `seq_lr_*`) im Unterordner `explainability`. |

## Phase F – Minimal XAI Hooks
- **Zweck**: Eingebaute Explainability-Klassen (ohne Eigenbau) einsetzen.
- **Aktionen**:
  1. Phase-E-Artefakte vorbereiten (`lo2_events.parquet`, T4-Parameter). Dann `python demo/lo2_phase_f_explainability.py --root demo/result/lo2 --if-contamination 0.45 --if-n-estimators 200 --shap-sample 200 --nn-top-k 50 --nn-normal-sample 200` ausführen. Das Skript reproduziert das T4-IF-Modell, erzeugt NN-Mappings (Top-K Anomalien + Normalstichprobe) und berechnet SHAP für LR/DT (optional Sequence-LR bei vorhandener `lo2_sequences.parquet`).
  2. NNExplainer erfordert `NNExplainer(df_pred, X_test, id_col="row_id", pred_col="pred_ano")`. Das Skript persistiert `if_nn_mapping.csv` und `if_false_positives.txt`; Inhalte stichprobenartig prüfen (`pl.read_csv(...)` oder `cat`).
  3. Für LR/DT entstehen SHAP-Summary- und Bar-Plots (`lr_shap_summary.png`, `dt_shap_bar.png` etc.), Feature-Toplisten (`lr_top_tokens.txt`, `dt_top_trigrams.txt`) und Kennzahlen (`metrics_*.json`). Sequenz-LR liefert analog `seq_lr_*`-Artefakte, sofern Sequenzen vorhanden sind.
  4. Nach dem Lauf prüfen, ob Plots lesbar gespeichert wurden (Matplotlib nutzt Agg-Backend) und False-Positive-Log kontextualisieren.
- **Check**: a) Mindestens ein True Positive besitzt ein sinnvolles Referenzlog im Mapping. b) SHAP-Barplots zeigen konsistente Top-Features. c) Ordner `demo/result/lo2/explainability/` enthält alle genannten Artefakte und wird in der Dokumentation referenziert.

- **Ausgeführter Befehl**: `python demo/lo2_phase_f_explainability.py --root demo/result/lo2 --if-contamination 0.45 --if-n-estimators 200 --shap-sample 200 --nn-top-k 50 --nn-normal-sample 200`
- **IsolationForest**: Accuracy 0.5027, F1 0.4444, AUCROC 0.5025 – bestätigt den FP-Hotspot um `light-oauth2-oauth2-client-1`.
- **Logistic Regression (Bag-of-Words)**: Accuracy 0.7502, F1 0.7113, AUCROC 0.8396; SHAP-Plots (`lr_shap_summary.png`, `lr_shap_bar.png`) und Topliste (`lr_top_tokens.txt`) gespeichert.
- **DecisionTree (Trigrams)**: Accuracy/F1/AUCROC = 1.0 – SHAP-Analyse (`dt_shap_summary.png`, `dt_shap_bar.png`) bestätigt hartes Overfitting.
- **Sequence-LR**: Accuracy 0.7778, F1 0.7407, AUCROC 0.8095; SHAP bleibt wegen rein numerischer Features deaktiviert (`seq_lr_shap_skipped.txt`).
- **NN-Mapping**: `if_nn_mapping.csv` & `if_false_positives.txt` generiert; Mapping listet Top-K Anomalien mit entsprechenden Normalfällen.
- **FP-Analyse**: `if_false_positives.txt` zeigt ausschließlich Runs aus `light-oauth2-oauth2-token-1` (correct). Maßnahmenplan: (a) Trainingssplit um diesen Service bereinigen, (b) zusätzliche Sequenz-/Run-Features einspeisen, (c) `contamination`/Ranking erneut prüfen und Phase F wiederholen.
- **Artefakte** (`demo/result/lo2/explainability/`): `lo2_if_predictions.parquet`, `metrics_lr.json`, `metrics_dt.json`, `if_nn_mapping.csv`, `if_false_positives.txt`, `lr_*`, `dt_*`, optional `seq_lr_shap_skipped.txt` + `metrics_seq_lr.json`.
- **Interpretation**: LR bleibt Referenzmodell; Explainability macht FP-Hotspot transparent. Sequence-Pfad benötigt zusätzliche Feature-Ingenieurung, bevor SHAP sinnvoll ist.

---
## Phase G – Iterative Tests & Tuning
- **Zweck**: Pipeline stabilisieren, bevor neue Daten reinkommen.
- **Aktionen**:
  1. IsolationForest-Parameter (`n_estimators`, `contamination`, `max_samples`) variieren und jeweils `sad.predict()` speichern.
  2. `item_list_col` wechseln (`"e_trigrams"`, `"e_event_drain_id"`) und vergleichen.
  3. Bestes Setting sichern: `joblib.dump((sad.model, sad.vec), "models/lo2_if.joblib")`.
  4. Optional Sequenz-Pipeline (`SequenceEnhancer`) mit eigenem `AnomalyDetector` (`numeric_cols=["seq_len","duration_sec"]`).
- **Check**: Wiederholte Läufe liefern konsistente Rankings; bestes Modell dokumentiert (Konfiguration + Speicherpfad).

---
## Phase H – Neue Runs testen
- **Zweck**: Trainiertes Modell auf neue Runs übertragen.
- **Aktionen**:
  1. Loader erneut ausführen (`--save-parquet`) → neue `lo2_events.parquet` einlesen.
  2. Enhancer-Schritte wiederholen (gleiches Skript wie in Phase C).
  3. `sad = AnomalyDetector(item_list_col="e_words")` initialisieren, `sad.train_df = df_events.filter(pl.col("test_case") == "correct")` setzen.
  4. Modell/Vektorizer laden (`model, vec = joblib.load("models/lo2_if.joblib")`), `sad.model = model`, `sad.prepare_train_test_data(vectorizer_class=vec)`.
  5. `sad.test_df` auf neue Events setzen, `pred_new = sad.predict()` und Ergebnisse speichern (`pred_new.write_parquet(...)`).
- **Check**: Ausgabe enthält Scores/Flags für neue Runs, keine Exceptions beim Laden.

---
## Phase I – Dokumentation & Nächste Schritte
- **Zweck**: Ergebnisse sichern, offene Punkte notieren.
- **Aktionen**:
  1. Kurze Markdown-Zusammenfassung (Inputpfad, Parameter, Scores, Top-K Beispiele, XAI-Auszüge).
  2. Offene Fragen/Verbesserungen sammeln (z.B. mehr Services, feinere Templates, SHAP-Automatisierung).
- **Check**: Dokumentation erlaubt Wiederholung und Weiterentwicklung.

---
**Iterative Empfehlung**
1. Phase A–C abschließen → Loader + Enhancer ohne Fehler (Warnung bzgl. TensorFlow ignorierbar).
2. Phase D mit kleinem Subset (`--runs 3`, `--single-service client`) debuggen; Modell/Vektorizer sichern, sobald Kennzahlen ok.
3. Erst wenn IF stabil, LR/DT + Explainability ergänzen (Phase E/F) und Artefakte dokumentieren.
4. Dokumentierter IF-FP-Hotspot (`light-oauth2-oauth2-token-1`) angehen: Trainingssplit filtern, zusätzliche Features testen, `contamination` feinjustieren, danach Phase F wiederholen.
4. Nach Änderungen erneut Phase C/D wiederholen, Feature-/Modell-Drift prüfen, Ergebnisse festhalten (Markdown/CSV).

Dieses Vorgehen hält den Fokus iterativ: Kleine, testbare Schritte, klare Artefakte, und jederzeit rückverfolgbar, wie aus den LO2-Sample-Logs Scores und Erklärungen entstehen.
