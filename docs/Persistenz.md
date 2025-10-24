# Persistenz der LO2-Pipeline

Diese Notiz beschreibt, welche Artefakte nach einem Pipeline-Run dauerhaft erhalten bleiben, wie sie gespeichert werden und wie sie in späteren Runs wiederverwendet werden können.

---
## Startvoraussetzungen
- Virtuelle Umgebung aktiv (die `loglead` installierte Umgebung).
- LO2-Sample-Verzeichnis vorhanden (`<pfad-zum-lo2-sample>`).
- Projekt-Root: `/Users/MTETTEN/Projects/LogLead`.

---
## Persistenz aktiv einschalten (nach einem Training)
Folgende Schritte sorgen dafür, dass ein fertig trainiertes Setup auch nach Laptop-Neustart weiterverwendet werden kann.

1. **Loader-Artefakte sichern** – beim ersten Lauf unbedingt mit `--save-parquet` arbeiten:
  ```bash
  python demo/run_lo2_loader.py --root <pfad-zum-lo2-sample> \
     --runs 5 --single-service client --save-parquet \
     --output-dir demo/result/lo2
  ```
  Dadurch landen `lo2_events.parquet` (und ggf. `lo2_sequences.parquet`) dauerhaft unter `demo/result/lo2/`.

2. **Angereicherte Events optional wegschreiben** – falls du nach den Enhancern nicht jedes Mal neu rechnen willst:
  ```python
  df_events = enhancer.normalize().words().trigrams()
  df_events = enhancer.length()
  df_events.write_parquet("demo/result/lo2/lo2_events_enriched.parquet")
  ```
  Dieser Schritt ist optional; wenn du ihn weglässt, werden die Enhancement-Schritte beim nächsten Lauf einfach erneut ausgeführt.

3. **Modell + Vectorizer sichern** – unmittelbar nach dem Training:
  ```python
  from pathlib import Path
  import joblib

  Path("models").mkdir(parents=True, exist_ok=True)
  joblib.dump((sad.model, sad.vec), "models/lo2_if.joblib")
  ```
  Die Datei `models/lo2_if.joblib` bleibt bestehen, bis du sie löscht. Damit ist das Modell “über Nacht” verfügbar.

4. **Predictions (optional)** – wenn du Scores/Rankings behalten willst:
  ```python
  pred_df.write_parquet("demo/result/lo2/lo2_if_predictions.parquet")
  ```
  Diese Parquet-Datei eignet sich für spätere Vergleiche oder Explainability.

---
## Wiederverwendung nach Neustart / neuem Run
1. **Parquet laden:**
  ```python
  import polars as pl
  df_events = pl.read_parquet("demo/result/lo2/lo2_events_enriched.parquet")
  # oder df_events = pl.read_parquet("demo/result/lo2/lo2_events.parquet")
  ```
  Falls du kein „enriched“-Parquet geschrieben hast, führst du jetzt einfach wieder `EventLogEnhancer` aus.

2. **Modell & Vectorizer laden:**
  ```python
  import joblib
  model, vec = joblib.load("models/lo2_if.joblib")
  ```

3. **`AnomalyDetector` vorbereiten:**
  ```python
  from loglead import AnomalyDetector

  sad = AnomalyDetector(item_list_col="e_event_drain_id", numeric_cols=["e_chars_len", "e_event_id_len", "e_words_len"])
  sad.train_df = df_events.filter(pl.col("test_case") == "correct")
  sad.test_df = df_events
  sad.prepare_train_test_data(vectorizer_class=vec)
  sad.model = model
  pred_df = sad.predict()
  ```
  `prepare_train_test_data(vectorizer_class=vec)` sorgt dafür, dass genau derselbe Vectorizer wie beim Training genutzt wird. Das geladene Modell steckt danach in `sad.model` – kein erneutes `train_*()` nötig.

4. **Neue Predictions sichern (optional):**
  ```python
  pred_df.write_parquet("demo/result/lo2/lo2_if_predictions_new.parquet")
  ```

Damit ist gewährleistet, dass das Modell auch nach einem Laptop-Neustart einsatzbereit bleibt, weil alle relevanten Artefakte (Parquet + joblib-Datei) auf der Platte liegen.

---
## Häufige Fragen (Mini-FAQ)
- **„Sad += sad?“** – Nein. Jedes Mal, wenn du ein Modell trainierst, steckt es in `sad.model`. Du speicherst es mit `joblib.dump`. Beim Wiederverwenden erzeugst du einen neuen `AnomalyDetector`, lädst Modell + Vectorizer und setzt sie ein – kein Addieren nötig.
- **„Was passiert ohne joblib?“** – Nach dem Beenden der Python-Session müsstest du jedes Modell neu trainieren. `joblib.dump` sorgt dafür, dass du das Ergebnis als Datei wieder laden kannst.
- **„Muss ich Enhancer jedes Mal laufen lassen?“** – Nur dann, wenn du die angereicherten Spalten nicht in eine Datei geschrieben hast. Ansonsten kannst du `lo2_events_enriched.parquet` direkt lesen.
- **„Wo liegen die Dateien?“** – Loader-Ausgabe: `demo/result/lo2/*.parquet`. Modell: `models/lo2_if.joblib`. Predictions: `demo/result/lo2/lo2_if_predictions*.parquet`. Explainability: `demo/result/lo2/explainability/`.
