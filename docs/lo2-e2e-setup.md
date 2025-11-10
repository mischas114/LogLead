---
title: LO2 e2e Setup Leitfaden
summary: Schritt-für-Schritt-Anleitung, um LogLead für die LO2-Ende-zu-Ende-Pipeline lokal einzurichten und komplett durchlaufen zu lassen.
last_updated: 2025-11-11
---

# LO2 e2e Chain Setup

Dieser Leitfaden ermöglicht dir, die komplette LO2-Pipeline (Loader → Enhancer → Isolation Forest & Registry → Explainability) auf einem sauberen macOS-/Linux-System aufzusetzen. Die folgenden Schritte sind so gestaltet, dass du nach einmaliger Einrichtung reproduzierbare Läufe und Artefakte unter `demo/result/lo2/` erhältst.

## 1. Python-Umgebung mit Conda vorbereiten

> Die folgenden Blöcke sind der verbindliche Startpunkt und sollten exakt in dieser Reihenfolge ausgeführt werden.

### Install Conda

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### Conda in Umgebungsvariablen

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.zshrc   # lädt Änderungen sofort
```

### Neue Conda-Umgebung mit Python 3.11

```bash
conda create -n loglead_env python=3.11
conda init
# TERMINAL NEU STARTEN
conda activate loglead_env
  # Am Ende die Umgebung mit 'conda deactivate' wieder deaktivieren
```

### LogLead-Paket installieren

```bash
python -m pip install loglead
```

### LogLead Repo clonen

```bash
git clone https://github.com/mischas114/LogLead.git

pip install -r LogLead/LogLead.egg-info/requires.txt
```

> **Hinweis:** Falls du bereits ein Fork oder lokales Verzeichnis besitzt, kannst du das bestehende Repo nutzen. Stelle lediglich sicher, dass das aktive Terminal weiterhin in der `loglead_env`-Umgebung läuft (`conda activate loglead_env`).

## 2. Repository initialisieren

1. Wechsle in das Projekt und (optional) installiere es im Entwicklungsmodus:
   ```bash
   cd ~/Projects/LogLead   # passe den Pfad an dein clone-Ziel an
   python -m pip install -e .
   ```
2. Prüfe, ob alle Skripte ausführbar sind:
   ```bash
   python -c "import loglead; print(loglead.__version__)"
   ```
3. Erstelle die Standard-Ausgabeordner:
   ```bash
   mkdir -p demo/result/lo2 models
   ```

## 3. LO2-Datenquelle vorbereiten

Die LO2-Skripte erwarten eine Verzeichnisstruktur wie folgt:

```
<LO2_ROOT>/
  run_0001/
    correct/
      oauth2-oauth2-client.log
      oauth2-oauth2-token.log
      ...
    error_invalid_grant/
      oauth2-oauth2-client.log
      ...
    metrics/
      latency.json
      ...
  run_0002/
    ...
```

- Kopiere oder mounte die Roh-Logs in ein lokales Verzeichnis, z. B. `~/data/lo2_runs`.
- Notiere den Pfad in einer Variablen:
  ```bash
  export LO2_ROOT=~/data/lo2_runs
  ```
- Falls du mit verschiedenen Samples experimentierst, ist ein dediziertes Verzeichnis pro Sample hilfreich (z. B. `~/data/lo2_runs_small`, `~/data/lo2_runs_full`).

## 4. Schnelles Sanity-Check nach dem Setup

```bash
conda activate loglead_env
cd ~/Projects/LogLead
python demo/lo2_e2e/run_lo2_loader.py --help | head -n 5
python demo/lo2_e2e/LO2_samples.py --version
```

So stellst du sicher, dass Loader und Pipeline-Skript erreichbar sind.

## 5. LO2 e2e Chain ausführen

### 5.1 Loader (Phase B)

```bash
python demo/lo2_e2e/run_lo2_loader.py \
  --root "$LO2_ROOT" \
  --runs 5 \
  --errors-per-run 1 \
  --service-types code token refresh-token \
  --save-parquet \
  --save-events \
  --save-base-sequences \
  --output-dir demo/result/lo2
```

- `--runs` und `--errors-per-run` begrenzen die Menge für Smoke-Tests. Lass beide weg, wenn du alles laden willst.
- `--save-parquet` erzeugt `lo2_sequences.parquet` und `lo2_sequences_enhanced.parquet` im Output-Ordner – diese Dateien benötigt der nächste Schritt.
- Nutze `--load-metrics`, wenn unter `run_*/**/metrics/*.json` zusätzlich Kennzahlen liegen, die du mitschneiden möchtest.

### 5.2 Enhancer + Modelle (Phasen C–E)

Da der Isolation Forest aktuell wenig Mehrwert liefert (Accuracy ≈0.45, F1 ≈0), empfiehlt es sich, ihn zunächst zu überspringen und sich auf die stabilen Registry-Modelle zu konzentrieren.

```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --root demo/result/lo2 \
  --skip-if \
  --if-item e_words \
  --if-numeric seq_len,duration_sec,e_words_len,e_trigrams_len \
  --if-contamination 0.15 \
  --if-holdout-fraction 0.2 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-model models/lo2_if.joblib \
  --dump-metadata \
  --models event_lr_words,event_dt_trigrams,event_xgb_words \
  --metrics-dir demo/result/lo2/metrics \
  --save-enhancers
```

- `--phase full` – Komplettlauf von Enhancern + Registry; du kannst später gezielt `--phase enhancers` oder `--phase if` wählen.
- `--skip-if` – überspringt den IF-Teil, damit nur die zuverlässigeren Registry-Modelle laufen.
- `--if-*` – bleiben drin, falls du den IF zu Vergleichszwecken später wieder aktivierst (Parameter zentral halten).
- `--models` – wähle bewusst 2–3 Schlüssel; alles zugleich macht Logs unübersichtlich (`--list-models` zeigt Alternativen).
- `--save-model` – speichert ein evtl. doch trainiertes IF-Bundle, falls `--skip-if` entfernt wird.
- `--dump-metadata` – legt ein YAML mit Parametern und Git-Commit ab, hilfreich für spätere Reproduktionen.
- `--save-enhancers` – persistiert Feature-Parquets für Notebook-/Analysezwecke.

### 5.3 Explainability (Phase F)

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words,event_dt_trigrams,event_xgb_words,sequence_shap_lr_words \
  --nn-source sequence_shap_lr_words \
  --nn-top-k 50 \
  --shap-sample 200 \
  --sup-holdout-fraction 0.2
```

- `MPLBACKEND=Agg` verhindert GUI-Abhängigkeiten und ermöglicht headless SHAP-Plots.
- Entferne `--skip-if`, wenn du den Isolation Forest in Phase F erneut trainieren willst (z. B. mit anderen Parametern).
- Mit `--nn-source` steuerst du, welches Modell für das Nearest-Neighbour-Mapping genutzt wird (`if`, `sequence_shap_lr_words`, …).

### 5.4 Optional: Artefakte nachträglich scannen

```bash
python tools/lo2_result_scan.py --root demo/result/lo2 --dry-run
```

Das Skript fasst zusammen, welche Parquets, JSONs und Explainability-Dateien vorhanden sind und hilft bei Regressions-Checks.

## 6. Erwartete Artefakte

| Datei | Zweck | Erzeugt durch |
| --- | --- | --- |
| `demo/result/lo2/lo2_events.parquet` | Event-Tabelle mit Labels, Run/Test-Service | Loader (`--save-events`) |
| `demo/result/lo2/lo2_sequences.parquet` | Sequenzen pro Run/Test/Service | Loader (`--save-parquet`) |
| `demo/result/lo2/lo2_sequences_enhanced.parquet` | Tokens, Drain-IDs, numerische Features | Loader + Enhancer (`--save-enhancers`) |
| `demo/result/lo2/lo2_if_predictions.parquet` | Isolation-Forest-Scores, Threshold, Ranking | `LO2_samples.py` |
| `demo/result/lo2/metrics/*.json` | Precision@k, FP-Rate, PSI und weitere Metriken | `LO2_samples.py --report-*` |
| `models/lo2_if.joblib` | Persistiertes IF-Modell inkl. Vectorizer | `LO2_samples.py --save-model` |
| `models/model.yml` | Snapshot mit Parametern, Contamination, Git-Commit | `LO2_samples.py --dump-metadata` |
| `demo/result/lo2/explainability/*` | NN-Mapping, SHAP-Plots, False-Positive-Listen | `lo2_phase_f_explainability.py` |

## 7. Wichtige Flags und Varianten

- **Loader:** `--single-service <token>` fokussiert auf einzelne Services; `--service-types code token` erlaubt mehrere. `--allow-duplicates/--no-duplicates` steuert, ob identische Fehler mehrfach aufgenommen werden.
- **Isolation Forest:** `--if-max-samples` begrenzt Trainingsdaten; `--if-threshold-percentile` setzt die Alert-Grenze über ein Hold-out-Quantil.
- **Supervised Modelle:** Halte `--sup-holdout-fraction` ≥0.2, damit SHAP/NN-Reports belastbar sind. Nutze `--predict-batch-size`, wenn GPU/CPU-Speicher knapp wird.
- **Explainability:** `--nn-normal-sample` begrenzt die Menge normaler Vergleichssequenzen. `--shap-sample` sollte reduziert werden (<200), falls Plot-Generierung zu viel RAM benötigt.

## 8. Validierung & Qualitätssicherung

1. **Loader-Preview prüfen:** Der Loader-Durchlauf gibt eine Kopfzeile für Events und Sequenzen aus. Achte darauf, dass mindestens `run`, `test_case`, `service`, `log`, `anomaly` befüllt sind.
2. **Metriken sichten:** Öffne `demo/result/lo2/metrics/*.json` oder verwende `python -m json.tool` zur schnellen Ansicht.
3. **SHAP/NN Checks:** Kontrolliere `demo/result/lo2/explainability/*.png` bzw. `*_nn_mapping.csv`, um sicherzugehen, dass Explainability-Artefakte generiert wurden.
4. **Regressionen erkennen:** Bewahre relevante Ausgaben (z. B. `summary-result.md`) auf, um künftige Läufe zu vergleichen. Für reproduzierbare Runs notiere die eingesetzten CLI-Parameter.

## 9. Troubleshooting

- **`ModuleNotFoundError: loglead`** – vergewissere dich, dass `conda activate loglead_env` aktiv ist und `python -m pip install -e .` erfolgreich war.
- **`Root directory not found` im Loader** – stimmt `LO2_ROOT`? Prüfe mit `ls "$LO2_ROOT"` und achte auf Groß-/Kleinschreibung.
- **Leere `lo2_sequences_enhanced.parquet`** – es wurden evtl. keine Sequenzen gefunden (z. B. nur Events ohne `correct`-Runs). Reduziere Filter (`--runs`, `--service-types`) oder prüfe die Datenbasis.
- **Isolation-Forest findet keine Anomalien** – erhöhe `--if-contamination`, ergänze numerische Features (`--if-numeric seq_len,duration_sec,...`), nutze Hold-out (`--if-holdout-fraction 0.2`) und setze `--if-threshold-percentile`.
- **Explainability hängt** – reduziere `--shap-sample` auf 50–100 und setze `MPLBACKEND=Agg`, falls nicht bereits geschehen.

## 10. Nächste Schritte

1. **Größere Samples laden:** Entferne `--runs`/`--errors-per-run`, um das komplette Dataset zu verarbeiten und belastbarere Kennzahlen zu erhalten.
2. **Modell-Sweeps automatisieren:** Experimentiere mit Kombinationen aus `--models`, `--if-*` Flags und dokumentiere Ergebnisse in `summary-result.md`.
3. **Artefakte teilen:** Zippe `demo/result/lo2/explainability` oder lade relevante Dateien in ein Artefakt-Repository hoch, um Ergebnisse mit anderen zu vergleichen.
4. **CI-/Agent-Runs vorbereiten:** Verpacke die obigen Befehle in Skripte oder GitHub Actions, sodass neue LO2-Datendrops automatisch durch die Pipeline laufen.

Mit diesen Schritten hast du eine vollständige, reproduzierbare Umgebung, um die LO2 e2e Chain lokal zu betreiben und weiterzuentwickeln.
