# LO2 e2e Chain Setup

komplette LO2-Pipeline (Loader → Enhancer → Isolation Forest & Registry → Explainability) aufsetzen

## 1. Python-Umgebung mit Conda vorbereiten

### Install Conda (Anleitung ist für MAC)

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

## 2. Repository initialisieren

1. Wechsle in das Projekt und (optional) installiere es im Entwicklungsmodus:
   ```bash
   cd ~/Projects/LogLead
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

- Kopiere oder mounte die Roh-Logs in ein lokales Verzeichnis

So stellst du sicher, dass Loader und Pipeline-Skript erreichbar sind.

## 4 LO2 Datensatz beziehen & vorbereiten

1. **Download zenodo :** https://zenodo.org/records/14938118
   - Klicke auf „Download newest version“, damit du die aktuelle Fassung erhältst.
   - Rechne mit mehreren Stunden Download (bei mir ≈5 h) und plane genügend Bandbreite ein.
2. **Entpacken:** Nach dem Entpacken beansprucht der Dump ~650 GB – sorgt für ausreichend freien SSD-Speicher.
3. **Dataset-Split herstellen:** Nutze die aktuelle Version des Extraktionsskripts (Python) und anschließend den Shell-Wrapper aus diesem Repo, um Archive in handliche Chunks zu teilen:
   ```bash
   python extract-split-resume-lo2.py               # passe den Dateinamen an deine Variante an
   MODE=newest SPLIT=4 DRY_RUN=0 bash tools/extract-split-resume-lo2.sh
   # Mit CHUNK=2 lässt sich z. B. nur die zweite Hälfte entpacken:
   MODE=newest SPLIT=4 CHUNK=2 DRY_RUN=0 bash tools/extract-split-resume-lo2.sh
   ```
4. **Pfad für LogLead merken:** Setze `LO2_ROOT` auf das entpackte Ziel; der Loader nutzt diesen Pfad als `--root`.

## 5. LO2 e2e Chain ausführen

### 5.1 Loader (Phase B)

```bash
python demo/lo2_e2e/run_lo2_loader.py \
  --root "$LO2_ROOT" \
  --runs 5 \ #ohne runs werden alle genommen
  --errors-per-run 1 \
  --service-types code token refresh-token \ #ich hab nur code
  --save-parquet \
  --save-events \ # weglassen
  --save-base-sequences \
  --output-dir demo/result/lo2
```

- `--runs` und `--errors-per-run` begrenzen die Menge für Smoke-Tests. Lass beide weg, wenn du alles laden willst.
- `--save-parquet` erzeugt `lo2_sequences.parquet` und `lo2_sequences_enhanced.parquet` im Output-Ordner – diese Dateien benötigt der nächste Schritt.
- Nutze `--load-metrics`, wenn unter `run_*/**/metrics/*.json` zusätzlich Kennzahlen liegen, die du mitschneiden möchtest.

### 5.2 Enhancer + Modelle (Phasen C–E)

```bash
python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --root demo/result/lo2 \
  --if-item e_words \
  --if-numeric seq_len,duration_sec,e_words_len,e_trigrams_len \
  --if-contamination 0.15 \
  --if-holdout-fraction 0.2 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-model models/lo2_if.joblib \
  --dump-metadata \
  --models event_lr_words,event_dt_trigrams,event_xgb_words,sequence_shap_lr_words \
  --metrics-dir demo/result/lo2/metrics \
  --save-enhancers
```

- `--phase full` umfasst Feature-Erzeugung, Isolation Forest und die angegebenen Registry-Modelle.
- Passe `--models` bei Bedarf an (`python demo/lo2_e2e/LO2_samples.py --list-models` zeigt alle verfügbaren Schlüssel).
- _Setze `--skip-if`, falls du einen Lauf ohne Isolation Forest starten möchtest (z. B. für reine Supervised-Benchmarks). -> sinnvoll, da IF Benchmarks schlecht sind_

### 5.3 Explainability (Phase F)

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --skip-if \
  --sup-models event_lr_words,event_dt_trigrams,event_xgb_words,sequence_shap_lr_words \
  --nn-source sequence_shap_lr_words \
  --nn-top-k 50 \
  --shap-sample 200 \
  --shap-background 256 \
  --shap-feature-threshold 2000 \
  --shap-cell-threshold 2000000 \
  --sup-holdout-fraction 0.2
```

- `MPLBACKEND=Agg` verhindert GUI-Abhängigkeiten und ermöglicht headless SHAP-Plots.
- Entferne `--skip-if`, wenn du den Isolation Forest in Phase F erneut trainieren willst (z. B. mit anderen Parametern).
- Mit `--nn-source` steuerst du, welches Modell für das Nearest-Neighbour-Mapping genutzt wird (`if`, `sequence_shap_lr_words`, …).
- `--shap-background` bestimmt, wie viele Sequenzen als Hintergrundverteilung für SHAP genutzt werden (Standard 256, `0` = komplettes Training).
- `--shap-feature-threshold` und `--shap-cell-threshold` schützen vor unhandlich großen SHAP-Matrizen. Hebe die Werte an, wenn ein Guard greift.
- Tree-basierte Modelle (RandomForest, XGBoost, IsolationForest) laufen automatisch über `shap.TreeExplainer`, lineare Modelle über `shap.LinearExplainer`. Für nicht unterstützte Modelle legt das Skript eine Hinweisdatei im Explainability-Ordner ab.

## 6. Erwartete Artefakte

| Datei                                            | Zweck                                              | Erzeugt durch                          |
| ------------------------------------------------ | -------------------------------------------------- | -------------------------------------- |
| `demo/result/lo2/lo2_events.parquet`             | Event-Tabelle mit Labels, Run/Test-Service         | Loader (`--save-events`)               |
| `demo/result/lo2/lo2_sequences.parquet`          | Sequenzen pro Run/Test/Service                     | Loader (`--save-parquet`)              |
| `demo/result/lo2/lo2_sequences_enhanced.parquet` | Tokens, Drain-IDs, numerische Features             | Loader + Enhancer (`--save-enhancers`) |
| `demo/result/lo2/lo2_if_predictions.parquet`     | Isolation-Forest-Scores, Threshold, Ranking        | `LO2_samples.py`                       |
| `demo/result/lo2/metrics/*.json`                 | Precision@k, FP-Rate, PSI und weitere Metriken     | `LO2_samples.py --report-*`            |
| `models/lo2_if.joblib`                           | Persistiertes IF-Modell inkl. Vectorizer           | `LO2_samples.py --save-model`          |
| `models/model.yml`                               | Snapshot mit Parametern, Contamination, Git-Commit | `LO2_samples.py --dump-metadata`       |
| `demo/result/lo2/explainability/*`               | NN-Mapping, SHAP-Plots, False-Positive-Listen      | `lo2_phase_f_explainability.py`        |

## 7. Wichtige Flags und Varianten

- **Loader:** `--single-service <token>` fokussiert auf einzelne Services; `--service-types code token` erlaubt mehrere. `--allow-duplicates/--no-duplicates` steuert, ob identische Fehler mehrfach aufgenommen werden.
- **Isolation Forest:** `--if-max-samples` begrenzt Trainingsdaten; `--if-threshold-percentile` setzt die Alert-Grenze über ein Hold-out-Quantil.
- **Supervised Modelle:** Halte `--sup-holdout-fraction` ≥0.2, damit SHAP/NN-Reports belastbar sind. Nutze `--predict-batch-size`, wenn GPU/CPU-Speicher knapp wird.
- **Explainability:** `--nn-normal-sample` begrenzt die Menge normaler Vergleichssequenzen. `--shap-sample` sollte reduziert werden (<200), falls Plot-Generierung zu viel RAM benötigt.
