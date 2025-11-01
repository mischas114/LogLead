

## Ausgangszustand (vor meinen Änderungen)
- Es gab bereits einen Loader (`demo/run_lo2_loader.py`) und ein kombiniertes Pipeline-Skript (`demo/LO2_samples.py`), aber beide lagen direkt unter `demo/` und waren nur locker mit älteren Prototyp-Skripten verknüpft (`lo2_if_baseline.py`, `lo2_feature_test.py`).
- Der Loader konnte Events/Sequenzen laden und als Parquet sichern, bot auch Service-Filter, aber schrieb seine Ergebnisse unter wechselnden Pfaden (`demo/result/lo2` relativ zum Skript) und war in den Abläufen nicht mit einer dokumentierten Pipeline verzahnt.
- `demo/LO2_samples.py` deckte Enhancement, Isolation Forest und **nur zwei fest verdrahtete Supervised-Modelle** (LogReg + DecisionTree) ab, bot aber keine Persistenz von Modellen/Enhancers, keine Kennzahlenreports, kein Holdout-/Threshold-Tuning, keine Metadatenablage und erzeugte keine Explainability-Artefakte.
- **Keine Modell-Registry:** Neue Modelle konnten nur durch direktes Code-Editing hinzugefügt werden; keine CLI-gesteuerte Modellauswahl.
- Phase F war nur durch manuelle Notebook- bzw. Prototype-Skripte möglich; NN-Mappings, SHAP-Plots oder False-Positive-Auswertungen mussten händisch durchgeführt werden.
- Dokumentation war auf mehrere ältere Dateien verteilt (`LO2_minimal_IF_XAI_workflow.md`, `LO2_prototype_pipeline.md`, `Persistenz.md`) und widersprach sich teilweise.

## Ziel der Erweiterung
- Eine reproduzierbare LO2 MVP Pipeline schaffen, die von Roh-Logs bis Explainability durch Skripte abgedeckt ist.
- **Modell-Registry implementieren:** Flexible Kombination von 14+ ML-Modellen über CLI ohne Code-Änderungen.
- Artefakte (Parquet, Modelle, Metriken, XAI-Plots) konsistent schreiben und in der Doku nachvollziehbar machen.

## Neue und überarbeitete Komponenten

| Datei | Status | Rolle im E2E-Prozess |
| --- | --- | --- |
| `demo/lo2_e2e/run_lo2_loader.py` | verlagert & ergänzt | Phase B: Loader nach `demo/lo2_e2e/` verschoben, Pfade vereinheitlicht, Service-Type-Filter und klare Output-Struktur (`demo/result/lo2/`). |
| `demo/lo2_e2e/LO2_samples.py` | **stark erweitert** | Phasen C–E: **Modell-Registry mit 14 konfigurierbaren Detektoren** (Supervised: LR, SVM, DT, RF, XGBoost; Unsupervised: LOF, OneClassSVM, KMeans, Rarity, OOV; Sequence: LR + SHAP). Holdout/Threshold-Tuning, Modell-Persistenz (`--save-model`), Enhancer-Export, Kennzahlen (`--report-*`), Metadata-Dump. CLI-gesteuert über `--models` / `--list-models` ohne Code-Änderungen. |
| `demo/lo2_e2e/metrics_utils.py` | neu | Kennzahlen-Helfer für Precision@k, FP-Rate@α, PSI auf den IF-Scores. |
| `demo/lo2_e2e/lo2_phase_f_explainability.py` | neu | Phase F: Reproduziert bestes IF-Setup, erstellt NN-Mapping, SHAP-Plots, False-Positive-Reports – erste automatisierte Explainability-Stufe. |
| `demo/lo2_e2e/README.md` | neu | Quickstart mit drei Kommandos über alle Phasen hinweg. |
| `docs/LO2_e2e_pipeline.md` | überarbeitet | Konsolidierte Schritt-für-Schritt-Anleitung, Artefakt-Index, Modell-Registry-Tabelle und Tuning-Hinweise (ersetzt ältere Prototyp-Dokumente). |
| `docs/modelle_overview.md` | neu | Vollständiger Modellkatalog mit Vergleichstabelle, Ausführungsbeispielen und Use-Case-Empfehlungen für alle 14 Registry-Modelle. |
| `docs/LO2_IF_E2E.md` | neu | Vertiefung zu Isolation-Forest-Training, Schwellenkalibrierung und Persistenz. |
| `docs/DATENVERARBEITUNG_INTEGRATION.md` | erweitert | Einbettung des LO2-Flows in Datenverarbeitung/Integrationskontext. |
| `models/model.yml` | neu erzeugt | Automatischer Metadaten-Snapshot zum gespeicherten Isolation-Forest (Parameter, Training-Stats, Git-Commit). |

## Resultierender LO2 E2E Ablauf
1. **Phase A/B – Loader:** `python demo/lo2_e2e/run_lo2_loader.py --root <logs> --save-parquet --output-dir demo/result/lo2`  
   → erzeugt `lo2_events.parquet` und `lo2_sequences.parquet`.
2. **Phasen C–E – Enhancement & Modelle:** `python demo/lo2_e2e/LO2_samples.py --phase full --models event_lr_words,event_rf_words,sequence_shap_lr_words --save-enhancers --save-model models/lo2_if.joblib --report-precision-at 100 --report-psi`  
   → schreibt Features, trainiert Isolation Forest + ausgewählte Registry-Modelle, erstellt `lo2_if_predictions.parquet`, Metrics und Modellpersistenz (`model.yml`).  
   → **Modellauswahl über CLI:** `--models` wählt beliebige Kombinationen aus 14 Detektoren; `--list-models` zeigt alle verfügbaren Optionen.
3. **Phase F – Explainability:** `MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py --root demo/result/lo2 --shap-sample 200 --nn-top-k 50`  
   → legt NN-Mappings, SHAP-Plots und XAI-Metriken unter `demo/result/lo2/explainability/` ab.

Damit ist der komplette Weg von LO2-Rohlogs bis Explainability reproduzierbar dokumentiert und automatisiert.

## Schlüssel-Innovation: Modell-Registry

Das zentrale Feature ist die **deklarative Modell-Registry** in `LO2_samples.py`:

```python
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "event_lr_words": {
        "level": "event",
        "item_list_col": "e_words",
        "train_method": "train_LR",
        ...
    },
    "event_rf_words": { ... },
    "sequence_shap_lr_words": { ... },
    # insgesamt 14 Modelle
}
```

**Vorteile:**
- **Keine Code-Änderungen:** Neue Modell-Kombinationen über CLI (`--models key1,key2`)
- **Event + Sequence Level:** Automatische Datenauswahl basierend auf `"level": "event"/"sequence"`
- **Flexible Feature-Kombinationen:** Token-Listen (`e_words`, `e_trigrams`, `e_event_drain_id`) + numerische Features
- **Training-Strategien:** Supervised (alle Daten), Unsupervised (`correct_only`), Semi-Supervised
- **SHAP-Integration:** Opt-in Explainability für ausgewählte Modelle via `"requires_shap": True`

**Beispiel-Workflow:**
```bash
# Liste alle verfügbaren Modelle
python demo/lo2_e2e/LO2_samples.py --list-models

# Teste Supervised-Paket
python demo/lo2_e2e/LO2_samples.py --phase full \
  --models event_lr_words,event_lsvm_words,event_rf_words,event_xgb_words

# Vergleiche Unsupervised-Methoden
python demo/lo2_e2e/LO2_samples.py --phase full \
  --models event_lof_words,event_oneclass_svm_words,event_kmeans_words

# Sequence-Analyse mit SHAP
python demo/lo2_e2e/LO2_samples.py --phase full \
  --models sequence_lr_numeric,sequence_shap_lr_words
```

## Dokumentations-Backlog (ToDos)
- [x] `docs/DATENVERARBEITUNG_INTEGRATION.md` nachziehen: Abschnitt zu konfigurierbaren Modellen ergänzen, Diagramm mit Phase-E-Varianten erweitern, offene Fragen aktualisieren.
- [x] `docs/LO2_e2e_pipeline.md` aktualisieren: `--models` Flag samt Beispiel aufnehmen, Default-Modelle tabellarisch erklären, Quickstart-Befehle vereinheitlichen.
- [x] `docs/LO2_IF_E2E.md` prüfen: Threshold-/Holdout-Beschreibung gegen aktuelle CLI-Parameter abgleichen (z. B. `--if-threshold-percentile`).
- [x] Architektur-Visuals durch Mermaid-Diagramme ersetzen (`docs/architektur-v1.md`, `docs/LO2_architektur_detail.md`).
- [x] `docs/NEXT_STEPS.md` um wartbare Dokumentations-Changelog-Sektion ergänzen, damit zukünftige Anpassungen nachvollziehbar bleiben.

----
