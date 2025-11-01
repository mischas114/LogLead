# Isolation Forest Next Steps

- **Expand training data:** Load all 666 new log files via `run_lo2_loader.py --save-parquet`; more correct runs improve the token statistics the Isolation Forest learns from.
- **Calibrate on recent runs:** Reserve the last 10 % of correct events with `--if-holdout-fraction 0.1` so score drift against fresh logs becomes visible before deployment.
- **Target alert rate:** Set `--if-threshold-percentile 99.5` to derive a data-driven score cutoff that yields a predictable alert volume.

---

## Schritt-für-Schritt: Isolation Forest Pipeline Befehle

1. **Lade alle Logdaten als Parquet:**
	 ```bash
	 python demo/lo2_e2e/run_lo2_loader.py \
		 --root /Volumes/LO2_DATA/lo2-extracted \
         --errors-per-run 1 \
         --service-types code token refresh-token \
		 --save-parquet \
		 --output-dir demo/result/lo2
	 ```

2. **Trainiere Isolation Forest & speichere Modell + Metriken:**
	 ```bash
	 MPLBACKEND=Agg python demo/lo2_e2e/LO2_samples.py \
        --phase full \
        --if-contamination 0.15 \
        --if-holdout-fraction 0.05 \
        --if-threshold-percentile 99.5 \
        --report-precision-at 200 \
        --report-fp-alpha 0.01 \
        --report-psi \
        --save-if demo/result/lo2/lo2_if_predictions.parquet \
        --save-model demo/result/lo2/models/lo2_if.joblib \
        --dump-metadata \
        --metrics-dir demo/result/lo2/metrics \
        --save-enhancers \
        --enhancers-output-dir demo/result/lo2/enhanced
	 ```
	 
	*Begründung der Parameterwahl:*
    - `--if-contamination 0.15`: Dein Dataset enthält mehrere Fehler-Testcases und nur ~50 % korrekte Events. 15 % erlaubt dem Isolation Forest, diese Fehlerdichte realistisch abzubilden; später kannst du zwischen 0.1–0.2 feintunen.
    - `--if-holdout-fraction 0.05`: Reserviert ca. 5 % der korrekten Events (≈7 000 Zeilen) für Score-Drift und Schwellenkalibrierung, ohne das Training zu stark zu verkleinern.
    - `--if-threshold-percentile 99.5`: Nutzt den Holdout, um einen Score-Cutoff für die Top 0,5 % der Scores abzuleiten – sinnvoll, wenn du nur wenige Alerts akzeptieren willst.
    - `--report-precision-at 200`: Bei 299 k Events sind die Top 200 ausreichend, um die Präzision deiner höchsten Scores auszuwerten; mit 100 wäre die Stichprobe unnötig klein.
    - `--report-fp-alpha 0.01`: Betrachtet die obersten 1 % der Scores (~3 000 Events) für eine belastbare False-Positive-Quote.
    - Speicherpfade (`--save-if`, `--save-model`, `--metrics-dir`, `--enhancers-output-dir`) zeigen alle auf `demo/result/lo2/...`, damit sämtliche Artefakte unter dem Loader-Output liegen und Phase F sie direkt findet.
    - `--dump-metadata`: dokumentiert Schwelle, Parameter und Git-Commit in `model.yml`, was spätere Reproduktionen vereinfacht.
    - **Hinweis:** Wenn `--save-enhancers` bei erneuten Läufen auf bestehende Dateien trifft, ergänze `--overwrite-enhancers`, um die Parquets zu aktualisieren.

3. **Erzeuge Explainability-Artefakte (optional):**
	 ```bash
	 MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
        --root demo/result/lo2 \
        --if-contamination 0.15 \
        --nn-top-k 50 \
        --shap-sample 200
	 ```

---

**Alle Outputs findest du unter `demo/result/lo2/` und `models/`. Die Reihenfolge ist wichtig!**
- **Persist context:** Use `--save-model` and `--dump-metadata` to bundle the model, threshold, dataset sizes, and git commit into `models/` for reproducible scoring across sessions.


## Phase F Review

Artefakte liegen unter `demo/result/lo2/explainability/`: `lo2_if_predictions.parquet`, `if_nn_mapping.csv`, `if_false_positives.txt`, SHAP-Plots (`lr_shap_*.png`, `dt_shap_*.png`) sowie Metrik-JSONs für LR/DT/Sequence-LR.

Isolation Forest bleibt schwach (letzter Lauf mit `--if-item e_words`, Contamination 0.15: Accuracy 0.80, F1 0.09, AUC 0.50); Scores unterscheiden Normal vs. Fehler praktisch nicht, daher dominieren False Positives in der Top‑k-Liste.

Logistic Regression auf Wort-Features liefert AUC ~0.86, aber F1 0.17 – gute Rankingfähigkeit, schwache Schwellenleistung; Decision Tree overfittet (F1 ≈ 1, aber wenig Aussagekraft für Deployment), Sequence-LR perfekt auf 21 Fällen → zu kleine Stichprobe, SHAP entfällt.

NN-Mapping und False-Positive-Liste zeigen, dass top-gerankte IF-Fälle überwiegend aus „correct“-Runs stammen; damit dokumentiert Phase F exakt, wo das Modell derzeit fehlschlägt.

**Warum die Ergebnisse schwach ausfallen**

- Training enthält weit über 50 % Fehler-Events; trotz `--if-contamination 0.15` sieht der IF kaum saubere Normalfälle. Zudem führen Token-Features (`e_words`) ohne zusätzliche numerische Kontexte zu wenig Trennschärfe.
- Der 5 %-Holdout entzieht weitere 9 k Normal-Events; das Modell trainiert effektiv auf ~175 k gemischten Zeilen – ein großer Teil davon anomal → Score verteilt sich kaum.
- Downsampling auf 200 k Events verwässert Sequenzen, so dass derselbe korrekte Token-Stream vielfach als Anomalie eingestuft wird.
- Logistic Regression zeigt mit AUC 0.86, dass die Features prinzipiell Informationen tragen; der schwache F1 ist primär ein Schwellenproblem. Entscheidungsträume memorieren Tokens, daher scheinbare Perfektion.

**MVP-Einordnung**

Die MVP-Pipeline beweist die Machbarkeit der Explainability: Loader → IF → Phase F läuft durch, erzeugt SHAP-Plots, NN-Mapping und False-Positive-Analysen. Damit kannst du erklärbare Ergebnisse vorlegen.

Inhaltlich zeigen die Artefakte aber, dass die aktuelle IF-Konfiguration keine nützliche Anomalie-Erkennung liefert. Für eine aussagekräftige Machbarkeitsstudie solltest du zusätzliche Feature-Sets (Drain-IDs, numerische Aggregationen), sauberere Normal-Daten oder alternative Kontaminationswerte testen, bis F1/AUC ansteigen.

---

## Experiment: Drain-ID + numerische Features (`--if-item e_event_drain_id`)

```bash
MPLBACKEND=Agg python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len \
  --if-contamination 0.25 \
  --if-holdout-fraction 0.05 \
  --if-threshold-percentile 99.0 \
  --report-precision-at 300 \
  --report-fp-alpha 0.015 \
  --report-psi \
  --save-if demo/result/lo2/lo2_if_predictions_drain.parquet \
  --save-model demo/result/lo2/models/lo2_if_drain.joblib \
  --dump-metadata \
  --metrics-dir demo/result/lo2/metrics \
  --save-enhancers \
  --enhancers-output-dir demo/result/lo2/enhanced
```

- Lauf abgebrochen, weil `--save-enhancers` bestehende Dateien nicht überschreiben darf → beim erneuten Test `--overwrite-enhancers` ergänzen oder den Flag weglassen, sofern die letzten Enhancer-Parquets beibehalten werden sollen.

Anschließend wurde Phase F mit dem erhöhten Contamination-Wert ausgeführt:

```bash
MPLBACKEND=Agg python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-contamination 0.25 \
  --nn-top-k 50 \
  --shap-sample 200
```

**Ergebnisse (Ausschnitt):**
- Isolation Forest (Drain-ID Features): Accuracy 0.71, F1 0.11, AUC 0.50 → trotz zusätzlicher numerischer Spalten weiterhin kaum Trennschärfe; höheres `--if-contamination` verschiebt Schwelle, erhöht aber die False-Positive-Rate.
- Logistic Regression bleibt stabil (Accuracy 0.93, F1 0.18, AUC 0.86), Decision Tree weiterhin nahezu perfekt (Überanpassung), Sequence-LR wegen kleiner Stichprobe unverändert bei 1.0.
- `if_false_positives.txt` enthält nun vor allem Drain-IDs aus `correct`-Runs – Indikator, dass Training immer noch zu viele Anomalie-Events sieht und Normalfälle unterrepräsentiert sind.

**Takeaways:**
- Drain-ID-Features plus numerische Längen senken die IF-AUC nicht weiter, liefern aber keine substanzielle Verbesserung. Höhere Kontamination verschiebt lediglich den Schwellenwert.
- Nächste Tests sollten Normalanteil im Training erhöhen (mehr `correct`-Runs laden oder Fehlerläufe drosseln), zusätzliche numerische Sequenzfeatures (`seq_len`, Dauer) zum IF hinzufügen und `--if-max-samples` begrenzen, um Übergewicht großer Runs zu vermeiden.
- Für Explainability genügt der MVP weiterhin – SHAP/NN-Mapping entstehen reproduzierbar. Die Modellqualität bleibt jedoch das zentrale Problem.


python demo/lo2_e2e/run_lo2_loader.py \
    --root /Volumes/LO2_DATA/lo2-extracted \
    --errors-per-run 1 \
    --service-types code token refresh-token \
    --save-parquet \
    --output-dir demo/result/lo2


python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --if-max-samples 100000 \
  --if-contamination 0.1 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/lo2_if_predictions.parquet \
  --metrics-dir demo/result/lo2/metrics \
  --save-model demo/result/lo2/models/lo2_if.joblib \
  --dump-metadata

python demo/lo2_e2e/LO2_samples.py \
  --phase full \
  --if-holdout-fraction 0.05 \
  --if-item e_event_drain_id \
  --if-numeric e_chars_len,e_event_id_len,e_words_len,e_lines_len \
  --if-max-samples 261748 \
  --if-contamination 0.25 \
  --if-threshold-percentile 99.5 \
  --report-precision-at 200 \
  --report-fp-alpha 0.01 \
  --save-if demo/result/lo2/lo2_if_predictions.parquet \
  --metrics-dir demo/result/lo2/metrics \
  --save-model demo/result/lo2/models/lo2_if.joblib \
  --overwrite-model \
  --dump-metadata

python demo/lo2_e2e/lo2_phase_f_explainability.py \
  --root demo/result/lo2 \
  --if-max-samples 100000 \
  --nn-top-k 0 \
  --nn-normal-sample 0 \
  --shap-sample 0


  Hier sind die aktualisierten Doku-Abschnitte, inkl. Quellen, PR-Text und Commit-Message.

docs/NEXT_STEPS_IF.md

<!-- AUTOGEN:IF_NEXT_STEPS:BEGIN -->
## Isolation Forest: Status & nächste Schritte

**Kurzfazit:** Der unsupervised IF liefert auf LO2 keine verwertbare Diskriminationsleistung (F1 ≈ 0.12; AUC ≈ 0.51), obwohl Drain-IDs, erweiterte numerische Features, `max_samples` = 261748 und ein 5 %-Holdout aktiv sind.

**Metriken (IF)**  
| Laufzeit | AUC | F1 | Precision | Recall | contamination | max_samples | n_samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-10-31 18:36:46Z | 0.51 | 0.12 | 0.08 | 0.25 | 0.25 | 261748 | 299115 |

**Vergleich (Supervised Benchmarks)**  
| Modell | AUC | F1 | Notizen |
|---|---:|---:|---|
| Logistic Regression | 1.00 | 0.88 | Training/Test auf kompletten Featuresatz (`e_words`); Schwellen-Tuning möglich |
| Decision Tree | 1.00 | 1.00 | Vollständige Trennbarkeit auf `e_trigrams` |

**Interpretation:** IF passt nicht zur LO2-Verteilung (viele Error-Runs, sehr ähnliche Text-Tokens). Das Log-Statement „Downsampling occurred: yes“ erklärt sich durch den 5 %-Holdout, nicht durch Trainings-Downsampling.

**Nächste Schritte:**  
- IF als Demonstrator belassen, Resultate dokumentieren.  
- Optional: IF-Sweep mit `contamination ∈ {0.30, 0.35}` durchführen.  
- Benchmarks: LR/DT als Referenz pflegen und für LR ein Schwellen-Tuning zur Alert-Steuerung zeigen.

**Quellen/Artefakte:**  
- `demo/result/lo2/metrics/if_metrics.json`  
- `demo/result/lo2/metrics/lr_full.json`  
- `demo/result/lo2/metrics/dt_full.json`  
- Commit: 1b9fcbc22c87b95051e82074e71ef3ae4e5a11e9  

## Aktueller Modellstatus

**Pipeline-Änderungen:**  
- Drain-IDs aktiv; numerische Features um `e_chars_len`, `e_event_id_len`, `e_words_len`, `e_lines_len` ergänzt.  
- `max_samples` für den IF auf 261748 hochgesetzt; 5 %-Holdout (temporal) eingeführt.  
- Benchmarks (LR/DT) erneuert und zentral abgelegt.

**Metrik-Überblick:**  
- Isolation Forest (unsupervised): AUC ≈ 0.51, F1 ≈ 0.12 → geringe Trennschärfe.  
- Logistic Regression: AUC ≈ 1.00, F1 ≈ 0.88.  
- Decision Tree: AUC ≈ 1.00, F1 ≈ 1.00.

**Hinweis zum Summary:**  
„Downsampling occurred: yes“ stammt vom 5 %-Holdout (Reserve), nicht von verkleinerten Trainingsdaten.

**Empfehlung:**  
LR/DT als produktionsnahe Benchmarks führen; der IF bleibt als Lehr-/Demo-Artefakt mit dokumentierten Grenzen bestehen.  

PR-Beschreibung (Kurzform)

Was: IF-Metriken mit Drain-ID/Feature-Erweiterungen eingepflegt, schwache Performance tabellarisch belegt. LR/DT-Benchmarks aus demo/result/lo2/metrics/*.json hervorgehoben. Holdout-Hinweis ergänzt („Downsampling occurred: yes“ = 5 % Reserve). Nächste Schritte (IF-contamination-Sweep, LR-Threshold-Tuning) dokumentiert.
Warum: Vollständige Nachvollziehbarkeit der aktuellen Runs, klare Handlungsempfehlungen trotz schwacher IF-Ergebnisse.
Quellen: demo/result/lo2/metrics/if_metrics.json, demo/result/lo2/metrics/lr_full.json, demo/result/lo2/metrics/dt_full.json, Commit 1b9fcbc22c87b95051e82074e71ef3ae4e5a11e9.
Commit-Message

