# NEXT_STEPS.md — LO2 Client-only Isolation Forest (IF) 🚀

## 0) Zielbild (1-Pager)
**In 1–2 Iterationen** liefern wir eine stabile IF-Baseline auf *Client*-Logs mit:
- **AUCROC ≥ 0.60** (Client-only), **Alert-Rate** fix bei **1–3 %**, **Precision@10** reportet  
- Reproduzierbare Artefakte (Modelle, Plots, Predictions, XAI)  
- Dokumentierte False-Positive-Muster + Gegenmaßnahmen

---

## Nach jedem MVP-Durchlauf (Quick Checklist)
- `demo/result/lo2/explainability/`: SHAP-Plots, `if_nn_mapping.csv`, `if_false_positives.txt` prüfen und Findings in `summary-result.md` oder Tickets dokumentieren.
- `python tools/lo2_result_scan.py` (`--dry-run` zum Testen) fasst die Artefakte automatisch zusammen und hängt das Ergebnis an `summary-result.md`; optional `--summary-file`/`--ticket-template` setzen.
- `result/lo2/enhanced/`: Enhanced-Parquets behalten (siehe `docs/LO2_enhanced_exports.md`) und bei Notebook-Analysen versionieren.
- Modell-Metriken (`metrics_*.json`) gegen Zielwerte halten; bei AUCROC < 0.6 oder Precision@10 < 0.2 erst Kontamination/Sample-Seed justieren, bevor weitere Daten geladen werden.
- Phase-F-Runs lassen sich direkt neu starten, z. B.:
  ```bash
  MPLBACKEND=Agg python demo/lo2_phase_f_explainability.py \
    --root demo/result/lo2 \
    --if-contamination 0.1 \
    --nn-top-k 50 \
    --shap-sample 200
  ```
  Anschließend den Result-Scan laufen lassen, um die neuen Kennzahlen zu protokollieren.
- Für schnellere Iterationen vor großen Datenläufen `LO2_samples.py --phase enhancers` (Feature-Check) ausführen und erst bei Bedarf mit `--phase if` bzw. `--phase full` in die Modellierung starten.
- Headless-Modus (`MPLBACKEND=Agg`) verwenden, um lange SHAP-Runs ohne manuelles Eingreifen durchlaufen zu lassen und Logs in `summary-result.md` zu referenzieren.

### False-Positive-Zahlen richtig einordnen
- Phase-F arbeitet auf Ereignis-Ebene (`lo2_events.parquet`). Jede Logzeile wird einzeln bewertet.
- `--if-contamination=0.45` erlaubt dem IsolationForest, bis zu 45 % aller Events als anomal zu markieren. Bei ~386 k Events entstehen so schnell >150 k Einträge in `if_false_positives.txt`, auch wenn Runs insgesamt korrekt sind. Mit `--if-contamination=0.1` sank die FP-Liste zuletzt auf ~44 k Events, blieb aber deutlich über Null, weil weiterhin einzelne Logzeilen in gesunden Runs auffallen.
- False Positives ≥ 0 sind normal und dienen als „Review-Queue“. Reduziere die Menge, indem du die Kontamination senkst (z. B. 0.05) oder eine Quantilschwelle gemäß Abschnitt „Schwellenkalibrierung“ nutzt.
- Für Run-/Sequence-Level-Alerts `lo2_sequences.parquet` verwenden; `LO2_samples.py --phase if` schreibt die IF-Scores je Sequenz und reduziert damit das FP-Volumen gegenüber Event-Level-Ausgaben.

## 1) Daten – konkret erweitern (ohne 540 GB zu ziehen)

**Aktueller Stand:** 5–10 Client-Runs (Sample).  
**Ziel kurzfristig (Iteration A):** **≥ 250 Runs** Client-only.  
**Ziel mittelfristig (Iteration B):** **≈ 500–800 Runs** Client-only.

> **Warum diese Größenordnung?**  
> - Ab ~250 Runs stabilisieren sich Vokabular und Score-Verteilungen merklich.  
> - 500–800 Runs reichen für 5 Seeds × kleines Parameterraster, ohne in IO-Hölle zu enden.

### 1.1 Was genau aus dem „großen“ Datensatz ziehen?
Nur **Client-Service** und nur **Events/Sequences**:
- **Zeitfenster:** 2–4 repräsentative Wochen (verteilt auf mind. 2 Monate)  
- **Testcase-Verteilung:** 50 % „correct“, 50 % „error“ **im Test**, **Training = nur correct**  
- **Pro Run Limit:** max. **10 k Log-Zeilen** (abschneiden), um Ausreißer-Runs zu zähmen  
- **Dateifilter:** exclude *debug-spam*, Health-Checks / Heartbeats (regex-Whitelist/Blacklist definieren)

**Heuristik für das erste Nachladen (Iteration A):**
- **250 Runs** total  
  - **Train:** 150 „correct“  
  - **Val (Normals):** 50 „correct“  
  - **Test (gemischt):** 50 (25 correct / 25 error)

**Für Iteration B (falls Ressourcen ok):**
- **800 Runs** total  
  - **Train:** 500 correct  
  - **Val (Normals):** 150 correct  
  - **Test:** 150 (75/75)

### 1.2 Wie groß ist das am Ende?
- Parquet-komprimiert (sparse Features): **~3–10 GB** für 800 Runs i. d. R. machbar.  
- **Wir brauchen nicht 540 GB**: striktes Service-/Zeit-/Zeilen-Subset + Parquet + Sparse.

### 1.3 Kürzen & Bereinigen (on repeat)
- **Zeilen-Cap pro Run:** `cap_lines_per_run=10_000`  
- **Dedup repetitive Lines** innerhalb eines Runs (gleicher Template-Hash + gleicher Payload → nur Zähler erhöhen)  
- **Drop Heartbeats/Health** (Regex wie `(?i)health|heartbeat|ping`)  
- **Trim Payload-IDs** (GUIDs, IPs, Timestamps) vor Tokenisierung  
- **Persistenz:** immer **Parquet** + **sparse CSR** für Textfeatures

**CLI-Beispiel (Pseudo):**
```bash
python run_lo2_loader.py   --service client   --date_from 2024-11-01 --date_to 2025-02-28   --cap_lines_per_run 10000   --drop_regex "(?i)health|heartbeat|ping"   --save-parquet out/lo2_events.parquet
```

---

## 2) Features – schlank und robust
- **Item-Spalte (A/B):**  
  A) `e_event_drain_id` (Default)  
  B) `e_words` (robust gegen Parser-Noise)  
- **Numeric minimal:** `e_chars_len`, `e_words_len`, optional `seq_len`, `duration_sec`  
- **Vektorisierung:** `CountVectorizer(max_features=100_000, min_df=3, ngram_range=(1,3))`  
- **Wichtig:** Vectorizer **nur auf Train fitten**, Test **nur transformen** (Vokabular freeze)

---

## 3) Modell – Isolation Forest (ohne contamination)
- `n_estimators ∈ {200, 400}` (Start: **400**)  
- `max_samples ∈ {256, 512}` (Start: **256**)  
- `max_features=1.0, bootstrap=False, random_state=SEED`  
- **Scoring:** `score_samples(X)` (je kleiner, desto anomaler)  
- **Schwelle:** **Quantil auf Validierungs-Normals**  
  - Startwerte: **0.5 % / 1.0 % / 1.5 % / 3.0 %** → Alert-Rate steuern

---

## 4) Evaluations-Setup (sauber & reproduzierbar)
- **Splits:**  
  - **Train:** nur correct  
  - **Val (Normals):** nur correct (Schwellenkalibrierung)  
  - **Test:** correct + error (nur Reporting)  
  - **Zeitlich/run-weise** trennen, keine Überlappung
- **Seeds:** **5** (0, 1, 2, 3, 4)  
- **Metriken:**  
  - **Primär:** ROC-AUC, **Alert-Rate vs. Recall**, **Precision@10**  
  - **Sekundär:** **F1@best threshold** (nur Vergleich)  
- **Plots/Artefakte:** ROC-Curve, Alert-Rate-Vs-Recall-Plot (Punkte bei 0.5/1/1.5/3 %), Top-10 Alerts (mit NN-Begründung)

---

## 5) Parameterraster (klein halten)
| Block | Parameter        | Werte                        |
|------:|------------------|------------------------------|
|   B1  | `n_estimators`   | 200, **400**                 |
|   B2  | `max_samples`    | **256**, 512                 |
|   B3  | Item-Spalte      | **`e_event_drain_id`**, `e_words` |
|   B4  | Numeric-Spalten  | off, **len + seq_len/duration** |
|   B5  | Seeds            | **5**                        |

**Max. Läufe:** 2×2×2×2×5 = **80** (Iteration B).  
**Iteration A:** nur B1×B3×Seeds = 2×2×5 = **20** Läufe.

---

## 6) Schwellenkalibrierung (operativ)
1. Scores **auf Val-Normals** sammeln  
2. Für `p ∈ {0.005, 0.01, 0.015, 0.03}` → **Schwelle = p-Quantil**  
3. Auf Test anwenden → Report: **Alert-Rate, Recall, Precision@10** je p  
4. **Auswahlregel:** p = 0.015 (Start) oder p, das **Precision@10** maximiert bei **Alert-Rate ≤ 3 %**

---

## 7) False Positives entschärfen (gezielt)
- **Hotspot**: `light-oauth2-oauth2-client-1`  
  - **In Training aufnehmen**: Top-10 wiederkehrende „korrekt“-Sequenzen (Outlier-Exposure light)  
  - **Cluster-Schwellen (leicht):** UMAP/HDBSCAN auf Val-Normals → **pro Cluster** p-Quantil anwenden  
  - **Nicht blocken**, sondern lernen lassen: häufige Drain-IDs als Normalbeispiele dem Train hinzufügen

---

## 8) Deliverables & Ordnerstruktur
```
models/
  lo2_if_{items}_n{n}_s{seed}.joblib
predictions/
  lo2_if_predictions_{items}.parquet
reports/
  grid_{date}.csv                  # Metriken je Lauf
  roc_{items}.png
  alert_curve_{items}.png
explainability/
  if_nn_topk_{items}.csv
  if_fp_hotspots_{service}.txt
docs/
  LO2_minimal_IF_XAI_workflow.md   # um Plots/Tabellen ergänzen
```

---

## 9) Konkrete Reihenfolge (Checkliste)

### Iteration A (heute–nächste Session, ~20 Läufe)
1. **Daten nachladen auf 250 Runs (Client)**  
   - Fenster 2–4 Wochen, `cap_lines_per_run=10k`, Heartbeats/Health droppen, Parquet schreiben  
2. **Features bauen** mit EventLogEnhancer (words, trigrams, drain, length; optional seq_len/duration)  
3. **Splits erstellen:** Train(correct)=150, Val-Normals(correct)=50, Test(mix)=50  
4. **IF trainieren (ohne contamination)** für Items ∈ {drain, words}, `n_estimators ∈ {200, 400}`, Seeds=5  
5. **Schwelle kalibrieren** (p ∈ {0.5, 1.0, 1.5, 3.0} %) auf Val-Normals  
6. **Evaluation & Plots:** ROC, Alert-Rate vs Recall, Precision@10, F1@best  
7. **Top-Alerts dokumentieren** (NN-Nachbarn + Drain-IDs), **FP-Hotspot-Liste** erzeugen

**Abnahme-Kriterien Iteration A:**  
- AUCROC **≥ 0.58** (mind. ein Setting)  
- **Alert-Rate 1–3 %** möglich + sinnvolle Precision@10  
- FP-Hotspot identifiziert & gelistet

### Iteration B (nach A, optional ~80 Läufe)
1. **Runs auf 500–800** erweitern (gleiches Filter-/Cap-Schema)  
2. **Parameterraster komplett** (B1–B5)  
3. **Cluster-Schwellen** je Val-Cluster (optional)  
4. **Outlier-Exposure light:** wiederkehrende korrekte Sequenzen des Hotspots in Train aufnehmen  
5. **Finale Reports + One-Pager** (KPIs, Plots, FP-Beispiele, Settings)

**Abnahme-Kriterien Iteration B:**  
- AUCROC **≥ 0.60** stabil über Seeds  
- **Alert-Rate 1–3 %** mit **Precision@10** praxistauglich  
- Dokumentierte FP-Reduktion am Hotspot

---

## 10) Risiken & Mitigation
- **Parser-Noise (Drain):** Parallel **`e_words`** laufen lassen (robust), Parser-Fehler loggen  
- **Memory/IO:** `max_features=100k`, `min_df=3`, sparse beibehalten, keine `.toarray()`  
- **Concept Drift:** Beim Nachladen **Schwellen (p-Quantile)** stets neu kalibrieren  
- **Metric-Illusion:** Accuracy ignorieren; **AUC + Alert-Kurve + Precision@10** als Leitplanken

---

## 11) Mini-Konfiguration (Defaults)
```yaml
data:
  service: client
  cap_lines_per_run: 10000
  drop_regex: "(?i)health|heartbeat|ping"
  runs_target_iteration_A: 250
  runs_target_iteration_B: 800

features:
  items: ["e_event_drain_id", "e_words"]
  numeric: ["e_chars_len", "e_words_len", "seq_len", "duration_sec"]
  vectorizer:
    max_features: 100000
    min_df: 3
    ngram_range: [1,3]

model:
  type: isolation_forest
  n_estimators: [200, 400]
  max_samples: [256, 512]
  max_features: 1.0
  contamination: null
  seeds: [0,1,2,3,4]

thresholds:
  p_quantiles: [0.005, 0.01, 0.015, 0.03]

report:
  metrics: ["roc_auc", "precision@10", "alert_rate_vs_recall", "f1@best_threshold"]
  artifacts: ["roc.png", "alert_curve.png", "grid.csv", "top_alerts.csv"]
```
