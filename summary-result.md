# LO2 Modeling & Explainability Summary

## Dataset & Run Context
- Pipeline: LO2 loader → default EventLogEnhancer steps (normalize, tokens, trigrams, Drain IDs, length features)
- Scope: Isolation Forest baseline plus supervised comparison (Logistic Regression, Decision Tree, Sequence Logistic Regression)
- Evaluation split: isolation forest trained on `test_case == "correct"`; supervised models trained and evaluated on full labelled sample (support shown per metric file)

## Model Performance
| Model | Feature Basis | Accuracy | F1 | AUC-ROC | Support | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Isolation Forest | Drain IDs + length features | ~0.52 | ~0.47 | ~0.50 | 142,916 | High false-positive concentration on `light-oauth2-oauth2-token-1` service (see NN explainability) |
| Logistic Regression | Bag-of-words tokens | 0.75 | 0.71 | 0.84 | 142,916 | Stable baseline; benefits from textual features that differentiate error flows |
| Decision Tree | Character trigrams | 1.00 | 1.00 | 1.00 | 142,916 | Perfect scores indicate overfitting on the sampled data |
| Sequence Logistic Regression | Sequence-level numeric features | 0.78 | 0.74 | 0.81 | 63 | Uses only `seq_len` and `duration_sec`; SHAP skipped because no vectorizer |

## ML vs XAI Artefacts
- **ML**: Performance metrics above (also persisted in `metrics_*.json`) zeigen lediglich, wie gut jedes Modell Klassen trennt.
- **XAI**: Dateien unter `demo/result/lo2/explainability/` (SHAP-Plots, `*_top_*`-Listen, NN-Mapping, False-Positive-Logs) erklären, welche Features oder Referenzbeispiele die Entscheidungen treiben.
- Zusammenspiel: ML liefert Scores und Kennzahlen; XAI prüft, ob die Grundlage der Vorhersagen plausibel ist und deckt Fehlverhalten auf.

## Explainability Highlights
### Isolation Forest (Nearest-Neighbour + False-Positive Audit)
- `demo/result/lo2/explainability/if_nn_mapping.csv` maps nearly every flagged anomaly to the same normal sequence ID (`128569`) from service `light-oauth2-oauth2-token-1`.
- `if_false_positives.txt` confirms the top-ranked anomalies are actually routine config-loading log lines, explaining the poor metrics. The NN explanations therefore surface a structural FP problem rather than useful anomalies.

### Decision Tree SHAP (Trigram features)
- Top SHAP contributors (`dt_top_trigrams.txt`) are short character n-grams such as `00:`, `},`, `Iki`, etc.
- The SHAP bar and beeswarm plots show almost all impact concentrated on the single feature `00:`, with negligible influence from the remaining 158k trigrams. This indicates that the tree latched onto very specific token patterns—consistent with its overfitting metrics.

### Logistic Regression SHAP (Token features)
- SHAP analysis (`lr_top_tokens.txt` and plots) highlights meaningful domain tokens: `task-1]`, `/oauth2/client`, `/oauth2/token`, `normalised`, `debug`, etc.
- Positive SHAP values align with OAuth endpoint references and task identifiers, suggesting the LR model distinguishes anomalous vs. normal flows based on request/response context rather than artefacts.

### Sequence Logistic Regression
- SHAP was skipped (`seq_lr_shap_skipped.txt`) because the model uses only numeric features and lacks a vectorizer with token names. Model still offers a modest accuracy lift over IF.

## XAI Interpretation
- **Isolation Forest**: NN-Explainability zeigt, dass nahezu alle „Anomalien" identische normale Sequenzen sind. Aussage: Modell markiert falsche Hotspots, XAI enttarnt damit den Fehlalarm.
- **Decision Tree**: SHAP konzentriert sich auf ein einziges Trigramm (`00:`); die restlichen Features tragen nichts bei. Aussage: Perfekte Metriken beruhen auf Overfitting, Modell ist nicht vertrauenswürdig.
- **Logistic Regression**: SHAP listet OAuth-spezifische Tokens als Haupttreiber. Aussage: Modell nutzt sinnvolle inhaltliche Signale, Empfehlungen lassen sich nachvollziehen.
- **Sequence-LR**: Keine XAI, da nur numerische Features; Modell bleibt interpretierbar über Feature-Signaturen, aber ohne SHAP-Beleg.

## Key Takeaways
- **Isolation Forest** needs refinement (feature filtering or service-specific training) before it can deliver reliable anomaly rankings; NN explainability currently documents its false positives.
- **Logistic Regression** provides the most interpretable and trustworthy baseline—SHAP pinpoints relevant OAuth-related tokens driving predictions.
- **Decision Tree** should not be used as-is despite perfect scores; SHAP reveals the model is memorizing trivial character patterns.
- Persisted explainability artefacts in `demo/result/lo2/explainability/` capture these insights for future tuning cycles.
