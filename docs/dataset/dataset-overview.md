---
title: LO2 Datensatzüberblick
summary: Herkunft, Struktur und Nutzungsschwerpunkte des Light-OAuth2-Datensatzes.
last_updated: 2025-11-03
---

# LO2 Datensatzüberblick

Der LO2-Datensatz kombiniert Logs, Metriken und Traces des Light-OAuth2-Microservice-Stacks. Für die aktuelle Pipeline nutzen wir ausschließlich ausgewählte Service-Logs, um eine kompakte, label-stabile Grundlage zu behalten.

## Quellen

- **Datensatz:** [Zenodo DOI 10.5281/zenodo.14938118](https://zenodo.org/records/14938118)
- **Paper:** *LO2: Multi-modal Dataset for Log Anomaly Detection in Microservice Systems* (ICPE 2025)
- **Light-OAuth2 Projekt:** archivierter Code auf GitHub (`networknt/light-oauth2`)

## Fokus der Pipeline

- Services: `token`, `code`, `refresh-token` (decken Authorization-Code-, PKCE-, Client-Credentials- und Refresh-Flows ab).
- Label-Logik: Pro Testzeitraum 1 × `correct` + 53 Fehlerfälle (`error1 … error53`), dadurch klare Normal/Anomal-Labels.
- Motivation: Token-Service liefert laut Paper den stärksten Signals, Key/Client/User-Logs bringen wenig Zusatznutzen.

## Struktur der Rohdaten

```
<root>/
  run_0001/
    correct/
      oauth2-code.log
      oauth2-token.log
      oauth2-refresh-token.log
    error_invalid_grant/
      ...
    ...
  run_0002/
    ...
  metrics/  (optional JSON pro Run/Test)
```

- **Runs:** bis zu 1 740 (`run_<id>`), jede Suite dauert ~600 s.
- **Testcases:** 54 pro Run (1 korrekt + 53 Fehler).
- **Dateitypen:** `.log` (Service-Container), Locust-Logs (Ground Truth), optionale `.json`/`.csv` für Metriken/Traces.

## Sequenz-Features (Enhanced Sequences)

Die Pipeline materialisiert pro Sequenz ein angereichertes Feature-Set (`lo2_sequences_enhanced.parquet`). Die wichtigsten Spalten im Überblick:

| Spalte | Typ | Beschreibung | Nutzen für Anomalieerkennung |
| --- | --- | --- | --- |
| `e_words` | List[Utf8] | Sequenzweite Worttokens aus `EventLogEnhancer.words()` | Primäre Eingabe für Bag-of-Words-Vektorisierer (`--if-item e_words`, `event_lr_words`, SHAP-Modelle) |
| `e_words_len` | Int | Gesamtanzahl der Wörter einer Sequenz | Volumenmerkmal; hohe Abweichungen deuten auf ungewöhnliche Ablaufkomplexität hin |
| `e_trigrams` | List[Utf8] | Zeichenbasierte 3-Gramme (`EventLogEnhancer.trigrams()`, danach Aggregation) | Robust gegen neue Schreibweisen und Case-Drift; alternative Token-Repräsentation |
| `e_trigrams_len` | Int | Anzahl der Trigramme pro Sequenz | Ergänzendes Längenfeature für Modelle mit `--if-numeric` oder `sequence_lr_numeric` |
| `e_event_drain_id` | List[Utf8] | Reihenfolge der Drain-Template-IDs aller Events | Sichtbarmachen ungewöhnlicher Templates oder Kontrollflussänderungen |
| `seq_len` | Int | Anzahl der Events innerhalb einer Sequenz | Kernmerkmal für `sequence_lr_numeric`; optional im Isolation Forest (`--if-numeric`) |
| `e_event_id_len` | Int | Alias zu `seq_len` für Legacy-Kompatibilität | Wird intern zur Harmonisierung älterer Pipelines verwendet |
| `start_time` / `end_time` | Datetime | Frühester beziehungsweise spätester Zeitstempel | Basis für zeitliche Splits (`--if-holdout-fraction`) und Trendanalysen |
| `duration` | Duration | Zeitspanne zwischen `start_time` und `end_time` | Manuelle Auswertung; dient als Rohwert für `duration_sec` |
| `duration_sec` | Float | Dauer in Sekunden | Direktes numerisches Feature für Performance- oder Delay-Anomalien |

> Hinweis: `SequenceEnhancer.tokens(token="...")` erzeugt sowohl die Tokenlisten als auch die dazugehörigen `_len`-Spalten. Mehrfaches Aufrufen (z. B. für Wörter und Trigramme) liefert parallele Repräsentationen derselben Sequenz.

**Praxis:** `demo/lo2_e2e/LO2_samples.py` nutzt standardmäßig `e_words` als Item-Feature und kombiniert `seq_len`, `duration_sec`, `e_words_len`, `e_trigrams_len` als numerische Zusatzsignale. Dadurch fließen sowohl semantische (Token) als auch strukturelle Merkmale (Länge, Dauer) in die Anomalieerkennung ein.

## Flow-Abdeckung (Hypothesen)

| Flow | Primäre Endpoints | Genutzte Logs | Bemerkung |
| --- | --- | --- | --- |
| Authorization Code ± PKCE | `GET /oauth2/code` → `POST /oauth2/token` | `code`, `token` | PKCE-Validierung im Token-Service |
| Client Credentials | `POST /oauth2/token` | `token` | Keine weiteren Services notwendig |
| Refresh Token Renewal | `POST /oauth2/token` (`grant_type=refresh_token`) | `token` | Refresh erfolgt am Token-Endpoint |
| Refresh-Token Verwaltung | `/oauth2/refresh-token/*` | `refresh-token` | Admin-Operationen (list, revoke) |

*Die Zuordnung basiert auf OAuth 2.0-Logik und Beobachtungen im Datensatz, nicht auf offizieller Light-OAuth2-Dokumentation.*

## Qualitätsnotizen

- Initialisierungsphasen erzeugen Label-Leakage. Der Loader schneidet standardmäßig die ersten 100 Zeilen jedes Logs ab (`trim_init_lines=True`).
- Traces enthalten oft nur einen Span; aktuell ungenutzt.
- Metriken (`node_load`, `memory`) sind vorhanden, werden aber noch nicht in der Pipeline verarbeitet.

## Nutzungsempfehlungen

- Loader mit Service-Filter (`--service-types code token refresh-token`) aufrufen.
- Für Experimente Runs/Fehler gezielt reduzieren (`--runs`, `--errors-per-run`), um Artefakte klein zu halten.
- Zusätzliche Services nur einbinden, wenn klarer Nutzen belegt ist (z. B. dedizierter Monitoring-Use-Case).

## Offene Fragen

- TODO: Validieren, ob Refresh-Renewal ausschließlich im Token-Log sichtbar ist oder ob weitere Services beteiligt sind.
- TODO: Prüfen, welche Fehlerklassen wiederholte False Positives verursachen und ob alternative Serviceauswahl hilft.
