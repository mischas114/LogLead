# lo2_preprocess_oauth_client

Kurze Anleitung für das Script `lo2_preprocess_oauth_client.py` zur Vorverarbeitung der Light-OAuth2-Logs (LO2) in ein sauberes, ML-taugliches Format für LogLead.

## Was macht das Script?
- Lädt hierarchisch organisierte LO2-Logs (z. B. `light-oauth2-data-<id>/<label>/…/*.log`).
- Parst einzelne Log-Events anhand eines Zeilen-Startmusters (Zeit, Thread, Request-ID, Level, Logger, Message).
- Extrahiert nützliche Features, u. a.:
  - HTTP-Methode, normalisierter Pfad, HTTP-Status
  - OAuth-spezifische Hinweise (Pfad-Treffer, `grant_type`, `client_id`-Vorkommen, Anzahl Scopes)
  - Exception-Klasse, Outcome-Label, Multi-Line-Indikator, Event-Template
  - Längen/Hash der Original-Message und optionaler Blob-Ref für lange Messages
- Maskiert PII/Secrets (JWT, Basic/Bearer Tokens, client_secret, E-Mail, IPs etc.).
- Schreibt das Ergebnis als Parquet (oder CSV) sowie optional eine CSV-Preview.

## Warum LO2 → LogLead Format?
LO2 erzeugt viele, thematisch getrennte Log-Dateien (z. B. `light-oauth2-oauth2-client-*.log`). Für Analyse- und ML-Pipelines (LogLead) ist jedoch ein tabellarisches, anonymisiertes Format ideal:
- Einheitliches Schema: jede Zeile = ein Event mit festen Spalten.
- Privatsphäre: sensible Felder werden maskiert/anonymisiert.
- Robustheit: volatile Teile (UUIDs, Zahlen, URLs) werden im `event_template` verallgemeinert.
- Leichte Integration: Parquet/CSV kann direkt von Pandas, Spark oder LogLead-Komponenten konsumiert werden.

## Eingaben
- `--root`: Wurzelordner der LO2-Logs, z. B. `data/lo2-sample/logs`.
- `--glob`: Dateinamen-Patterns (kommagetrennt), die gefiltert werden (Match auf Basisname). Standard: `light-oauth2-oauth2-client-*.log`.
  - Beispiel für mehrere Patterns: `--glob 'light-oauth2-oauth2-client-*.log,light-oauth2-oauth2-token-*.log'`

## Ausgaben
- Hauptdatei als Parquet (Standard) oder CSV (`--format csv`).
- Optional: `*_preview.csv` mit den ersten 200 Zeilen (`--write-csv-preview`).
- Optional: Blob-Dateien (vollständige Messages > 4096 Bytes) unter `--blob-root`.

## Wichtigste Spalten
- **Kanonische Zeitstempel** (neu):
  - `event_datetime`: Zeitstempel in Europe/Berlin Zeitzone
  - `event_datetime_utc`: Zeitstempel in UTC (Referenz für stabile Sortierung)
- Kontext: `session_id`, `event_sequence_id`, `label_primary`, `label_path`, `segment_id`, `line_number`
- Log-Metadaten: `timestamp`, `log_level`, `logger`, `thread`, `request_id`
- HTTP: `http_method`, `http_path` (normalisiert), `http_status`, `status_family`, `outcome`
- OAuth: `oauth_path_hit`, `oauth_grant_type`, `oauth_client_id_present`, `oauth_scope_count`
- Qualität/Template: `exception_class`, `has_multiline`, `event_template`, `logger_root`
- Message: `message` (maskiert), `message_len`, `message_truncated`, `message_sha1`, `blob_ref`
- **Extrahierte JSON-Payloads** (neu): `payload_statusCode`, `payload_code`, `payload_message`, `payload_description`, `payload_severity` (wenn vorhanden)

## Beispiel (Sample-Daten)
### Schnellstart (LO2 Sample → LogLead)

Führe im Projekt-Root (parallel zu `data/`) aus:

```zsh
python scripts/lo2_preprocess_oauth_client.py \
  --root data/lo2-sample/logs \
  --out data/light-oauth2-oauth2-client-1.parquet \
  --format parquet \
  --write-csv-preview \
  --glob 'light-oauth2-oauth2-client-*.log'
```

Damit werden die LO2-Sample-Daten für LogLead vorbereitet und zusätzlich eine `_preview.csv` mit den wichtigsten Spalten geschrieben.

## Optionen im Überblick
- `--root <PATH>`: Wurzelordner der Logs (erforderlich)
- `--out <FILE>`: Ausgabe-Datei (erforderlich; Endung bestimmt nicht das Format)
- `--format parquet|csv`: Ausgabeformat (Standard: `parquet`)
- `--max-message-len <N>`: Trunkiert gespeicherte Message im Haupt-Table (Standard: 16384 Bytes)
- `--blob-root <DIR>`: Optionaler Ordner für vollständige (lange) Messages als separate Dateien
- `--write-csv-preview`: Zusätzlich eine kleine Preview-CSV (erste 200 Zeilen) schreiben
- `--glob <PATTERNS>`: Kommagetrennte Dateinamen-Patterns (Match auf Basename)

## Verbesserungen für LogLead-Kompatibilität (v1.1)

### Kanonische Zeitstempel
Das Script erstellt nun vollständige, timezone-aware Zeitstempel:
- `session_id` (Unix-Epoch) wird als Sessionsdatum interpretiert
- Kombiniert mit der Tageszeit aus `timestamp` zu `event_datetime_utc` (UTC)
- Konvertiert zu lokaler Zeit (Europe/Berlin) als `event_datetime`
- Ermöglicht zuverlässiges Sorting und zeitbasierte Analysen in LogLead

### Stabile Sortierung
Events werden nach:
1. `event_datetime_utc` (chronologisch)
2. `line_number` (konsistente Ordnung bei gleicher Millisekunde)

sortiert. Dies garantiert reproduzierbare Reihenfolge über mehrere Läufe.

### Automatische JSON-Payload-Extraktion
Eingebettete JSON in Messages wird automatisch extrahiert:
- Sucht nach erstem `{...}`-Objekt in der Message
- Extrahiert OAuth-relevante Felder: `statusCode`, `code`, `message`, `description`, `severity`
- Speichert als separate Spalten mit `payload_`-Präfix (z. B. `payload_statusCode`)
- Verhindert CSV-Escaping-Fehler durch embedded Quotes
- Tolerant: ignoriert ungültige JSON (gibt leere Spalten zurück)

### Verbessertes Output-Format
- Saubere, korrekt formatierte CSV-Ausgabe ohne Quote-Fehler
- Alle Spalten korrekt ausgerichtet
- Perfekt für LogLead-Ingestion und Machine-Learning-Pipelines

## Hinweise & Tipps
- Wenn „No events parsed" erscheint, prüfe `--root` und ob `--glob` zum Datensatz passt (z. B. `light-oauth2-oauth2-client-*.log`).
- Für weitere Komponenten zusätzliches Pattern ergänzen (z. B. `...-token-*.log`).
- Parquet ist kompakt und schnell; CSV ist als Fallback/Debug nützlich.
- Die Parser-Regex `START_RE` ist auf typische Zeilen wie `HH:MM:SS.mmm [Thread] reqid LEVEL logger rest` ausgelegt. Bei Abweichungen ggf. anpassen.
- Nutze `--write-csv-preview` zur Validierung der Output-Struktur vor der Vollverarbeitung.
