#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lo2_preprocess_oauth_client.py
--------------------------------
Preprocess hierarchically organized log datasets (lo2-sample style) for LogLead,
focused on OAuth client flows. Extracts essential features, masks PII/secrets,
and outputs a tidy Parquet/CSV suitable for ML & LogLead ingestion.

Usage example:
  python scripts/lo2_preprocess_oauth_client.py \
    --root data/lo2-sample/logs \
    --out data/light-oauth2-oauth2-client-1.parquet \
    --format parquet \
    --write-csv-preview
"""


from __future__ import annotations
import sys
import os
from pathlib import Path
import re
import fnmatch
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlsplit

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional dependency
    def tqdm(iterable=None, **_kwargs):
        return iterable

# ---------- Configurable Patterns ----------

# Session dir: light-oauth2-data-<digits>
SESSION_DIR_RE = re.compile(r"light-oauth2-data-(?P<sid>\d+)")

# Log file discovery (default pattern tuned to dataset). Supports comma-separated patterns via --glob.
DEFAULT_LOG_GLOB = "light-oauth2-oauth2-client-*.log"

# Event start line (time, thread, reqid, level, logger, rest-of-message)
# Example:
# 16:37:11.870 [XNIO-1 task-1]  eOElTeoVQ9qY7ct3iz8glw DEBUG io.undertow.request.error-response debugf - Setting error code 500 ...
START_RE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+\[(?P<thread>[^\]]+)\]\s+(?P<reqid>\S+)\s+(?P<level>DEBUG|INFO|WARN|ERROR)\s+(?P<logger>\S+)\s+(?P<rest>.*)$"
)

# HTTP method & path
HTTP_RE = re.compile(r"\b(?P<method>GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(?P<path>/\S+)")

# Status code extraction (from payload or message text patterns)
STATUS_FROM_TEXT_RE = re.compile(
    r"""(?:
           (?:error\s*code|status\s*code)\s*(?P<status1>[1-5]\d{2})
        |  setStatusCode\(\s*(?P<status2>[1-5]\d{2})\s*\)
        )""",
    re.I | re.X,
)

# OAuth-specific cues (adjust as needed)
OAUTH_TOKEN_PATH_RE = re.compile(r"/oauth2/(?:token|client|introspect|authorize)\b", re.I)
GRANT_TYPE_RE = re.compile(r"grant_type=([a-zA-Z0-9_]+)")
CLIENT_ID_RE = re.compile(r"(?:client_id|clientId)=(\S+)")
CLIENT_ID_IN_PATH_RE = re.compile(r"/oauth2/client/(?P<cid>[0-9a-fA-F-]{36})\b")
SCOPE_RE = re.compile(r"(?:scope|scopes?)=([a-zA-Z0-9_ \-:.]+)")
NORMALIZED_PATH_RE = re.compile(r"normalised\s*=\s*(?P<path>/\S+)")

# UUID/IP/URL/Email etc for masking + templating
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
NUM_RE = re.compile(r"\b\d+\b")

# Secrets to mask
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+\b")
BASIC_AUTH_RE = re.compile(r"Basic\s+[A-Za-z0-9+/=]+")
BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._-]+")
SECRET_KEYS = re.compile(r"(?i)\b(client_secret|api_key|token|password)\b\s*[:=]\s*\S+")

# Exception class at start of message or stack lines
EXCEPTION_RE = re.compile(r"^(?P<ex>[a-zA-Z0-9_.]+(?:Exception|Error))\b")

# Stack trace fragments occasionally precede the structured log line. Capture them
# so that we can associate the diagnostic block with the following event.
STACKTRACE_LINE_RE = re.compile(r"^\s+(?:at|\.\.\.|Caused by:)\b")
EXCEPTION_HEADER_RE = re.compile(r"^(?:Caused by:\s+)?[A-Za-z0-9_.$]+(?:Exception|Error|Throwable)(?::.*)?$")

# ---------- Utilities ----------

def sha1_hex(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()

def mask_value(text: str) -> str:
    # Standard masking for PII/secrets
    text = JWT_RE.sub("JWT_*", text)
    text = BASIC_AUTH_RE.sub("Basic_*", text)
    text = BEARER_RE.sub("Bearer_*", text)
    text = SECRET_KEYS.sub(lambda m: f"{m.group(1)}=***", text)
    text = EMAIL_RE.sub("u***@d***.tld", text)
    # Anonymize IPs to /24 style (just zero out last octet)
    text = IP_RE.sub(lambda m: ".".join(m.group(0).split(".")[:3] + ["0"]), text)
    return text

def template_from_message(text: str) -> str:
    t = mask_value(text)
    t = UUID_RE.sub("*", t)
    t = URL_RE.sub("*", t)
    # Replace numbers that are likely volatile (keep status codes later separately)
    t = NUM_RE.sub("*", t)
    return t

def normalize_path(path: str) -> str:
    # Replace path segments that are UUIDs or numbers with *
    path = urlsplit(path).path  # strip query/fragment
    segs = path.split("/")
    out = []
    for s in segs:
        if UUID_RE.fullmatch(s) or NUM_RE.fullmatch(s):
            out.append("*")
        else:
            out.append(s)
    return "/".join(out) or "/"

def extract_status(message: str, payload: dict | None = None) -> int | None:
    """Extract HTTP status code from payload or message text patterns."""
    if payload and isinstance(payload.get("statusCode"), (int, float)):
        sc = int(payload["statusCode"])
        if 100 <= sc <= 599:
            return sc
    m = STATUS_FROM_TEXT_RE.search(message or "")
    if m:
        for g in ("status1", "status2"):
            if m.group(g):
                return int(m.group(g))
    return None


def extract_json_from_message(s: str) -> dict:
    """Safely extract first complete JSON object from message string."""
    if not isinstance(s, str):
        return {}
    i = s.find("{")
    if i < 0:
        return {}
    depth = 0
    for j, ch in enumerate(s[i:], start=i):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[i:j+1])
                except Exception:
                    return {}
    return {}

# ---------- Parsing ----------

def discover_log_files(root: Path, patterns: list[str] | None = None):
    # If patterns provided, iterate all files and match via fnmatch for each pattern.
    # This keeps compatibility across Python versions for multiple globs.
    pats = patterns or [DEFAULT_LOG_GLOB]
    # Efficiently scan all files under root; filter by patterns
    for p in root.rglob("*.log"):
        if not p.is_file():
            continue
        name = p.name
        for pat in pats:
            if fnmatch.fnmatch(name, pat):
                yield p
                break

def extract_metadata_from_path(p: Path):
    # Expect .../<session_dir>/<label...>/<file>
    parts = list(p.parts)
    session_id = None
    session_idx = None
    for i, comp in enumerate(parts):
        match = SESSION_DIR_RE.fullmatch(comp)
        if match:
            session_id = match.group("sid")
            session_idx = i
            break

    label_primary = None
    label_path = None
    if session_idx is not None:
        sub = parts[session_idx + 1 : -1]
        if sub:
            label_primary = sub[0]
            label_path = "/".join(sub)
    segment_id = p.stem
    return session_id, label_primary, label_path, segment_id

def parse_events(file_path: Path):
    events = []
    current = None
    pending_prefix: list[str] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = START_RE.match(line)
            if m:
                if current is not None:
                    events.append(current)
                message = m.group("rest").strip()
                if pending_prefix:
                    suffix = "\n".join(pending_prefix)
                    message = f"{message}\n{suffix}" if message else suffix
                    pending_prefix.clear()
                current = {
                    "timestamp": m.group("time"),
                    "thread": m.group("thread"),
                    "request_id": m.group("reqid"),
                    "log_level": m.group("level"),
                    "logger": m.group("logger"),
                    "message": message,
                }
                continue

            if current is not None:
                current["message"] += ("\n" if current["message"] else "") + line
                continue

            # No current event yet; preserve stack trace fragments to attach to the next one
            if STACKTRACE_LINE_RE.match(line) or EXCEPTION_HEADER_RE.match(line):
                pending_prefix.append(line)
            elif line.strip() == "" and pending_prefix:
                pending_prefix.append(line)
            else:
                pending_prefix.clear()

        if current is not None:
            events.append(current)
    return events

def derive_features(evt: dict, max_message_len: int, make_blob_ref: bool, blob_root: Path, event_idx: int, session_id: str, event_sequence_id: str):
    msg = evt.get("message", "") or ""
    # compute sha1 on original (before truncation/masking) for auditability
    orig_bytes = msg.encode("utf-8", errors="replace")
    message_sha1 = sha1_hex(orig_bytes)
    message_len = len(orig_bytes)

    payload = extract_json_from_message(evt.get("message", ""))

    # optional blob store
    blob_ref = None
    if make_blob_ref and message_len > 4096 and blob_root:
        blob_dir = blob_root / (session_id or "no-session") / (event_sequence_id or "no-seq")
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob_file = blob_dir / f"{event_idx}.log"
        blob_file.write_bytes(orig_bytes)
        blob_ref = str(blob_file)

    # truncate for main table
    truncated = False
    if max_message_len > 0 and message_len > max_message_len:
        msg = orig_bytes[:max_message_len].decode("utf-8", errors="replace")
        truncated = True

    # mask PII/secrets in stored message
    masked_message = mask_value(msg)

    # base fields
    out = dict(evt)
    out["message"] = masked_message
    out["message_len"] = message_len
    out["message_truncated"] = truncated
    out["message_sha1"] = message_sha1
    if blob_ref:
        out["blob_ref"] = blob_ref
    out["payload"] = payload if payload else None

    # HTTP/OAuth
    http_m = HTTP_RE.search(evt.get("message", ""))
    if http_m:
        out["http_method"] = http_m.group("method")
        out["http_path"] = normalize_path(http_m.group("path"))
    else:
        out["http_method"] = None
        out["http_path"] = None

    if not out["http_path"]:
        norm_match = NORMALIZED_PATH_RE.search(evt.get("message", ""))
        if norm_match:
            out["http_path"] = normalize_path(norm_match.group("path"))

    # Extract status from payload or message text patterns
    msg = evt.get("message", "")
    out["http_status"] = extract_status(msg, payload=payload)

    out["oauth_path_hit"] = bool(OAUTH_TOKEN_PATH_RE.search(msg))
    gt = GRANT_TYPE_RE.search(msg)
    out["oauth_grant_type"] = gt.group(1) if gt else None
    cid = CLIENT_ID_RE.search(msg) or CLIENT_ID_IN_PATH_RE.search(msg)
    out["oauth_client_id_present"] = bool(cid)
    sc = SCOPE_RE.search(evt.get("message", ""))
    if sc:
        # store only count (no raw scopes for privacy)
        out["oauth_scope_count"] = len([s for s in re.split(r"[\s,]+", sc.group(1).strip()) if s])
    else:
        out["oauth_scope_count"] = None

    # Exceptions
    # First line of message often contains class; also check subsequent lines
    ex = None
    first_line = evt.get("message","").split("\n",1)[0]
    m1 = EXCEPTION_RE.match(first_line)
    if m1:
        ex = m1.group("ex")
    else:
        for line in evt.get("message","").split("\n"):
            m2 = EXCEPTION_RE.match(line.strip())
            if m2:
                ex = m2.group("ex"); break
    out["exception_class"] = ex
    out["has_multiline"] = "\n" in evt.get("message","")

    # Template
    out["event_template"] = template_from_message(evt.get("message",""))

    return out

# ---------- Main ETL ----------

def run(
    root: Path,
    out_path: Path,
    fmt: str = "parquet",
    max_message_len: int = 16384,
    write_csv_preview: bool = False,
    blob_root: Path | None = None,
    glob_patterns: list[str] | None = None,
):
    rows = []
    file_count = 0
    log_iter = discover_log_files(root, patterns=glob_patterns)
    for log_file in tqdm(log_iter, desc="Parsing log files", unit="file"):
        file_count += 1
        session_id, label_primary, label_path, segment_id = extract_metadata_from_path(log_file)
        event_sequence_id = f"{session_id}-{label_primary}" if session_id and label_primary else session_id
        events = parse_events(log_file)
        if not events:
            continue

        event_iter = enumerate(events, start=1)
        for idx, evt in tqdm(
            event_iter,
            desc=log_file.name,
            unit="event",
            leave=False,
            total=len(events),
        ):
            feat = derive_features(
                evt,
                max_message_len=max_message_len,
                make_blob_ref=bool(blob_root),
                blob_root=blob_root or Path(""),
                event_idx=idx,
                session_id=session_id or "no-session",
                event_sequence_id=event_sequence_id or "no-seq",
            )
            # schema mapping
            row = {
                "session_id": session_id,
                "event_sequence_id": event_sequence_id,
                "label_primary": label_primary,
                "label_path": label_path,
                "segment_id": segment_id,
                "line_number": idx,
                "timestamp": feat.get("timestamp"),
                "log_level": feat.get("log_level"),
                "logger": feat.get("logger"),
                "thread": feat.get("thread"),
                "request_id": feat.get("request_id"),
                "message": feat.get("message"),
                "message_len": feat.get("message_len"),
                "message_truncated": feat.get("message_truncated"),
                "message_sha1": feat.get("message_sha1"),
                "blob_ref": feat.get("blob_ref"),
                "http_method": feat.get("http_method"),
                "http_path": feat.get("http_path"),
                "http_status": feat.get("http_status"),
                "oauth_path_hit": feat.get("oauth_path_hit"),
                "oauth_grant_type": feat.get("oauth_grant_type"),
                "oauth_client_id_present": feat.get("oauth_client_id_present"),
                "oauth_scope_count": feat.get("oauth_scope_count"),
                "exception_class": feat.get("exception_class"),
                "has_multiline": feat.get("has_multiline"),
                "event_template": feat.get("event_template"),
                "payload": feat.get("payload"),
                "source_file": str(log_file),
                "file_mtime": pd.to_datetime(log_file.stat().st_mtime, unit="s", utc=True),
            }
            rows.append(row)

    if not rows:
        print("No events parsed. Check patterns and root path.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)

    # === Canonical timestamp (supports both structured and flat logs) ===
    # Convert session_id (unix epoch seconds) to date, then combine with time_of_day
    # If session_id is None/missing, fall back to file mtime
    df["session_date"] = pd.to_datetime(df["session_id"], unit="s", utc=True)
    df["time_delta"] = pd.to_timedelta(df["timestamp"], errors="coerce")
    df["event_datetime_utc"] = (
        df["session_date"].dt.floor("D")
        .fillna(df["file_mtime"].dt.floor("D"))
        + df["time_delta"]
    )
    df["event_datetime"] = df["event_datetime_utc"].dt.tz_convert("Europe/Berlin")

    # Stable sort: first by time, then by line_number
    df.sort_values(["event_datetime_utc", "line_number"], inplace=True, kind="mergesort")

    # === JSON payload extraction (tolerant, depth-aware) ===
    def extract_json_from_message(s: str) -> dict:
        """Safely extract first complete JSON object from message string."""
        if not isinstance(s, str):
            return {}
        i = s.find("{")
        if i < 0:
            return {}
        depth = 0
        for j, ch in enumerate(s[i:], start=i):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:j+1])
                    except Exception:
                        return {}
        return {}

    details = pd.json_normalize(df["payload"].map(lambda v: v if isinstance(v, dict) else {}))
    # Keep only payload fields relevant to OAuth flows
    payload_fields = ["statusCode", "code", "message", "description", "severity"]
    keep_cols = [c for c in payload_fields if c in details.columns]
    if keep_cols:
        df = pd.concat([df, details[keep_cols].add_prefix("payload_")], axis=1)

    if "payload_statusCode" in df.columns:
        df["http_status"] = df["http_status"].fillna(df["payload_statusCode"])

    df["status_family"] = df["http_status"].map(lambda v: f"{int(v) // 100}xx" if pd.notna(v) else None)

    df["logger_root"] = df["logger"].str.split(".", n=1).str[0]

    df["outcome"] = np.select(
        [df["http_status"].between(200, 299, inclusive="both"), df["http_status"].between(400, 499, inclusive="both"), df["http_status"].between(500, 599, inclusive="both")],
        ["success", "client_error", "server_error"],
        default=None,
    )

    # === Cleanup & final column ordering ===
    df.drop(columns=[c for c in ["session_date", "time_delta", "file_mtime"] if c in df.columns], inplace=True)

    # Dtypes & ordering
    columns = [
        "event_datetime", "event_datetime_utc",
        "session_id", "label_primary", "label_path",
        "event_sequence_id", "segment_id", "line_number", "timestamp",
        "http_method", "http_path", "http_status", "status_family", "outcome",
        "oauth_path_hit", "oauth_grant_type", "oauth_client_id_present", "oauth_scope_count",
        "log_level", "logger", "logger_root", "exception_class", "has_multiline",
        "payload_code", "payload_message", "payload_description", "payload_severity",
        "message", "message_len", "message_truncated", "message_sha1", "blob_ref",
        "request_id", "thread",
        "event_template", "source_file"
    ]
    # Add any extracted payload columns
    payload_cols = [c for c in df.columns if c.startswith("payload_")]
    for c in payload_cols:
        if c not in columns:
            columns.append(c)

    df = df[[c for c in columns if c in df.columns]]

    if "payload" in df.columns:
        df.drop(columns=["payload"], inplace=True)

    # Preserve nullable integer semantics for downstream consumers
    for nullable_int in ["http_status", "oauth_scope_count"]:
        if nullable_int in df:
            df[nullable_int] = df[nullable_int].astype("Int64")

    if "payload_statusCode" in df.columns:
        df["payload_statusCode"] = df["payload_statusCode"].astype("Int64")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt.lower() == "parquet":
        try:
            df.to_parquet(out_path, index=False)
        except Exception as e:
            print(f"Parquet write failed ({e}); falling back to CSV.", file=sys.stderr)
            df.to_csv(out_path.with_suffix(".csv"), index=False)
    else:
        df.to_csv(out_path, index=False)

    if write_csv_preview:
        preview_cols = [
            "event_datetime", "event_datetime_utc",
            "session_id", "label_primary", "label_path",
            "event_sequence_id", "segment_id", "line_number",
            "http_method", "http_path", "http_status", "status_family",
            "log_level", "logger", "exception_class", "has_multiline",
            "payload_code", "payload_message", "payload_description", "payload_severity",
            "request_id", "thread",
            "message",
            "source_file",
        ]
        df_preview = df[[c for c in preview_cols if c in df.columns]].copy()
        df_preview.head(200).to_csv(out_path.with_name(out_path.stem + "_preview.csv"), index=False)

    print(f"Wrote {len(df):,} events from {file_count} files to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Preprocess lo2 logs (OAuth client focus)")
    ap.add_argument("--root", required=True, help="Root directory of lo2-sample/logs")
    ap.add_argument("--out", required=True, help="Output file path (parquet or csv)")
    ap.add_argument("--format", choices=["parquet","csv"], default="parquet")
    ap.add_argument("--max-message-len", type=int, default=16384, help="Truncate message to this many bytes in main table")
    ap.add_argument("--blob-root", default=None, help="Optional directory to store full message blobs (for long events)")
    ap.add_argument("--write-csv-preview", action="store_true", help="Also write a small CSV preview (first 200 rows)")
    ap.add_argument(
        "--glob",
        default=DEFAULT_LOG_GLOB,
        help="Comma-separated filename patterns to include (matched against basenames). Default: %(default)s",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    blob_root = Path(args.blob_root) if args.blob_root else None

    patterns = [s.strip() for s in (args.glob or DEFAULT_LOG_GLOB).split(",") if s.strip()]
    run(
        root=root,
        out_path=out,
        fmt=args.format,
        max_message_len=args.max_message_len,
        write_csv_preview=args.write_csv_preview,
        blob_root=blob_root,
        glob_patterns=patterns,
    )

if __name__ == "__main__":
    main()

    
