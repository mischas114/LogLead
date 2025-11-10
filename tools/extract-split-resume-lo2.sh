#!/usr/bin/env bash
# macOS-kompatibler, resumierbarer Extractor für light-oauth2-data-*.tar
# Anpassungen:
#  - Default-Datenpfade: ~/Desktop/lo2-data (anpassbar via BASE_DIR/SRC/OUT)
#  - Parallelität default 4 (über JOBS steuerbar)
#  - progress.txt als menschenlesbare Fortschrittsanzeige
#  - Pro TAR nur 'correct/' + genau ein zufälliger error
#  - Keine Doppelordner mehr (nutzt --strip-components=1 wenn sinnvoll)

set -euo pipefail

# Basispfade
DEFAULT_BASE="$HOME/Desktop"
BASE_DIR="${BASE_DIR:-$DEFAULT_BASE}"
SRC="${SRC:-$BASE_DIR/lo2-data}"
OUT="${OUT:-$BASE_DIR/lo2-extracted}"

# Auswahl/Modus (wie zuvor)
MODE="${MODE:-random}"        # random | oldest | newest | every2
DRY_RUN="${DRY_RUN:-1}"
JOBS="${JOBS:-4}"             # <— jetzt 4 Standard-Jobs
COUNT="${COUNT:-}"            # optional
SPLIT="${SPLIT:-}"            # optional
CHUNK="${CHUNK:-}"            # optional
SEED="${SEED:-}"              # optional

echo "Quelle: $SRC"
echo "Ziel:   $OUT"
echo "Modus:  $MODE"
echo "DryRun: $DRY_RUN  |  Jobs: $JOBS"
[[ -n "$COUNT" ]] && echo "COUNT: $COUNT"
[[ -n "$SPLIT" || -n "$CHUNK" ]] && echo "SPLIT/CHUNK: $SPLIT / $CHUNK"
[[ -n "$SEED" ]] && echo "SEED: $SEED"

if ! [ -d "$SRC" ]; then
  echo "SRC existiert nicht: $SRC" >&2; exit 1
fi

mkdir -p "$OUT"

# Liste der TARs (null-separiert, macOS-kompatibel)
LISTFILE="$(mktemp)"
find "$SRC" -maxdepth 1 -type f -name 'light-oauth2-data-*.tar' -print0 > "$LISTFILE"

# Zählen (für Anzeige)
N=$(find "$SRC" -maxdepth 1 -type f -name 'light-oauth2-data-*.tar' | wc -l | tr -d ' ')
if [[ "$N" == "0" ]]; then echo "Keine light-oauth2-data-*.tar gefunden."; rm -f "$LISTFILE"; exit 1; fi

# Lauf-Signatur für Status (wie zuvor)
sig_input="$SRC|$MODE|${COUNT:-_}|${SPLIT:-_}|${CHUNK:-_}|${SEED:-_}"
RUN_SIG=$(printf "%s" "$sig_input" | shasum | awk '{print $1}')
STATE_DIR="$OUT/.lo2-extract-state/$RUN_SIG"
DONE_DIR="$STATE_DIR/done"
MANIFEST="$STATE_DIR/manifest.txt"
LOCKFILE="$STATE_DIR/.lock"
PROGRESS="$STATE_DIR/progress.txt"

mkdir -p "$STATE_DIR" "$DONE_DIR"

# Einfaches Lock
if [ -e "$LOCKFILE" ]; then
  echo "Hinweis: Ein anderer Lauf mit derselben Signatur scheint aktiv ($LOCKFILE existiert)."
  echo "Falls nicht korrekt, Datei löschen und neu starten."
fi
date +%s > "$LOCKFILE" || true

# Manifest bauen/verwenden (wie zuvor)
if [ -s "$MANIFEST" ]; then
  echo "Manifest existiert bereits: $MANIFEST"
else
  echo "Erstelle Manifest (Auswahl & Reihenfolge) …"
  python3 - "$MODE" "$LISTFILE" "$COUNT" "$SPLIT" "$CHUNK" "$SEED" > "$MANIFEST" <<'PY'
import os, sys, random, re, math
mode = sys.argv[1]; listfile = sys.argv[2]
COUNT=os.environ.get("COUNT","").strip()
SPLIT=os.environ.get("SPLIT","").strip()
CHUNK=os.environ.get("CHUNK","").strip()
SEED=os.environ.get("SEED","").strip()
with open(listfile,'rb') as f: paths=[p for p in f.read().split(b'\x00') if p]
paths=[p.decode('utf-8') for p in paths]
def ts(p): import re,os; m=re.search(r'-(\d+)\.tar$',os.path.basename(p)); return int(m.group(1)) if m else -1
if mode=='random':
    random.seed(int(SEED)) if SEED else random.seed()
    random.shuffle(paths)
elif mode=='oldest': paths=sorted(paths,key=ts)
elif mode=='newest': paths=sorted(paths,key=ts,reverse=True)
elif mode=='every2': paths=[p for i,p in enumerate(sorted(paths)) if i%2==0]
else: print(f"Unbekannter MODE: {mode}", file=sys.stderr); sys.exit(2)
n=len(paths)
if COUNT:
    k=max(0,min(int(COUNT),n)); sel=paths[:k]
elif SPLIT and CHUNK:
    import math
    S=max(1,int(SPLIT)); K=max(1,min(int(CHUNK),S))
    per=math.ceil(n/S); start=(K-1)*per; end=min(n,K*per); sel=paths[start:end]
else:
    sel=paths[: n//2 ]
print("\n".join(sel))
PY
fi

# Fortschritt am Start anzeigen (wo weitermachen?)
OK_COUNT=$(find "$DONE_DIR" -type f -name '*.ok' | wc -l | tr -d ' ')
SEL_COUNT=$(wc -l < "$MANIFEST" | tr -d ' ')
echo "Gefunden: $N Archive  ->  Auswahl gemäß Manifest: $SEL_COUNT"
echo "Bereits erledigt (.ok): $OK_COUNT"
if [ -s "$PROGRESS" ]; then
  echo "Letzte Einträge progress.txt:"
  tail -n 5 "$PROGRESS" | sed 's/^/ - /'
fi
echo "Manifest: $MANIFEST"
echo "Auswahl (erste 20 Zeilen):"
nl -ba "$MANIFEST" | head -n 20 | sed 's/^/ - /'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY-RUN: nichts entpackt. Setze DRY_RUN=0 zum Ausführen."
  exit 0
fi

# TODO-Liste aus Manifest abzüglich bereits erledigter Dateien
TODO_FILE="$(mktemp)"
> "$TODO_FILE"
while IFS= read -r f; do
  key=$(printf "%s" "$f" | shasum | awk '{print $1}')
  if [ -f "$DONE_DIR/$key.ok" ]; then :; else printf "%s\0" "$f" >> "$TODO_FILE"; fi
done < "$MANIFEST"

if [ ! -s "$TODO_FILE" ]; then
  echo "Nichts zu tun – alle Dateien aus dem Manifest sind bereits verarbeitet."
  exit 0
fi

# --- Extraktion einer Datei: immer 'correct/' + 1 zufälliger weiterer Top-Level-Ordner ---
extract_one () {
  local f="$1"
  local out="$OUT"
  local done_dir="$DONE_DIR"
  local state_dir="$STATE_DIR"

  local base dir key start_marker ok_marker err_marker
  base="$(basename "$f")"
  dir="$out/${base%.tar}"
  mkdir -p "$dir"   # EIN Zielordner pro Archiv
  key=$(printf "%s" "$f" | shasum | awk '{print $1}')
  start_marker="$done_dir/$key.start"
  ok_marker="$done_dir/$key.ok"
  err_marker="$done_dir/$key.err"
  : > "$start_marker"

  # Mitgliederliste lesen
  local list rel_list tmp_list strip topdirs topdir_name chosen_other pattern
  list="$(mktemp)"; rel_list="$(mktemp)"; tmp_list="$(mktemp)"
  tar -tf "$f" > "$list"

  # Prüfen, ob ein gemeinsamer Top-Level existiert (zur Vermeidung Doppelordner)
  topdirs=$(awk -F/ 'NF{print $1}' "$list" | sort -u | wc -l | tr -d ' ')
  if [[ "$topdirs" == "1" ]]; then
    strip="--strip-components=1"
    topdir_name=$(awk -F/ 'NF{print $1}' "$list" | sort -u)
    # Pfade für die Top-Level-Auswertung ohne den ersten Ordner erzeugen
    awk -F/ 'NF{ $1=""; sub(/^\/+/,""); print }' OFS='/' "$list" > "$rel_list"
  else
    strip=""
    cp "$list" "$rel_list"
    topdir_name=""
  fi

  # Kandidaten: Top-Level-Ordnernamen (nach optionalem Strip), 'correct' muss existieren
  # -> wähle 1 zufälligen Ordner != correct
  toplevels=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    toplevels+=("$line")
  done < <(awk -F/ 'NF{print $1}' "$rel_list" | sed '/^$/d' | sort -u)

  has_correct=0
  for name in "${toplevels[@]}"; do
    if [[ "$name" == "correct" ]]; then
      has_correct=1
      break
    fi
  done

  if [[ "$has_correct" -eq 0 ]]; then
    : > "$err_marker"
    echo "FEHLER: In $base kein 'correct/'-Ordner gefunden." >&2
    rm -f "$start_marker" "$list" "$rel_list" "$tmp_list"
    return 1
  fi

  # Liste der „anderen“ Ordner (ohne correct)
  others=()
  for name in "${toplevels[@]}"; do
    [[ "$name" == "correct" ]] && continue
    others+=("$name")
  done

  if [[ "${#others[@]}" -gt 0 ]]; then
    if [[ -n "$SEED" ]]; then
      # deterministische Wahl: lexikographisch kleinster Name
      chosen_other=$(printf "%s\n" "${others[@]}" | sort | head -n1)
    else
      if command -v gshuf >/dev/null 2>&1; then
        chosen_other=$(printf "%s\n" "${others[@]}" | gshuf -n1)
      else
        chosen_other=$(printf "%s\n" "${others[@]}" | shuf -n1)
      fi
    fi
  else
    chosen_other=""
  fi

  # Filter-Pattern für die Original-Liste bauen
  if [[ -n "$topdir_name" ]]; then
    # Beispiel: ^topdir/(correct|<chosen>)(/|$)
    if [[ -n "$chosen_other" ]]; then
      pattern="^${topdir_name}/(${chosen_other}|correct)(/|$)"
    else
      pattern="^${topdir_name}/correct(/|$)"
    fi
  else
    # Beispiel: ^(correct|<chosen>)(/|$)
    if [[ -n "$chosen_other" ]]; then
      pattern="^(${chosen_other}|correct)(/|$)"
    else
      pattern="^correct(/|$)"
    fi
  fi

  # Inklusionsliste erzeugen (alle Einträge unter correct/ + ggf. unter <chosen_other>/)
  grep -E "$pattern" "$list" > "$tmp_list" || true

  if [[ ! -s "$tmp_list" ]]; then
    : > "$err_marker"
    echo "FEHLER: In $base nichts Passendes gefunden (Pattern: $pattern)." >&2
    rm -f "$start_marker" "$list" "$rel_list" "$tmp_list"
    return 1
  fi

  # In temporäres Arbeitsverzeichnis extrahieren, dann sauber nach $dir verschieben
  local workdir="$dir.__extracting__"
  rm -rf "$workdir"
  mkdir -p "$workdir"

  if bsdtar -xpf "$f" -C "$workdir" $strip -T "$tmp_list"; then
    {
      ts=$(date +"%Y-%m-%d %H:%M:%S")
      echo "$ts  OK  $base   chosen_other=${chosen_other:-none}"
    } >> "$state_dir/progress.txt"
    : > "$ok_marker"
    rm -f "$err_marker" "$start_marker" 2>/dev/null || true

    # Sicherstellen, dass nur EIN Zielordner existiert (keine Doppelverschachtelung)
    shopt -s dotglob nullglob
    rm -rf "$dir"/*
    mv "$workdir"/* "$dir"/ 2>/dev/null || true
    rmdir "$workdir" 2>/dev/null || true
    shopt -u dotglob nullglob

    echo "OK: $base -> $dir   (correct/ + ${chosen_other:-kein weiterer Ordner})"
  else
    : > "$err_marker"
    echo "FEHLER beim Entpacken: $base" >&2
    rm -rf "$workdir"
    rm -f "$start_marker"
    rm -f "$list" "$rel_list" "$tmp_list"
    return 1
  fi

  rm -f "$list" "$rel_list" "$tmp_list"
}


export -f extract_one
export OUT DONE_DIR STATE_DIR SEED

echo "Starte Extraktion …"
xargs -0 -n1 -P "$JOBS" bash -c 'extract_one "$1"' _ < "$TODO_FILE"

# Abschluss-Info
OK_COUNT_END=$(find "$DONE_DIR" -type f -name '*.ok' | wc -l | tr -d ' ')
echo "Fertig. Fortschritt gespeichert unter: $STATE_DIR"
echo "Erfolgreich erledigt: $OK_COUNT_END / $SEL_COUNT"
echo "Siehe: $STATE_DIR/progress.txt"
