#!/usr/bin/env python3
"""Scan LO2 explainability outputs and append a compact summary."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REQUIRED_FILES = (
    "if_nn_mapping.csv",
    "if_false_positives.txt",
)

SHAP_EXTENSIONS = (".png", ".svg", ".pdf", ".txt", ".json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise LO2 explainability artefacts and write a markdown snippet.",
    )
    parser.add_argument(
        "--explainability-dir",
        default="demo/result/lo2/explainability",
        help="Directory with Phase-F outputs.",
    )
    parser.add_argument(
        "--summary-file",
        default="summary-result.md",
        help="Markdown file to append the summary to (created if missing).",
    )
    parser.add_argument(
        "--ticket-template",
        default=None,
        help="Optional path that will receive a ticket-ready summary copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary without writing files.",
    )
    return parser.parse_args()


def collect_required_files(base_dir: Path) -> Tuple[List[str], List[str]]:
    present, missing = [], []
    for name in REQUIRED_FILES:
        path = base_dir / name
        if path.exists():
            present.append(name)
        else:
            missing.append(name)
    return present, missing


def collect_shap_artifacts(base_dir: Path) -> List[str]:
    shap_files: List[str] = []
    for ext in SHAP_EXTENSIONS:
        for path in base_dir.glob(f"*shap*{ext}"):
            shap_files.append(path.name)
    return sorted(set(shap_files))


def load_metrics(base_dir: Path) -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []
    for path in sorted(base_dir.glob("metrics_*.json")):
        entry: Dict[str, object] = {"file": path.name}
        try:
            entry["metrics"] = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - defensive guard
            entry["error"] = str(exc)
        metrics.append(entry)
    return metrics


def summarise_metrics(entries: Iterable[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for entry in entries:
        label = entry.get("file", "metrics")
        if "error" in entry:
            lines.append(f"  * {label}: error loading ({entry['error']})")
            continue
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            lines.append(f"  * {label}: unexpected payload type {type(metrics).__name__}")
            continue
        score_snippets = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                score_snippets.append(f"{key}={value:.4f}")
            else:
                score_snippets.append(f"{key}={value}")
        if not score_snippets:
            score_snippets.append("no scalar metrics found")
        lines.append(f"  * {label}: " + ", ".join(score_snippets))
    if not lines:
        lines.append("  * no metrics files detected")
    return lines


def summarise_false_positives(base_dir: Path) -> str:
    fp_path = base_dir / "if_false_positives.txt"
    if not fp_path.exists():
        return "not available"
    try:
        lines = [line.strip() for line in fp_path.read_text().splitlines() if line.strip()]
    except Exception:  # pragma: no cover - robust fallback
        return "failed to read"
    return f"{len(lines)} listed" if lines else "file present but empty"


def build_summary(
    explain_dir: Path,
    required_present: List[str],
    required_missing: List[str],
    shap_files: List[str],
    metrics_summary: List[str],
    false_positive_status: str,
) -> List[str]:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        f"## LO2 explainability scan ({timestamp})",
        "",
        f"- Directory: `{explain_dir}`",
        f"- Required files present: {', '.join(required_present) if required_present else 'none'}",
    ]
    if required_missing:
        lines.append(f"- Missing artefacts: {', '.join(required_missing)}")
    else:
        lines.append("- Missing artefacts: none")
    lines.extend(
        [
            f"- SHAP artefacts: {len(shap_files)} file(s)",
            f"- False positives list: {false_positive_status}",
            "- Metrics:",
        ]
    )
    lines.extend(metrics_summary)
    lines.append("")
    if shap_files:
        lines.append("### SHAP files")
        for name in shap_files:
            lines.append(f"- {name}")
        lines.append("")
    return lines


def write_outputs(lines: List[str], summary_file: Path, ticket_path: Path | None, dry_run: bool) -> None:
    joined = "\n".join(lines).rstrip() + "\n"
    if dry_run or summary_file is None:
        print(joined)
    else:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with summary_file.open("a", encoding="utf-8") as handle:
            handle.write(joined + "\n")
    if ticket_path is not None and not dry_run:
        ticket_path.parent.mkdir(parents=True, exist_ok=True)
        ticket_path.write_text(joined + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    explain_dir = Path(args.explainability_dir).expanduser().resolve()
    summary_path = Path(args.summary_file).expanduser().resolve()
    ticket_path = Path(args.ticket_template).expanduser().resolve() if args.ticket_template else None

    if not explain_dir.exists():
        raise FileNotFoundError(f"Explainability directory not found: {explain_dir}")

    present, missing = collect_required_files(explain_dir)
    shap_files = collect_shap_artifacts(explain_dir)
    metrics_entries = load_metrics(explain_dir)
    metrics_summary = summarise_metrics(metrics_entries)
    fp_status = summarise_false_positives(explain_dir)

    lines = build_summary(
        explain_dir,
        present,
        missing,
        shap_files,
        metrics_summary,
        fp_status,
    )

    write_outputs(lines, summary_path, ticket_path, args.dry_run)


if __name__ == "__main__":
    main()
