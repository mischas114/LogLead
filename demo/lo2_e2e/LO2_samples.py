"""LO2 demo pipeline for enhancement, anomaly detection, and explainability."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from pathlib import Path
from datetime import datetime

import polars as pl

from loglead import AnomalyDetector
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
import loglead.explainer as ex
import joblib
import numpy as np

from metrics_utils import (
    false_positive_rate_at_alpha,
    population_stability_index,
    precision_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LO2 enhancement pipeline with optional anomaly detection phases."
    )
    parser.add_argument(
        "--phase",
        choices=["enhancers", "if", "full"],
        default="full",
        help="Use 'enhancers' to stop after feature generation; 'if' trainiert IsolationForest; 'full' ergänzt LR/DT + XAI.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for sampling enhanced records and optional down-sampling.",
    )
    parser.add_argument(
        "--if-contamination",
        type=float,
        default=0.1,
        help="IsolationForest contamination (Anteil erwarteter Anomalien).",
    )
    parser.add_argument(
        "--if-n-estimators",
        type=int,
        default=200,
        help="Anzahl Trees für IsolationForest.",
    )
    parser.add_argument(
        "--if-max-samples",
        default="auto",
        help="max_samples für IsolationForest (Ganzzahl oder 'auto').",
    )
    parser.add_argument(
        "--if-item",
        default="e_words",
        help="Spalte mit Tokenlisten für IsolationForest (z.B. e_words, e_trigrams, e_event_drain_id).",
    )
    parser.add_argument(
        "--if-numeric",
        default="",
        help="Kommagetrennte numerische Zusatzfeatures (z.B. e_chars_len).",
    )
    parser.add_argument(
        "--save-if",
        type=Path,
        default=Path("result/lo2/lo2_if_predictions.parquet"),
        help="Pfad für IsolationForest-Ergebnis (Parquet oder CSV).",
    )
    parser.add_argument(
        "--save-enhancers",
        action="store_true",
        help="Persist enhanced event/sequence tables to Parquet files.",
    )
    parser.add_argument(
        "--enhancers-output-dir",
        type=Path,
        default=Path("result/lo2/enhanced"),
        help="Directory used when --save-enhancers is active (relative paths resolve against the original working directory).",
    )
    parser.add_argument(
        "--overwrite-enhancers",
        action="store_true",
        help="Allow replacing existing enhancer export files.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path for persisting the trained IsolationForest model and vectorizer via joblib.",
    )
    parser.add_argument(
        "--overwrite-model",
        action="store_true",
        help="Allow replacing an existing model dump when --save-model is provided.",
    )
    parser.add_argument(
        "--if-holdout-fraction",
        type=float,
        default=0.0,
        help="Optional fraction (0-0.5) of 'correct' events reserved as temporal hold-out.",
    )
    parser.add_argument(
        "--if-threshold-percentile",
        type=float,
        default=None,
        help="Optional percentile (e.g. 99.5) to derive a score threshold from the hold-out set.",
    )
    parser.add_argument(
        "--report-precision-at",
        type=int,
        default=None,
        help="Report Precision@k for the IF scores (requires anomaly labels).",
    )
    parser.add_argument(
        "--report-fp-alpha",
        type=float,
        default=None,
        help="Report False-Positive rate at the top alpha fraction (e.g. 0.005 for 0.5%%).",
    )
    parser.add_argument(
        "--report-psi",
        action="store_true",
        help="Report Population Stability Index between train and hold-out scores.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("result/lo2/metrics"),
        help="Directory for optional metric reports (CSV/JSON).",
    )
    parser.add_argument(
        "--dump-metadata",
        action="store_true",
        help="Write a model.yml snapshot alongside the joblib artefact.",
    )
    return parser.parse_args()


def _transform_with_detector(detector: AnomalyDetector, df: pl.DataFrame):
    """Vectorize a new dataframe using an already-fitted detector."""
    if df is None or df.is_empty():
        return None
    X, _, _ = detector._prepare_data(df, detector.vec)  # type: ignore[attr-defined]
    return X


def _dict_to_yaml_lines(payload: dict, indent: int = 0) -> list[str]:
    """Minimal YAML serializer (avoids extra dependency)."""
    lines: list[str] = []
    pad = "  " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.extend(_dict_to_yaml_lines(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{pad}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{pad}  -")
                    lines.extend(_dict_to_yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return lines


def _log_train_fraction(label: str, train_rows: int, total_rows: int) -> None:
    frac = train_rows / max(total_rows, 1)
    print(f"[TrainStats] {label}: train_rows={train_rows} total_rows={total_rows} fraction={frac:.4f}")


def main() -> None:
    args = parse_args()

    # Keep working directory stable so relative paths resolve against this script location.
    script_dir = Path(__file__).resolve().parent
    orig_cwd = Path.cwd()
    os.chdir(script_dir)

    # Expected loader output locations. Result artefacts live under demo/result/lo2 relative to repo root.
    loader_output = (script_dir / "../result/lo2").resolve()
    events_path = loader_output / "lo2_events.parquet"
    seq_path = loader_output / "lo2_sequences.parquet"

    if not events_path.exists():
        raise SystemExit(
            "Missing Parquet export. Run run_lo2_loader.py with --save-parquet before executing this script."
        )

    print(f"Reading LO2 events from {events_path}")
    df_events = pl.read_parquet(events_path)
    print(f"Number of LO2 events: {len(df_events)}")
    if "anomaly" in df_events.columns:
        event_ano = int(df_events["anomaly"].sum())
        print(f"Event anomalies: {event_ano} ({event_ano / max(len(df_events), 1) * 100:.2f}%)")

    # Optional sequence-level table.
    df_seqs = None
    if seq_path.exists():
        print(f"Reading LO2 sequences from {seq_path}")
        df_seqs = pl.read_parquet(seq_path)
        if len(df_seqs):
            seq_ano = int(df_seqs["anomaly"].sum()) if "anomaly" in df_seqs.columns else 0
            print(f"Sequence anomalies: {seq_ano} ({seq_ano / max(len(df_seqs), 1) * 100:.2f}%)")
    else:
        print("No lo2_sequences.parquet found; continuing with event-only workflow.")

    downsampling_performed = False
    train_stats: list[tuple[str, int, int]] = []

    print("\nEnhancing events (normalization, tokens, parsers, lengths)...")
    enhancer = EventLogEnhancer(df_events)
    df_events = enhancer.normalize()
    df_events = enhancer.words()
    df_events = enhancer.trigrams()
    try:
        df_events = enhancer.parse_drain()
    except Exception as exc:  # drain parser can fail if templates missing
        print(f"Drain parsing skipped: {exc}")

    df_events = enhancer.length()

    random.seed(args.sample_seed)
    rand_idx = random.randint(0, len(df_events) - 1)
    print("\nSample enhanced record:")
    print(f"Original:  {df_events['m_message'][rand_idx]}")
    if "e_message_normalized" in df_events.columns:
        print(f"Normalized: {df_events['e_message_normalized'][rand_idx]}")
    print(f"Words:     {df_events['e_words'][rand_idx]}")
    print(f"Trigrams:  {df_events['e_trigrams'][rand_idx]}")
    if "e_event_drain_id" in df_events.columns:
        print(f"Drain ID: {df_events['e_event_drain_id'][rand_idx]}")
    print(f"Len chars: {df_events['e_chars_len'][rand_idx]}")

    if df_seqs is not None and len(df_seqs):
        print("\nAggregating to sequence level...")
        seq_enhancer = SequenceEnhancer(df=df_events, df_seq=df_seqs)
        df_seqs = seq_enhancer.seq_len()
        df_seqs = seq_enhancer.duration()
        df_seqs = seq_enhancer.tokens(token="e_words")
        df_seqs = seq_enhancer.tokens(token="e_trigrams")

    if args.save_enhancers:
        enhancer_dir = args.enhancers_output_dir
        if not enhancer_dir.is_absolute():
            enhancer_dir = (orig_cwd / enhancer_dir).resolve()
        enhancer_dir.mkdir(parents=True, exist_ok=True)

        events_out = enhancer_dir / "lo2_events_enhanced.parquet"
        if events_out.exists() and not args.overwrite_enhancers:
            raise SystemExit(
                f"Enhanced events already exist at {events_out}. Use --overwrite-enhancers to replace them."
            )
        df_events.write_parquet(events_out)
        print(f"Enhanced events gespeichert unter {events_out}")

        if df_seqs is not None and len(df_seqs):
            seqs_out = enhancer_dir / "lo2_sequences_enhanced.parquet"
            if seqs_out.exists() and not args.overwrite_enhancers:
                raise SystemExit(
                    f"Enhanced sequences already exist at {seqs_out}. Use --overwrite-enhancers to replace them."
                )
            df_seqs.write_parquet(seqs_out)
            print(f"Enhanced sequences gespeichert unter {seqs_out}")

    if args.phase == "enhancers":
        print("\nEnhancer phase complete. Skipping anomaly detection and explainability.")
        return

    # Isolation Forest baseline (Phase D)
    print("\nTraining Isolation Forest on event words (Phase D)")
    numeric_cols = [col.strip() for col in args.if_numeric.split(",") if col.strip()]
    sad_if = AnomalyDetector(item_list_col=args.if_item, numeric_cols=numeric_cols or None)
    # IsolationForest learns only from normal runs; keep anomalies in test_df for evaluation.
    correct_events = df_events.filter(pl.col("test_case") == "correct")
    holdout_fraction = min(max(args.if_holdout_fraction, 0.0), 0.5)
    holdout_df = None
    if holdout_fraction > 0 and correct_events.height > 1:
        if "m_timestamp" in correct_events.columns:
            sorted_correct = correct_events.sort("m_timestamp")
        else:
            sorted_correct = correct_events.sort("seq_id")
        holdout_size = max(1, int(sorted_correct.height * holdout_fraction))
        if holdout_size >= sorted_correct.height:
            holdout_size = sorted_correct.height - 1
        if holdout_size > 0:
            holdout_df = sorted_correct.tail(holdout_size)
            correct_events = sorted_correct.head(sorted_correct.height - holdout_size)
            print(
                f"Using temporal hold-out: {holdout_size} events reserved ({holdout_fraction * 100:.2f}% of correct runs)."
            )
            downsampling_performed = True
    sad_if.train_df = correct_events
    sad_if.test_df = df_events
    sad_if.prepare_train_test_data()

    max_samples = args.if_max_samples
    if isinstance(max_samples, str) and max_samples != "auto":
        if max_samples.isdigit():
            max_samples = int(max_samples)
        else:
            raise SystemExit("--if-max-samples muss 'auto' oder eine Ganzzahl sein.")

    sad_if.train_IsolationForest(
        filter_anos=True,
        n_estimators=args.if_n_estimators,
        contamination=args.if_contamination,
        max_samples=max_samples,
    )
    pred_if = sad_if.predict()

    # Add raw anomaly scores and dense ranking for inspection.
    score_if = (-sad_if.model.score_samples(sad_if.X_test)).tolist()
    pred_if = pred_if.with_columns(
        pl.Series(name="score_if", values=score_if)
    ).with_columns(
        pl.col("score_if").rank("dense", descending=True).alias("rank_if")
    )
    print("Top 5 IF-Runs (höchster Score zuerst):")
    print(pred_if.sort("score_if", descending=True).head(5))

    train_scores = (-sad_if.model.score_samples(sad_if.X_train_no_anos)).tolist()
    holdout_scores = None
    if holdout_df is not None:
        holdout_matrix = _transform_with_detector(sad_if, holdout_df)
        if holdout_matrix is not None:
            holdout_scores = (-sad_if.model.score_samples(holdout_matrix)).tolist()

    threshold_value = None
    threshold_percentile = None
    if args.if_threshold_percentile is not None:
        percentile = args.if_threshold_percentile
        if percentile <= 1:
            percentile *= 100
        percentile = max(0.0, min(percentile, 100.0))
        source_scores = holdout_scores or train_scores
        if source_scores:
            threshold_value = float(np.percentile(source_scores, percentile))
            threshold_percentile = percentile / 100.0
            print(
                f"Derived IF score threshold: {threshold_value:.6f} (percentile {percentile:.2f})"
            )
        else:
            print("Threshold percentile requested, but no scores available to calibrate.")

    if threshold_value is not None:
        pred_if = pred_if.with_columns(
            (pl.col("score_if") >= threshold_value).alias("pred_if_threshold")
        )

    save_if_path = args.save_if
    if not save_if_path.is_absolute():
        save_if_path = (orig_cwd / save_if_path).resolve()
    save_if_path.parent.mkdir(parents=True, exist_ok=True)
    if save_if_path.suffix == ".csv":
        pred_if.write_csv(save_if_path)
    else:
        pred_if.write_parquet(save_if_path)
    print(f"IsolationForest-Ergebnis gespeichert unter {save_if_path}")

    metrics_results = {}
    if threshold_value is not None:
        metrics_results["threshold_value"] = threshold_value
        metrics_results["threshold_percentile"] = threshold_percentile
    if args.report_precision_at:
        precision_val = precision_at_k(pred_if, args.report_precision_at)
        if precision_val is not None:
            metrics_results[f"precision_at_{args.report_precision_at}"] = precision_val
        else:
            print("Precision@k requested, but insufficient data to compute.")

    if args.report_fp_alpha:
        fp_val = false_positive_rate_at_alpha(pred_if, args.report_fp_alpha)
        if fp_val is not None:
            metrics_results[f"fp_rate_at_{args.report_fp_alpha}"] = fp_val
        else:
            print("FP-rate@alpha requested, but insufficient data to compute.")

    if args.report_psi:
        if holdout_scores:
            psi_val = population_stability_index(train_scores, holdout_scores)
            if psi_val is not None:
                metrics_results["psi_train_vs_holdout"] = psi_val
        else:
            print("PSI requested, but hold-out scores are unavailable.")

    if metrics_results:
        metrics_dir = args.metrics_dir
        if not metrics_dir.is_absolute():
            metrics_dir = (orig_cwd / metrics_dir).resolve()
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_json = metrics_dir / "if_metrics.json"
        metrics_csv = metrics_dir / "if_metrics.csv"
        with metrics_json.open("w", encoding="utf-8") as fh:
            json.dump(metrics_results, fh, indent=2)
        with metrics_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for key, value in metrics_results.items():
                writer.writerow([key, value])
        print(f"IF metrics gespeichert unter {metrics_json} und {metrics_csv}")

    if args.save_model:
        model_path = args.save_model
        if not model_path.is_absolute():
            model_path = (orig_cwd / model_path).resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists() and not args.overwrite_model:
            raise SystemExit(
                f"Modelldatei existiert bereits unter {model_path}. Verwende --overwrite-model, um sie zu ersetzen."
            )
        joblib.dump((sad_if.model, sad_if.vec), model_path)
        print(f"IsolationForest-Modell + Vectorizer gespeichert unter {model_path}")

        if args.dump_metadata:
            metadata = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "training_rows": sad_if.train_df.height if sad_if.train_df is not None else 0,
                "holdout_rows": holdout_df.height if holdout_df is not None else 0,
                "if_params": {
                    "item_list_col": args.if_item,
                    "numeric_cols": numeric_cols,
                    "contamination": args.if_contamination,
                    "n_estimators": args.if_n_estimators,
                    "max_samples": args.if_max_samples,
                },
                "threshold": threshold_value,
                "threshold_percentile": threshold_percentile,
                "metrics": metrics_results,
            }
            try:
                git_commit = (
                    subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout.strip()
                )
                metadata["git_commit"] = git_commit
            except Exception:
                metadata["git_commit"] = "unknown"

            metadata_lines = _dict_to_yaml_lines(metadata)
            metadata_path = model_path.with_name("model.yml")
            with metadata_path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(metadata_lines) + "\n")
            print(f"Metadata YAML gespeichert unter {metadata_path}")
    elif args.dump_metadata:
        print("Warnung: --dump-metadata benötigt --save-model, wird übersprungen.")

    if args.phase == "if":
        print("\nIsolation Forest abgeschlossen. Weitere Modelle übersprungen.")
        return

    print("\nTraining anomaly detector on events (words)")
    sad = AnomalyDetector()
    sad.item_list_col = "e_words"
    sad.train_df = df_events
    sad.test_df = df_events
    sad.prepare_train_test_data()
    sad.train_LR()
    df_pred = sad.predict()
    print("Event-level predictions ready.")
    train_stats.append(("event_lr_words", sad.train_df.height if sad.train_df is not None else 0, df_events.height))
    _log_train_fraction("event_lr_words", sad.train_df.height if sad.train_df is not None else 0, df_events.height)

    print("Switching to trigrams + DecisionTree")
    sad.item_list_col = "e_trigrams"
    sad.train_df = df_events
    sad.test_df = df_events
    sad.prepare_train_test_data()
    sad.train_DT()
    df_pred = sad.predict()
    train_stats.append(("event_dt_trigrams", sad.train_df.height if sad.train_df is not None else 0, df_events.height))
    _log_train_fraction("event_dt_trigrams", sad.train_df.height if sad.train_df is not None else 0, df_events.height)

    if df_seqs is not None and len(df_seqs):
        print("\nSequence-level anomaly detection with duration + length")
        sad_seq = AnomalyDetector()
        sad_seq.numeric_cols = ["seq_len", "duration_sec"]
        sad_seq.train_df = df_seqs
        sad_seq.test_df = df_seqs
        sad_seq.prepare_train_test_data()
        sad_seq.train_LR()
        seq_pred = sad_seq.predict()
        print("Sequence-level predictions ready.")
        train_stats.append(
            ("sequence_lr_numeric", sad_seq.train_df.height if sad_seq.train_df is not None else 0, df_seqs.height)
        )
        _log_train_fraction(
            "sequence_lr_numeric", sad_seq.train_df.height if sad_seq.train_df is not None else 0, df_seqs.height
        )

        print("\nExplaining sequence model via SHAP (words vectorizer)")
        sad_seq.item_list_col = "e_words"
        sad_seq.numeric_cols = None
        sad_seq.train_df = df_seqs
        sad_seq.test_df = df_seqs
        sad_seq.prepare_train_test_data()
        sad_seq.train_LR()
        seq_pred = sad_seq.predict()
        explainer = ex.ShapExplainer(sad_seq, ignore_warning=True, plot_featurename_len=18)
        explainer.calc_shapvalues()
        explainer.plot(plottype="summary")
    else:
        print("\nNo sequence table available; skipping sequence-level AD and XAI.")

    print("\n[Summary] Full-data pipeline diagnostics:")
    for label, train_rows, total_rows in train_stats:
        frac = train_rows / max(total_rows, 1)
        print(f"  {label}: train_rows={train_rows} total_rows={total_rows} fraction={frac:.4f}")
    print(f"[Summary] Downsampling occurred: {'yes' if downsampling_performed else 'no'}")

    print("\nLO2 sample pipeline complete.")


if __name__ == "__main__":
    main()
