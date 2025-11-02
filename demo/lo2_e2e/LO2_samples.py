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
from typing import Any, Dict, List

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

DEFAULT_SUPERVISED_MODELS: List[str] = [
    "event_lr_words",
    "event_dt_trigrams",
    "sequence_lr_numeric",
    "sequence_shap_lr_words",
]

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "event_lr_words": {
        "description": "LogisticRegression auf Event-Worttokens (Bag-of-Words).",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LR",
        "stat_label": "event_lr_words",
    },
    "event_dt_trigrams": {
        "description": "DecisionTree auf Event-Trigrams.",
        "level": "event",
        "item_list_col": "e_trigrams",
        "numeric_cols": [],
        "train_method": "train_DT",
        "stat_label": "event_dt_trigrams",
    },
    "event_lsvm_words": {
        "description": "LinearSVM auf Event-Worttokens.",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LSVM",
        "stat_label": "event_lsvm_words",
    },
    "event_rf_words": {
        "description": "RandomForest auf Event-Worttokens.",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_RF",
        "stat_label": "event_rf_words",
    },
    "event_xgb_words": {
        "description": "XGBoost Klassifikator auf Event-Worttokens.",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_XGB",
        "stat_label": "event_xgb_words",
    },
    "event_lof_words": {
        "description": "LocalOutlierFactor (novelty) auf Event-Worttokens (trainiert nur auf korrekten Runs).",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LOF",
        "train_kwargs": {"filter_anos": True},
        "train_selector": "correct_only",
        "stat_label": "event_lof_words",
    },
    "event_kmeans_words": {
        "description": "KMeans Clustering auf Event-Worttokens (2 Cluster).",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_KMeans",
        "stat_label": "event_kmeans_words",
    },
    "event_oneclass_svm_words": {
        "description": "OneClassSVM auf Event-Worttokens (trainiert nur auf korrekten Runs).",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_OneClassSVM",
        "train_selector": "correct_only",
        "stat_label": "event_oneclass_svm_words",
    },
    "event_rarity_words": {
        "description": "RarityModel auf Event-Worttokens.",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_RarityModel",
        "stat_label": "event_rarity_words",
    },
    "event_oov_words": {
        "description": "OOVDetector für seltene Tokens (trainiert nur auf korrekten Runs).",
        "level": "event",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_OOVDetector",
        "train_kwargs": {"filter_anos": True},
        "train_selector": "correct_only",
        "stat_label": "event_oov_words",
    },
    "sequence_lr_numeric": {
        "description": "LogisticRegression auf Sequenz-Längen und Dauerfeatures.",
        "level": "sequence",
        "item_list_col": None,
        "numeric_cols": ["seq_len", "duration_sec"],
        "train_method": "train_LR",
        "stat_label": "sequence_lr_numeric",
    },
    "sequence_lr_words": {
        "description": "LogisticRegression auf Sequenz-Worttokens.",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LR",
        "stat_label": "sequence_lr_words",
    },
    "sequence_shap_lr_words": {
        "description": "LogisticRegression auf Sequenz-Worttokens mit SHAP-Erklärung.",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LR",
        "stat_label": "sequence_shap_lr_words",
        "requires_shap": True,
        "shap_kwargs": {"ignore_warning": True, "plot_featurename_len": 18},
        "shap_plot_type": "summary",
    },
}


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
        "--load-model",
        type=Path,
        default=None,
        help="Optional path to an existing IsolationForest+vectorizer bundle to reuse and skip retraining.",
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
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_SUPERVISED_MODELS),
        help="Kommagetrennte Liste an Schlüsselwörtern für zusätzliche Modelle (siehe --list-models).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Verfügbare Modellschlüssel ausgeben und beenden.",
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

    if args.list_models:
        print("Verfügbare Modelle:")
        for key in sorted(MODEL_REGISTRY):
            spec = MODEL_REGISTRY[key]
            level = spec.get("level", "event")
            print(f"  {key} ({level}): {spec['description']}")
        return

    selected_models = [entry.strip() for entry in args.models.split(",") if entry.strip()]
    if selected_models:
        unknown_models = [m for m in selected_models if m not in MODEL_REGISTRY]
        if unknown_models:
            raise SystemExit(f"Unbekannte Modellschlüssel: {', '.join(unknown_models)}")
    else:
        selected_models = []

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
    print("\nTraining/Loading Isolation Forest on event words (Phase D)")
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

    # Try loading an existing bundle if provided
    model_loaded = False
    if args.load_model is not None:
        load_path = args.load_model
        if not load_path.is_absolute():
            load_path = (orig_cwd / load_path).resolve()
        if load_path.exists():
            try:
                loaded = joblib.load(load_path)
                # Support both tuple and dict-style bundles
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    model, vec = loaded
                elif isinstance(loaded, dict):
                    model = loaded.get("model")
                    vec = loaded.get("vectorizer") or loaded.get("vec")
                else:
                    raise ValueError("Unrecognized model bundle format")
                sad_if.model = model
                sad_if.vec = vec
                model_loaded = True
                print(f"Loaded existing IF model bundle from {load_path}")
            except Exception as exc:
                print(f"[WARN] Could not load model bundle from {load_path}: {exc}. Will train a new model.")
        else:
            print(f"[INFO] No existing model found at {load_path}; training a new model.")

    # Prepare features (reuses existing vectorizer if present)
    sad_if.prepare_train_test_data()

    max_samples = args.if_max_samples
    if isinstance(max_samples, str) and max_samples != "auto":
        if max_samples.isdigit():
            max_samples = int(max_samples)
        else:
            raise SystemExit("--if-max-samples muss 'auto' oder eine Ganzzahl sein.")

    if not model_loaded:
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

    if not selected_models:
        print("\nKeine zusätzlichen Modelle in --models angegeben; überspringe Phase E/F.")
    else:
        print("\nStarte konfigurierbare Anomalie-Detektoren (Phase E/F)")
    for model_key in selected_models:
        spec = MODEL_REGISTRY[model_key]
        level = spec.get("level", "event")
        dataset = df_events if level == "event" else df_seqs
        if dataset is None or dataset.is_empty():
            requirement = "Sequenzdaten" if level == "sequence" else "Eventdaten"
            print(f"\n[{model_key}] übersprungen (benötigt {requirement}).")
            continue

        train_df = dataset
        if spec.get("train_selector") == "correct_only":
            if "test_case" in dataset.columns:
                filtered = dataset.filter(pl.col("test_case") == "correct")
                if filtered.is_empty():
                    print(f"\n[{model_key}] übersprungen (keine 'correct'-Beispiele vorhanden).")
                    continue
                train_df = filtered
            else:
                print(f"\n[{model_key}] übersprungen (Spalte 'test_case' fehlt für Filterung).")
                continue

        print(f"\n[{model_key}] {spec['description']}")
        detector = AnomalyDetector()
        detector.item_list_col = spec.get("item_list_col")
        numeric_cols = spec.get("numeric_cols")
        detector.numeric_cols = numeric_cols if numeric_cols is not None else []
        detector.train_df = train_df
        detector.test_df = dataset
        detector.prepare_train_test_data()

        train_kwargs = spec.get("train_kwargs", {})
        getattr(detector, spec["train_method"])(**train_kwargs)
        detector.predict()
        train_rows = detector.train_df.height if detector.train_df is not None else 0
        total_rows = dataset.height
        train_stats.append((spec["stat_label"], train_rows, total_rows))
        _log_train_fraction(spec["stat_label"], train_rows, total_rows)

        if spec.get("requires_shap"):
            shap_kwargs = spec.get("shap_kwargs", {})
            shap_plot_type = spec.get("shap_plot_type", "summary")
            print("  -> SHAP-Erklärungen werden berechnet.")
            explainer = ex.ShapExplainer(detector, **shap_kwargs)
            explainer.calc_shapvalues()
            explainer.plot(plottype=shap_plot_type)

    if df_seqs is None or df_seqs.is_empty():
        print("\nNo sequence table available; skipping sequence-level models.")

    print("\n[Summary] Full-data pipeline diagnostics:")
    for label, train_rows, total_rows in train_stats:
        frac = train_rows / max(total_rows, 1)
        print(f"  {label}: train_rows={train_rows} total_rows={total_rows} fraction={frac:.4f}")
    print(f"[Summary] Downsampling occurred: {'yes' if downsampling_performed else 'no'}")

    print("\nLO2 sample pipeline complete.")


if __name__ == "__main__":
    main()
