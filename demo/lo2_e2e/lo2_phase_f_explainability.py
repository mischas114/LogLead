#!/usr/bin/env python3
"""Phase F helper to generate explainability artefacts for the LO2 MVP pipeline.

The script replays the best IsolationForest configuration (T4) and the supervised
baselines from Phase E, then saves NNExplainer mappings and SHAP plots/feature logs
under ``demo/result/lo2/explainability/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # Headless plot generation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

from loglead import AnomalyDetector
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
from loglead.explainer import NNExplainer, ShapExplainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explainability artefacts for Phase F.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("demo/result/lo2"),
        help="Ordner mit den von Phase B erzeugten Parquet-Dateien.",
    )
    parser.add_argument(
        "--if-contamination",
        type=float,
        default=0.45,
        help="IsolationForest contamination Wert (Phase-D-Tuning).",
    )
    parser.add_argument(
        "--if-n-estimators",
        type=int,
        default=200,
        help="Anzahl Trees für den IsolationForest.",
    )
    parser.add_argument(
        "--if-max-samples",
        default="auto",
        help="max_samples Parameter für den IsolationForest.",
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=0,
        help="Maximale Anzahl Beispiele für die SHAP-Berechnung (0 = keine Begrenzung).",
    )
    parser.add_argument(
        "--nn-top-k",
        type=int,
        default=0,
        help="Wie viele anomal markierte Events für das NN-Mapping berücksichtigt werden (0 = alle).",
    )
    parser.add_argument(
        "--nn-normal-sample",
        type=int,
        default=0,
        help="Wie viele Normalfälle ergänzend berücksichtigt werden (0 = alle).",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Optional: vorhandenes IF+Vectorizer-Bundle laden und Training überspringen.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_dense(matrix: np.ndarray) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_event_enhancers(df_events: pl.DataFrame) -> pl.DataFrame:
    enhancer = EventLogEnhancer(df_events)
    df_events = enhancer.normalize()
    df_events = enhancer.words()
    df_events = enhancer.trigrams()
    try:
        df_events = enhancer.parse_drain()
    except Exception as exc:  # Drain Templates optional
        print(f"[WARN] Drain parsing ausgelassen: {exc}")
    df_events = enhancer.length()
    return df_events


def train_if(df_events: pl.DataFrame, args: argparse.Namespace):
    numeric_cols = ["e_chars_len", "e_event_id_len", "e_words_len"]
    item_col = "e_event_drain_id" if "e_event_drain_id" in df_events.columns else "e_words"
    if item_col != "e_event_drain_id":
        print("[INFO] Drain IDs unavailable; falling back to e_words for IsolationForest.")
    sad_if = AnomalyDetector(item_list_col=item_col, numeric_cols=numeric_cols)
    sad_if.train_df = df_events.filter(pl.col("test_case") == "correct")
    sad_if.test_df = df_events

    # Optional: vorhandenes Bundle laden
    model_loaded = False
    if getattr(args, "load_model", None):
        load_path = args.load_model
        if not load_path.is_absolute():
            load_path = load_path.resolve()
        if load_path.exists():
            try:
                loaded = joblib.load(load_path)
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
                print(f"[INFO] Bestehendes IF-Modell geladen: {load_path}")
            except Exception as exc:
                print(f"[WARN] Konnte Modellbundle nicht laden ({load_path}): {exc}. Trainiere neu.")

    # Features vorbereiten (nutzt vorhandenen Vektorizer falls gesetzt)
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
    # Score & Ranking analog Phase D
    score_if = (-sad_if.model.score_samples(sad_if.X_test)).tolist()
    pred_if = pred_if.with_columns(pl.Series("score_if", score_if))
    pred_if = pred_if.with_columns(pl.col("score_if").rank("dense", descending=True).alias("rank_if"))
    if "row_id" in pred_if.columns:
        pred_if = pred_if.drop("row_id")
    pred_if = pred_if.with_row_index(name="row_id")
    return sad_if, pred_if


def build_nn_mapping(
    pred_if: pl.DataFrame,
    sad_if: AnomalyDetector,
    out_dir: Path,
    top_k: int,
    normal_sample: int,
) -> tuple[int, int]:
    anomalies = pred_if.filter(pl.col("pred_ano") == 1).sort("score_if", descending=True)
    if top_k > 0 and anomalies.height > top_k:
        anomalies = anomalies.head(top_k)
    if anomalies.is_empty():
        print("[WARN] Keine Anomalien für das NN-Mapping vorhanden – Schritt übersprungen.")
        return (0, 0)

    normals = pred_if.filter(pl.col("pred_ano") == 0).sort("score_if", descending=True)
    if normal_sample > 0 and normals.height > normal_sample:
        normals = normals.head(normal_sample)

    actual_top = anomalies.height
    actual_normals = normals.height

    subset = pl.concat([anomalies, normals], how="vertical").unique("row_id")
    indices = [int(idx) for idx in subset["row_id"].to_list()]

    X_full = sad_if.X_test
    if hasattr(X_full, "tocsr"):
        X_full = X_full.tocsr()
    X_subset = X_full[indices]

    nn = NNExplainer(subset, to_dense(X_subset), id_col="row_id", pred_col="pred_ano")
    mapping_path = out_dir / "if_nn_mapping.csv"
    nn.mapping.write_csv(mapping_path)
    print(f"[INFO] NN-Mapping gespeichert: {mapping_path}")

    # False-Positive-Analyse sichern
    fp_rows = pred_if.filter((pl.col("pred_ano") == 1) & (pl.col("anomaly") == 0))
    if fp_rows.height == 0:
        print("[INFO] Keine False Positives gefunden.")
    else:
        fp_text = []
        for row in fp_rows.iter_rows(named=True):
            words = " ".join(row["e_words"])
            fp_text.append(
                f"row_id={row['row_id']} | seq_id={row['seq_id']} | service={row['service']} | score_if={row['score_if']:.6f}\n{words}"
            )
        write_lines(out_dir / "if_false_positives.txt", fp_text)
        print(f"[INFO] False-Positive-Liste geschrieben: {out_dir / 'if_false_positives.txt'}")

    print(
        f"[Explainability] NN mapping uses anomalies={actual_top} normals={actual_normals} "
        f"(requested top_k={top_k}, normal_sample={normal_sample})"
    )
    return actual_top, actual_normals


def compute_metrics(sad: AnomalyDetector) -> dict:
    y_true = np.asarray(sad.labels_test, dtype=int)
    y_pred = sad.model.predict(sad.X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(y_true.size),
    }
    if hasattr(sad.model, "predict_proba"):
        y_prob = sad.model.predict_proba(sad.X_test)[:, 1]
        metrics["aucroc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def save_top_features(explainer: ShapExplainer, limit: int, out_path: Path) -> None:
    feature_names = explainer.sorted_featurenames()[:limit]
    write_lines(out_path, [f"{idx+1}. {name}" for idx, name in enumerate(feature_names)])


def plot_shap(explainer: ShapExplainer, out_prefix: Path) -> None:
    shap_vals = explainer.Svals
    data = explainer.shapdata
    dense_data = to_dense(data)

    summary_path = out_prefix.parent / f"{out_prefix.name}_summary.png"
    if hasattr(shap_vals, "values"):
        shap.summary_plot(shap_vals, show=False, max_display=20)
    else:
        shap.summary_plot(shap_vals, dense_data, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()
    print(f"[INFO] SHAP Summary Plot: {summary_path}")

    bar_path = out_prefix.parent / f"{out_prefix.name}_bar.png"
    shap.plots.bar(shap_vals, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[INFO] SHAP Bar Plot: {bar_path}")


def run_event_lr_shap(
    df_events: pl.DataFrame, out_dir: Path, sample_size: int
) -> tuple[dict, int, int]:
    sad = AnomalyDetector()
    sad.item_list_col = "e_words"
    sad.train_df = df_events
    sad.test_df = df_events
    sad.prepare_train_test_data()
    sad.train_LR()
    _ = sad.predict()

    shap_expl = ShapExplainer(sad, ignore_warning=True, plot_featurename_len=18)
    total = sad.X_test.shape[0]
    size = total if sample_size <= 0 else min(sample_size, total)
    shap_expl.calc_shapvalues(custom_slice=slice(0, size))
    save_top_features(shap_expl, 20, out_dir / "lr_top_tokens.txt")
    plot_shap(shap_expl, out_dir / "lr_shap")
    metrics = compute_metrics(sad)
    save_json(out_dir / "metrics_lr.json", metrics)
    print(f"[Explainability] LR SHAP samples: {size} of {total} (requested {sample_size})")
    return metrics, size, total


def run_event_dt_shap(
    df_events: pl.DataFrame, out_dir: Path, sample_size: int
) -> tuple[dict, int, int]:
    sad = AnomalyDetector()
    sad.item_list_col = "e_trigrams"
    sad.train_df = df_events
    sad.test_df = df_events
    sad.prepare_train_test_data()
    sad.train_DT()
    _ = sad.predict()

    shap_expl = ShapExplainer(sad, ignore_warning=True, plot_featurename_len=18)
    total = sad.X_test.shape[0]
    size = total if sample_size <= 0 else min(sample_size, total)
    shap_expl.calc_shapvalues(custom_slice=slice(0, size))
    save_top_features(shap_expl, 20, out_dir / "dt_top_trigrams.txt")
    plot_shap(shap_expl, out_dir / "dt_shap")
    metrics = compute_metrics(sad)
    save_json(out_dir / "metrics_dt.json", metrics)
    print(f"[Explainability] DT SHAP samples: {size} of {total} (requested {sample_size})")
    return metrics, size, total


def run_sequence_lr_shap(
    df_events: pl.DataFrame, df_seq: pl.DataFrame | None, out_dir: Path, sample_size: int
) -> tuple[dict | None, int, int]:
    if df_seq is None or df_seq.is_empty():
        print("[INFO] Keine Sequenz-Parquet-Datei gefunden – Sequenz-SHAP übersprungen.")
        return None, 0, 0

    seq_enhancer = SequenceEnhancer(df=df_events, df_seq=df_seq)
    df_seq = seq_enhancer.seq_len()
    df_seq = seq_enhancer.duration()
    df_seq = seq_enhancer.tokens(token="e_words")

    sad = AnomalyDetector()
    sad.numeric_cols = ["seq_len", "duration_sec"]
    sad.test_train_split(df_seq, test_frac=0.90)
    sad.prepare_train_test_data()
    sad.train_LR()
    _ = sad.predict()

    metrics = compute_metrics(sad)
    save_json(out_dir / "metrics_seq_lr.json", metrics)

    if sad.vec is None:
        note_path = out_dir / "seq_lr_shap_skipped.txt"
        write_lines(
            note_path,
            [
                "SHAP wurde übersprungen, weil das Sequence-LR-Modell nur numerische Features (seq_len, duration_sec) nutzt",
                "und daher kein Vectorizer mit Feature-Namen vorhanden ist.",
            ],
        )
        print(f"[INFO] Sequence-SHAP übersprungen (nur numerische Features); Hinweis unter {note_path}")
        return metrics, 0, sad.X_test.shape[0]

    shap_expl = ShapExplainer(sad, ignore_warning=True, plot_featurename_len=18)
    total = sad.X_test.shape[0]
    size = total if sample_size <= 0 else min(sample_size, total)
    shap_expl.calc_shapvalues(custom_slice=slice(0, size))
    save_top_features(shap_expl, 20, out_dir / "seq_lr_top_features.txt")
    plot_shap(shap_expl, out_dir / "seq_lr_shap")
    print(f"[Explainability] Sequence LR SHAP samples: {size} of {total} (requested {sample_size})")
    return metrics, size, total


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    events_path = root / "lo2_events.parquet"
    seq_path = root / "lo2_sequences.parquet"
    if not events_path.exists():
        raise SystemExit(f"lo2_events.parquet nicht gefunden unter {events_path}")

    out_dir = ensure_dir(root / "explainability")
    print(f"[INFO] Artefakte werden unter {out_dir} abgelegt.")

    print("[INFO] Lade Events …")
    df_events = pl.read_parquet(events_path)
    df_events = run_event_enhancers(df_events)

    print("[INFO] Trainiere IsolationForest (Phase D Setting)…")
    sad_if, pred_if = train_if(df_events, args)
    pred_if_path = out_dir / "lo2_if_predictions.parquet"
    pred_if.write_parquet(pred_if_path)
    print(f"[INFO] IF-Predictions gespeichert: {pred_if_path}")

    print("[INFO] Berechne NNExplainer Mapping …")
    nn_top_used, nn_normal_used = build_nn_mapping(
        pred_if,
        sad_if,
        out_dir,
        top_k=args.nn_top_k,
        normal_sample=args.nn_normal_sample,
    )

    print("[INFO] Berechne SHAP für Logistic Regression …")
    lr_metrics, lr_shap_used, lr_total = run_event_lr_shap(df_events, out_dir, args.shap_sample)
    print(f"[INFO] LR-Metriken: {lr_metrics}")

    print("[INFO] Berechne SHAP für Decision Tree …")
    dt_metrics, dt_shap_used, dt_total = run_event_dt_shap(df_events, out_dir, args.shap_sample)
    print(f"[INFO] DT-Metriken: {dt_metrics}")

    df_seq = pl.read_parquet(seq_path) if seq_path.exists() else None
    seq_metrics, seq_shap_used, seq_total = run_sequence_lr_shap(df_events, df_seq, out_dir, args.shap_sample)
    if seq_metrics:
        print(f"[INFO] Sequence-LR-Metriken: {seq_metrics}")

    total_anomalies = pred_if.filter(pl.col("pred_ano") == 1).height
    total_normals = pred_if.filter(pl.col("pred_ano") == 0).height
    downsampling_performed = False
    if (lr_total and lr_shap_used < lr_total) or (dt_total and dt_shap_used < dt_total):
        downsampling_performed = True
    if seq_total and seq_shap_used < seq_total:
        downsampling_performed = True
    if nn_top_used and nn_top_used < total_anomalies:
        downsampling_performed = True
    if nn_normal_used and nn_normal_used < total_normals:
        downsampling_performed = True

    print("\n[Summary] Explainability diagnostics:")
    print(
        f"  lr_shap_samples={lr_shap_used} total={lr_total} | dt_shap_samples={dt_shap_used} total={dt_total}"
    )
    print(f"  seq_shap_samples={seq_shap_used} total={seq_total}")
    print(
        f"  nn_top_used={nn_top_used} of {total_anomalies} anomalies | nn_normals_used={nn_normal_used} of {total_normals}"
    )
    print(f"[Summary] Downsampling occurred: {'yes' if downsampling_performed else 'no'}")

    print("[INFO] Phase-F-Artefakte fertig.")


if __name__ == "__main__":
    main()
