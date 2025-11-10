#!/usr/bin/env python3
"""Phase F helper to generate explainability artefacts for the LO2 MVP pipeline.

The script replays the best IsolationForest configuration (T4) and the supervised
baselines from Phase E, then saves NNExplainer mappings and SHAP plots/feature logs
under ``demo/result/lo2/explainability/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from LO2_samples import (
    MODEL_REGISTRY,
    DEFAULT_PREDICT_BATCH_SIZE as BASE_PREDICT_BATCH_SIZE,
    _prepare_model_configs,
    _run_based_holdout_split,
    _detect_available_ram_gb,
)
VECTOR_KWARGS = {
    "max_features": 5000,
    "min_df": 5,
    "binary": True,
    "dtype": np.float32,
    "strip_accents": "unicode",
}

DT_TRAIN_KWARGS = {
    "max_depth": 8,
    "min_samples_leaf": 10,
    "min_samples_split": 20,
    "max_leaf_nodes": 256,
    "max_features": 0.3,
    "random_state": 42,
}

DEFAULT_PREDICT_BATCH_SIZE = BASE_PREDICT_BATCH_SIZE
SHAP_CAPABLE_METHODS = {"train_LR", "train_DT", "train_RF", "train_XGB", "train_LSVM"}
RUN_DT_BASELINE = False


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
    parser.add_argument(
        "--skip-if",
        action="store_true",
        help="IsolationForest in Phase F überspringen (nur supervised Modelle verwenden).",
    )
    parser.add_argument(
        "--nn-source",
        default="if",
        help="Welches Modell für NN-Mapping und False-Positive-Liste genutzt wird (\"if\" oder Modellschlüssel).",
    )
    parser.add_argument(
        "--sup-models",
        default="",
        help="Kommagetrennte Liste an Modellschlüsseln (siehe --list-models), die für Explainability erneut trainiert werden.",
    )
    parser.add_argument(
        "--sup-holdout-fraction",
        type=float,
        default=0.0,
        help="Optionaler Hold-out-Anteil für supervised Modelle (0 = deaktiviert).",
    )
    parser.add_argument(
        "--sup-holdout-min-groups",
        type=int,
        default=1,
        help="Minimale Anzahl Gruppen pro Bucket für den Hold-out-Split.",
    )
    parser.add_argument(
        "--sup-holdout-shuffle",
        action="store_true",
        help="Hold-out-Split mischen statt nach Startzeit zu sortieren.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Deterministischer Seed für Vektorisierung und Hold-out.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=DEFAULT_PREDICT_BATCH_SIZE,
        help="Batchgröße für predict/predict_proba (0 = keine Chunking-Strategie).",
    )
    parser.add_argument(
        "--disable-memory-guard",
        action="store_true",
        help="Deaktiviert die RAM-basierten Schutzanpassungen beim Training der supervised Modelle.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Verfügbare Modellschlüssel ausgeben und beenden.",
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


def attach_row_ids(df: pl.DataFrame) -> pl.DataFrame:
    if "row_id" in df.columns:
        df = df.drop("row_id")
    return df.with_row_index(name="row_id")


def append_scores(pred_df: pl.DataFrame, detector: AnomalyDetector, model_key: str) -> tuple[pl.DataFrame, str | None]:
    score_values: list[float] | None = None
    score_col = None
    try:
        if hasattr(detector.model, "predict_proba"):
            raw = detector._batched_call(detector.model.predict_proba, detector.X_test)
            if raw.ndim > 1:
                score_values = raw[:, 1].tolist()
            else:
                score_values = raw.tolist()
        elif hasattr(detector.model, "decision_function"):
            raw = detector._batched_call(detector.model.decision_function, detector.X_test)
            score_values = raw.tolist()
        elif hasattr(detector.model, "score_samples"):
            raw = detector.model.score_samples(detector.X_test)
            score_values = (-raw).tolist()
    except Exception as exc:
        print(f"[WARN:{model_key}] Scores konnten nicht berechnet werden: {exc}")

    if score_values is None:
        return pred_df, None

    score_col = f"score_{model_key}"
    pred_df = pred_df.with_columns(pl.Series(score_col, score_values))
    pred_df = pred_df.with_columns(pl.col(score_col).rank("dense", descending=True).alias(f"rank_{model_key}"))
    return pred_df, score_col


def explain_detector_with_shap(
    detector: AnomalyDetector,
    model_key: str,
    out_dir: Path,
    sample_size: int,
    shap_kwargs: dict | None = None,
) -> tuple[int, int]:
    shap_kwargs = shap_kwargs or {}
    shap_expl = ShapExplainer(detector, **shap_kwargs)
    total = detector.X_test.shape[0]
    size = total if sample_size <= 0 else min(sample_size, total)
    shap_expl.calc_shapvalues(custom_slice=slice(0, size))
    save_top_features(shap_expl, 20, out_dir / f"{model_key}_top_features.txt")
    plot_shap(shap_expl, out_dir / f"{model_key}_shap")
    return size, total


def train_registry_models(
    df_seq: pl.DataFrame,
    model_keys: list[str],
    out_dir: Path,
    *,
    shap_sample: int,
    sample_seed: int,
    predict_batch_size: int,
    holdout_fraction: float,
    holdout_min_groups: int,
    holdout_shuffle: bool,
    available_ram_gb: float | None,
    memory_guard_enabled: bool,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not model_keys:
        return results

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            print(f"[WARN] Überspringe unbekanntes Modell '{model_key}'.")
            continue
        spec = MODEL_REGISTRY[model_key]
        dataset = df_seq
        if dataset is None or dataset.is_empty():
            print(f"[WARN:{model_key}] Sequenzdataset leer – Modell übersprungen.")
            continue

        train_df = dataset
        eval_df = dataset
        holdout_meta = {
            "applied": False,
            "holdout_rows": 0,
            "holdout_groups": 0,
            "reason": "",
        }
        if spec.get("train_selector") == "correct_only":
            if "test_case" in dataset.columns:
                filtered = dataset.filter(pl.col("test_case") == "correct")
                if filtered.is_empty():
                    print(f"[{model_key}] übersprungen (keine 'correct'-Beispiele für train_selector).")
                    continue
                train_df = filtered
                print(f"[{model_key}] train_selector=correct_only aktiv ({train_df.height} Zeilen).")
            else:
                print(f"[{model_key}] übersprungen (Spalte 'test_case' fehlt für train_selector).")
                continue
        elif holdout_fraction > 0:
            train_candidate, holdout_candidate, candidate_meta = _run_based_holdout_split(
                dataset,
                holdout_fraction,
                shuffle=holdout_shuffle,
                min_per_bucket=holdout_min_groups,
                rng_seed=sample_seed,
            )
            holdout_meta.update(candidate_meta)
            if candidate_meta.get("applied"):
                train_df = train_candidate
                eval_df = holdout_candidate
                print(
                    f"[{model_key}] Hold-out aktiv: {holdout_meta['holdout_groups']} Gruppen, "
                    f"{holdout_meta['holdout_rows']} Zeilen."
                )
            elif candidate_meta.get("reason"):
                print(f"[{model_key}] Hold-out übersprungen: {candidate_meta['reason']}")

        print(f"[{model_key}] {spec['description']}")
        train_kwargs = dict(spec.get("train_kwargs", {}))
        vectorizer_kwargs = dict(spec.get("vectorizer_kwargs", {})) if spec.get("vectorizer_kwargs") else None
        train_kwargs, vectorizer_kwargs, guard_notes = _prepare_model_configs(
            model_key,
            train_kwargs,
            vectorizer_kwargs,
            use_vectorizer=bool(spec.get("item_list_col")),
            available_ram_gb=available_ram_gb,
            memory_guard_enabled=memory_guard_enabled,
        )
        for note in guard_notes:
            print(f"  -> {note}")

        detector = AnomalyDetector(
            item_list_col=spec.get("item_list_col"),
            numeric_cols=spec.get("numeric_cols") or [],
            vectorizer_kwargs=vectorizer_kwargs,
            random_state=sample_seed,
            predict_batch_size=predict_batch_size,
        )
        detector.train_df = train_df
        detector.test_df = eval_df
        detector.prepare_train_test_data()

        train_kwargs_final = train_kwargs.copy()
        if spec.get("train_method") == "train_XGB":
            if holdout_meta.get("applied") and detector.labels_test:
                eval_y = np.asarray(detector.labels_test, dtype=np.int32)
                train_kwargs_final.setdefault("eval_set", [(detector.X_test, eval_y)])
                train_kwargs_final.setdefault("verbose", False)
            else:
                train_kwargs_final.pop("early_stopping_rounds", None)

        getattr(detector, spec["train_method"])(**train_kwargs_final)
        pred_df = detector.predict()
        pred_df = attach_row_ids(pred_df)
        pred_df, score_col = append_scores(pred_df, detector, model_key)
        metrics = compute_metrics(detector)
        save_json(out_dir / f"metrics_{model_key}.json", metrics)
        pred_path = out_dir / f"{model_key}_predictions.parquet"
        pred_df.write_parquet(pred_path)
        print(f"[INFO] Predictions gespeichert: {pred_path}")
        shap_used = 0
        shap_total = pred_df.height
        shap_supported = spec.get("requires_shap") or spec.get("train_method") in SHAP_CAPABLE_METHODS
        if shap_supported:
            try:
                shap_kwargs = spec.get("shap_kwargs", {})
                shap_used, shap_total = explain_detector_with_shap(
                    detector,
                    model_key,
                    out_dir,
                    shap_sample,
                    shap_kwargs=shap_kwargs,
                )
                print(f"[INFO] {model_key} SHAP samples: {shap_used} von {shap_total}")
            except Exception as exc:
                shap_used = 0
                print(f"[WARN:{model_key}] SHAP-Erzeugung fehlgeschlagen: {exc}")
        else:
            note = out_dir / f"{model_key}_shap_skipped.txt"
            write_lines(
                note,
                [
                    "SHAP wurde übersprungen, weil das Modell nicht von ShapExplainer unterstützt wird.",
                    f"Trainingsmethode: {spec.get('train_method')}",
                ],
            )
            print(f"[INFO] {model_key}: SHAP nicht verfügbar (Hinweis unter {note})")

        results[model_key] = {
            "detector": detector,
            "predictions": pred_df,
            "score_col": score_col,
            "metrics": metrics,
            "shap_used": shap_used,
            "shap_total": shap_total,
            "holdout": holdout_meta,
        }
    return results

def enhance_event_features(df_events: pl.DataFrame) -> pl.DataFrame:
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


def aggregate_sequence_features(df_seq: pl.DataFrame, df_events: pl.DataFrame) -> pl.DataFrame:
    seq_enhancer = SequenceEnhancer(df=df_events, df_seq=df_seq)
    df_seq = seq_enhancer.seq_len()
    df_seq = seq_enhancer.start_time()
    df_seq = seq_enhancer.duration()
    df_seq = seq_enhancer.tokens(token="e_words")
    df_seq = seq_enhancer.tokens(token="e_trigrams")
    if "e_event_drain_id" in df_events.columns:
        df_seq = seq_enhancer.events("e_event_drain_id")
    return df_seq


def load_sequence_dataset(root: Path) -> pl.DataFrame:
    seq_enhanced_path = root / "lo2_sequences_enhanced.parquet"
    seq_path = root / "lo2_sequences.parquet"
    events_path = root / "lo2_events.parquet"

    if seq_enhanced_path.exists():
        print(f"[INFO] Lade Sequenzen inkl. Features aus {seq_enhanced_path}")
        df_seq = pl.read_parquet(seq_enhanced_path)
    else:
        if not seq_path.exists():
            raise SystemExit(
                "Weder lo2_sequences_enhanced.parquet noch lo2_sequences.parquet gefunden. "
                "Führe run_lo2_loader.py mit --save-parquet aus."
            )
        print(f"[INFO] Lade Basis-Sequenzen aus {seq_path} und berechne Features on-the-fly.")
        df_seq = pl.read_parquet(seq_path)
        if not events_path.exists():
            raise SystemExit(
                "lo2_sequences_enhanced.parquet fehlt und es gibt kein lo2_events.parquet für das Feature-Engineering. "
                "Bitte run_lo2_loader.py mit --save-parquet (ggf. --save-events) ausführen."
            )
        df_events = pl.read_parquet(events_path)
        df_events = enhance_event_features(df_events)
        df_seq = aggregate_sequence_features(df_seq, df_events)

    if df_seq.is_empty():
        raise SystemExit("Sequenz-Tabelle ist leer – nichts zu erklären.")
    return df_seq


def train_if(df_seq: pl.DataFrame, args: argparse.Namespace):
    if "e_words" not in df_seq.columns:
        raise SystemExit("Sequenzdaten enthalten keine Spalte e_words. Führe run_lo2_loader.py erneut mit --save-parquet aus.")

    item_col = "e_words"
    numeric_candidates = ["seq_len", "duration_sec", "e_words_len", "e_trigrams_len"]
    numeric_cols = [col for col in numeric_candidates if col in df_seq.columns]
    sad_if = AnomalyDetector(
        item_list_col=item_col,
        numeric_cols=numeric_cols or None,
        vectorizer_kwargs=VECTOR_KWARGS.copy(),
        random_state=args.sample_seed,
        predict_batch_size=args.predict_batch_size,
    )
    sad_if.train_df = df_seq.filter(pl.col("test_case") == "correct")
    sad_if.test_df = df_seq

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
    predictions: pl.DataFrame,
    detector: AnomalyDetector,
    out_dir: Path,
    top_k: int,
    normal_sample: int,
    *,
    score_col: str | None,
    prefix: str,
) -> tuple[int, int]:
    anomalies = predictions.filter(pl.col("pred_ano") == 1)
    if score_col and score_col in anomalies.columns:
        anomalies = anomalies.sort(score_col, descending=True)
    else:
        anomalies = anomalies.sort("row_id")
    if top_k > 0 and anomalies.height > top_k:
        anomalies = anomalies.head(top_k)
    if anomalies.is_empty():
        print("[WARN] Keine Anomalien für das NN-Mapping vorhanden – Schritt übersprungen.")
        return (0, 0)

    normals = predictions.filter(pl.col("pred_ano") == 0)
    if score_col and score_col in normals.columns:
        normals = normals.sort(score_col, descending=True)
    else:
        normals = normals.sort("row_id")
    if normal_sample > 0 and normals.height > normal_sample:
        normals = normals.head(normal_sample)

    actual_top = anomalies.height
    actual_normals = normals.height

    subset = pl.concat([anomalies, normals], how="vertical").unique("row_id")
    indices = [int(idx) for idx in subset["row_id"].to_list()]

    X_full = detector.X_test
    if hasattr(X_full, "tocsr"):
        X_full = X_full.tocsr()
    X_subset = X_full[indices]

    nn = NNExplainer(subset, to_dense(X_subset), id_col="row_id", pred_col="pred_ano")
    mapping_path = out_dir / f"{prefix}_nn_mapping.csv"
    nn.mapping.write_csv(mapping_path)
    print(f"[INFO] NN-Mapping gespeichert: {mapping_path}")

    # False-Positive-Analyse sichern
    fp_rows = predictions.filter((pl.col("pred_ano") == 1) & (pl.col("anomaly") == 0))
    if fp_rows.height == 0:
        print("[INFO] Keine False Positives gefunden.")
    else:
        fp_text = []
        for row in fp_rows.iter_rows(named=True):
            token_list = row.get("e_words") or []
            if isinstance(token_list, list):
                words = " ".join(token_list)
            else:
                words = str(token_list)
            score_fragment = ""
            if score_col and score_col in fp_rows.columns:
                score_val = row.get(score_col)
                if isinstance(score_val, (int, float)):
                    score_fragment = f" | {score_col}={score_val:.6f}"
            fp_text.append(
                f"row_id={row['row_id']} | seq_id={row.get('seq_id')} | service={row.get('service')} | pred_ano={row['pred_ano']}{score_fragment}\n{words}"
            )
        fp_path = out_dir / f"{prefix}_false_positives.txt"
        write_lines(fp_path, fp_text)
        print(f"[INFO] False-Positive-Liste geschrieben: {fp_path}")

    print(
        f"[Explainability:{prefix}] NN mapping uses anomalies={actual_top} normals={actual_normals} "
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


def run_sequence_lr_tokens_shap(
    df_seq: pl.DataFrame, out_dir: Path, sample_size: int, predict_batch_size: int
) -> tuple[dict | None, int, int]:
    if "e_words" not in df_seq.columns:
        print("[INFO] e_words nicht vorhanden – LR-SHAP (Tokens) übersprungen.")
        return None, 0, 0

    sad = AnomalyDetector(
        item_list_col="e_words",
        vectorizer_kwargs=VECTOR_KWARGS.copy(),
        random_state=42,
        predict_batch_size=predict_batch_size,
    )
    sad.train_df = df_seq
    sad.test_df = df_seq
    sad.prepare_train_test_data()
    sad.train_LR()
    _ = sad.predict()

    shap_expl = ShapExplainer(sad, ignore_warning=True, plot_featurename_len=18)
    total = sad.X_test.shape[0]
    size = total if sample_size <= 0 else min(sample_size, total)
    shap_expl.calc_shapvalues(custom_slice=slice(0, size))
    save_top_features(shap_expl, 20, out_dir / "seq_lr_tokens_top_features.txt")
    plot_shap(shap_expl, out_dir / "seq_lr_tokens_shap")
    metrics = compute_metrics(sad)
    save_json(out_dir / "metrics_seq_lr_tokens.json", metrics)
    print(f"[Explainability] Sequence LR (Tokens) SHAP samples: {size} of {total} (requested {sample_size})")
    return metrics, size, total


def run_sequence_dt_shap(
    df_seq: pl.DataFrame, out_dir: Path, sample_size: int, predict_batch_size: int
) -> tuple[dict | None, int, int]:
    if "e_trigrams" not in df_seq.columns:
        print("[INFO] e_trigrams nicht vorhanden – Decision-Tree-SHAP übersprungen.")
        return None, 0, 0

    sad = AnomalyDetector(
        item_list_col="e_trigrams",
        vectorizer_kwargs=VECTOR_KWARGS.copy(),
        random_state=42,
        predict_batch_size=predict_batch_size,
    )
    sad.train_df = df_seq
    sad.test_df = df_seq
    sad.prepare_train_test_data()
    sad.train_DT(**DT_TRAIN_KWARGS)
    _ = sad.predict()

    shap_expl = ShapExplainer(sad, ignore_warning=True, plot_featurename_len=18)
    total = sad.X_test.shape[0]
    size = total if sample_size <= 0 else min(sample_size, total)
    shap_expl.calc_shapvalues(custom_slice=slice(0, size))
    save_top_features(shap_expl, 20, out_dir / "seq_dt_top_trigrams.txt")
    plot_shap(shap_expl, out_dir / "seq_dt_shap")
    metrics = compute_metrics(sad)
    save_json(out_dir / "metrics_seq_dt.json", metrics)
    print(f"[Explainability] Sequence DT SHAP samples: {size} of {total} (requested {sample_size})")
    return metrics, size, total


def run_sequence_lr_numeric_shap(
    df_seq: pl.DataFrame, out_dir: Path, sample_size: int, predict_batch_size: int
) -> tuple[dict | None, int, int]:
    required_cols = [col for col in ["seq_len", "duration_sec"] if col in df_seq.columns]
    if not required_cols:
        print("[INFO] Keine numerischen Sequenzfeatures (seq_len/duration_sec) vorhanden – Numeric-LR-SHAP übersprungen.")
        return None, 0, 0

    sad = AnomalyDetector(
        numeric_cols=required_cols,
        random_state=42,
        predict_batch_size=predict_batch_size,
    )
    sad.train_df = df_seq
    sad.test_df = df_seq
    sad.prepare_train_test_data()
    sad.train_LR()
    _ = sad.predict()

    metrics = compute_metrics(sad)
    save_json(out_dir / "metrics_seq_lr_numeric.json", metrics)

    if sad.vec is None:
        note_path = out_dir / "seq_lr_numeric_shap_skipped.txt"
        write_lines(
            note_path,
            [
                "SHAP wurde übersprungen, weil das Sequence-LR-Modell nur numerische Features nutzt",
                "und daher kein Vectorizer mit Feature-Namen vorhanden ist.",
            ],
        )
        print(f"[INFO] Sequence-LR (numeric) SHAP übersprungen; Hinweis unter {note_path}")
        return metrics, 0, sad.X_test.shape[0]

    shap_expl = ShapExplainer(sad, ignore_warning=True, plot_featurename_len=18)
    total = sad.X_test.shape[0]
    size = total if sample_size <= 0 else min(sample_size, total)
    shap_expl.calc_shapvalues(custom_slice=slice(0, size))
    save_top_features(shap_expl, 20, out_dir / "seq_lr_numeric_top_features.txt")
    plot_shap(shap_expl, out_dir / "seq_lr_numeric_shap")
    print(f"[Explainability] Sequence LR (numeric) SHAP samples: {size} of {total} (requested {sample_size})")
    return metrics, size, total


def main() -> None:
    args = parse_args()
    if args.list_models:
        print("Verfügbare Modelle für --sup-models / --nn-source:")
        for key in sorted(MODEL_REGISTRY):
            spec = MODEL_REGISTRY[key]
            print(f"  {key}: {spec['description']}")
        return

    os_seed = str(args.sample_seed)
    os.environ["PYTHONHASHSEED"] = os_seed
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)

    selected_models = [entry.strip() for entry in args.sup_models.split(",") if entry.strip()]
    nn_source = args.nn_source.strip()
    if nn_source and nn_source not in ("if", "none"):
        if nn_source not in MODEL_REGISTRY:
            raise SystemExit(f"Unbekannter nn-source Schlüssel: {nn_source}")
        if nn_source not in selected_models:
            selected_models.append(nn_source)
    selected_models = list(dict.fromkeys(selected_models))

    available_ram_gb = _detect_available_ram_gb()
    memory_guard_enabled = not args.disable_memory_guard
    if available_ram_gb is not None:
        print(f"[Guard] Verfügbare RAM (ca.): {available_ram_gb:.1f} GB")
    elif memory_guard_enabled:
        print("[Guard] psutil nicht verfügbar – Ressourcen-Guards verwenden Standardlimits.")

    if args.skip_if and nn_source == "if":
        if selected_models:
            nn_source = selected_models[0]
            print(f"[INFO] NN-Quelle automatisch auf '{nn_source}' gesetzt, da IsolationForest deaktiviert ist.")
        else:
            nn_source = "none"
            print("[WARN] --skip-if aktiv, aber keine supervised Modelle angegeben. NN-Mapping wird übersprungen.")

    root = args.root.resolve()
    out_dir = ensure_dir(root / "explainability")
    print(f"[INFO] Artefakte werden unter {out_dir} abgelegt.")

    print("[INFO] Lade Sequenzen …")
    df_seq = load_sequence_dataset(root)

    sad_if = None
    pred_if = None
    nn_top_used = 0
    nn_normal_used = 0
    nn_total_anomalies = 0
    nn_total_normals = 0
    mapping_source = None

    if args.skip_if:
        print("[INFO] IsolationForest-Lauf übersprungen (--skip-if).")
    else:
        print("[INFO] Trainiere IsolationForest (Phase D Setting)…")
        sad_if, pred_if = train_if(df_seq, args)
        pred_if_path = out_dir / "lo2_if_predictions.parquet"
        pred_if.write_parquet(pred_if_path)
        print(f"[INFO] IF-Predictions gespeichert: {pred_if_path}")

    print("[INFO] Berechne SHAP für Sequence-LR (Tokens) …")
    lr_metrics, lr_shap_used, lr_total = run_sequence_lr_tokens_shap(
        df_seq, out_dir, args.shap_sample, args.predict_batch_size
    )
    if lr_metrics:
        print(f"[INFO] Sequence-LR (Tokens) Metriken: {lr_metrics}")

    if RUN_DT_BASELINE:
        print("[INFO] Berechne SHAP für Sequence-Decision-Tree …")
        dt_metrics, dt_shap_used, dt_total = run_sequence_dt_shap(
            df_seq, out_dir, args.shap_sample, args.predict_batch_size
        )
        if dt_metrics:
            print(f"[INFO] Sequence-DT Metriken: {dt_metrics}")
    else:
        dt_metrics, dt_shap_used, dt_total = (None, 0, 0)
        print("[INFO] Sequence-Decision-Tree SHAP übersprungen (RUN_DT_BASELINE=False).")

    seq_num_metrics, seq_num_shap_used, seq_num_total = run_sequence_lr_numeric_shap(
        df_seq, out_dir, args.shap_sample, args.predict_batch_size
    )
    if seq_num_metrics:
        print(f"[INFO] Sequence-LR (numeric) Metriken: {seq_num_metrics}")

    registry_results = train_registry_models(
        df_seq,
        selected_models,
        out_dir,
        shap_sample=args.shap_sample,
        sample_seed=args.sample_seed,
        predict_batch_size=args.predict_batch_size,
        holdout_fraction=args.sup_holdout_fraction,
        holdout_min_groups=args.sup_holdout_min_groups,
        holdout_shuffle=args.sup_holdout_shuffle,
        available_ram_gb=available_ram_gb,
        memory_guard_enabled=memory_guard_enabled,
    )

    if nn_source == "if":
        if pred_if is None or sad_if is None:
            print("[WARN] NN-Mapping für IF nicht möglich (IsolationForest deaktiviert).")
        else:
            mapping_source = "if"
            nn_total_anomalies = pred_if.filter(pl.col("pred_ano") == 1).height
            nn_total_normals = pred_if.filter(pl.col("pred_ano") == 0).height
            nn_top_used, nn_normal_used = build_nn_mapping(
                pred_if,
                sad_if,
                out_dir,
                top_k=args.nn_top_k,
                normal_sample=args.nn_normal_sample,
                score_col="score_if",
                prefix="if",
            )
    elif nn_source == "none":
        print("[INFO] NN-Mapping deaktiviert (--nn-source=none).")
    else:
        model_entry = registry_results.get(nn_source)
        if not model_entry:
            print(f"[WARN] NN-Mapping-Quelle '{nn_source}' konnte nicht geladen werden (kein Ergebnis).")
        else:
            mapping_source = nn_source
            pred_sup = model_entry["predictions"]
            detector_sup = model_entry["detector"]
            nn_total_anomalies = pred_sup.filter(pl.col("pred_ano") == 1).height
            nn_total_normals = pred_sup.filter(pl.col("pred_ano") == 0).height
            nn_top_used, nn_normal_used = build_nn_mapping(
                pred_sup,
                detector_sup,
                out_dir,
                top_k=args.nn_top_k,
                normal_sample=args.nn_normal_sample,
                score_col=model_entry.get("score_col"),
                prefix=nn_source,
            )

    downsampling_performed = False
    if lr_total and lr_shap_used < lr_total:
        downsampling_performed = True
    if dt_total and dt_shap_used < dt_total:
        downsampling_performed = True
    if seq_num_total and seq_num_shap_used < seq_num_total:
        downsampling_performed = True
    if nn_top_used and nn_total_anomalies and nn_top_used < nn_total_anomalies:
        downsampling_performed = True
    if nn_normal_used and nn_total_normals and nn_normal_used < nn_total_normals:
        downsampling_performed = True
    for entry in registry_results.values():
        if entry["shap_total"] and entry["shap_used"] < entry["shap_total"]:
            downsampling_performed = True

    print("\n[Summary] Explainability diagnostics:")
    print(f"  seq_lr_tokens_shap_samples={lr_shap_used} total={lr_total}")
    print(f"  seq_dt_shap_samples={dt_shap_used} total={dt_total}")
    print(f"  seq_lr_numeric_shap_samples={seq_num_shap_used} total={seq_num_total}")
    for key, entry in registry_results.items():
        print(f"  {key}_shap_samples={entry['shap_used']} total={entry['shap_total']}")
        print(f"    metrics={entry['metrics']}")
    if mapping_source:
        print(
            f"  nn_top_used={nn_top_used} of {nn_total_anomalies} anomalies | "
            f"nn_normals_used={nn_normal_used} of {nn_total_normals} (source={mapping_source})"
        )
    else:
        print("  nn_mapping=skipped")
    print(f"[Summary] Downsampling occurred: {'yes' if downsampling_performed else 'no'}")

    print("[INFO] Phase-F-Artefakte fertig.")


if __name__ == "__main__":
    main()
