"""LO2 demo pipeline for enhancement, anomaly detection, and explainability."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import polars as pl

from loglead import AnomalyDetector
from loglead.enhancers import EventLogEnhancer, SequenceEnhancer
import loglead.explainer as ex
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

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

HARD_LIMITS = {
    "max_depth": 20,
    "n_estimators": 300,
    "max_features": 0.5,
    "max_leaf_nodes": 512,
    "max_leaves": 512,
}

VECTORIZER_DEFAULTS = {
    "dtype": np.float32,
    "binary": True,
    "strip_accents": "unicode",
    "max_df": 0.9,
    "min_df": 2,
    "max_features": 100_000,
}

DEFAULT_PREDICT_BATCH_SIZE = 50_000

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "event_lr_words": {
        "description": "LogisticRegression auf Sequenz-Worttokens (Bag-of-Words).",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LR",
        "stat_label": "event_lr_words",
    },
    "event_dt_trigrams": {
        "description": "DecisionTree auf Sequenz-Trigrams.",
        "level": "sequence",
        "item_list_col": "e_trigrams",
        "numeric_cols": [],
        "train_method": "train_DT",
        "train_kwargs": {
            "max_depth": 8,
            "min_samples_leaf": 10,
            "min_samples_split": 20,
            "max_leaf_nodes": 256,
            "max_features": 0.3,
            "random_state": 42,
        },
        "vectorizer_kwargs": {"max_features": 40000, "min_df": 5},
        "stat_label": "event_dt_trigrams",
    },
    "event_lsvm_words": {
        "description": "LinearSVM auf Sequenz-Worttokens.",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LSVM",
        "stat_label": "event_lsvm_words",
    },
    "event_rf_words": {
        "description": "RandomForest auf Sequenz-Worttokens.",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_RF",
        "train_kwargs": {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_leaf": 10,
            "min_samples_split": 20,
            "max_features": 0.3,
            "bootstrap": True,
            "n_jobs": 1,
            "random_state": 42,
        },
        "vectorizer_kwargs": {"max_features": 40000, "min_df": 5},
        "stat_label": "event_rf_words",
    },
    "event_xgb_words": {
        "description": "XGBoost Klassifikator auf Sequenz-Worttokens.",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_XGB",
        "train_kwargs": {
            "tree_method": "hist",
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "n_estimators": 120,
            "max_leaves": 256,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "random_state": 42,
        },
        "vectorizer_kwargs": {"max_features": 40000, "min_df": 5},
        "stat_label": "event_xgb_words",
    },
    "event_lof_words": {
        "description": "LocalOutlierFactor (novelty) auf Sequenz-Worttokens (trainiert nur auf korrekten Runs).",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_LOF",
        "train_kwargs": {"filter_anos": True},
        "train_selector": "correct_only",
        "stat_label": "event_lof_words",
    },
    "event_kmeans_words": {
        "description": "KMeans Clustering auf Sequenz-Worttokens (2 Cluster).",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_KMeans",
        "stat_label": "event_kmeans_words",
    },
    "event_oneclass_svm_words": {
        "description": "OneClassSVM auf Sequenz-Worttokens (trainiert nur auf korrekten Runs).",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_OneClassSVM",
        "train_selector": "correct_only",
        "stat_label": "event_oneclass_svm_words",
    },
    "event_rarity_words": {
        "description": "RarityModel auf Sequenz-Worttokens.",
        "level": "sequence",
        "item_list_col": "e_words",
        "numeric_cols": [],
        "train_method": "train_RarityModel",
        "stat_label": "event_rarity_words",
    },
    "event_oov_words": {
        "description": "OOVDetector für seltene Sequenz-Tokens (trainiert nur auf korrekten Runs).",
        "level": "sequence",
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


def _detect_available_ram_gb() -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


def _clamp_numeric(
    model_key: str,
    params: Dict[str, Any],
    key: str,
    adjustments: List[str],
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    clamp_int: bool = False,
) -> None:
    if key not in params:
        return
    value = params[key]
    if value is None:
        return
    if isinstance(value, str):
        return
    new_value = value
    if min_val is not None and value < min_val:
        new_value = min_val
    if max_val is not None and value > max_val:
        new_value = max_val
    if clamp_int:
        new_value = int(new_value)
    if new_value != value:
        params[key] = new_value
        adjustments.append(
            f"[Guard:{model_key}] {key} adjusted from {value} to {new_value}"
        )


def _sanitize_train_kwargs(model_key: str, raw_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    sanitized = raw_kwargs.copy()
    notes: List[str] = []

    _clamp_numeric(model_key, sanitized, "max_depth", notes, max_val=HARD_LIMITS["max_depth"])
    _clamp_numeric(model_key, sanitized, "n_estimators", notes, max_val=HARD_LIMITS["n_estimators"], clamp_int=True)
    _clamp_numeric(model_key, sanitized, "max_features", notes, min_val=0.05, max_val=HARD_LIMITS["max_features"])
    _clamp_numeric(model_key, sanitized, "max_leaf_nodes", notes, max_val=HARD_LIMITS["max_leaf_nodes"], clamp_int=True)
    _clamp_numeric(model_key, sanitized, "max_leaves", notes, max_val=HARD_LIMITS["max_leaves"], clamp_int=True)

    if sanitized.get("min_samples_leaf") is not None:
        _clamp_numeric(model_key, sanitized, "min_samples_leaf", notes, min_val=2, clamp_int=True)
    if sanitized.get("min_samples_split") is not None:
        _clamp_numeric(model_key, sanitized, "min_samples_split", notes, min_val=4, clamp_int=True)

    if model_key == "event_rf_words":
        sanitized.setdefault("max_samples", 0.8)
        sanitized.setdefault("warm_start", True)
        sanitized.setdefault("n_jobs", 1)
    if model_key == "event_xgb_words":
        sanitized.setdefault("tree_method", "hist")
        sanitized.setdefault("max_bin", 256)
        sanitized.setdefault("grow_policy", "lossguide")
        sanitized.setdefault("early_stopping_rounds", 30)
        sanitized.setdefault("eval_metric", "logloss")
        sanitized.setdefault("n_jobs", 1)
        _clamp_numeric(model_key, sanitized, "max_bin", notes, min_val=32, max_val=256, clamp_int=True)
        _clamp_numeric(model_key, sanitized, "subsample", notes, min_val=0.5, max_val=0.9)
        _clamp_numeric(model_key, sanitized, "colsample_bytree", notes, min_val=0.4, max_val=0.9)

    return sanitized, notes


def _sanitize_vectorizer_kwargs(
    model_key: str,
    raw_kwargs: Dict[str, Any] | None,
    *,
    use_vectorizer: bool,
) -> Tuple[Dict[str, Any] | None, List[str]]:
    if not use_vectorizer:
        return None, []
    sanitized = VECTORIZER_DEFAULTS.copy()
    if raw_kwargs:
        sanitized.update(raw_kwargs)
    notes: List[str] = []

    max_features = sanitized.get("max_features")
    if max_features is not None:
        max_features = int(max_features)
        if max_features > VECTORIZER_DEFAULTS["max_features"]:
            notes.append(
                f"[Guard:{model_key}] vectorizer max_features adjusted from {max_features} to {VECTORIZER_DEFAULTS['max_features']}"
            )
            max_features = VECTORIZER_DEFAULTS["max_features"]
        sanitized["max_features"] = max_features

    min_df = sanitized.get("min_df", VECTORIZER_DEFAULTS["min_df"])
    if isinstance(min_df, (int, float)) and min_df < VECTORIZER_DEFAULTS["min_df"]:
        notes.append(
            f"[Guard:{model_key}] vectorizer min_df raised to {VECTORIZER_DEFAULTS['min_df']}"
        )
        sanitized["min_df"] = VECTORIZER_DEFAULTS["min_df"]

    sanitized.setdefault("token_pattern", r"(?u)\b\w\w+\b")
    sanitized.setdefault("binary", True)
    sanitized.setdefault("dtype", np.float32)

    return sanitized, notes


def _apply_memory_guard(
    model_key: str,
    train_kwargs: Dict[str, Any],
    vectorizer_kwargs: Dict[str, Any] | None,
    available_gb: float | None,
) -> List[str]:
    adjustments: List[str] = []
    if available_gb is None:
        return adjustments

    if available_gb < 8:
        if train_kwargs.get("max_depth") and train_kwargs["max_depth"] > 10:
            adjustments.append(f"[Guard:{model_key}] max_depth tightened for low RAM ({available_gb:.1f} GB available)")
            train_kwargs["max_depth"] = 10
        if vectorizer_kwargs and vectorizer_kwargs.get("max_features", 0) > 30000:
            adjustments.append(f"[Guard:{model_key}] vectorizer max_features tightened to 30000 due to RAM constraints")
            vectorizer_kwargs["max_features"] = 30000
    if available_gb < 4:
        if train_kwargs.get("n_estimators") and train_kwargs["n_estimators"] > 120:
            adjustments.append(f"[Guard:{model_key}] n_estimators cut to 120 due to very low RAM")
            train_kwargs["n_estimators"] = 120
        if vectorizer_kwargs and vectorizer_kwargs.get("max_features", 0) > 20000:
            vectorizer_kwargs["max_features"] = 20000
            adjustments.append(f"[Guard:{model_key}] vectorizer max_features capped at 20000")
    if train_kwargs.get("n_jobs", 1) != 1:
        train_kwargs["n_jobs"] = 1
        adjustments.append(f"[Guard:{model_key}] n_jobs forced to 1 to control peak memory")
    return adjustments


def _prepare_model_configs(
    model_key: str,
    raw_train_kwargs: Dict[str, Any],
    raw_vectorizer_kwargs: Dict[str, Any] | None,
    *,
    use_vectorizer: bool,
    available_ram_gb: float | None,
    memory_guard_enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None, List[str]]:
    train_kwargs, notes = _sanitize_train_kwargs(model_key, raw_train_kwargs)
    vectorizer_kwargs, vector_notes = _sanitize_vectorizer_kwargs(
        model_key,
        raw_vectorizer_kwargs,
        use_vectorizer=use_vectorizer,
    )
    adjustments = notes + vector_notes
    if memory_guard_enabled:
        adjustments += _apply_memory_guard(model_key, train_kwargs, vectorizer_kwargs, available_ram_gb)
    return train_kwargs, vectorizer_kwargs, adjustments


def _estimate_model_size_mb(model: Any) -> float | None:
    buffer = io.BytesIO()
    try:
        joblib.dump(model, buffer, compress=3)
    except Exception:
        return None
    size_mb = buffer.tell() / (1024 ** 2)
    return round(size_mb, 4)


def _collect_tree_stats(model: Any) -> List[str]:
    stats: List[str] = []
    if hasattr(model, "get_depth"):
        try:
            stats.append(f"depth={model.get_depth()}")
        except Exception:
            pass
    if hasattr(model, "estimators_"):
        depths = []
        for est in getattr(model, "estimators_", []):
            with suppress(Exception):
                depths.append(est.get_depth())
        if depths:
            stats.append(f"avg_depth={np.mean(depths):.1f}")
            stats.append(f"trees={len(depths)}")
    if hasattr(model, "n_estimators"):
        try:
            stats.append(f"n_estimators={model.n_estimators}")
        except Exception:
            pass
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            stats.append(f"trees={booster.best_ntree_limit or booster.num_boosted_rounds()}")
        except Exception:
            pass
    return stats


def _log_model_resource_stats(
    label: str,
    detector: AnomalyDetector,
    train_kwargs: Dict[str, Any],
    vectorizer_kwargs: Dict[str, Any] | None,
    elapsed: float,
) -> None:
    feature_count = detector.X_train.shape[1] if detector.X_train is not None else None
    vocab_size = None
    vec = detector.vec
    if vec is not None and hasattr(vec, "vocabulary_") and vec.vocabulary_ is not None:
        vocab_size = len(vec.vocabulary_)
    model_size_mb = _estimate_model_size_mb(detector.model)

    parts = [f"time={elapsed:.2f}s"]
    if feature_count is not None:
        parts.append(f"features={feature_count}")
    if vocab_size is not None:
        parts.append(f"vocab={vocab_size}")
    if model_size_mb is not None:
        parts.append(f"size_mb={model_size_mb:.2f}")
    parts.extend(_collect_tree_stats(detector.model))
    if vectorizer_kwargs and "max_features" in vectorizer_kwargs:
        parts.append(f"vec_max_features={vectorizer_kwargs['max_features']}")
    if train_kwargs.get("n_estimators"):
        parts.append(f"n_estimators={train_kwargs['n_estimators']}")
    print(f"[Resource] {label}: " + ", ".join(parts))


def _infer_time_column(df: pl.DataFrame) -> str | None:
    """Return the most specific timestamp column available for temporal ordering."""
    for candidate in ("start_time", "m_timestamp", "event_time", "timestamp"):
        if candidate in df.columns:
            return candidate
    return None


def _run_based_holdout_split(
    df: pl.DataFrame,
    fraction: float,
    *,
    shuffle: bool = False,
    min_per_bucket: int = 1,
    rng_seed: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, dict[str, Any]]:
    """
    Split a DataFrame into train/hold-out partitions based on run groupings.

    Groups are formed on (service, run, test_case) when present to avoid leakage between
    services or label strata. The newest groups per bucket (based on timestamps) are
    reserved for hold-out evaluation unless ``shuffle`` is requested.
    """
    meta: dict[str, Any] = {
        "applied": False,
        "reason": "",
        "total_groups": 0,
        "holdout_groups": 0,
        "train_groups": 0,
        "holdout_rows": 0,
        "train_rows": df.height,
    }
    if fraction <= 0:
        meta["reason"] = "Hold-out deaktiviert (Bruchteil ≤ 0)."
        return df, None, meta
    if df.is_empty():
        meta["reason"] = "Dataset leer."
        return df, None, meta
    if "run" not in df.columns:
        if fraction <= 0:
            meta["reason"] = "Spalte 'run' fehlt und Hold-out deaktiviert."
            return df, None, meta
        fallback_key = None
        for candidate in ("seq_id", "id"):
            if candidate in df.columns:
                fallback_key = candidate
                break
        if fallback_key is None:
            meta["reason"] = "Spalte 'run' fehlt und kein eindeutiger Schlüssel vorhanden."
            return df, None, meta

        df_indexed = df.with_row_count("__row_id")
        rng = random.Random(rng_seed)
        holdout_ids: list[int] = []

        if "anomaly" in df.columns:
            anomaly_values = df["anomaly"].unique().to_list()
            for value in anomaly_values:
                group = df_indexed.filter(pl.col("anomaly") == value)
                group_ids = group["__row_id"].to_list()
                if len(group_ids) <= 1:
                    continue
                if shuffle:
                    rng.shuffle(group_ids)
                else:
                    group_ids.sort()
                group_holdout = max(1, int(len(group_ids) * fraction))
                if group_holdout >= len(group_ids):
                    group_holdout = len(group_ids) - 1
                if group_holdout > 0:
                    holdout_ids.extend(group_ids[-group_holdout:])

        if not holdout_ids:
            all_ids = df_indexed["__row_id"].to_list()
            if len(all_ids) <= 1:
                meta["reason"] = "Dataset zu klein für Hold-out ohne 'run'."
                return df, None, meta
            if shuffle:
                rng.shuffle(all_ids)
            else:
                all_ids.sort()
            fallback_holdout = max(1, int(len(all_ids) * fraction))
            if fallback_holdout >= len(all_ids):
                fallback_holdout = len(all_ids) - 1
            holdout_ids = all_ids[-fallback_holdout:]

        holdout_df = df_indexed.filter(pl.col("__row_id").is_in(holdout_ids)).drop("__row_id")
        train_df = df_indexed.filter(~pl.col("__row_id").is_in(holdout_ids)).drop("__row_id")
        if holdout_df.is_empty() or train_df.is_empty():
            meta["reason"] = "Fallback-Hold-out ohne 'run' führte zu leerem Split."
            return df, None, meta

        meta.update(
            {
                "applied": True,
                "reason": "Fallback-Hold-out ohne 'run' genutzt.",
                "holdout_groups": 1,
                "train_groups": 1,
                "holdout_rows": holdout_df.height,
                "train_rows": train_df.height,
            }
        )
        return train_df, holdout_df, meta

    fraction = max(0.0, min(float(fraction), 0.5))
    min_per_bucket = max(1, int(min_per_bucket))

    group_cols: list[str] = []
    if "service" in df.columns:
        group_cols.append("service")
    group_cols.append("run")
    if "test_case" in df.columns:
        group_cols.append("test_case")

    time_col = _infer_time_column(df)
    if time_col:
        group_meta = df.group_by(group_cols).agg(pl.col(time_col).min().alias("_first_ts"))
    else:
        group_meta = df.group_by(group_cols).agg(pl.len().alias("_group_size"))
    meta_rows = group_meta.to_dicts()
    total_groups = len(meta_rows)
    meta["total_groups"] = total_groups
    if total_groups <= 1:
        meta["reason"] = "Nur eine Gruppe vorhanden."
        return df, None, meta

    bucket_cols = [col for col in group_cols if col != "run"]
    if not bucket_cols:
        bucket_cols = ["__ALL__"]

    rng = random.Random(rng_seed)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in meta_rows:
        bucket_key = (
            tuple(row[col] for col in bucket_cols)
            if bucket_cols != ["__ALL__"]
            else ("__ALL__",)
        )
        buckets.setdefault(bucket_key, []).append(row)

    holdout_groups: list[dict[str, Any]] = []
    train_groups: list[dict[str, Any]] = []
    for rows in buckets.values():
        if shuffle:
            rng.shuffle(rows)
        else:
            if time_col and "_first_ts" in rows[0]:
                rows.sort(key=lambda item: (item.get("_first_ts"), item["run"]))
            else:
                rows.sort(key=lambda item: item["run"])
        if len(rows) <= 1:
            train_groups.extend(rows)
            continue
        holdout_size = max(min_per_bucket, int(len(rows) * fraction))
        if holdout_size >= len(rows):
            holdout_size = len(rows) - 1
        if holdout_size <= 0:
            train_groups.extend(rows)
            continue
        train_groups.extend(rows[:-holdout_size])
        holdout_groups.extend(rows[-holdout_size:])

    if not holdout_groups:
        meta["reason"] = "Zu wenige Gruppen für Hold-out."
        return df, None, meta

    key_cols = [col for col in group_cols]
    holdout_key_dicts = [{col: row[col] for col in key_cols} for row in holdout_groups]
    holdout_key_df = pl.DataFrame(holdout_key_dicts)
    holdout_df = df.join(holdout_key_df, on=key_cols, how="inner")
    train_df = df.join(holdout_key_df, on=key_cols, how="anti")
    if holdout_df.is_empty() or train_df.is_empty():
        meta["reason"] = "Hold-out würde Training oder Test entleeren."
        return df, None, meta

    meta.update(
        {
            "applied": True,
            "holdout_groups": len(holdout_groups),
            "train_groups": len(train_groups),
            "holdout_rows": holdout_df.height,
            "train_rows": train_df.height,
        }
    )
    return train_df, holdout_df, meta


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
        "--predict-batch-size",
        type=int,
        default=DEFAULT_PREDICT_BATCH_SIZE,
        help="Batchgröße für predict/predict_proba (0 = keine Chunking-Strategie).",
    )
    parser.add_argument(
        "--sup-holdout-fraction",
        type=float,
        default=0.2,
        help="Anteil der Run-Gruppen (pro Service/Test-Case), die für supervised Modelle als Hold-out reserviert werden. 0 deaktiviert den Split.",
    )
    parser.add_argument(
        "--sup-holdout-min-groups",
        type=int,
        default=1,
        help="Mindestanzahl von Gruppen pro Bucket, die in den Hold-out fallen dürfen (gesetzt auf ≥1).",
    )
    parser.add_argument(
        "--sup-holdout-shuffle",
        action="store_true",
        help="Wählt Hold-out-Gruppen zufällig (statt zeitbasierter Auswahl). Nutzt --sample-seed.",
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
        "--skip-if",
        action="store_true",
        help="Überspringt den IsolationForest-Baseline-Schritt (Phase D).",
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
        help="Persist the enhanced sequence table to Parquet.",
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
        help="Optional fraction (0-0.5) of 'correct' sequences reserved as temporal hold-out.",
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
        "--disable-memory-guard",
        action="store_true",
        help="Deaktiviert die adaptive RAM-Guard-Logik für Baum- und Vectorizer-Parameter.",
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

    os.environ["PYTHONHASHSEED"] = str(args.sample_seed)
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    available_ram_gb = _detect_available_ram_gb()
    memory_guard_enabled = not args.disable_memory_guard
    if available_ram_gb is not None:
        print(f"[Guard] Available RAM (approx.): {available_ram_gb:.1f} GB")
    if memory_guard_enabled and available_ram_gb is None:
        print("[Guard] psutil nicht verfügbar – Ressourcenbegrenzungen basieren auf statischen Limits.")

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
    seq_enhanced_path = loader_output / "lo2_sequences_enhanced.parquet"

    df_events: pl.DataFrame | None = None
    sequences_enhanced = False

    if seq_enhanced_path.exists():
        print(f"Reading enhanced LO2 sequences from {seq_enhanced_path}")
        df_seqs = pl.read_parquet(seq_enhanced_path)
        sequences_enhanced = True
    else:
        if not seq_path.exists():
            raise SystemExit(
                "Missing sequence export. Run run_lo2_loader.py with --save-parquet to generate "
                "lo2_sequences_enhanced.parquet."
            )
        print(f"Reading base LO2 sequences from {seq_path}")
        df_seqs = pl.read_parquet(seq_path)
        if events_path.exists():
            print(f"Reading LO2 events from {events_path} for enhancement")
            df_events = pl.read_parquet(events_path)
        else:
            raise SystemExit(
                "Sequences need enhancement (words/trigrams), but no enhanced parquet or event table found. "
                "Re-run run_lo2_loader.py with --save-parquet so that lo2_sequences_enhanced.parquet is created; "
                "if necessary add --save-events (and optionally --save-base-sequences)."
            )

    if df_seqs.is_empty():
        raise SystemExit("Sequence table is empty; cannot continue with sequence-based pipeline.")

    seq_ano = int(df_seqs["anomaly"].sum()) if "anomaly" in df_seqs.columns else 0
    print(f"Sequence anomalies: {seq_ano} ({seq_ano / max(len(df_seqs), 1) * 100:.2f}%)")

    downsampling_performed = False
    train_stats: list[dict[str, Any]] = []

    if not sequences_enhanced:
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

        print("\nAggregating to sequence level...")
        seq_enhancer = SequenceEnhancer(df=df_events, df_seq=df_seqs)
        df_seqs = seq_enhancer.seq_len()
        df_seqs = seq_enhancer.start_time()
        df_seqs = seq_enhancer.duration()
        df_seqs = seq_enhancer.tokens(token="e_words")
        df_seqs = seq_enhancer.tokens(token="e_trigrams")
        if "e_event_drain_id" in df_events.columns:
            df_seqs = seq_enhancer.events("e_event_drain_id")
        df_events = None

    random.seed(args.sample_seed)
    sample_seq = df_seqs.sample(n=1, seed=args.sample_seed)
    sample_dict = sample_seq.to_dicts()[0]
    words = sample_dict.get("e_words") or []
    trigrams = sample_dict.get("e_trigrams") or []
    print("\nSample sequence (post-enhancement):")
    print(
        f"Seq ID: {sample_dict.get('seq_id')} | service: {sample_dict.get('service')} | "
        f"test_case: {sample_dict.get('test_case')}"
    )
    print(f"Length: seq_len={sample_dict.get('seq_len')} duration_sec={sample_dict.get('duration_sec')}")
    print(f"Words[{len(words)}]: {words[:10]}")
    print(f"Trigrams[{len(trigrams)}]: {trigrams[:10]}")

    if args.save_enhancers:
        enhancer_dir = args.enhancers_output_dir
        if not enhancer_dir.is_absolute():
            enhancer_dir = (orig_cwd / enhancer_dir).resolve()
        enhancer_dir.mkdir(parents=True, exist_ok=True)

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

    run_if_phase = args.phase in ("if", "full") and not args.skip_if
    if run_if_phase:
        print("\nTraining/Loading Isolation Forest on sequence tokens (Phase D)")
        numeric_cols = [col.strip() for col in args.if_numeric.split(",") if col.strip()]
        if_vectorizer_kwargs, if_vectorizer_notes = _sanitize_vectorizer_kwargs(
            "if_baseline",
            {"max_features": 60000, "min_df": 3},
            use_vectorizer=bool(args.if_item),
        )
        for note in if_vectorizer_notes:
            print(f"  -> {note}")
        sad_if = AnomalyDetector(
            item_list_col=args.if_item,
            numeric_cols=numeric_cols or None,
            vectorizer_kwargs=if_vectorizer_kwargs,
            random_state=args.sample_seed,
            predict_batch_size=args.predict_batch_size,
        )
        # IsolationForest learns only from normal runs; keep anomalies in test_df for evaluation.
        if "test_case" in df_seqs.columns:
            correct_sequences = df_seqs.filter(pl.col("test_case") == "correct")
        else:
            print("Warnung: Spalte 'test_case' fehlt im Sequenz-Export; verwende Labels für Training, falls verfügbar.")
            if "anomaly" in df_seqs.columns:
                correct_sequences = df_seqs.filter(pl.col("anomaly") == 0)
            else:
                print("  -> Keine Anomalielabels gefunden; benutze alle Sequenzen für das IF-Training.")
                correct_sequences = df_seqs
        holdout_fraction = min(max(args.if_holdout_fraction, 0.0), 0.5)
        holdout_df = None
        if holdout_fraction > 0 and correct_sequences.height > 1:
            if "start_time" in correct_sequences.columns:
                sorted_correct = correct_sequences.sort("start_time")
            else:
                sorted_correct = correct_sequences.sort("seq_id")
            holdout_size = max(1, int(sorted_correct.height * holdout_fraction))
            if holdout_size >= sorted_correct.height:
                holdout_size = sorted_correct.height - 1
            if holdout_size > 0:
                holdout_df = sorted_correct.tail(holdout_size)
                correct_sequences = sorted_correct.head(sorted_correct.height - holdout_size)
                print(
                    f"Using temporal hold-out: {holdout_size} sequences reserved ({holdout_fraction * 100:.2f}% of correct runs)."
                )
                downsampling_performed = True
        sad_if.train_df = correct_sequences
        sad_if.test_df = df_seqs

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

        if_train_kwargs = {
            "n_estimators": args.if_n_estimators,
            "contamination": args.if_contamination,
            "max_samples": max_samples,
        }
        fit_elapsed = 0.0
        if not model_loaded:
            start_fit = time.perf_counter()
            sad_if.train_IsolationForest(
                filter_anos=True,
                n_estimators=args.if_n_estimators,
                contamination=args.if_contamination,
                max_samples=max_samples,
            )
            fit_elapsed = time.perf_counter() - start_fit
        if hasattr(sad_if, "model") and sad_if.model is not None:
            _log_model_resource_stats("if_baseline", sad_if, if_train_kwargs, if_vectorizer_kwargs, fit_elapsed)
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
            joblib.dump((sad_if.model, sad_if.vec), model_path, compress=3)
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
    else:
        if args.phase in ("if", "full"):
            print("\nIsolation Forest übersprungen (--skip-if aktiviert).")
            if args.dump_metadata:
                print("  -> Hinweis: --dump-metadata greift nur, wenn ein IF-Modell gespeichert wird.")

    if args.phase == "if":
        if args.skip_if:
            print("\nIsolation Forest wurde übersprungen (--skip-if); keine weiteren Phasen aktiv.")
        else:
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

        original_rows = dataset.height
        train_df = dataset
        eval_df = dataset
        holdout_meta: dict[str, Any] = {
            "applied": False,
            "holdout_rows": 0,
            "holdout_groups": 0,
            "reason": "",
        }
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
        elif args.sup_holdout_fraction > 0:
            train_candidate, holdout_candidate, holdout_meta_candidate = _run_based_holdout_split(
                dataset,
                args.sup_holdout_fraction,
                shuffle=args.sup_holdout_shuffle,
                min_per_bucket=args.sup_holdout_min_groups,
                rng_seed=args.sample_seed,
            )
            if holdout_meta_candidate.get("reason") and not holdout_meta_candidate["applied"]:
                print(f"  -> Hold-out übersprungen: {holdout_meta_candidate['reason']}")
            if holdout_meta_candidate["applied"]:
                split_valid = True
                reason = ""
                if "anomaly" in train_candidate.columns:
                    train_anomalies = int(train_candidate["anomaly"].sum())
                    holdout_anomalies = int(holdout_candidate["anomaly"].sum())
                    if train_anomalies == 0:
                        split_valid = False
                        reason = "keine Anomalien im Training"
                    elif holdout_anomalies == 0:
                        split_valid = False
                        reason = "keine Anomalien im Hold-out"
                if split_valid:
                    train_df = train_candidate
                    eval_df = holdout_candidate
                    holdout_meta = holdout_meta_candidate
                    print(
                        f"  -> Hold-out aktiv: {holdout_meta['holdout_groups']} Gruppen, "
                        f"{holdout_meta['holdout_rows']} Zeilen."
                    )
                else:
                    print(f"  -> Hold-out verworfen ({reason}).")

        print(f"\n[{model_key}] {spec['description']}")
        item_list_col = spec.get("item_list_col")
        numeric_cols = spec.get("numeric_cols")
        raw_train_kwargs = dict(spec.get("train_kwargs", {}))
        raw_vectorizer_kwargs = dict(spec.get("vectorizer_kwargs", {})) if spec.get("vectorizer_kwargs") else None
        train_kwargs, vectorizer_kwargs, guard_notes = _prepare_model_configs(
            model_key,
            raw_train_kwargs,
            raw_vectorizer_kwargs,
            use_vectorizer=bool(item_list_col),
            available_ram_gb=available_ram_gb,
            memory_guard_enabled=memory_guard_enabled,
        )
        for note in guard_notes:
            print(f"  -> {note}")

        detector = AnomalyDetector(
            item_list_col=item_list_col,
            numeric_cols=numeric_cols if numeric_cols is not None else [],
            vectorizer_kwargs=vectorizer_kwargs,
            random_state=args.sample_seed,
            predict_batch_size=args.predict_batch_size,
        )
        detector.train_df = train_df
        detector.test_df = eval_df
        detector.prepare_train_test_data()

        train_kwargs_final = train_kwargs.copy()
        if spec["train_method"] == "train_XGB":
            if holdout_meta.get("applied") and detector.labels_test:
                eval_y = np.asarray(detector.labels_test, dtype=np.int32)
                train_kwargs_final.setdefault("eval_set", [(detector.X_test, eval_y)])
                train_kwargs_final.setdefault("verbose", False)
            else:
                train_kwargs_final.pop("early_stopping_rounds", None)

        start_fit = time.perf_counter()
        getattr(detector, spec["train_method"])(**train_kwargs_final)
        fit_elapsed = time.perf_counter() - start_fit
        detector.predict()
        _log_model_resource_stats(model_key, detector, train_kwargs_final, vectorizer_kwargs, fit_elapsed)
        if holdout_meta.get("applied") and detector.labels_train and detector.labels_test:
            try:
                train_pred = detector._batched_call(detector.model.predict, detector.X_train)
                holdout_pred = detector._batched_call(detector.model.predict, detector.X_test)
                train_acc = accuracy_score(detector.labels_train, train_pred)
                holdout_acc = accuracy_score(detector.labels_test, holdout_pred)
                if train_acc - holdout_acc > 0.01:
                    drop = train_acc - holdout_acc
                    print(
                        f"[Guard:{model_key}] Hold-out Accuracy drop {drop:.3f} (train={train_acc:.3f}, holdout={holdout_acc:.3f})"
                    )
            except Exception:
                pass
        train_rows = detector.train_df.height if detector.train_df is not None else 0
        test_rows = detector.test_df.height if detector.test_df is not None else 0
        entry = {
            "label": spec["stat_label"],
            "train_rows": train_rows,
            "test_rows": test_rows,
            "original_rows": original_rows,
            "holdout_applied": holdout_meta["applied"],
            "holdout_rows": holdout_meta["holdout_rows"],
            "holdout_groups": holdout_meta["holdout_groups"],
            "guard_notes": list(guard_notes),
        }
        train_stats.append(entry)
        log_total_rows = (
            train_rows + holdout_meta["holdout_rows"]
            if holdout_meta["applied"]
            else original_rows
        )
        _log_train_fraction(spec["stat_label"], train_rows, log_total_rows)

        if spec.get("requires_shap"):
            shap_kwargs = spec.get("shap_kwargs", {})
            shap_plot_type = spec.get("shap_plot_type", "summary")
            print("  -> SHAP-Erklärungen werden berechnet.")
            explainer = ex.ShapExplainer(detector, **shap_kwargs)
            explainer.calc_shapvalues()
            explainer.plot(plottype=shap_plot_type)

    if df_seqs is None or df_seqs.is_empty():
        print("\nNo sequence table available; skipping sequence-level models.")

    if train_stats:
        print("\n[Summary] Full-data pipeline diagnostics:")
        for entry in train_stats:
            label = entry["label"]
            train_rows = entry["train_rows"]
            test_rows = entry["test_rows"]
            msg = f"  {label}: train_rows={train_rows} test_rows={test_rows}"
            if entry.get("holdout_applied"):
                msg += (
                    f" holdout_rows={entry['holdout_rows']} "
                    f"holdout_groups={entry['holdout_groups']}"
                )
            else:
                msg += " (kein Hold-out)"
            print(msg)
            guard_notes_entry = entry.get("guard_notes") or []
            for note in guard_notes_entry:
                print(f"    {note}")
    print(f"[Summary] Downsampling occurred: {'yes' if downsampling_performed else 'no'}")

    print("\nLO2 sample pipeline complete.")


if __name__ == "__main__":
    main()
