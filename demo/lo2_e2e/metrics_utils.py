"""Utility metrics for LO2 Isolation Forest evaluation."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import polars as pl


def _to_array(series: Sequence[float]) -> np.ndarray:
    """Convert a sequence into a numpy array."""
    if isinstance(series, np.ndarray):
        return series
    return np.asarray(list(series), dtype=float)


def precision_at_k(
    df: pl.DataFrame,
    k: int,
    *,
    score_col: str = "score_if",
    label_col: str = "anomaly",
) -> Optional[float]:
    """Compute Precision@k given anomaly scores."""
    if k <= 0 or df.is_empty():
        return None
    subset_size = min(k, df.height)
    tops = df.sort(score_col, descending=True).head(subset_size)
    positives = (
        tops.select(pl.col(label_col).cast(pl.Int64)).to_series(0).sum()
    )
    if subset_size == 0:
        return None
    return float(positives) / float(subset_size)


def false_positive_rate_at_alpha(
    df: pl.DataFrame,
    alpha: float,
    *,
    score_col: str = "score_if",
    label_col: str = "anomaly",
) -> Optional[float]:
    """Estimate False-Positive rate at the top alpha fraction."""
    if df.is_empty() or alpha <= 0:
        return None
    alpha = min(alpha, 1.0)
    total = df.height
    cutoff = max(1, math.ceil(total * alpha))
    sorted_df = df.sort(score_col, descending=True).head(cutoff)
    label_series = (
        sorted_df.select(pl.col(label_col).cast(pl.Int64)).to_series(0)
    )
    false_positives = cutoff - int(label_series.sum())
    normals = df.filter(pl.col(label_col) == False).height
    if normals == 0:
        return None
    return float(false_positives) / float(normals)


def population_stability_index(
    base_scores: Iterable[float],
    target_scores: Iterable[float],
    *,
    bins: int = 10,
) -> Optional[float]:
    """Compute PSI between baseline and target score distributions."""
    base_arr = _to_array(base_scores)
    target_arr = _to_array(target_scores)
    if base_arr.size == 0 or target_arr.size == 0:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(base_arr, quantiles)
    # ensure open intervals at extremes
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_hist, _ = np.histogram(base_arr, bins=edges)
    target_hist, _ = np.histogram(target_arr, bins=edges)

    base_ratio = base_hist / max(base_hist.sum(), 1)
    target_ratio = target_hist / max(target_hist.sum(), 1)

    # avoid division by zero / log(0)
    epsilon = 1e-8
    psi_components = (target_ratio - base_ratio) * np.log(
        (target_ratio + epsilon) / (base_ratio + epsilon)
    )
    return float(np.sum(psi_components))
