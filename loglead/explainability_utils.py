"""Shared helpers for LogLead explainability workflows.

The LO2 demo scripts historically duplicated plotting and vectorizer defaults,
which quickly drifted apart.  Centralizing the helpers keeps lightweight,
float32-friendly defaults in one place and makes it easier to reuse the SHAP /
nearest-neighbour tooling across phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import shap

DEFAULT_VECTORIZER_KWARGS: dict[str, object] = {
    "dtype": np.float32,
    "binary": True,
    "strip_accents": "unicode",
    "max_df": 0.9,
    "min_df": 2,
    "max_features": 100_000,
}


def vectorizer_with_defaults(overrides: dict | None = None) -> dict:
    """Return a copy of the standard vectorizer kwargs merged with overrides.

    Parameters
    ----------
    overrides:
        Optional dict with user provided settings.  Handled conservatively
        to keep float32-friendly defaults and a compact vocabulary.
    """
    params = DEFAULT_VECTORIZER_KWARGS.copy()
    if overrides:
        params.update(overrides)
    return params


def to_dense(matrix):
    """Convert sparse matrices (scipy / sklearn) into numpy.ndarray when needed."""
    if matrix is None:
        return None
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "todense"):
        return np.asarray(matrix.todense())
    return matrix


def write_lines(path: Path, lines: Iterable[str]) -> None:
    """Write text lines with a trailing newline (used for feature logs)."""
    payload = "\n".join(lines)
    if payload and not payload.endswith("\n"):
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def save_top_features(explainer, limit: int, out_path: Path) -> Sequence[str]:
    """Persist the feature ranking emitted by a SHAP explainer."""
    feature_names = explainer.sorted_featurenames()[:limit]
    write_lines(out_path, [f"{idx + 1}. {name}" for idx, name in enumerate(feature_names)])
    return feature_names


def plot_shap(explainer, out_prefix: Path, *, max_display: int = 20) -> None:
    """Generate summary + bar SHAP plots without leaking pyplot configuration."""
    shap_vals = explainer.Svals
    data = to_dense(explainer.shapdata)

    # Import pyplot lazily so callers can set MPLBACKEND before touching helpers.
    import matplotlib.pyplot as plt

    summary_path = out_prefix.parent / f"{out_prefix.name}_summary.png"
    if hasattr(shap_vals, "values"):
        shap.summary_plot(shap_vals, show=False, max_display=max_display)
    else:
        shap.summary_plot(shap_vals, data, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    bar_path = out_prefix.parent / f"{out_prefix.name}_bar.png"
    shap.plots.bar(shap_vals, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()


__all__ = [
    "DEFAULT_VECTORIZER_KWARGS",
    "plot_shap",
    "save_top_features",
    "to_dense",
    "vectorizer_with_defaults",
    "write_lines",
]
