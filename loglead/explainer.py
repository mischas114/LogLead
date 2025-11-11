from __future__ import annotations

from typing import Callable

from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
import numpy as np
import umap
import plotly.express as px

import shap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from loglead.explainability_utils import to_dense

class NNExplainer:
    """Assistants for nearest-neighbour based explainability.

    The LO2 workflow only needs a small slice of the prediction table for
    explainability.  Computing the full NN mapping lazily avoids materialising
    large cosine-similarity matrices until the caller explicitly asks for them.

    Parameters
    ----------
    df:
        Polars dataframe that contains predictions (`pred_col`) and unique ids.
    X:
        Feature matrix aligned with `df`.  Can be numpy arrays or sparse CSR
        matrices.
    backend:
        Optional callable that receives `(anomalies, normals)` matrices and
        returns the index of the closest normal row for every anomaly.  By
        default cosine similarity is used, but this hook makes it trivial to
        plug in FAISS/Annoy/HNSW backends.
    auto_dense:
        Densify sparse matrices before calling the default backend.  Set to
        `False` when the provided backend operates on sparse inputs directly.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        X,
        id_col: str,
        pred_col: str,
        *,
        backend: Callable | None = None,
        auto_dense: bool = True,
    ) -> None:
        self.df = df
        self.X = X
        self.id_column = id_col
        self.prediction_column = pred_col
        self.auto_dense = auto_dense
        self._backend = backend
        self._mapping: pl.DataFrame | None = None

    def build_mapping(self, *, force: bool = False) -> pl.DataFrame:
        """Compute (or recompute) the anomalous→normal lookup table."""
        if self._mapping is None or force:
            self._mapping = self._get_normal_mapping()
        return self._mapping

    @property
    def mapping(self) -> pl.DataFrame:
        """Expose the cached mapping while keeping the computation lazy."""
        return self.build_mapping()

    def clear_cache(self) -> None:
        """Drop cached results so the mapping will be recomputed on demand."""
        self._mapping = None

    def _resolve_backend(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if self._backend is None:
            return self._cosine_backend
        if callable(self._backend):
            return self._backend
        if hasattr(self._backend, "query"):
            return getattr(self._backend, "query")
        raise TypeError("NNExplainer backend must be callable or expose a 'query' method.")

    def _prepare_matrix(self, matrix):
        data = matrix
        if self.auto_dense and hasattr(matrix, "toarray"):
            data = matrix.toarray()
        if isinstance(data, np.ndarray):
            return data.astype(np.float32, copy=False)
        if hasattr(data, "astype"):
            return data.astype(np.float32)
        return np.asarray(data, dtype=np.float32)

    def _prediction_mask(self) -> np.ndarray:
        series = self.df.select(pl.col(self.prediction_column)).to_series()
        return np.asarray(series.to_list(), dtype=bool)

    def _normalise_backend_output(self, output, max_index: int) -> np.ndarray:
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, list):
            if output and isinstance(output[0], (list, tuple, np.ndarray)):
                return np.asarray([row[0] for row in output], dtype=int)
            return np.asarray(output, dtype=int)
        arr = np.asarray(output)
        if arr.ndim > 1:
            arr = arr[:, 0]
        arr = arr.astype(int, copy=False)
        arr = np.atleast_1d(arr)
        if arr.size:
            arr = np.clip(arr, 0, max_index)
        return arr

    def _get_normal_mapping(self) -> pl.DataFrame:
        mask = self._prediction_mask()
        anomaly_idx = np.flatnonzero(mask)
        normal_idx = np.flatnonzero(~mask)

        if anomaly_idx.size == 0 or normal_idx.size == 0:
            return pl.DataFrame({"anomalous_id": [], "normal_id": []})

        anomalies = self._prepare_matrix(self.X[anomaly_idx])
        normals = self._prepare_matrix(self.X[normal_idx])
        backend = self._resolve_backend()
        matches = backend(anomalies, normals)
        matches = self._normalise_backend_output(matches, normal_idx.size - 1)

        id_series = self.df.select(pl.col(self.id_column)).to_series().to_list()
        anomaly_ids = [id_series[idx] for idx in anomaly_idx]
        normal_ids = [id_series[idx] for idx in normal_idx]
        matched_normals = [normal_ids[idx] for idx in matches]
        return pl.DataFrame({"anomalous_id": anomaly_ids, "normal_id": matched_normals})

    def _cosine_backend(self, anomalies: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Default similarity backend operating on float32 dense matrices."""
        sims = cosine_similarity(anomalies, normals)
        return sims.argmax(axis=1)

    def print_log_content_from_nn_mapping(self) -> None:
        """Prints the log content of the anomalous and the closest normal instances in the mapping.
        The content is defined to be the list in the column e_words of the Polars DataFrame.
        """
        assert "e_words" in self.df.columns, "The column e_words is not present in the DataFrame."
        assert self.df.select(pl.col("e_words")).dtypes[0].is_nested(), "The column e_words is not nested data type."

        for anomaly, normal in self.mapping.rows():
            anomaly_words = self.df.filter(pl.col(self.id_column) == anomaly).select(pl.col("e_words")).to_series().to_list()[0]
            print(f"Anomaly sequence:{' '*8}{' '.join(anomaly_words)}")

            normal_words = self.df.filter(pl.col(self.id_column) == normal).select(pl.col("e_words")).to_series().to_list()[0]
            print(f"Closest normal sequence: {' '.join(normal_words)}\n")


    def print_features_from_nn_mapping(self, feature_cols: list[str]) -> None:
        """Prints the given features of the anomalous and the closest normal instances.

        Args:
            feature_cols (list[str]): The list of feature columns to be printed.
        """
        for anomaly, normal in self.mapping.rows():
            print(f"Features of anomaly {anomaly}: {self.df.filter(pl.col(self.id_column) == anomaly).select(pl.col(feature_cols)).to_pandas().values}")
            print(f"Features of closest normal {normal}: {self.df.filter(pl.col(self.id_column) == normal).select(pl.col(feature_cols)).to_pandas().values}")
            print("\n"*2)


    def print_false_positive_content(self, ground_truth_col: str):
        """Prints the content of the false positive instances in the log data. The false positive
        instances are the instances that are predicted to be anomalous but are not according to
        the ground truth labels.

        Args:
            ground_truth_col (str): The column name for the ground truth labels.
        """
        false_positives = self.df.filter((pl.col(self.prediction_column) == True) & (pl.col(ground_truth_col) == False)).select(pl.col(self.id_column), pl.col("e_words"))
        print("False positive sequences:")
        for row in false_positives.rows():
            print(f"{row[0]}: {' '.join(row[1])}")

    
    def print_false_negative_content(self, ground_truth_col: str):
        """Prints the content of the false negative instances in the log data. The false negative
        instances are the instances that are predicted to be normal but are anomalous according to
        the ground truth labels.

        Args:
            ground_truth_col (str): The column name for the ground truth labels.
        """
        false_negatives = self.df.filter((pl.col(self.prediction_column) == False) & (pl.col(ground_truth_col) == True)).select(pl.col(self.id_column), pl.col("e_words"))
        print("False negative sequences:")
        for row in false_negatives.rows():
            print(f"{row[0]}: {' '.join(row[1])}")


    def plot_features_in_two_dimensions(self, ground_truth_col: str = None) -> None:
        """Plots the features of the instances in 2D UMAP space. The instances are colored by whether
        they are predicted to be anomalous or not. If ground_truth_col is provided, the instances are
        also symbolized by the ground truth labels. The visualization is interactive and can be used to
        explore the instances in the 2D space.

        Args:
            ground_truth_col (str, optional): The column name for the ground truth labels. Defaults to None.
        """
        embeddings = umap.UMAP().fit_transform(self.X)
        df_vis = pl.DataFrame(embeddings, schema=["UMAP-1", "UMAP-2"])
        df_vis = df_vis.with_columns(
            self.df.select(pl.col(self.id_column)).to_series().alias(self.id_column),
            self.df.select(pl.col(self.prediction_column)).to_series().alias(self.prediction_column)
        )
        if ground_truth_col:
            symbol_col = "ground_truth"
            df_vis = df_vis.with_columns(ground_truth=self.df.select(pl.col(ground_truth_col)).to_series())
        else:
            symbol_col = None
        
        df_vis = df_vis.join(self.mapping, left_on=self.id_column, right_on="anomalous_id", how="left")
        df_vis = df_vis.with_columns(pl.when(pl.col("normal_id").is_null()).then(pl.lit("None")).otherwise(pl.col("normal_id")).alias("nearest_normal"))

        fig = px.scatter(
            data_frame=df_vis, 
            color=self.prediction_column, 
            x="UMAP-1", y="UMAP-2", 
            hover_data=[self.id_column, "nearest_normal"],
            title="Logs visualized in 2D UMAP space", 
            symbol=symbol_col,
            symbol_map={True: "cross", False: "circle"},)
        fig.show()


class ShapExplainer:
    """Resource-aware wrapper around SHAP explainers.

    The LO2 stack often runs in constrained CI environments, so this wrapper
    keeps float32-friendly defaults, samples the SHAP background distribution
    automatically, and exposes knobs for tightening the guardrails.
    """

    TREE_MODELS = (IsolationForest, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier)

    def __init__(
        self,
        sad,
        *,
        ignore_warning: bool = False,
        plot_featurename_len: int = 16,
        feature_warning_threshold: int = 1500,
        sample_warning_threshold: int | None = None,
        background_sample_size: int | None = 512,
    ) -> None:
        """
        Parameters
        ----------
        sad:
            An ``AnomalyDetector`` instance (or compatible object) with ``model``,
            ``X_train``/``X_test`` matrices and optional ``vectorizer`` / ``numeric_cols``.
        ignore_warning:
            Disable resource guardrails (not recommended for notebook usage).
        plot_featurename_len:
            Max length of feature labels in plots.
        feature_warning_threshold:
            Trigger a ``ResourceWarning`` once the number of features exceeds
            this limit.
        sample_warning_threshold:
            Optional total-cell-count guard for the SHAP matrix.  Defaults to
            ``feature_warning_threshold * 1000`` to mimic the legacy heuristic.
        background_sample_size:
            Number of rows sampled from the training set to form the SHAP
            background distribution.  Set to ``None`` to use the whole dataset.
        """
        self.model = sad.model
        self.X_train = sad.X_train
        self.X_test = sad.X_test
        self.vec = getattr(sad, "vectorizer", None)
        self.numeric_feature_names = getattr(sad, "numeric_cols", None) or []
        self.random_state = getattr(sad, "random_state", 42)

        self.warn = not ignore_warning
        self.feature_warning_threshold = feature_warning_threshold
        self.sample_warning_threshold = (
            sample_warning_threshold if sample_warning_threshold is not None else feature_warning_threshold * 1000
        )
        self.background_sample_size = background_sample_size

        self.Svals = None
        self.expl = None
        self.istree = False
        self.truncatelen = plot_featurename_len
        self.shapdata = None
        self.index = None

        self._feature_names = self._resolve_feature_names()
        self._background_cache = None
        self._explainer_factory = self._select_backend()

    def _select_backend(self):
        if isinstance(self.model, (LogisticRegression, LinearSVC)):
            return self._build_linear_explainer
        if self._is_tree_model():
            self.istree = True
            return self._build_tree_explainer
        if hasattr(self.model, "predict"):
            # KernelExplainer tolerates arbitrary callable predict functions.
            return self._build_kernel_explainer
        return self._build_plain_explainer

    def _is_tree_model(self) -> bool:
        if isinstance(self.model, self.TREE_MODELS):
            return True
        return hasattr(self.model, "tree_") or hasattr(self.model, "estimators_")

    def _build_linear_explainer(self):
        background = self._ensure_float32(self._get_background_data())
        if background is None:
            raise ValueError("Linear SHAP explainers require training data.")
        self.expl = shap.LinearExplainer(self.model, background, feature_names=self._truncate_feature_names(self.truncatelen))
        return self.expl

    def _build_tree_explainer(self):
        background = self._ensure_float32(to_dense(self._get_background_data()), force_dense=True)
        self.expl = shap.TreeExplainer(self.model, data=background, feature_names=self._truncate_feature_names(self.truncatelen))
        return self.expl

    def _build_kernel_explainer(self):
        background = self._ensure_float32(self._get_background_data())
        if background is None:
            raise ValueError("Kernel SHAP explainers require training data.")
        predict_fn = (
            getattr(self.model, "predict_proba", None)
            or getattr(self.model, "decision_function", None)
            or getattr(self.model, "predict", None)
        )
        self.expl = shap.KernelExplainer(predict_fn, background, feature_names=self._truncate_feature_names(self.truncatelen))
        return self.expl

    def _build_plain_explainer(self):
        self.expl = shap.Explainer(self.model, feature_names=self._truncate_feature_names(self.truncatelen))
        return self.expl

    def _resolve_feature_names(self) -> np.ndarray:
        if self.vec is not None and hasattr(self.vec, "get_feature_names_out"):
            return self.vec.get_feature_names_out()
        if hasattr(self.model, "feature_names_in_"):
            return np.asarray(self.model.feature_names_in_)
        if self.numeric_feature_names:
            return np.asarray(self.numeric_feature_names)
        width = None
        for candidate in (self.X_train, self.X_test):
            if candidate is not None and hasattr(candidate, "shape"):
                width = candidate.shape[1]
                break
        if not width:
            return np.asarray([])
        return np.asarray([f"feature_{idx}" for idx in range(width)])

    def _truncate_feature_names(self, length: int):
        if self._feature_names.size == 0:
            return None
        return self._feature_names.astype(f"<U{length}", copy=False)

    def _get_background_data(self):
        if self._background_cache is not None:
            return self._background_cache
        source = self.X_train if self.X_train is not None else self.X_test
        if source is None:
            return None
        sampled = self._sample_rows(source, self.background_sample_size)
        self._background_cache = sampled
        return sampled

    def _sample_rows(self, data, target_size: int | None):
        if target_size is None or target_size <= 0:
            return data
        total = getattr(data, "shape", (0,))[0]
        if not total or total <= target_size:
            return data
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(total, size=target_size, replace=False)
        return data[indices]

    def _ensure_float32(self, data, *, force_dense: bool = False):
        if data is None:
            return None
        payload = to_dense(data) if force_dense else data
        if isinstance(payload, np.ndarray):
            return payload.astype(np.float32, copy=False)
        if hasattr(payload, "astype"):
            return payload.astype(np.float32)
        return np.asarray(payload, dtype=np.float32)

    def calc_shapvalues(self, test_data=None, custom_slice: slice | None = None):
        """Calculate SHAP values for a vectorised dataset."""
        if test_data is None:
            test_data = self.X_test
        if test_data is None:
            raise ValueError("No dataset provided and detector.X_test is empty.")
        if custom_slice:
            test_data = test_data[custom_slice]

        prepared = self._ensure_float32(test_data, force_dense=self.istree)
        feature_count = getattr(prepared, "shape", (0, 0))[1]
        sample_size = getattr(prepared, "shape", (0,))[0] if hasattr(prepared, "shape") else len(prepared)

        if self.warn:
            if self.feature_warning_threshold and feature_count >= self.feature_warning_threshold:
                raise ResourceWarning("Feature count exceeds configured SHAP guard.")
            total_cells = sample_size * max(feature_count, 1)
            if self.sample_warning_threshold and total_cells >= self.sample_warning_threshold:
                raise ResourceWarning("SHAP matrix would allocate too many cells.")

        self.shapdata = prepared
        expl = self.expl or self._explainer_factory()
        self.expl = expl
        values = expl(prepared)
        self.Svals = self._coerce_tree_output(values, prepared)
        return self.Svals

    def _coerce_tree_output(self, values, prepared):
        if not self.istree or isinstance(self.model, IsolationForest):
            return values
        payload = values.values if hasattr(values, "values") else values
        if payload.ndim < 3:
            return values
        positive_class = payload[:, :, 1]
        if hasattr(values, "values"):
            base_values = getattr(values, "base_values", None)
            if base_values is not None and base_values.ndim > 1:
                base_values = base_values[:, 1]
            return shap.Explanation(
                values=positive_class,
                base_values=base_values,
                data=getattr(values, "data", prepared),
                feature_names=getattr(values, "feature_names", self._truncate_feature_names(self.truncatelen)),
                output_names=getattr(values, "output_names", None),
            )
        return positive_class

    @property
    def shap_values(self):
        return self.Svals

    @property
    def feature_names(self):
        return self._feature_names

    def _shap_value_array(self):
        if self.Svals is None:
            return None
        return self.Svals.values if hasattr(self.Svals, "values") else self.Svals

    def sorted_shapvalues(self):
        values = self._shap_value_array()
        if values is None:
            return None
        importance = np.sum(np.abs(values), axis=0)
        if self.index is None:
            self.index = np.argsort(importance)
        return np.array([values[:, idx] for idx in self.index][::-1])

    def sorted_featurenames(self):
        values = self._shap_value_array()
        if values is None:
            return []
        importance = np.sum(np.abs(values), axis=0)
        self.index = np.argsort(importance)
        names = self.feature_names
        if names is None or len(names) == 0:
            names = np.asarray([f"feature_{idx}" for idx in range(importance.shape[0])])
        return [names[idx] for idx in self.index][::-1]

    def plot(self, data=None, plottype: str = "summary", custom_slice: slice | None = None, displayed: int = 16):
        """Plot SHAP results with lightweight defaults."""
        if data is not None or self.Svals is None or custom_slice:
            self.calc_shapvalues(data, custom_slice)
            plotdata = self.shapdata
        elif custom_slice:
            plotdata = self.shapdata[custom_slice]
        else:
            plotdata = self.shapdata

        fullnames = self.sorted_featurenames()
        print("====================================")
        for i in range(min(displayed, len(fullnames))):
            print(fullnames[i])
        print("====================================")

        if plottype == "summary":
            shap.summary_plot(self.Svals, plotdata, max_display=displayed)
        elif plottype == "bar":
            shap.plots.bar(self.Svals, max_display=displayed)
        elif plottype == "beeswarm":
            shap.plots.beeswarm(self.Svals, max_display=displayed)
