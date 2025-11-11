import unittest
from unittest import mock

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from loglead.explainer import NNExplainer, ShapExplainer


class ExplainabilityTests(unittest.TestCase):
    def test_nnexplainer_lazy_mapping_with_custom_backend(self):
        df = pl.DataFrame(
            {
                "row_id": [0, 1, 2, 3],
                "pred_ano": [1, 0, 1, 0],
                "e_words": [["a"], ["b"], ["c"], ["d"]],
            }
        )
        X = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )

        calls = {"count": 0}

        def backend(anomalies, normals):
            calls["count"] += 1
            return np.zeros(anomalies.shape[0], dtype=int)

        explainer = NNExplainer(df, X, id_col="row_id", pred_col="pred_ano", backend=backend)
        self.assertEqual(calls["count"], 0, "backend should not be invoked during __init__")

        mapping = explainer.mapping
        self.assertEqual(calls["count"], 1, "lazy mapping should execute backend once")
        self.assertEqual(mapping.height, 2)

        _ = explainer.mapping
        self.assertEqual(calls["count"], 1, "cached mapping should be reused")

        explainer.build_mapping(force=True)
        self.assertEqual(calls["count"], 2, "force=True should recompute the mapping")

    @mock.patch("loglead.explainer.shap.TreeExplainer")
    def test_shapexplainer_routes_xgb_to_tree_backend(self, mock_tree):
        class DummyTreeExplainer:
            def __call__(self, data):
                rows, cols = data.shape
                return np.zeros((rows, cols, 2), dtype=np.float32)

        mock_tree.return_value = DummyTreeExplainer()

        class DummyDetector:
            def __init__(self):
                self.model = XGBClassifier()
                self.X_train = np.ones((8, 3), dtype=np.float32)
                self.X_test = np.ones((4, 3), dtype=np.float32)
                self.vectorizer = None
                self.numeric_cols = ["f0", "f1", "f2"]
                self.random_state = 0

        detector = DummyDetector()
        explainer = ShapExplainer(
            detector,
            background_sample_size=2,
            feature_warning_threshold=100,
            sample_warning_threshold=1000,
        )
        result = explainer.calc_shapvalues()

        self.assertTrue(mock_tree.called, "TreeExplainer should be used for XGBClassifier")
        _, kwargs = mock_tree.call_args
        self.assertEqual(kwargs["data"].shape[0], 2, "background sampling should respect CLI default")
        self.assertEqual(result.shape, (detector.X_test.shape[0], detector.X_test.shape[1]))


if __name__ == "__main__":
    unittest.main()
