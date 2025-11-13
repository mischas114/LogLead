"""Test that pipeline runs without IF branching.

This test verifies that the pipeline executes steps based on configuration
and registry lookups, with no conditional branching in the runner.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import to trigger registrations
from demo.lo2_e2e.core.registry import STEP_REGISTRY, get_step
from demo.lo2_e2e.core.runner import run_pipeline
from demo.lo2_e2e import steps  # noqa
from demo.lo2_e2e.core import adapters  # noqa - register adapters
from demo.lo2_e2e.core import explainers  # noqa - register explainers


def test_runner_has_no_if_branching():
    """Verify that the runner module contains no IF-based step selection."""
    from demo.lo2_e2e.core import runner
    
    # Check runner source code doesn't contain IF-based model selection
    import inspect
    source = inspect.getsource(runner.run_pipeline)
    
    # Should not have model-type specific IF statements
    assert "if model_type ==" not in source.lower()
    assert "if adapter_type ==" not in source.lower()
    assert "if explainer ==" not in source.lower()
    
    # Should use registry lookup
    assert "get_step" in source or "STEP_REGISTRY" in source
    
    print("✓ Runner uses registry-based dispatch without IF branching")


def test_steps_registered():
    """Verify that expected steps are registered."""
    expected_steps = ["load_data", "preprocess", "predict", "explain"]
    
    for step_name in expected_steps:
        assert step_name in STEP_REGISTRY, f"Step '{step_name}' not registered"
        
        # Verify we can retrieve it
        step_func = get_step(step_name)
        assert callable(step_func)
    
    print(f"✓ All {len(expected_steps)} expected steps are registered")


def test_pipeline_execution_via_config():
    """Test that pipeline executes based on config without IF branching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic data
        n_samples = 50
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Create feature columns
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        data_dict = {col: X[:, i] for i, col in enumerate(feature_cols)}
        data_dict["anomaly"] = y
        data_dict["seq_id"] = [f"seq_{i}" for i in range(n_samples)]
        
        df = pl.DataFrame(data_dict)
        
        # Save data
        data_path = tmpdir / "test_sequences.parquet"
        df.write_parquet(data_path)
        
        # Train a simple model
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X, y)
        
        model_path = tmpdir / "test_model.joblib"
        joblib.dump(clf, model_path)
        
        # Create models config
        models_config_path = tmpdir / "models.yaml"
        models_config_path.write_text(f"""
models:
  test_dt:
    adapter: decision_tree
    path: "{model_path}"
""")
        
        # Create pipeline config
        pipeline_config = {
            "pipeline": [
                {
                    "step": "load_data",
                    "with": {
                        "sequences_path": str(data_path)
                    }
                },
                {
                    "step": "preprocess",
                    "with": {
                        "feature_columns": feature_cols
                    }
                },
                {
                    "step": "predict",
                    "with": {
                        "model": "test_dt",
                        "models_config": str(models_config_path)
                    }
                }
            ]
        }
        
        # Run pipeline
        context = run_pipeline(pipeline_config)
        
        # Verify results
        assert "df_sequences" in context
        assert "predictions" in context
        assert "adapter" in context
        assert len(context["predictions"]) == n_samples
        
        print("✓ Pipeline executed successfully via config-driven dispatch")


def test_adding_new_model_no_code_change():
    """Test that adding a new model requires only config changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create data and models
        n_samples = 30
        X = np.random.randn(n_samples, 3)
        y = np.random.randint(0, 2, n_samples)
        
        feature_cols = ["f0", "f1", "f2"]
        data_dict = {col: X[:, i] for i, col in enumerate(feature_cols)}
        data_dict["anomaly"] = y
        data_dict["seq_id"] = [f"seq_{i}" for i in range(n_samples)]
        
        df = pl.DataFrame(data_dict)
        data_path = tmpdir / "data.parquet"
        df.write_parquet(data_path)
        
        # Create two different models
        dt_model = DecisionTreeClassifier(max_depth=2, random_state=42)
        dt_model.fit(X, y)
        dt_path = tmpdir / "dt.joblib"
        joblib.dump(dt_model, dt_path)
        
        rf_model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
        rf_model.fit(X, y)
        rf_path = tmpdir / "rf.joblib"
        joblib.dump(rf_model, rf_path)
        
        # Config with both models
        models_config_path = tmpdir / "models.yaml"
        models_config_path.write_text(f"""
models:
  my_dt:
    adapter: decision_tree
    path: "{dt_path}"
  my_rf:
    adapter: random_forest
    path: "{rf_path}"
""")
        
        # Test running with first model
        config1 = {
            "pipeline": [
                {"step": "load_data", "with": {"sequences_path": str(data_path)}},
                {"step": "preprocess", "with": {"feature_columns": feature_cols}},
                {"step": "predict", "with": {"model": "my_dt", "models_config": str(models_config_path)}}
            ]
        }
        
        context1 = run_pipeline(config1)
        assert context1["adapter"].model_type == "decision_tree"
        
        # Test running with second model - no code changes, just config
        config2 = {
            "pipeline": [
                {"step": "load_data", "with": {"sequences_path": str(data_path)}},
                {"step": "preprocess", "with": {"feature_columns": feature_cols}},
                {"step": "predict", "with": {"model": "my_rf", "models_config": str(models_config_path)}}
            ]
        }
        
        context2 = run_pipeline(config2)
        assert context2["adapter"].model_type == "random_forest"
        
        print("✓ Different models can be used via config-only changes")


if __name__ == "__main__":
    print("Running runner tests...")
    test_runner_has_no_if_branching()
    test_steps_registered()
    test_pipeline_execution_via_config()
    test_adding_new_model_no_code_change()
    print("\nAll runner tests passed! ✓")
