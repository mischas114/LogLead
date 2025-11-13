#!/usr/bin/env python3
"""End-to-end demonstration of the declarative pipeline.

This script creates synthetic data, trains models, and runs the complete
pipeline to demonstrate the refactored architecture.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import to register components
from demo.lo2_e2e.core import adapters, explainers
from demo.lo2_e2e import steps
from demo.lo2_e2e.core.runner import run_pipeline


def create_synthetic_data(n_samples=200, n_features=5, random_state=42):
    """Create synthetic log sequence data."""
    np.random.seed(random_state)
    
    # Generate features that simulate log characteristics
    seq_len = np.random.randint(10, 500, n_samples)
    duration_sec = np.random.uniform(0.1, 120.0, n_samples)
    error_count = np.random.poisson(2, n_samples)
    word_count = seq_len * np.random.uniform(5, 15, n_samples)
    cpu_usage = np.random.uniform(0, 100, n_samples)
    
    # Create anomaly labels based on rules
    # Anomalies: long duration + high error count OR very short sequences with errors
    anomaly = np.zeros(n_samples, dtype=int)
    anomaly[(duration_sec > 80) & (error_count > 3)] = 1
    anomaly[(seq_len < 30) & (error_count > 1)] = 1
    
    # Create DataFrame
    data = {
        "seq_id": [f"seq_{i:04d}" for i in range(n_samples)],
        "seq_len": seq_len.astype(int),
        "duration_sec": duration_sec,
        "error_count": error_count.astype(int),
        "word_count": word_count.astype(int),
        "cpu_usage": cpu_usage,
        "anomaly": anomaly,
        "service": np.random.choice(["api", "database", "cache"], n_samples),
        "test_case": np.where(anomaly == 1, "error", "correct")
    }
    
    df = pl.DataFrame(data)
    return df


def train_models(df, output_dir):
    """Train different model types on the data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare features
    feature_cols = ["seq_len", "duration_sec", "error_count", "word_count", "cpu_usage"]
    X = df.select(feature_cols).to_numpy()
    y = df["anomaly"].to_numpy()
    
    print(f"Training models on {len(X)} samples with {X.shape[1]} features")
    print(f"  Anomalies: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")
    
    # Train Decision Tree
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
    dt.fit(X, y)
    dt_path = output_dir / "dt_demo.joblib"
    joblib.dump(dt, dt_path)
    print(f"  Saved to {dt_path}")
    print(f"  Depth: {dt.tree_.max_depth}, Nodes: {dt.tree_.node_count}")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=20, max_depth=4, min_samples_split=10, random_state=42)
    rf.fit(X, y)
    rf_path = output_dir / "rf_demo.joblib"
    joblib.dump(rf, rf_path)
    print(f"  Saved to {rf_path}")
    print(f"  Trees: {len(rf.estimators_)}")
    
    # Try to train XGBoost if available
    xgb_path = None
    try:
        import xgboost as xgb
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=20, 
            max_depth=4, 
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X, y)
        xgb_path = output_dir / "xgb_demo.joblib"
        joblib.dump(xgb_model, xgb_path)
        print(f"  Saved to {xgb_path}")
    except ImportError:
        print("\nXGBoost not available, skipping")
    
    return dt_path, rf_path, xgb_path, feature_cols


def create_pipeline_configs(data_path, model_paths, feature_cols, output_dir):
    """Create pipeline configuration files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dt_path, rf_path, xgb_path, _ = model_paths
    
    # Models config
    models_config = {
        "models": {
            "dt_demo": {
                "adapter": "decision_tree",
                "path": str(dt_path),
                "description": "Decision tree on synthetic log sequences"
            },
            "rf_demo": {
                "adapter": "random_forest",
                "path": str(rf_path),
                "description": "Random forest on synthetic log sequences"
            }
        }
    }
    
    if xgb_path:
        models_config["models"]["xgb_demo"] = {
            "adapter": "xgboost",
            "path": str(xgb_path),
            "description": "XGBoost on synthetic log sequences"
        }
    
    models_config_path = output_dir / "models_demo.yaml"
    import yaml
    with models_config_path.open("w") as f:
        yaml.dump(models_config, f, default_flow_style=False)
    
    print(f"\nCreated models config: {models_config_path}")
    
    # Pipeline configs for each model
    pipeline_configs = {}
    
    for model_key in models_config["models"].keys():
        config = {
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
                        "model": model_key,
                        "models_config": str(models_config_path)
                    }
                },
                {
                    "step": "explain",
                    "with": {
                        "model": model_key,
                        "max_samples": 5,
                        "feature_names": feature_cols,
                        "output_file": str(output_dir / f"explanations_{model_key}.jsonl")
                    }
                }
            ]
        }
        
        pipeline_config_path = output_dir / f"pipeline_{model_key}.yaml"
        with pipeline_config_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        pipeline_configs[model_key] = pipeline_config_path
        print(f"Created pipeline config: {pipeline_config_path}")
    
    return pipeline_configs


def run_demo():
    """Run the complete end-to-end demo."""
    print("="*70)
    print("E2E Demo: Declarative Pipeline with Explainability")
    print("="*70)
    
    # Setup
    demo_dir = Path(__file__).parent / "demo_output"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data
    print("\n1. Creating synthetic data...")
    df = create_synthetic_data(n_samples=200)
    data_path = demo_dir / "synthetic_sequences.parquet"
    df.write_parquet(data_path)
    print(f"   Created {len(df)} sequences")
    print(f"   Saved to {data_path}")
    
    # Train models
    print("\n2. Training models...")
    model_paths = train_models(df, demo_dir / "models")
    feature_cols = ["seq_len", "duration_sec", "error_count", "word_count", "cpu_usage"]
    
    # Create configs
    print("\n3. Creating pipeline configurations...")
    pipeline_configs = create_pipeline_configs(
        data_path, 
        model_paths, 
        feature_cols,
        demo_dir / "config"
    )
    
    # Run pipelines
    print("\n4. Running declarative pipelines...")
    print("="*70)
    
    for model_key, config_path in pipeline_configs.items():
        print(f"\n[{model_key.upper()}] Running pipeline...")
        print("-"*70)
        
        # Load config
        import yaml
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        
        # Run pipeline
        context = run_pipeline(config)
        
        # Print summary
        print(f"\n[{model_key.upper()}] Results:")
        print(f"  Predictions: {len(context['predictions'])} samples")
        print(f"  Model type: {context['adapter'].model_type}")
        
        if "explanations" in context:
            n_local = len(context["explanations"]["local"])
            print(f"  Explanations: {n_local} local, 1 global")
            
            # Show sample explanation text
            if n_local > 0:
                sample_exp = context["explanations"]["local"][0]
                print(f"\n  Sample Explanation (instance {sample_exp['instance_id']}):")
                text_lines = sample_exp["text"].split("\n")
                for line in text_lines[:8]:  # First 8 lines
                    print(f"    {line}")
                if len(text_lines) > 8:
                    print(f"    ... ({len(text_lines) - 8} more lines)")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print(f"\nOutputs saved to: {demo_dir}")
    print("\nGenerated files:")
    for path in sorted(demo_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(demo_dir)}")
    print("="*70)


if __name__ == "__main__":
    run_demo()
