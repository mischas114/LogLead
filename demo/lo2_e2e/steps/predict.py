"""Prediction step for pipeline."""

from typing import Any, Dict
from pathlib import Path
import yaml

from ..core.registry import register_step, get_model_adapter_class


@register_step("predict")
def run(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Run predictions using a model adapter.
    
    This step loads a model, creates an adapter, and runs predictions.
    No IF branching - adapter selection is via registry lookup.
    
    Args:
        context: Pipeline context dictionary
        **kwargs: Step parameters
            - model: Model key to look up in models config
            - models_config: Path to models configuration file
            - output_key: Key to store predictions (default: 'predictions')
            
    Returns:
        Updated context with predictions and adapter
    """
    df_sequences = context.get("df_sequences")
    if df_sequences is None:
        raise ValueError("predict step requires 'df_sequences' in context")
    
    model_key = kwargs.get("model")
    if not model_key:
        raise ValueError("predict step requires 'model' parameter")
    
    models_config_path = kwargs.get("models_config", "demo/lo2_e2e/config/models.yaml")
    output_key = kwargs.get("output_key", "predictions")
    
    # Load models configuration
    config_path = Path(models_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {models_config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        models_config = yaml.safe_load(f)
    
    if "models" not in models_config:
        raise ValueError(f"Models config missing 'models' key: {models_config_path}")
    
    # Get model configuration
    model_config = models_config["models"].get(model_key)
    if not model_config:
        available = list(models_config["models"].keys())
        raise ValueError(f"Model '{model_key}' not found in config. Available: {available}")
    
    # Get adapter class from registry (no IF branching)
    adapter_type = model_config.get("adapter")
    if not adapter_type:
        raise ValueError(f"Model config for '{model_key}' missing 'adapter' field")
    
    adapter_class = get_model_adapter_class(adapter_type)
    
    # Load model and create adapter
    model_path = model_config.get("path")
    if not model_path:
        raise ValueError(f"Model config for '{model_key}' missing 'path' field")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"[predict] Loading model '{model_key}' from {model_path}")
    adapter = adapter_class.load(str(model_path), name=model_key)
    
    # Prepare features for prediction
    # This is simplified - in practice, you'd need vectorization etc.
    # For now, assume the model was trained with compatible preprocessing
    feature_columns = context.get("feature_columns", [])
    
    if not feature_columns:
        # Try to use all numeric columns
        import polars as pl
        numeric_cols = [col for col in df_sequences.columns 
                       if df_sequences[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        feature_columns = numeric_cols
    
    print(f"[predict] Using feature columns: {feature_columns}")
    
    # Convert to numpy for prediction
    X = df_sequences.select(feature_columns).to_numpy()
    
    print(f"[predict] Running predictions with {adapter_type} adapter")
    predictions = adapter.predict(X)
    
    # Store in context
    context[output_key] = predictions
    context["adapter"] = adapter
    context["model_key"] = model_key
    context["X"] = X
    
    print(f"[predict] Generated {len(predictions)} predictions")
    
    return context
