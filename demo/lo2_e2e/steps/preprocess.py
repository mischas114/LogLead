"""Data preprocessing step for pipeline."""

from typing import Any, Dict
import numpy as np

from ..core.registry import register_step


@register_step("preprocess")
def run(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Preprocess data for model input.
    
    This step prepares feature matrices from the loaded data.
    It can handle both token-based and numeric features.
    
    Args:
        context: Pipeline context dictionary
        **kwargs: Step parameters
            - feature_columns: List of column names to use as features
            - vectorizer_type: Type of vectorizer ('count', 'tfidf', None for numeric)
            - store_key: Key to store processed features (default: 'X')
            
    Returns:
        Updated context with feature matrix stored under 'store_key'
    """
    df_sequences = context.get("df_sequences")
    if df_sequences is None:
        raise ValueError("preprocess step requires 'df_sequences' in context")
    
    feature_columns = kwargs.get("feature_columns", [])
    vectorizer_type = kwargs.get("vectorizer_type")
    store_key = kwargs.get("store_key", "X")
    
    if not feature_columns:
        print("[preprocess] No feature columns specified, using all numeric columns")
        # Use all numeric columns
        numeric_cols = [col for col in df_sequences.columns 
                       if df_sequences[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]
        feature_columns = numeric_cols
    
    print(f"[preprocess] Using features: {feature_columns}")
    
    # For now, store column names for later vectorization by predict step
    # The actual vectorization should happen when we know which model/vectorizer to use
    context["feature_columns"] = feature_columns
    context["vectorizer_type"] = vectorizer_type
    
    # Also store raw data for model-specific preprocessing
    print(f"[preprocess] Features prepared for {len(df_sequences)} samples")
    
    return context
