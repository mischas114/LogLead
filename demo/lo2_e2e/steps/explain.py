"""Explanation step for pipeline."""

from typing import Any, Dict, List
import json

from ..core.registry import register_step, get_explainer_class


@register_step("explain")
def run(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Generate explanations for predictions.
    
    This step creates an explainer based on the model type and generates
    both local (per-instance) and global explanations.
    
    Args:
        context: Pipeline context dictionary
        **kwargs: Step parameters
            - model: Model key (optional, uses context['model_key'] if not provided)
            - sample_indices: Indices of samples to explain (default: all)
            - output_file: Path to save explanations as JSONL (optional)
            - feature_names: List of feature names (optional)
            
    Returns:
        Updated context with explanations
    """
    adapter = context.get("adapter")
    if adapter is None:
        raise ValueError("explain step requires 'adapter' in context (run predict step first)")
    
    X = context.get("X")
    if X is None:
        raise ValueError("explain step requires 'X' in context")
    
    df_sequences = context.get("df_sequences")
    
    # Get model key
    model_key = kwargs.get("model") or context.get("model_key")
    if not model_key:
        raise ValueError("explain step requires 'model' parameter or model_key in context")
    
    # Get explainer class based on adapter type (no IF branching)
    explainer_type = adapter.model_type
    explainer_class = get_explainer_class(explainer_type)
    
    # Get feature names if available
    feature_names = kwargs.get("feature_names")
    if not feature_names:
        feature_names = context.get("feature_columns")
    
    print(f"[explain] Using {explainer_type} explainer")
    explainer = explainer_class(feature_names=feature_names)
    
    # Generate global explanation
    print(f"[explain] Generating global explanation")
    y = df_sequences["anomaly"].to_numpy() if df_sequences and "anomaly" in df_sequences.columns else None
    global_explanation = explainer.explain_global(adapter, X, y)
    
    # Generate local explanations for sample indices
    sample_indices = kwargs.get("sample_indices")
    if sample_indices is None:
        # Explain all samples (limit to reasonable number)
        max_samples = kwargs.get("max_samples", 100)
        sample_indices = list(range(min(len(X), max_samples)))
    
    print(f"[explain] Generating local explanations for {len(sample_indices)} samples")
    local_explanations = []
    
    for idx in sample_indices:
        X_row = X[idx]
        explanation = explainer.explain_local(adapter, X_row)
        
        # Build explanation record
        record = {
            "model": model_key,
            "instance_id": int(idx),
            "prediction": context["predictions"][idx].item() if hasattr(context["predictions"][idx], "item") else int(context["predictions"][idx]),
            "explanation": {
                "local": explanation.local,
                "global": None  # Global is same for all, save separately
            },
            "text": explanation.text,
            "metadata": explanation.metadata
        }
        
        local_explanations.append(record)
    
    # Store in context
    context["explanations"] = {
        "local": local_explanations,
        "global": global_explanation,
        "model": model_key
    }
    
    # Optionally save to file
    output_file = kwargs.get("output_file")
    if output_file:
        print(f"[explain] Saving explanations to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            # Write local explanations as JSONL
            for record in local_explanations:
                f.write(json.dumps(record) + "\n")
            
            # Write global explanation as last line
            global_record = {
                "model": model_key,
                "instance_id": "global",
                "explanation": {
                    "local": None,
                    "global": global_explanation.global_
                },
                "text": global_explanation.text,
                "metadata": global_explanation.metadata
            }
            f.write(json.dumps(global_record) + "\n")
        
        print(f"[explain] Saved {len(local_explanations) + 1} explanation records")
    
    return context
