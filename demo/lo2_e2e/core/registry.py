"""Registry system for pipeline components.

This module provides a registry-based dispatch mechanism that eliminates
IF-based control flow. Components are registered by name and looked up
at runtime based on configuration.
"""

from typing import Any, Callable, Dict

# Global registries for pipeline components
STEP_REGISTRY: Dict[str, Callable] = {}
MODEL_REGISTRY: Dict[str, type] = {}
EXPLAINER_REGISTRY: Dict[str, type] = {}


def register_step(name: str):
    """Register a pipeline step function.
    
    Args:
        name: Unique name for the step
        
    Example:
        @register_step("load_data")
        def load_data_step(context, **kwargs):
            ...
    """
    def decorator(func: Callable):
        STEP_REGISTRY[name] = func
        return func
    return decorator


def register_model(name: str):
    """Register a model adapter class.
    
    Args:
        name: Unique name for the model adapter
        
    Example:
        @register_model("decision_tree")
        class DecisionTreeAdapter(ModelAdapter):
            ...
    """
    def decorator(cls: type):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_explainer(name: str):
    """Register an explainer class.
    
    Args:
        name: Unique name for the explainer
        
    Example:
        @register_explainer("decision_tree")
        class DecisionTreeExplainer:
            ...
    """
    def decorator(cls: type):
        EXPLAINER_REGISTRY[name] = cls
        return cls
    return decorator


def get_step(name: str) -> Callable:
    """Get a registered step by name.
    
    Args:
        name: Name of the step
        
    Returns:
        The registered step function
        
    Raises:
        KeyError: If step is not registered
    """
    if name not in STEP_REGISTRY:
        raise KeyError(f"Step '{name}' not registered. Available: {list(STEP_REGISTRY.keys())}")
    return STEP_REGISTRY[name]


def get_model_adapter_class(name: str) -> type:
    """Get a registered model adapter class by name.
    
    Args:
        name: Name of the model adapter
        
    Returns:
        The registered adapter class
        
    Raises:
        KeyError: If adapter is not registered
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model adapter '{name}' not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


def get_explainer_class(name: str) -> type:
    """Get a registered explainer class by name.
    
    Args:
        name: Name of the explainer
        
    Returns:
        The registered explainer class
        
    Raises:
        KeyError: If explainer is not registered
    """
    if name not in EXPLAINER_REGISTRY:
        raise KeyError(f"Explainer '{name}' not registered. Available: {list(EXPLAINER_REGISTRY.keys())}")
    return EXPLAINER_REGISTRY[name]
