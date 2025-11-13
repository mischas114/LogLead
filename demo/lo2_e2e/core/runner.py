"""Pipeline runner for configuration-driven execution.

This module provides a declarative pipeline runner that eliminates IF-based
control flow. Steps are selected and executed based on configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml

from .registry import get_step


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def run_pipeline(config: Dict[str, Any], initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a pipeline based on configuration.
    
    This function executes pipeline steps in sequence based on the configuration.
    No IF branching is used - all step selection is done via registry lookups.
    
    Args:
        config: Pipeline configuration with 'pipeline' key containing list of steps
        initial_context: Initial context dictionary (optional)
        
    Returns:
        Final context after all steps have executed
        
    Example:
        config = {
            'pipeline': [
                {'step': 'load_data', 'with': {'path': 'data.parquet'}},
                {'step': 'predict', 'with': {'model': 'rf_v1'}},
                {'step': 'explain', 'with': {'model': 'rf_v1'}}
            ]
        }
        context = run_pipeline(config)
    """
    context = initial_context or {}
    
    # Get pipeline steps from config
    pipeline_steps = config.get("pipeline", [])
    
    if not pipeline_steps:
        raise ValueError("Configuration must contain 'pipeline' key with list of steps")
    
    # Execute each step in sequence
    for step_config in pipeline_steps:
        step_name = step_config.get("step")
        if not step_name:
            raise ValueError(f"Step configuration missing 'step' key: {step_config}")
        
        # Get step function from registry (no IF branching)
        step_func = get_step(step_name)
        
        # Extract step parameters
        step_params = step_config.get("with", {})
        
        # Execute step - step updates and returns context
        context = step_func(context, **step_params)
        
        if context is None:
            raise RuntimeError(f"Step '{step_name}' returned None context")
    
    return context


def run_pipeline_from_file(config_path: str, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration from file and run pipeline.
    
    Args:
        config_path: Path to YAML configuration file
        initial_context: Initial context dictionary (optional)
        
    Returns:
        Final context after pipeline execution
    """
    config = load_config(config_path)
    return run_pipeline(config, initial_context)
