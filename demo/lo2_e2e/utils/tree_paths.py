"""Utilities for extracting decision paths from sklearn trees.

This module provides helper functions for working with sklearn decision trees,
including path extraction and visualization.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def extract_decision_path(tree, X_sample: np.ndarray, feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Extract the decision path for a single sample through a tree.
    
    Args:
        tree: sklearn tree object (e.g., from estimator.tree_)
        X_sample: Single sample (1D array)
        feature_names: Optional feature names
        
    Returns:
        List of node dictionaries with path information
    """
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)
    
    # Get node indicator matrix
    node_indicator = tree.decision_path(X_sample).toarray()[0]
    node_ids = np.where(node_indicator)[0]
    
    path = []
    for node_id in node_ids:
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        
        node_info = {
            "node_id": int(node_id),
            "feature_index": int(feature_idx),
            "threshold": float(threshold) if feature_idx >= 0 else None,
            "impurity": float(tree.impurity[node_id]),
            "n_samples": int(tree.n_node_samples[node_id]),
            "value": tree.value[node_id].tolist()
        }
        
        if feature_names and feature_idx >= 0:
            node_info["feature_name"] = feature_names[feature_idx]
        
        if feature_idx >= 0:  # Internal node
            feature_value = X_sample[0, feature_idx]
            node_info["feature_value"] = float(feature_value)
            node_info["direction"] = "left" if feature_value <= threshold else "right"
        
        path.append(node_info)
    
    return path


def format_decision_path_text(path: List[Dict[str, Any]], prediction: Any) -> str:
    """Format decision path as human-readable text.
    
    Args:
        path: List of node dictionaries from extract_decision_path
        prediction: Final prediction value
        
    Returns:
        Formatted text representation
    """
    lines = ["Decision Path:"]
    
    for i, node in enumerate(path):
        if node.get("threshold") is not None:
            feat_name = node.get("feature_name", f"feature_{node['feature_index']}")
            lines.append(
                f"  {i+1}. Node {node['node_id']}: "
                f"{feat_name} = {node['feature_value']:.4f} "
                f"{'<=' if node['direction'] == 'left' else '>'} {node['threshold']:.4f}"
            )
        else:
            lines.append(f"  {i+1}. Leaf {node['node_id']}: value={node['value']}")
    
    lines.append(f"\nPrediction: {prediction}")
    return "\n".join(lines)


def get_tree_structure_summary(tree) -> Dict[str, Any]:
    """Get summary statistics about a tree structure.
    
    Args:
        tree: sklearn tree object
        
    Returns:
        Dictionary with tree statistics
    """
    n_nodes = tree.node_count
    n_leaves = np.sum(tree.children_left == -1)
    
    return {
        "n_nodes": int(n_nodes),
        "n_leaves": int(n_leaves),
        "max_depth": int(tree.max_depth),
        "n_features": int(tree.n_features)
    }


def get_feature_splits(tree, feature_names: Optional[List[str]] = None) -> Dict[str, int]:
    """Count how many times each feature is used for splits.
    
    Args:
        tree: sklearn tree object
        feature_names: Optional feature names
        
    Returns:
        Dictionary mapping features to split counts
    """
    feature_counts = {}
    
    for node_id in range(tree.node_count):
        feature_idx = tree.feature[node_id]
        if feature_idx >= 0:  # Internal node
            feat_name = feature_names[feature_idx] if feature_names else f"feature_{feature_idx}"
            feature_counts[feat_name] = feature_counts.get(feat_name, 0) + 1
    
    return dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))
