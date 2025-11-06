"""Explainability layer for model predictions.

Provides unified interface for explaining predictions from different model types,
with graceful degradation when optional dependencies are not available.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
import numpy as np

from .registry import register_explainer


@dataclass
class Explanation:
    """Container for explanation data.
    
    Attributes:
        local: Per-instance explanation data
        global_: Global model explanation data
        text: Human-readable summary
        metadata: Additional metadata (e.g., shap_used flag)
    """
    local: Optional[Dict[str, Any]] = None
    global_: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Explainer(Protocol):
    """Protocol for explainer implementations."""
    
    def explain_local(self, adapter: Any, X_row: np.ndarray, **kwargs) -> Explanation:
        """Generate local (per-instance) explanation.
        
        Args:
            adapter: Model adapter
            X_row: Single instance features (1D array)
            **kwargs: Additional explainer-specific options
            
        Returns:
            Explanation with local data and text
        """
        ...
    
    def explain_global(self, adapter: Any, X: np.ndarray, 
                      y: Optional[np.ndarray] = None, **kwargs) -> Explanation:
        """Generate global model explanation.
        
        Args:
            adapter: Model adapter
            X: Training/test features
            y: Optional labels
            **kwargs: Additional explainer-specific options
            
        Returns:
            Explanation with global data
        """
        ...


@register_explainer("decision_tree")
class DecisionTreeExplainer:
    """Explainer for decision tree models.
    
    Provides glass-box explanations showing the exact decision path
    taken through the tree, including feature thresholds, node IDs,
    impurity values, and class distributions.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """Initialize the explainer.
        
        Args:
            feature_names: Optional list of feature names for readable output
        """
        self.feature_names = feature_names
    
    def explain_local(self, adapter: Any, X_row: np.ndarray, **kwargs) -> Explanation:
        """Explain a single prediction by tracing the decision path.
        
        Args:
            adapter: DecisionTreeAdapter instance
            X_row: Single instance (1D array)
            **kwargs: Additional options
            
        Returns:
            Explanation with decision path details
        """
        if X_row.ndim == 1:
            X_row = X_row.reshape(1, -1)
        
        tree = adapter.tree_
        decision_path = adapter.decision_path(X_row)
        node_indicator = decision_path.toarray()[0]
        
        # Get nodes along the path
        path_nodes = np.where(node_indicator)[0]
        
        # Build detailed path information
        path_details = []
        for node_id in path_nodes:
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            impurity = tree.impurity[node_id]
            n_samples = tree.n_node_samples[node_id]
            value = tree.value[node_id]
            
            node_info = {
                "node_id": int(node_id),
                "feature_index": int(feature_idx),
                "threshold": float(threshold) if feature_idx >= 0 else None,
                "impurity": float(impurity),
                "n_samples": int(n_samples),
                "value": value.tolist()
            }
            
            # Add feature name if available
            if self.feature_names and feature_idx >= 0:
                node_info["feature_name"] = self.feature_names[feature_idx]
            
            # Determine direction taken (only for internal nodes)
            if feature_idx >= 0:  # Not a leaf
                feature_value = X_row[0, feature_idx]
                node_info["feature_value"] = float(feature_value)
                node_info["direction"] = "left" if feature_value <= threshold else "right"
            
            path_details.append(node_info)
        
        # Generate human-readable text
        text_lines = ["Decision Path:"]
        for i, node in enumerate(path_details):
            if node.get("threshold") is not None:
                feat_name = node.get("feature_name", f"feature_{node['feature_index']}")
                text_lines.append(
                    f"  {i+1}. Node {node['node_id']}: "
                    f"{feat_name} = {node['feature_value']:.4f} "
                    f"{'<=' if node['direction'] == 'left' else '>'} {node['threshold']:.4f} "
                    f"(impurity={node['impurity']:.4f}, n={node['n_samples']})"
                )
            else:
                # Leaf node
                text_lines.append(
                    f"  {i+1}. Leaf {node['node_id']}: "
                    f"value={node['value']} "
                    f"(n={node['n_samples']})"
                )
        
        prediction = adapter.predict(X_row)[0]
        text_lines.append(f"\nPrediction: {prediction}")
        
        return Explanation(
            local={
                "path_nodes": [int(n) for n in path_nodes],
                "path_details": path_details,
                "prediction": int(prediction) if np.issubdtype(type(prediction), np.integer) else float(prediction)
            },
            text="\n".join(text_lines),
            metadata={"explainer_type": "decision_tree"}
        )
    
    def explain_global(self, adapter: Any, X: np.ndarray, 
                      y: Optional[np.ndarray] = None, **kwargs) -> Explanation:
        """Generate global explanation of the tree structure.
        
        Args:
            adapter: DecisionTreeAdapter instance
            X: Feature matrix
            y: Optional labels (not used for trees)
            **kwargs: Additional options
            
        Returns:
            Explanation with tree statistics and feature importances
        """
        tree = adapter.tree_
        
        # Feature importances
        importances = adapter.model.feature_importances_
        importance_dict = {}
        for idx, imp in enumerate(importances):
            if imp > 0:
                feat_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                importance_dict[feat_name] = float(imp)
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        global_info = {
            "n_nodes": int(tree.node_count),
            "n_leaves": int(np.sum(tree.children_left == -1)),
            "max_depth": int(tree.max_depth),
            "feature_importances": dict(sorted_features[:20]),  # Top 20
            "n_features": int(tree.n_features)
        }
        
        text_lines = ["Decision Tree Summary:"]
        text_lines.append(f"  Nodes: {global_info['n_nodes']}")
        text_lines.append(f"  Leaves: {global_info['n_leaves']}")
        text_lines.append(f"  Max Depth: {global_info['max_depth']}")
        text_lines.append("\nTop Feature Importances:")
        for feat, imp in sorted_features[:10]:
            text_lines.append(f"  {feat}: {imp:.4f}")
        
        return Explanation(
            global_=global_info,
            text="\n".join(text_lines),
            metadata={"explainer_type": "decision_tree"}
        )


@register_explainer("random_forest")
class RandomForestExplainer:
    """Explainer for random forest models.
    
    Provides local explanations through aggregated tree paths and
    optional SHAP values. Global explanations include feature importances
    and ensemble statistics.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None, use_shap: bool = True):
        """Initialize the explainer.
        
        Args:
            feature_names: Optional list of feature names
            use_shap: Whether to attempt SHAP explanations (gracefully degrades)
        """
        self.feature_names = feature_names
        self.use_shap = use_shap
        self._shap_available = False
        
        if use_shap:
            try:
                import shap
                self._shap_available = True
            except ImportError:
                pass
    
    def explain_local(self, adapter: Any, X_row: np.ndarray, **kwargs) -> Explanation:
        """Explain a single prediction using tree paths or SHAP.
        
        Args:
            adapter: RandomForestAdapter instance
            X_row: Single instance (1D array)
            **kwargs: Additional options
            
        Returns:
            Explanation with local feature contributions
        """
        if X_row.ndim == 1:
            X_row = X_row.reshape(1, -1)
        
        prediction = adapter.predict(X_row)[0]
        proba = adapter.predict_proba(X_row)
        
        local_data = {
            "prediction": int(prediction) if np.issubdtype(type(prediction), np.integer) else float(prediction),
        }
        
        if proba is not None:
            local_data["probabilities"] = proba[0].tolist()
        
        # Try SHAP if available
        shap_used = False
        if self._shap_available and self.use_shap:
            try:
                import shap
                explainer = shap.TreeExplainer(adapter.model)
                shap_values = explainer.shap_values(X_row)
                
                # Handle multi-class or binary
                if isinstance(shap_values, list):
                    shap_values = shap_values[int(prediction)]
                
                local_data["shap_values"] = shap_values[0].tolist()
                shap_used = True
                
                # Build feature contribution text
                contributions = []
                for idx, val in enumerate(shap_values[0]):
                    feat_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                    contributions.append((feat_name, float(val)))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                local_data["top_contributions"] = contributions[:10]
                
            except Exception as e:
                # SHAP failed, fall back to basic explanation
                pass
        
        # Fallback: use feature importances as proxy
        if not shap_used:
            importances = adapter.feature_importances_
            contributions = []
            for idx, imp in enumerate(importances):
                if imp > 0:
                    feat_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                    # Scale by feature value as rough contribution
                    contrib = imp * X_row[0, idx]
                    contributions.append((feat_name, float(contrib)))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            local_data["approximate_contributions"] = contributions[:10]
        
        # Generate text
        text_lines = [f"Random Forest Prediction: {prediction}"]
        if proba is not None:
            text_lines.append(f"Probabilities: {[f'{p:.4f}' for p in proba[0]]}")
        
        if shap_used and "top_contributions" in local_data:
            text_lines.append("\nTop SHAP Contributions:")
            for feat, val in local_data["top_contributions"][:5]:
                text_lines.append(f"  {feat}: {val:+.4f}")
        elif "approximate_contributions" in local_data:
            text_lines.append("\nTop Approximate Contributions (importance × value):")
            for feat, val in local_data["approximate_contributions"][:5]:
                text_lines.append(f"  {feat}: {val:+.4f}")
        
        return Explanation(
            local=local_data,
            text="\n".join(text_lines),
            metadata={"explainer_type": "random_forest", "shap_used": shap_used}
        )
    
    def explain_global(self, adapter: Any, X: np.ndarray, 
                      y: Optional[np.ndarray] = None, **kwargs) -> Explanation:
        """Generate global explanation of the forest.
        
        Args:
            adapter: RandomForestAdapter instance
            X: Feature matrix
            y: Optional labels
            **kwargs: Additional options
            
        Returns:
            Explanation with ensemble statistics and importances
        """
        importances = adapter.feature_importances_
        importance_dict = {}
        for idx, imp in enumerate(importances):
            if imp > 0:
                feat_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                importance_dict[feat_name] = float(imp)
        
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        global_info = {
            "n_estimators": len(adapter.estimators_),
            "feature_importances": dict(sorted_features[:20]),
            "n_features": X.shape[1]
        }
        
        text_lines = ["Random Forest Summary:"]
        text_lines.append(f"  Trees: {global_info['n_estimators']}")
        text_lines.append(f"  Features: {global_info['n_features']}")
        text_lines.append("\nTop Feature Importances:")
        for feat, imp in sorted_features[:10]:
            text_lines.append(f"  {feat}: {imp:.4f}")
        
        return Explanation(
            global_=global_info,
            text="\n".join(text_lines),
            metadata={"explainer_type": "random_forest"}
        )


@register_explainer("xgboost")
class XGBoostExplainer:
    """Explainer for XGBoost models.
    
    Uses pred_contribs for SHAP-like local explanations when available,
    falls back to feature importances otherwise. Provides multiple
    importance types (gain, weight, cover) for global explanations.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """Initialize the explainer.
        
        Args:
            feature_names: Optional list of feature names
        """
        self.feature_names = feature_names
    
    def explain_local(self, adapter: Any, X_row: np.ndarray, **kwargs) -> Explanation:
        """Explain a single prediction using contributions or importances.
        
        Args:
            adapter: XGBoostAdapter instance
            X_row: Single instance (1D array)
            **kwargs: Additional options
            
        Returns:
            Explanation with local feature contributions
        """
        if X_row.ndim == 1:
            X_row = X_row.reshape(1, -1)
        
        prediction = adapter.predict(X_row)[0]
        proba = adapter.predict_proba(X_row)
        
        local_data = {
            "prediction": int(prediction) if np.issubdtype(type(prediction), np.integer) else float(prediction),
        }
        
        if proba is not None:
            local_data["probabilities"] = proba[0].tolist()
        
        # Try to get contributions (SHAP-like)
        contributions = adapter.predict_contributions(X_row)
        contribs_used = False
        
        if contributions is not None:
            # contributions shape: (n_samples, n_features + 1) where last is bias
            contribs = contributions[0]
            local_data["contributions"] = contribs.tolist()
            contribs_used = True
            
            # Build feature contribution ranking
            contrib_list = []
            for idx in range(len(contribs) - 1):  # Exclude bias term
                feat_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                contrib_list.append((feat_name, float(contribs[idx])))
            contrib_list.sort(key=lambda x: abs(x[1]), reverse=True)
            local_data["top_contributions"] = contrib_list[:10]
            local_data["bias"] = float(contribs[-1])
        else:
            # Fallback: use global importances
            try:
                importance_scores = adapter.get_score(importance_type="gain")
                local_data["fallback_mode"] = "global_importances"
                local_data["global_importances"] = importance_scores
            except Exception:
                local_data["fallback_mode"] = "unavailable"
        
        # Generate text
        text_lines = [f"XGBoost Prediction: {prediction}"]
        if proba is not None:
            text_lines.append(f"Probabilities: {[f'{p:.4f}' for p in proba[0]]}")
        
        if contribs_used and "top_contributions" in local_data:
            text_lines.append("\nTop Feature Contributions:")
            for feat, val in local_data["top_contributions"][:5]:
                text_lines.append(f"  {feat}: {val:+.4f}")
            if "bias" in local_data:
                text_lines.append(f"  Bias: {local_data['bias']:+.4f}")
        elif "fallback_mode" in local_data:
            text_lines.append(f"\nNote: Local contributions unavailable, using {local_data['fallback_mode']}")
        
        return Explanation(
            local=local_data,
            text="\n".join(text_lines),
            metadata={"explainer_type": "xgboost", "contributions_used": contribs_used}
        )
    
    def explain_global(self, adapter: Any, X: np.ndarray, 
                      y: Optional[np.ndarray] = None, **kwargs) -> Explanation:
        """Generate global explanation with multiple importance types.
        
        Args:
            adapter: XGBoostAdapter instance
            X: Feature matrix
            y: Optional labels
            **kwargs: Additional options
            
        Returns:
            Explanation with importance scores
        """
        global_info = {"n_features": X.shape[1]}
        
        # Collect different importance types
        for imp_type in ["gain", "weight", "cover"]:
            try:
                scores = adapter.get_score(importance_type=imp_type)
                if scores:
                    # Convert feature indices to names if available
                    named_scores = {}
                    for feat_id, score in scores.items():
                        if self.feature_names:
                            try:
                                feat_idx = int(feat_id.replace("f", ""))
                                feat_name = self.feature_names[feat_idx]
                            except (ValueError, IndexError):
                                feat_name = feat_id
                        else:
                            feat_name = feat_id
                        named_scores[feat_name] = float(score)
                    
                    sorted_scores = sorted(named_scores.items(), key=lambda x: x[1], reverse=True)
                    global_info[f"importance_{imp_type}"] = dict(sorted_scores[:20])
            except Exception:
                pass
        
        # Generate text
        text_lines = ["XGBoost Summary:"]
        text_lines.append(f"  Features: {global_info['n_features']}")
        
        for imp_type in ["gain", "weight", "cover"]:
            key = f"importance_{imp_type}"
            if key in global_info:
                text_lines.append(f"\nTop Features by {imp_type.capitalize()}:")
                items = list(global_info[key].items())[:5]
                for feat, score in items:
                    text_lines.append(f"  {feat}: {score:.4f}")
        
        return Explanation(
            global_=global_info,
            text="\n".join(text_lines),
            metadata={"explainer_type": "xgboost"}
        )
