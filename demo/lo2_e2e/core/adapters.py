"""Model adapters for unified prediction interface.

Adapters provide a consistent interface for different model types,
allowing the pipeline to work with various models without IF branching.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import joblib

from .registry import register_model


@dataclass
class ModelAdapter:
    """Base interface for model adapters.
    
    Attributes:
        name: Identifier for this model
        model: The underlying trained model object
        model_type: Type of the model (e.g., 'decision_tree', 'random_forest')
    """
    name: str
    model: Any
    model_type: str
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Generate probability predictions if supported.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Probability predictions (n_samples, n_classes) or None
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
    
    @classmethod
    def load(cls, path: str, name: str) -> "ModelAdapter":
        """Load a model from disk.
        
        Args:
            path: Path to the model file
            name: Name to assign to this adapter
            
        Returns:
            Initialized adapter with loaded model
        """
        model = joblib.load(path)
        return cls(name=name, model=model)


@register_model("decision_tree")
class DecisionTreeAdapter(ModelAdapter):
    """Adapter for scikit-learn DecisionTree models.
    
    Provides access to tree structure for explainability.
    """
    
    def __init__(self, name: str, model: Any):
        super().__init__(name=name, model=model, model_type="decision_tree")
    
    @property
    def tree_(self):
        """Access the underlying tree structure."""
        return self.model.tree_
    
    def decision_path(self, X: np.ndarray):
        """Get decision paths through the tree.
        
        Args:
            X: Input features
            
        Returns:
            Sparse indicator matrix of nodes visited
        """
        return self.model.decision_path(X)
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf indices for samples.
        
        Args:
            X: Input features
            
        Returns:
            Leaf node indices (n_samples,)
        """
        return self.model.apply(X)


@register_model("random_forest")
class RandomForestAdapter(ModelAdapter):
    """Adapter for scikit-learn RandomForest models.
    
    Provides access to ensemble properties and individual trees.
    """
    
    def __init__(self, name: str, model: Any):
        super().__init__(name=name, model=model, model_type="random_forest")
    
    @property
    def estimators_(self):
        """Access individual trees in the forest."""
        return self.model.estimators_
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances from the ensemble."""
        return self.model.feature_importances_
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf indices for all trees.
        
        Args:
            X: Input features
            
        Returns:
            Leaf indices (n_samples, n_estimators)
        """
        return self.model.apply(X)


@register_model("xgboost")
class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost models.
    
    Handles both XGBClassifier and Booster objects.
    """
    
    def __init__(self, name: str, model: Any):
        super().__init__(name=name, model=model, model_type="xgboost")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.
        
        Handles both Booster and sklearn-compatible interfaces.
        """
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        # Booster interface
        try:
            import xgboost as xgb
            if isinstance(X, np.ndarray):
                dmatrix = xgb.DMatrix(X)
                return self.model.predict(dmatrix)
        except ImportError:
            pass
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Generate probability predictions."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
    
    def predict_contributions(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get SHAP-like feature contributions if available.
        
        Args:
            X: Input features
            
        Returns:
            Contribution matrix or None if not available
        """
        try:
            import xgboost as xgb
            if hasattr(self.model, "predict"):
                # XGBClassifier interface
                if hasattr(self.model, "get_booster"):
                    booster = self.model.get_booster()
                    dmatrix = xgb.DMatrix(X)
                    return booster.predict(dmatrix, pred_contribs=True)
                # Try direct prediction with contributions
                return self.model.predict(X, pred_contribs=True)
        except (ImportError, AttributeError, TypeError):
            pass
        return None
    
    def get_score(self, importance_type: str = "gain") -> dict:
        """Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Dictionary mapping feature names to scores
        """
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
            return booster.get_score(importance_type=importance_type)
        elif hasattr(self.model, "get_score"):
            return self.model.get_score(importance_type=importance_type)
        return {}
