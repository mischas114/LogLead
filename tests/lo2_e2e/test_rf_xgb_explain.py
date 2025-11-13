"""Test RandomForest and XGBoost explainability.

This test verifies that RF and XGB explainers provide both local and global
explanations with appropriate fallbacks when SHAP is unavailable.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.lo2_e2e.core.adapters import RandomForestAdapter, XGBoostAdapter
from demo.lo2_e2e.core.explainers import RandomForestExplainer, XGBoostExplainer


def test_random_forest_local_explanation():
    """Test local explanations for random forest."""
    X = np.random.randn(40, 5)
    y = np.random.randint(0, 2, 40)
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    
    adapter = RandomForestAdapter(name="test_rf", model=clf)
    feature_names = [f"feature_{i}" for i in range(5)]
    explainer = RandomForestExplainer(feature_names=feature_names, use_shap=True)
    
    explanation = explainer.explain_local(adapter, X[0])
    
    # Verify structure
    assert explanation.local is not None
    assert "prediction" in explanation.local
    assert explanation.metadata is not None
    assert "shap_used" in explanation.metadata
    
    # Should have either SHAP values or approximate contributions
    has_shap = "shap_values" in explanation.local
    has_approx = "approximate_contributions" in explanation.local
    assert has_shap or has_approx
    
    # Should have probabilities for classifier
    assert "probabilities" in explanation.local
    
    # Text should be present
    assert explanation.text is not None
    assert "Random Forest Prediction" in explanation.text
    
    print(f"✓ RF local explanation works (SHAP used: {explanation.metadata['shap_used']})")


def test_random_forest_global_explanation():
    """Test global explanations for random forest."""
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    
    adapter = RandomForestAdapter(name="test_rf", model=clf)
    feature_names = [f"f{i}" for i in range(4)]
    explainer = RandomForestExplainer(feature_names=feature_names)
    
    explanation = explainer.explain_global(adapter, X, y)
    
    # Verify structure
    assert explanation.global_ is not None
    assert "n_estimators" in explanation.global_
    assert "feature_importances" in explanation.global_
    assert "n_features" in explanation.global_
    
    # Check importances
    importances = explanation.global_["feature_importances"]
    assert len(importances) > 0
    
    for feat, imp in importances.items():
        assert 0 <= imp <= 1
    
    # Text summary
    assert explanation.text is not None
    assert "Random Forest Summary" in explanation.text
    
    print("✓ RF global explanation includes feature importances")


def test_xgboost_available():
    """Test XGBoost explainer when xgboost is available."""
    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        xgb_available = False
        print("⚠ XGBoost not available, skipping xgboost tests")
        return
    
    if xgb_available:
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50)
        
        clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
        clf.fit(X, y)
        
        adapter = XGBoostAdapter(name="test_xgb", model=clf)
        feature_names = [f"feat_{i}" for i in range(4)]
        explainer = XGBoostExplainer(feature_names=feature_names)
        
        # Local explanation
        explanation = explainer.explain_local(adapter, X[0])
        
        assert explanation.local is not None
        assert "prediction" in explanation.local
        
        # Should have contributions or fallback
        has_contribs = "contributions" in explanation.local
        has_fallback = "fallback_mode" in explanation.local
        assert has_contribs or has_fallback
        
        # Text present
        assert explanation.text is not None
        assert "XGBoost Prediction" in explanation.text
        
        print(f"✓ XGBoost local explanation works (contributions: {has_contribs})")


def test_xgboost_global_explanation():
    """Test global explanation for XGBoost."""
    try:
        import xgboost as xgb
    except ImportError:
        print("⚠ XGBoost not available, skipping test")
        return
    
    X = np.random.randn(60, 5)
    y = np.random.randint(0, 2, 60)
    
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X, y)
    
    adapter = XGBoostAdapter(name="test_xgb", model=clf)
    feature_names = [f"feature_{i}" for i in range(5)]
    explainer = XGBoostExplainer(feature_names=feature_names)
    
    explanation = explainer.explain_global(adapter, X, y)
    
    # Should have importance scores
    assert explanation.global_ is not None
    assert "n_features" in explanation.global_
    
    # Should have at least one importance type
    has_gain = "importance_gain" in explanation.global_
    has_weight = "importance_weight" in explanation.global_
    has_cover = "importance_cover" in explanation.global_
    assert has_gain or has_weight or has_cover
    
    # Text
    assert explanation.text is not None
    assert "XGBoost Summary" in explanation.text
    
    print("✓ XGBoost global explanation includes importance scores")


def test_shap_graceful_degradation():
    """Test that RF explainer degrades gracefully without SHAP."""
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)
    
    clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    clf.fit(X, y)
    
    adapter = RandomForestAdapter(name="test", model=clf)
    
    # Explicitly disable SHAP
    explainer = RandomForestExplainer(use_shap=False)
    
    explanation = explainer.explain_local(adapter, X[0])
    
    # Should work without SHAP
    assert explanation.local is not None
    assert explanation.metadata["shap_used"] == False
    
    # Should provide approximate contributions
    assert "approximate_contributions" in explanation.local
    
    print("✓ RF explainer degrades gracefully without SHAP")


def test_explanation_shapes_consistency():
    """Test that explanation outputs have consistent shapes."""
    X = np.random.randn(40, 6)
    y = np.random.randint(0, 2, 40)
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    
    adapter = RandomForestAdapter(name="test", model=clf)
    explainer = RandomForestExplainer()
    
    # Get multiple local explanations
    explanations = []
    for i in range(5):
        exp = explainer.explain_local(adapter, X[i])
        explanations.append(exp)
    
    # All should have consistent structure
    for exp in explanations:
        assert exp.local is not None
        assert "prediction" in exp.local
        assert exp.text is not None
        assert exp.metadata is not None
    
    # All should have same metadata keys
    first_keys = set(explanations[0].metadata.keys())
    for exp in explanations[1:]:
        assert set(exp.metadata.keys()) == first_keys
    
    print("✓ Explanation shapes are consistent across samples")


def test_multiclass_rf_explanation():
    """Test RF explanation with multiclass problem."""
    X = np.random.randn(60, 4)
    y = np.random.randint(0, 3, 60)  # 3 classes
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    
    adapter = RandomForestAdapter(name="test", model=clf)
    explainer = RandomForestExplainer()
    
    explanation = explainer.explain_local(adapter, X[0])
    
    # Should handle multiclass
    assert explanation.local is not None
    assert explanation.local["prediction"] in [0, 1, 2]
    
    # Probabilities should have 3 classes
    if "probabilities" in explanation.local:
        assert len(explanation.local["probabilities"]) == 3
    
    print("✓ RF explainer handles multiclass classification")


def test_feature_names_in_explanations():
    """Test that feature names appear in explanations when provided."""
    X = np.random.randn(30, 4)
    y = np.random.randint(0, 2, 30)
    
    clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    clf.fit(X, y)
    
    adapter = RandomForestAdapter(name="test", model=clf)
    feature_names = ["length", "width", "height", "weight"]
    explainer = RandomForestExplainer(feature_names=feature_names, use_shap=False)
    
    # Global explanation
    global_exp = explainer.explain_global(adapter, X, y)
    
    # Feature names should appear in importances
    importance_keys = set(global_exp.global_["feature_importances"].keys())
    assert any(name in importance_keys for name in feature_names)
    
    # Local explanation
    local_exp = explainer.explain_local(adapter, X[0])
    
    # Feature names should appear in contributions
    if "approximate_contributions" in local_exp.local:
        contrib_feats = [feat for feat, _ in local_exp.local["approximate_contributions"]]
        assert any(name in contrib_feats for name in feature_names)
    
    print("✓ Feature names are used in explanations when provided")


if __name__ == "__main__":
    print("Running RF and XGBoost explanation tests...")
    test_random_forest_local_explanation()
    test_random_forest_global_explanation()
    test_xgboost_available()
    test_xgboost_global_explanation()
    test_shap_graceful_degradation()
    test_explanation_shapes_consistency()
    test_multiclass_rf_explanation()
    test_feature_names_in_explanations()
    print("\nAll RF/XGB explanation tests passed! ✓")
