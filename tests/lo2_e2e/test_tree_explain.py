"""Test decision tree explainability with readable paths.

This test verifies that decision tree explanations include ordered nodes,
feature thresholds, directions, and human-readable text summaries.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.lo2_e2e.core.adapters import DecisionTreeAdapter
from demo.lo2_e2e.core.explainers import DecisionTreeExplainer


def test_decision_tree_path_extraction():
    """Test that decision paths are correctly extracted from trees."""
    # Create simple dataset
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    y = np.array([0, 0, 1, 1])
    
    # Train tree
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)
    
    # Create adapter
    adapter = DecisionTreeAdapter(name="test_dt", model=clf)
    
    # Create explainer with feature names
    feature_names = ["feature_a", "feature_b"]
    explainer = DecisionTreeExplainer(feature_names=feature_names)
    
    # Get explanation for first sample
    X_row = X[0]
    explanation = explainer.explain_local(adapter, X_row)
    
    # Verify structure
    assert explanation.local is not None
    assert "path_nodes" in explanation.local
    assert "path_details" in explanation.local
    assert "prediction" in explanation.local
    
    # Verify path details have required fields
    path_details = explanation.local["path_details"]
    assert len(path_details) > 0
    
    for node in path_details:
        assert "node_id" in node
        assert "impurity" in node
        assert "n_samples" in node
        assert "value" in node
        
        # Internal nodes should have feature info
        if node.get("threshold") is not None:
            assert "feature_index" in node
            assert "feature_name" in node
            assert "feature_value" in node
            assert "direction" in node
            assert node["direction"] in ["left", "right"]
    
    # Verify text is readable
    assert explanation.text is not None
    assert "Decision Path:" in explanation.text
    assert "Prediction:" in explanation.text
    
    # Should mention feature names
    assert "feature_a" in explanation.text or "feature_b" in explanation.text
    
    print("✓ Decision path extraction works correctly")


def test_decision_tree_explanation_json_serializable():
    """Test that explanations can be serialized to JSON."""
    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, 20)
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    adapter = DecisionTreeAdapter(name="test", model=clf)
    explainer = DecisionTreeExplainer(feature_names=[f"f{i}" for i in range(4)])
    
    explanation = explainer.explain_local(adapter, X[0])
    
    # Should be JSON serializable
    try:
        json_str = json.dumps({
            "local": explanation.local,
            "text": explanation.text,
            "metadata": explanation.metadata
        })
        assert len(json_str) > 0
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Explanation not JSON serializable: {e}")
    
    print("✓ Explanations are JSON serializable")


def test_decision_tree_path_ordered():
    """Test that nodes in path are in traversal order."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    y = np.array([0, 1, 1])
    
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)
    
    adapter = DecisionTreeAdapter(name="test", model=clf)
    explainer = DecisionTreeExplainer()
    
    explanation = explainer.explain_local(adapter, X[0])
    
    # Node IDs should be in increasing order (tree traversal order)
    node_ids = explanation.local["path_nodes"]
    assert len(node_ids) > 1
    assert node_ids[0] == 0  # Root node
    
    # Each node should be child of previous (monotonic path)
    for i in range(len(node_ids) - 1):
        assert node_ids[i] < node_ids[i+1]
    
    print("✓ Decision path nodes are in correct traversal order")


def test_decision_tree_global_explanation():
    """Test global tree explanation with feature importances."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    
    adapter = DecisionTreeAdapter(name="test", model=clf)
    feature_names = [f"feature_{i}" for i in range(5)]
    explainer = DecisionTreeExplainer(feature_names=feature_names)
    
    explanation = explainer.explain_global(adapter, X, y)
    
    # Verify global structure
    assert explanation.global_ is not None
    assert "n_nodes" in explanation.global_
    assert "n_leaves" in explanation.global_
    assert "max_depth" in explanation.global_
    assert "feature_importances" in explanation.global_
    
    # Verify importances
    importances = explanation.global_["feature_importances"]
    assert len(importances) > 0
    
    # Importances should be between 0 and 1
    for feat, imp in importances.items():
        assert 0 <= imp <= 1
        assert any(fname in feat for fname in feature_names)
    
    # Verify text summary
    assert explanation.text is not None
    assert "Decision Tree Summary" in explanation.text
    assert "Feature Importances" in explanation.text
    
    print("✓ Global tree explanation includes importances and summary")


def test_decision_tree_multiclass():
    """Test that explainer handles multiclass classification."""
    X = np.random.randn(60, 3)
    y = np.random.randint(0, 3, 60)  # 3 classes
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    adapter = DecisionTreeAdapter(name="test", model=clf)
    explainer = DecisionTreeExplainer()
    
    explanation = explainer.explain_local(adapter, X[0])
    
    # Should still produce valid explanation
    assert explanation.local is not None
    assert "prediction" in explanation.local
    assert explanation.local["prediction"] in [0, 1, 2]
    
    # Path should still be valid
    assert len(explanation.local["path_details"]) > 0
    
    print("✓ Decision tree explainer handles multiclass problems")


def test_compact_text_summary():
    """Test that text summary is compact and readable."""
    X = np.random.randn(40, 4)
    y = np.random.randint(0, 2, 40)
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    adapter = DecisionTreeAdapter(name="test", model=clf)
    feature_names = ["length", "duration", "word_count", "error_rate"]
    explainer = DecisionTreeExplainer(feature_names=feature_names)
    
    explanation = explainer.explain_local(adapter, X[0])
    
    text = explanation.text
    lines = text.split("\n")
    
    # Should have header and prediction
    assert any("Decision Path" in line for line in lines)
    assert any("Prediction" in line for line in lines)
    
    # Should have node descriptions
    node_lines = [l for l in lines if "Node" in l or "Leaf" in l]
    assert len(node_lines) > 0
    
    # Each node line should be reasonably compact (less than 150 chars)
    for line in node_lines:
        assert len(line) < 150
    
    # Should use friendly feature names
    text_lower = text.lower()
    assert any(name.lower() in text_lower for name in feature_names)
    
    print("✓ Text summaries are compact and readable")


if __name__ == "__main__":
    print("Running decision tree explanation tests...")
    test_decision_tree_path_extraction()
    test_decision_tree_explanation_json_serializable()
    test_decision_tree_path_ordered()
    test_decision_tree_global_explanation()
    test_decision_tree_multiclass()
    test_compact_text_summary()
    print("\nAll tree explanation tests passed! ✓")
