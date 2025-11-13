#!/usr/bin/env python3
"""Display a sample decision tree explanation to showcase glass-box explainability."""

import sys
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.lo2_e2e.core.adapters import DecisionTreeAdapter
from demo.lo2_e2e.core.explainers import DecisionTreeExplainer


def create_sample_tree():
    """Create a simple decision tree for demonstration."""
    # Create interpretable synthetic data
    np.random.seed(42)
    n = 100
    
    # Features: sequence length, duration, error count
    seq_len = np.random.randint(10, 500, n)
    duration = np.random.uniform(0.1, 120, n)
    errors = np.random.poisson(2, n)
    
    X = np.column_stack([seq_len, duration, errors])
    
    # Simple rule: anomaly if (duration > 80 and errors > 3) or (seq_len < 30 and errors > 1)
    y = np.zeros(n)
    y[(duration > 80) & (errors > 3)] = 1
    y[(seq_len < 30) & (errors > 1)] = 1
    
    # Train shallow tree
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)
    clf.fit(X, y)
    
    return clf, X, y, ["seq_len", "duration_sec", "error_count"]


def main():
    print("="*70)
    print("Glass-Box Decision Tree Explanation Demo")
    print("="*70)
    
    # Create sample tree
    clf, X, y, feature_names = create_sample_tree()
    
    print("\n📊 Model Statistics:")
    print(f"  Tree depth: {clf.tree_.max_depth}")
    print(f"  Number of nodes: {clf.tree_.node_count}")
    print(f"  Number of leaves: {np.sum(clf.tree_.children_left == -1)}")
    print(f"  Training samples: {len(X)}")
    print(f"  Features: {', '.join(feature_names)}")
    
    # Create adapter and explainer
    adapter = DecisionTreeAdapter(name="demo_dt", model=clf)
    explainer = DecisionTreeExplainer(feature_names=feature_names)
    
    # Pick an interesting sample (one that's an anomaly)
    anomaly_idx = np.where(y == 1)[0]
    if len(anomaly_idx) > 0:
        sample_idx = anomaly_idx[0]
    else:
        sample_idx = 0
    
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]
    
    print(f"\n🔍 Sample Instance #{sample_idx}:")
    print(f"  True label: {'ANOMALY' if y_sample == 1 else 'NORMAL'}")
    print(f"  Features:")
    for i, (name, val) in enumerate(zip(feature_names, X_sample)):
        print(f"    {name}: {val:.2f}")
    
    # Get explanation
    explanation = explainer.explain_local(adapter, X_sample)
    
    print("\n" + "="*70)
    print("🌳 DECISION PATH (Glass-Box Explanation)")
    print("="*70)
    print()
    print(explanation.text)
    
    print("\n" + "="*70)
    print("📋 STRUCTURED PATH DATA (JSON-ready)")
    print("="*70)
    
    print("\nPath nodes:", explanation.local["path_nodes"])
    print(f"\nNumber of nodes traversed: {len(explanation.local['path_details'])}")
    
    print("\n🔸 Node Details:")
    for i, node in enumerate(explanation.local["path_details"]):
        print(f"\n  Node {i} (ID: {node['node_id']}):")
        if node.get("threshold") is not None:
            print(f"    Feature: {node.get('feature_name', 'N/A')}")
            print(f"    Threshold: {node['threshold']:.4f}")
            print(f"    Sample value: {node['feature_value']:.4f}")
            print(f"    Direction: {node['direction']} (sample goes {'left' if node['direction'] == 'left' else 'right'})")
        else:
            print(f"    Type: LEAF NODE")
        print(f"    Impurity: {node['impurity']:.4f}")
        print(f"    Training samples at this node: {node['n_samples']}")
        print(f"    Class distribution: {node['value']}")
    
    # Global explanation
    print("\n" + "="*70)
    print("🌍 GLOBAL MODEL EXPLANATION")
    print("="*70)
    
    global_exp = explainer.explain_global(adapter, X, y)
    print()
    print(global_exp.text)
    
    print("\n" + "="*70)
    print("✅ Demo Complete")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Complete decision path from root to leaf")
    print("  ✓ Feature names, thresholds, and directions")
    print("  ✓ Node-level statistics (impurity, samples)")
    print("  ✓ Both human-readable and JSON-structured output")
    print("  ✓ Global feature importance rankings")
    print()


if __name__ == "__main__":
    main()
