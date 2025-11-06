# Explanation Output Format

This document describes the explanation output format for different model types in the refactored pipeline.

## Overview

Explanations are generated in a unified format that includes:
- **Local explanations**: Per-instance predictions and feature contributions
- **Global explanations**: Model-wide statistics and feature importances
- **Text summaries**: Human-readable descriptions
- **Metadata**: Information about the explainer and available features

## Output Format

Explanations are saved as JSONL (JSON Lines) format, with one JSON object per line.

### Local Explanation Record

```json
{
  "model": "rf_v1",
  "instance_id": 42,
  "prediction": 1,
  "explanation": {
    "local": { ... },
    "global": null
  },
  "text": "Human-readable summary...",
  "metadata": {
    "explainer_type": "random_forest",
    "shap_used": true
  }
}
```

### Global Explanation Record

```json
{
  "model": "rf_v1",
  "instance_id": "global",
  "explanation": {
    "local": null,
    "global": { ... }
  },
  "text": "Model summary...",
  "metadata": {
    "explainer_type": "random_forest"
  }
}
```

## Decision Tree Explanations

### Local Explanation

Decision tree local explanations include the complete decision path from root to leaf.

```json
{
  "model": "dt_v1",
  "instance_id": 0,
  "prediction": 1,
  "explanation": {
    "local": {
      "path_nodes": [0, 1, 3, 7],
      "path_details": [
        {
          "node_id": 0,
          "feature_index": 2,
          "feature_name": "duration_sec",
          "feature_value": 45.3,
          "threshold": 30.0,
          "direction": "right",
          "impurity": 0.48,
          "n_samples": 100,
          "value": [[40, 60]]
        },
        {
          "node_id": 3,
          "feature_index": 0,
          "feature_name": "seq_len",
          "feature_value": 120.0,
          "threshold": 150.0,
          "direction": "left",
          "impurity": 0.32,
          "n_samples": 60,
          "value": [[10, 50]]
        },
        {
          "node_id": 7,
          "feature_index": -2,
          "threshold": null,
          "impurity": 0.12,
          "n_samples": 25,
          "value": [[2, 23]]
        }
      ],
      "prediction": 1
    }
  },
  "text": "Decision Path:\n  1. Node 0: duration_sec = 45.3000 > 30.0000 (impurity=0.4800, n=100)\n  2. Node 3: seq_len = 120.0000 <= 150.0000 (impurity=0.3200, n=60)\n  3. Leaf 7: value=[[2, 23]] (n=25)\n\nPrediction: 1",
  "metadata": {
    "explainer_type": "decision_tree"
  }
}
```

**Key Fields:**
- `path_nodes`: Ordered list of node IDs traversed
- `path_details`: Detailed information for each node
  - `node_id`: Unique node identifier
  - `feature_name`: Name of the split feature (or null for leaf)
  - `feature_value`: Value of the feature for this instance
  - `threshold`: Split threshold
  - `direction`: "left" or "right" for internal nodes
  - `impurity`: Node impurity (Gini, entropy, etc.)
  - `n_samples`: Number of training samples at this node
  - `value`: Class distribution at this node

### Global Explanation

```json
{
  "model": "dt_v1",
  "instance_id": "global",
  "explanation": {
    "global": {
      "n_nodes": 15,
      "n_leaves": 8,
      "max_depth": 4,
      "n_features": 5,
      "feature_importances": {
        "duration_sec": 0.4523,
        "seq_len": 0.3211,
        "error_count": 0.1456,
        "word_count": 0.0810
      }
    }
  },
  "text": "Decision Tree Summary:\n  Nodes: 15\n  Leaves: 8\n  Max Depth: 4\n\nTop Feature Importances:\n  duration_sec: 0.4523\n  seq_len: 0.3211\n  error_count: 0.1456",
  "metadata": {
    "explainer_type": "decision_tree"
  }
}
```

## Random Forest Explanations

### Local Explanation (with SHAP)

When SHAP is available, Random Forest explanations include SHAP values:

```json
{
  "model": "rf_v1",
  "instance_id": 5,
  "prediction": 1,
  "explanation": {
    "local": {
      "prediction": 1,
      "probabilities": [0.23, 0.77],
      "shap_values": [0.12, -0.08, 0.34, -0.02, 0.15],
      "top_contributions": [
        ["duration_sec", 0.34],
        ["error_rate", 0.15],
        ["seq_len", 0.12],
        ["word_count", -0.08],
        ["log_size", -0.02]
      ]
    }
  },
  "text": "Random Forest Prediction: 1\nProbabilities: ['0.2300', '0.7700']\n\nTop SHAP Contributions:\n  duration_sec: +0.3400\n  error_rate: +0.1500\n  seq_len: +0.1200\n  word_count: -0.0800\n  log_size: -0.0200",
  "metadata": {
    "explainer_type": "random_forest",
    "shap_used": true
  }
}
```

### Local Explanation (without SHAP)

When SHAP is not available, approximate contributions are provided:

```json
{
  "model": "rf_v1",
  "instance_id": 5,
  "prediction": 1,
  "explanation": {
    "local": {
      "prediction": 1,
      "probabilities": [0.23, 0.77],
      "approximate_contributions": [
        ["duration_sec", 12.56],
        ["error_rate", 8.34],
        ["seq_len", 5.12]
      ]
    }
  },
  "text": "Random Forest Prediction: 1\nProbabilities: ['0.2300', '0.7700']\n\nTop Approximate Contributions (importance × value):\n  duration_sec: +12.5600\n  error_rate: +8.3400\n  seq_len: +5.1200",
  "metadata": {
    "explainer_type": "random_forest",
    "shap_used": false
  }
}
```

### Global Explanation

```json
{
  "model": "rf_v1",
  "instance_id": "global",
  "explanation": {
    "global": {
      "n_estimators": 100,
      "n_features": 5,
      "feature_importances": {
        "duration_sec": 0.3245,
        "error_rate": 0.2834,
        "seq_len": 0.1923,
        "word_count": 0.1234,
        "log_size": 0.0764
      }
    }
  },
  "text": "Random Forest Summary:\n  Trees: 100\n  Features: 5\n\nTop Feature Importances:\n  duration_sec: 0.3245\n  error_rate: 0.2834\n  seq_len: 0.1923",
  "metadata": {
    "explainer_type": "random_forest"
  }
}
```

## XGBoost Explanations

### Local Explanation (with contributions)

When `pred_contribs` is available:

```json
{
  "model": "xgb_v1",
  "instance_id": 3,
  "prediction": 1,
  "explanation": {
    "local": {
      "prediction": 1,
      "probabilities": [0.15, 0.85],
      "contributions": [0.23, -0.12, 0.45, 0.08, -0.03, 0.5],
      "top_contributions": [
        ["duration_sec", 0.45],
        ["seq_len", 0.23],
        ["error_count", 0.08],
        ["word_freq", -0.12]
      ],
      "bias": 0.5
    }
  },
  "text": "XGBoost Prediction: 1\nProbabilities: ['0.1500', '0.8500']\n\nTop Feature Contributions:\n  duration_sec: +0.4500\n  seq_len: +0.2300\n  error_count: +0.0800\n  word_freq: -0.1200\n  Bias: +0.5000",
  "metadata": {
    "explainer_type": "xgboost",
    "contributions_used": true
  }
}
```

### Local Explanation (fallback mode)

When contributions are not available:

```json
{
  "model": "xgb_v1",
  "instance_id": 3,
  "prediction": 1,
  "explanation": {
    "local": {
      "prediction": 1,
      "probabilities": [0.15, 0.85],
      "fallback_mode": "global_importances",
      "global_importances": {
        "f0": 245.3,
        "f1": 189.2,
        "f2": 156.8
      }
    }
  },
  "text": "XGBoost Prediction: 1\nProbabilities: ['0.1500', '0.8500']\n\nNote: Local contributions unavailable, using global_importances",
  "metadata": {
    "explainer_type": "xgboost",
    "contributions_used": false
  }
}
```

### Global Explanation

XGBoost provides multiple importance types:

```json
{
  "model": "xgb_v1",
  "instance_id": "global",
  "explanation": {
    "global": {
      "n_features": 5,
      "importance_gain": {
        "duration_sec": 234.56,
        "error_rate": 189.23,
        "seq_len": 145.67
      },
      "importance_weight": {
        "duration_sec": 87,
        "seq_len": 65,
        "error_rate": 54
      },
      "importance_cover": {
        "duration_sec": 1245.3,
        "error_rate": 892.1,
        "seq_len": 678.4
      }
    }
  },
  "text": "XGBoost Summary:\n  Features: 5\n\nTop Features by Gain:\n  duration_sec: 234.5600\n  error_rate: 189.2300\n  seq_len: 145.6700\n\nTop Features by Weight:\n  duration_sec: 87.0000\n  seq_len: 65.0000",
  "metadata": {
    "explainer_type": "xgboost"
  }
}
```

## Usage Examples

### Reading Explanations

```python
import json

# Read JSONL file
with open("explanations.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        
        if record["instance_id"] == "global":
            # Global explanation
            print("Global importance:", record["explanation"]["global"])
        else:
            # Local explanation
            print(f"Instance {record['instance_id']}: {record['text']}")
```

### Filtering by Model Type

```python
import json

def load_explanations(filepath, model_type=None):
    explanations = []
    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line)
            if model_type is None or record["metadata"]["explainer_type"] == model_type:
                explanations.append(record)
    return explanations

# Get only decision tree explanations
dt_explanations = load_explanations("explanations.jsonl", model_type="decision_tree")
```

## Metadata Fields

All explanations include a `metadata` dictionary with the following common fields:

- `explainer_type`: Type of explainer used ("decision_tree", "random_forest", "xgboost")
- Additional fields vary by explainer:
  - Random Forest: `shap_used` (boolean)
  - XGBoost: `contributions_used` (boolean)

## Notes

1. **Feature Names**: When feature names are not provided, features are identified as `feature_0`, `feature_1`, etc.

2. **SHAP Availability**: Random Forest explainer attempts to use SHAP if available, but gracefully degrades to approximate contributions based on feature importances.

3. **XGBoost Contributions**: XGBoost local explanations prefer `pred_contribs` for SHAP-like contributions but fall back to global importances if unavailable.

4. **Binary vs Multiclass**: The format supports both binary and multiclass classification. For multiclass:
   - `probabilities` contains values for all classes
   - SHAP values may be class-specific
   - Decision tree `value` shows distribution across all classes

5. **JSON Compatibility**: All numeric values are converted to Python float/int for JSON serialization. NumPy types are converted automatically.
