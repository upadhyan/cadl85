"""
demo.py — Quick demonstration of the cadl85 Python API.

Generates a synthetic binary classification dataset, fits a CADL85 tree,
and prints predictions, training error, and search statistics.

Run:
    python demo.py
"""

import numpy as np
from cadl85 import CADL85

# ---------------------------------------------------------------------------
# 1. Create a synthetic binary dataset
# ---------------------------------------------------------------------------
# Simple rule: label = 1 iff (feature_0 == 1) AND (feature_2 == 1)
rng = np.random.default_rng(42)
n_samples, n_features = 200, 6

X = rng.integers(0, 2, size=(n_samples, n_features), dtype=np.int64)
y = ((X[:, 0] == 1) & (X[:, 2] == 1)).astype(np.int64)

# Train / test split (80 / 20)
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------------------------------------------------------
# 2. Fit the model
# ---------------------------------------------------------------------------
model = CADL85(max_depth=4, min_support=1, timeout=30.0, heuristic="information_gain")
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
test_accuracy = (y_pred == y_test).mean()

print(f"Training misclassification rate : {model.error_:.4f}")
print(f"Test  accuracy                  : {test_accuracy:.4f}")

# ---------------------------------------------------------------------------
# 4. Search statistics
# ---------------------------------------------------------------------------
stats = model.statistics_
print("\nSearch statistics:")
for key, val in stats.items():
    if isinstance(val, float):
        print(f"  {key:<25} {val:.4f}")
    else:
        print(f"  {key:<25} {val}")

# ---------------------------------------------------------------------------
# 5. Inspect the fitted tree (nested dict)
# ---------------------------------------------------------------------------
def print_tree(node: dict, indent: int = 0) -> None:
    pad = "  " * indent
    if "feature" in node:
        print(f"{pad}if feature[{node['feature']}] == 1:")
        print_tree(node["right"], indent + 1)
        print(f"{pad}else:")
        print_tree(node["left"], indent + 1)
    else:
        print(f"{pad}predict {node['output']}  (leaf error {node['error']:.4f})")

print("\nFitted tree:")
print_tree(model.tree_)
