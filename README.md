# CADL8.5: Complete Anytime DL8.5

CADL8.5 is a **complete anytime framework** for decision tree learning that extends **DL8.5** with **rule-based restart strategies** inspired by **Complete Anytime Beam Search (CABS)**.
It quickly finds high-quality solutions and improves them over time, while still guaranteeing **optimal solutions**.
Compared to DL8.5 and Blossom, CADL8.5 often solves **more instances to optimality** and performs at least as well as greedy baselines in early stages.

---

## Python Installation

Pre-built wheels are attached to each [GitHub Release](../../releases) — no Rust required.

```bash
pip install cadl85 --find-links https://github.com/<org>/cadl85/releases/expanded_assets/v0.1.0
```

Or install a specific wheel directly (replace the Python/platform tags to match your environment):

```bash
pip install https://github.com/<org>/cadl85/releases/download/v0.1.0/cadl85-0.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

**Usage:**

```python
import numpy as np
from cadl85 import CADL85

model = CADL85(max_depth=4, min_support=1, timeout=60.0, heuristic="information_gain")
model.fit(X_train, y_train)   # X: binary 0/1 array, y: integer labels
preds = model.predict(X_test)
print(model.error_)           # training misclassification rate
print(model.statistics_)      # search stats: duration, cache_size, restarts, …
```

`X` and `y` accept numpy arrays, pandas DataFrames/Series, or any array-like. Labels can be any non-negative integers. Available heuristics: `"information_gain"` (default), `"gini"`, `"none"`.

To build from source (requires Rust):

```bash
pip install maturin
maturin develop --features python
```

---

## Rust CLI Installation and Execution

This project is implemented in **Rust**, and `cargo` is required to build and run the CLI.
Clone and build:

```bash
git clone <repository-url>
cd CADL8.5
cargo build --release
```

All algorithms are provided as examples in the examples/ folder. You can run them as follows:

```bash
cargo run --release --example $ALGO -- \
    --input $DATASET \
    --depth $DEPTH \
    --support $SUPPORT \
    --timeout $TIMEOUT \
    --heuristic information-gain \
    --result $OUTPUT_DIR \
    --fast-d2 enabled
```

Where:

$ALGO is the example file name (e.g., discrepancy, topk, gain, ...)

$DATASET is the dataset path (.txt)

$DEPTH is the maximum decision tree depth

$SUPPORT is the minimum support threshold (integer)

$TIMEOUT is the time limit in seconds

$OUTPUT_DIR is the folder where results are stored

**Example** :

```bash
cargo run --release --example discrepancy -- \
    --input test_data/anneal.txt \
    --depth 6 \
    --support 1 \
    --timeout 300 \
    --result .
    --fast-d2 enabled

```


All the benchmarks results are generated with `all.sh`
