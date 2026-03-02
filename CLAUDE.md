# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CADL8.5 is a **Complete Anytime Decision Tree Learning** framework in Rust, extending the DL8.5 algorithm with rule-based restart strategies inspired by Complete Anytime Beam Search (CABS). It finds high-quality decision tree solutions quickly and improves them iteratively while guaranteeing eventual optimality.

## Commands

```bash
# Build
cargo build --release

# Run tests
cargo test
cargo test -- --include-ignored       # include slow tests
cargo test -- --test-threads=1        # run sequentially

# Format / lint
cargo fmt
cargo fmt -- --check
cargo clippy

# Run an example algorithm
cargo run --release --example gain -- \
    --input test_data/anneal.txt \
    --depth 6 \
    --support 1 \
    --timeout 300 \
    --result .
```

Available examples: `basic`, `discrepancy`, `gain`, `gainlds`, `gaintopk`, `lds`, `purity`, `restart`, `topk`.

## Architecture

### Core Modules (`src/`)

**`algorithms/`** — All search algorithms, organized as:
- `optimal/dl85/` — Main DL85 algorithm (branch-and-bound with caching and restart rules). Entry point for all optimal search.
- `optimal/depth2/` — Fast optimal solver specialized for depth-2 subtrees; used as a subroutine by DL85.
- `optimal/rules/` — Modular restart/pruning rules: `DiscrepancyRule`, `GainRule`, `PurityRule`, `TopkRule`. Rules can be composed (see `CompositeRule`).
- `greedy/` — LGDT greedy baseline for fast initial solutions.
- `common/` — Shared types, `Config`, error functions (misclassification, information gain), and heuristics (`NoHeuristic`, `InformationGain`, `GiniIndex`).

**`cover/`** — Efficient bitset-based dataset representation. Encodes which rows are covered by a given attribute assignment. Supports reversible (backtrackable) operations via `search_trail`.

**`caching/`** — Trie-based memoization of subproblems. The `Caching` trait abstracts different cache backends; `CacheEntry` stores lower/upper bounds and best trees found.

**`tree/`** — Vector-indexed binary tree (`Tree`, `TreeNode`). Nodes store test attribute, error, class output, and optional gain metrics. Uses a `NodeUpdater` builder for mutations.

**`reader/`** — Parses CSV/TSV/space-separated datasets into `Cover` objects ready for the algorithm.

**`parser.rs`** — CLI argument parsing (via `clap`) and JSON result serialization.

**`globals.rs`** — Shared utility functions (entropy, attribute index mapping).

### Algorithm Flow

1. `Reader` loads dataset → `Cover` (bitset representation)
2. `DL85Builder` configures the search (depth, support, heuristic, rules, time limit)
3. DL85 runs branch-and-bound, maintaining a `Caching` trie and applying `Rules` to prune/restart
4. Depth-2 solver handles leaf subproblems optimally and fast
5. Anytime solutions are serialized to JSON in the `--result` directory as they improve

### Rule System

Rules implement a trait that the DL85 loop queries at each node to decide whether to skip, prune, or restart. Rules are composable—examples combine `GainRule + LDSRule` or `GainRule + TopkRule`. The discrepancy strategies (`Monotonic`, `Exponential`, `Luby`) control the restart sequence.

## Key CLI Options

| Option | Description |
|---|---|
| `--input <PATH>` | Dataset file |
| `--depth <N>` | Max tree depth |
| `--support <N>` | Min leaf support (default 5) |
| `--timeout <SECS>` | Time limit (default 300) |
| `--heuristic` | `none`, `information-gain`, `gini-index` |
| `--step` | Discrepancy strategy: `monotonic`, `exponential`, `luby` |
| `--fast-d2` | `enabled` / `disabled` (depth-2 optimization) |
| `--result <DIR>` | Output directory for JSON results |

## Test Data

`test_data/` contains 19+ benchmark datasets (anneal, audiology, iris, mushroom, vote, etc.). Integration tests in `tests/dl85.rs` use macro-generated test cases that validate expected error values with floating-point tolerance.

## Code Style

`rustfmt.toml` sets `max_width = 100`. Pre-commit hooks enforce formatting, YAML/TOML validity, and trailing whitespace. Run `cargo fmt` before committing.
