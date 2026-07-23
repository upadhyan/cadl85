"""
cadl85 — Complete Anytime Decision Tree Learning

A scikit-learn compatible decision tree classifier based on the DL8.5 algorithm
with anytime restart strategies.

Example
-------
>>> import numpy as np
>>> from cadl85 import CADL85
>>> X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64)
>>> y = np.array([1, 0, 1, 0], dtype=np.int64)
>>> model = CADL85(max_depth=2, min_support=1, timeout=10.0)
>>> model.fit(X, y)
CADL85(max_depth=2, min_support=1, timeout=10.0, heuristic='information_gain')
>>> model.predict(X)
array([1, 0, 1, 0])
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from cadl85._cadl85 import CADL85 as _CADL85

__all__ = ["CADL85"]

# Accepted input types: numpy arrays, pandas DataFrames/Series, or any array-like
_ArrayLike = Union[npt.ArrayLike, "pd.DataFrame", "pd.Series"]


def _to_int64(arr: _ArrayLike, name: str) -> npt.NDArray[np.int64]:
    """Convert any array-like (including pandas objects) to a C-contiguous int64 array."""
    try:
        import pandas as pd  # optional dependency
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            arr = arr.to_numpy()
    except ImportError:
        pass
    out = np.asarray(arr, dtype=np.int64)
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out, dtype=np.int64)
    return out


def _require_binary(arr: npt.NDArray[np.int64]) -> None:
    """Raise ValueError unless every value in arr is 0 or 1."""
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(
            "X must be binary (values 0 and 1 only). CADL8.5 operates on "
            "binarized features — apply QuantileBinarizer before fitting."
        )


def _tree_depth(node) -> int:
    if node is None or "output" in node:
        return 0
    return 1 + max(_tree_depth(node["left"]), _tree_depth(node["right"]))


def _n_leaves(node) -> int:
    if node is None or "output" in node:
        return 1
    return _n_leaves(node["left"]) + _n_leaves(node["right"])


class CADL85(ClassifierMixin, BaseEstimator):
    """Complete Anytime Decision Tree Learning classifier.

    Finds optimal binary decision trees using the DL8.5 branch-and-bound
    algorithm. Provides anytime behaviour: produces valid trees quickly and
    improves them until the time limit or optimality is proven.

    Parameters
    ----------
    max_depth : int, default=4
        Maximum depth of the decision tree.
    min_support : int, default=1
        Minimum number of samples required at a leaf node.
    timeout : float, default=300.0
        Time limit in seconds. The best tree found within this time is returned.
    heuristic : {"information_gain", "gini", "none"}, default="information_gain"
        Attribute selection heuristic used to order candidates during search.

    Attributes
    ----------
    error_ : float
        Training misclassification rate after calling ``fit``.
    tree_ : dict
        The fitted decision tree as a nested dict.  Internal nodes have keys
        ``"feature"`` (int), ``"left"`` (subtree for feature==0),
        ``"right"`` (subtree for feature==1), and ``"error"`` (float).
        Leaves have keys ``"output"`` (int class label) and ``"error"``.
    statistics_ : dict
        Search statistics: ``cache_size``, ``cache_hits``, ``restarts``,
        ``search_space_size``, ``tree_error``, ``duration`` (seconds),
        ``num_attributes``, ``num_samples``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    history_ : pandas.DataFrame
        Optimization history across restarts (one row per restart), with columns
        ``"elapsed_time"``, ``"error"``, ``"restart"``, ``"cache_size"``,
        ``"search_space_size"``.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        max_depth: int = 4,
        min_support: int = 1,
        timeout: float = 300.0,
        heuristic: str = "information_gain",
    ) -> None:
        self.max_depth = max_depth
        self.min_support = min_support
        self.timeout = timeout
        self.heuristic = heuristic
        self._model = _CADL85(
            max_depth=max_depth,
            min_support=min_support,
            timeout=timeout,
            heuristic=heuristic,
        )

    def fit(self, X: _ArrayLike, y: _ArrayLike) -> "CADL85":
        """Fit the decision tree to binary training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix — values must be 0 or 1.
            Accepts numpy arrays, pandas DataFrames, or any array-like.
        y : array-like of shape (n_samples,)
            Integer class labels (non-negative).
            Accepts numpy arrays, pandas Series, or any array-like.

        Returns
        -------
        self : CADL85
            Fitted estimator (for method chaining).
        """
        X_arr = _to_int64(X, "X")
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_arr.shape}")
        _require_binary(X_arr)

        # Extract raw labels (handle pandas Series) without forcing int64 — the
        # engine only accepts integer labels, so we LabelEncode to 0..n-1 and
        # map predictions back through self.le. For already-integer y this is a
        # no-op round-trip.
        try:
            import pandas as pd  # optional dependency
            if isinstance(y, pd.Series):
                y = y.values
        except ImportError:
            pass
        y_raw = np.asarray(y)
        if y_raw.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y_raw.shape}")

        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y_raw)
        self.classes_ = self.le.classes_
        y_arr = np.ascontiguousarray(y_enc, dtype=np.int64)

        self._model.fit(X_arr, y_arr)
        return self

    def predict(self, X: _ArrayLike) -> npt.NDArray[np.int64]:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary feature matrix — values must be 0 or 1.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X_arr = _to_int64(X, "X")
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X_arr.shape}")
        _require_binary(X_arr)
        encoded = np.asarray(self._model.predict(X_arr))
        return self.le.inverse_transform(encoded)

    # ------------------------------------------------------------------
    # Post-fit attributes (mirror the Rust accessors)
    # ------------------------------------------------------------------

    @property
    def error_(self) -> float:
        """Training misclassification rate (fraction of misclassified samples)."""
        return self._model.error_

    @property
    def tree_(self) -> dict:
        """The fitted decision tree as a nested dict."""
        return self._model.tree_

    @property
    def statistics_(self) -> dict:
        """Search statistics dict."""
        return self._model.statistics_

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit."""
        return self._model.n_features_in_

    @property
    def history_(self):
        """Optimization history across restarts as a pandas DataFrame.

        Columns:
          ``elapsed_time``       – elapsed seconds since ``fit()`` was called
          ``error``              – training misclassification rate (fraction)
          ``restart``            – restart number (1-indexed)
          ``cache_size``         – number of cached subproblems at this point
          ``search_space_size``  – cumulative nodes explored at this point

        If the search completes in a single pass (e.g. small depth/dataset),
        the DataFrame will have exactly one row.
        """
        import pandas as pd
        return pd.DataFrame(self._model.history_)

    # ------------------------------------------------------------------
    # sklearn benchmark introspection
    # ------------------------------------------------------------------

    def is_solved(self) -> bool:
        """True when the search proved optimality.

        CADL8.5's anytime restart loop stops only when the search space is
        exhausted (provably optimal) or the time limit is hit. So not hitting
        the timeout is equivalent to a completed, optimal search — the same
        "optimal iff it finished before the clock" rule used for DPDT/GOSDT.
        """
        return not bool(self._model.has_timeout_)

    def get_depth(self) -> int:
        """Depth of the fitted tree (0 for a single-leaf tree)."""
        return _tree_depth(self.tree_)

    def get_n_leaves(self) -> int:
        """Number of leaves in the fitted tree."""
        return _n_leaves(self.tree_)

    @property
    def total_time_(self) -> float:
        """Total search wall-clock time in seconds."""
        return self._model.total_time_

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CADL85(max_depth={self.max_depth}, min_support={self.min_support}, "
            f"timeout={self.timeout}, heuristic={self.heuristic!r})"
        )
