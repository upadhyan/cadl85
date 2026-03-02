use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::algorithms::common::errors::NativeError;
use crate::algorithms::common::heuristics::{GiniIndex, InformationGain, NoHeuristic};
use crate::algorithms::common::types::{OptimalDepth2Policy, SearchStatistics};
use crate::algorithms::optimal::depth2::ErrorMinimizer;
use crate::algorithms::optimal::dl85::DL85Builder;
use crate::algorithms::TreeSearchAlgorithm;
use crate::caching::Trie;
use crate::cover::Cover;
use crate::tree::Tree;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn tree_to_pyobject(py: Python, tree: &Tree, node_idx: usize, label_map: &[i64]) -> PyObject {
    let dict = PyDict::new(py);
    if let Some(node) = tree.get_node(node_idx) {
        match node.value.test {
            None => {
                // Leaf node
                let raw_class = node.value.out.unwrap_or(0.0) as usize;
                let mapped = label_map.get(raw_class).copied().unwrap_or(raw_class as i64);
                dict.set_item("output", mapped).unwrap();
                dict.set_item("error", node.value.error).unwrap();
            }
            Some(attr) => {
                // Internal node
                dict.set_item("feature", attr).unwrap();
                let left = tree_to_pyobject(py, tree, node.left, label_map);
                let right = tree_to_pyobject(py, tree, node.right, label_map);
                // left child = feature value 0, right child = feature value 1
                dict.set_item("left", left).unwrap();
                dict.set_item("right", right).unwrap();
                dict.set_item("error", node.value.error).unwrap();
            }
        }
    }
    dict.into()
}

fn predict_one(tree: &Tree, sample: &[i64], label_map: &[i64]) -> i64 {
    let mut idx = tree.get_root_index();
    loop {
        let node = match tree.get_node(idx) {
            Some(n) => n,
            None => return label_map.first().copied().unwrap_or(0),
        };
        match node.value.test {
            None => {
                let raw = node.value.out.unwrap_or(0.0) as usize;
                return label_map.get(raw).copied().unwrap_or(raw as i64);
            }
            Some(attr) => {
                if attr < sample.len() && sample[attr] == 1 {
                    idx = node.right; // feature present → right
                } else {
                    idx = node.left; // feature absent → left
                }
            }
        }
    }
}

macro_rules! run_dl85 {
    ($cover:expr, $max_depth:expr, $min_support:expr, $timeout:expr, $heuristic:expr) => {{
        let error_fn = Box::<NativeError>::default();
        let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));
        let mut algo = DL85Builder::default()
            .max_depth($max_depth)
            .min_support($min_support)
            .max_time($timeout)
            .specialization(OptimalDepth2Policy::Enabled)
            .cache(Box::<Trie>::default())
            .heuristic($heuristic)
            .depth2_search(depth2)
            .error_function(error_fn)
            .build()
            .map_err(PyRuntimeError::new_err)?;
        algo.fit($cover)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tree = algo.tree().clone();
        let stats = algo.statistics().clone();
        (tree, stats)
    }};
}

// ---------------------------------------------------------------------------
// Public PyO3 class
// ---------------------------------------------------------------------------

#[pyclass(name = "CADL85")]
pub struct PyCadl85 {
    max_depth: usize,
    min_support: usize,
    timeout: f64,
    heuristic: String,
    // Populated after fit()
    tree: Option<Tree>,
    statistics: Option<SearchStatistics>,
    label_map: Vec<i64>,
    n_features_in: usize,
    n_samples: usize,
}

#[pymethods]
impl PyCadl85 {
    #[new]
    #[pyo3(signature = (max_depth=4, min_support=1, timeout=300.0, heuristic="information_gain"))]
    fn new(max_depth: usize, min_support: usize, timeout: f64, heuristic: &str) -> Self {
        PyCadl85 {
            max_depth,
            min_support,
            timeout,
            heuristic: heuristic.to_string(),
            tree: None,
            statistics: None,
            label_map: vec![],
            n_features_in: 0,
            n_samples: 0,
        }
    }

    /// Fit the decision tree on binary feature matrix X and label vector y.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features), dtype int/bool
    ///     Binary feature matrix (values must be 0 or 1).
    /// y : array-like of shape (n_samples,), dtype int
    ///     Class labels (non-negative integers).
    ///
    /// Returns self for method chaining.
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i64>,
        y: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<PyObject> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();

        let (mut cover, label_map) = Cover::from_arrays(x_arr, y_arr)
            .map_err(|e| PyValueError::new_err(e))?;

        self.n_features_in = x_arr.ncols();
        self.n_samples = x_arr.nrows();

        let (tree, stats) = match self.heuristic.as_str() {
            "gini_index" | "gini" => run_dl85!(
                &mut cover,
                self.max_depth,
                self.min_support,
                self.timeout,
                Box::<GiniIndex>::default()
            ),
            "none" => run_dl85!(
                &mut cover,
                self.max_depth,
                self.min_support,
                self.timeout,
                Box::<NoHeuristic>::default()
            ),
            _ => run_dl85!(
                &mut cover,
                self.max_depth,
                self.min_support,
                self.timeout,
                Box::<InformationGain>::default()
            ),
        };

        self.tree = Some(tree);
        self.statistics = Some(stats);
        self.label_map = label_map;

        Ok(py.None())
    }

    /// Predict class labels for samples in X.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features), dtype int/bool
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,) with predicted class labels.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Call fit() before predict()"))?;

        let x_arr = x.as_array();
        let n = x_arr.nrows();
        let mut preds = Vec::with_capacity(n);

        for row in 0..n {
            let sample: Vec<i64> = x_arr.row(row).iter().copied().collect();
            preds.push(predict_one(tree, &sample, &self.label_map));
        }

        Ok(preds.into_pyarray(py).into())
    }

    /// Training error as fraction of misclassified samples.
    #[getter]
    fn error_(&self) -> PyResult<f64> {
        let stats = self
            .statistics
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Call fit() first"))?;
        if self.n_samples == 0 {
            return Ok(0.0);
        }
        Ok(stats.tree_error / self.n_samples as f64)
    }

    /// The fitted decision tree as a nested dict.
    ///
    /// Internal nodes: {"feature": int, "left": ..., "right": ..., "error": float}
    /// Leaves:         {"output": int, "error": float}
    ///
    /// "left" is taken when feature == 0, "right" when feature == 1.
    #[getter]
    fn tree_(&self, py: Python) -> PyResult<PyObject> {
        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Call fit() first"))?;
        Ok(tree_to_pyobject(py, tree, tree.get_root_index(), &self.label_map))
    }

    /// Search statistics dict with keys: cache_size, cache_hits, restarts,
    /// search_space_size, tree_error, duration, num_attributes, num_samples.
    #[getter]
    fn statistics_(&self, py: Python) -> PyResult<PyObject> {
        let stats = self
            .statistics
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Call fit() first"))?;
        let dict = PyDict::new(py);
        dict.set_item("cache_size", stats.cache_size)?;
        dict.set_item("cache_hits", stats.cache_hits)?;
        dict.set_item("restarts", stats.restarts)?;
        dict.set_item("search_space_size", stats.search_space_size)?;
        dict.set_item("tree_error", stats.tree_error)?;
        dict.set_item("duration", stats.duration)?;
        dict.set_item("num_attributes", stats.num_attributes)?;
        dict.set_item("num_samples", stats.num_samples)?;
        Ok(dict.into())
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> usize {
        self.n_features_in
    }

    fn __repr__(&self) -> String {
        format!(
            "CADL85(max_depth={}, min_support={}, timeout={}, heuristic='{}')",
            self.max_depth, self.min_support, self.timeout, self.heuristic
        )
    }
}
