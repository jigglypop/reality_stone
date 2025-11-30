use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyfunction]
pub fn householder_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let x = x.as_array().to_owned();
        let y = crate::ops::householder_chain_apply_from_key(key, dim, num, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}

#[pyfunction]
pub fn householder_chain_apply_transpose_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let x = x.as_array().to_owned();
        let y = crate::ops::householder_chain_apply_transpose_from_key(key, dim, num, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}

#[pyfunction]
pub fn givens_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let x = x.as_array().to_owned();
        let y = crate::ops::givens_chain_apply_from_key(key, dim, num, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}

#[pyfunction]
pub fn lowrank_plus_diag_apply_from_key(
    key_u: &str,
    key_v: &str,
    s_diag: PyReadonlyArray1<f32>,
    rank: usize,
    x: PyReadonlyArray1<f32>,
) -> Py<PyArray1<f32>> {
    Python::with_gil(|py| {
        let s = s_diag.as_array().to_owned();
        let x = x.as_array().to_owned();
        let y = crate::ops::lowrank_plus_diag_apply_from_key(key_u, key_v, &s, rank, &x);
        PyArray1::from_vec(py, y.to_vec()).to_owned()
    })
}
#[pyfunction]
pub fn rotate_metric_factor_block<'py>(
    py: Python<'py>,
    key: &str,
    l: PyReadonlyArray2<f32>,
    global_dim: usize,
) -> &'py PyArray2<f32> {
    let l = l.as_array().to_owned();
    let out = crate::ops::rotate_metric_factor_block(key, &l, global_dim);
    PyArray2::from_owned_array(py, out)
}

#[pyfunction]
pub fn spd_metric_from_key_weighted<'py>(
    py: Python<'py>,
    key: &str,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
    mass: f32,
) -> &'py PyArray2<f32> {
    let g = crate::ops::spd_metric_from_key_weighted(key, dim, min_lambda, max_lambda, mass);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn compose_layers_gravity<'py>(
    py: Python<'py>,
    keys: Vec<String>,
    masses: Vec<f32>,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> &'py PyArray2<f32> {
    let t = crate::ops::compose_layers_gravity(&keys, &masses, dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, t)
}

// f64 high-precision variants
#[pyfunction]
pub fn compose_layers_gravity_f64<'py>(
    py: Python<'py>,
    keys: Vec<String>,
    masses: Vec<f64>,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
) -> &'py PyArray2<f64> {
    let t = crate::ops::compose_layers_gravity_f64(&keys, &masses, dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, t)
}

#[pyfunction]
pub fn apply_linear_f64<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
    vecs: PyReadonlyArray2<f64>,
) -> &'py PyArray2<f64> {
    let matrix = matrix.as_array().to_owned();
    let vecs = vecs.as_array().to_owned();
    let out = crate::ops::apply_linear_f64(&matrix, &vecs);
    PyArray2::from_owned_array(py, out)
}

#[pyfunction]
pub fn effective_metric_from_transform_f64<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
) -> &'py PyArray2<f64> {
    let t_arr = t.as_array().to_owned();
    let g = crate::ops::effective_metric_from_transform_f64(&t_arr);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn metric_factor_cholesky_f64<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<'py, f64>,
) -> &'py PyArray2<f64> {
    let g_arr = g.as_array().to_owned();
    let u = crate::ops::metric_factor_cholesky_f64(&g_arr);
    PyArray2::from_owned_array(py, u)
}

#[pyfunction]
pub fn compose_layers_gravity_compact_f64<'py>(
    py: Python<'py>,
    master_key: &str,
    num_layers: usize,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
    mass_base: f64,
    mass_step: f64,
) -> &'py PyArray2<f64> {
    let t = crate::ops::compose_layers_gravity_compact_f64(
        master_key, num_layers, dim, min_lambda, max_lambda, mass_base, mass_step,
    );
    PyArray2::from_owned_array(py, t)
}

// Collapsed transform wrapper (F32) for order-preserving layer composition
#[pyclass]
pub struct CollapsedTransformF32 {
    t: Array2<f32>,
    dim: usize,
}

// High-precision collapsed transform (F64) for strict verification/inference
#[pyclass]
pub struct CollapsedTransformF64 {
    t: ndarray::Array2<f64>,
    dim: usize,
}

#[pymethods]
impl CollapsedTransformF64 {
    #[new]
    fn new(t: PyReadonlyArray2<f64>) -> Self {
        let t_arr = t.as_array().to_owned();
        let dim = t_arr.dim().1;
        Self { t: t_arr, dim }
    }

    #[staticmethod]
    fn from_keys(
        keys: Vec<String>,
        masses: Vec<f64>,
        dim: usize,
        min_lambda: f64,
        max_lambda: f64,
    ) -> Self {
        let t = crate::ops::compose_layers_gravity_f64(&keys, &masses, dim, min_lambda, max_lambda);
        Self { t, dim }
    }

    #[staticmethod]
    fn from_master_key_compact(
        master_key: &str,
        num_layers: usize,
        dim: usize,
        min_lambda: f64,
        max_lambda: f64,
        mass_base: f64,
        mass_step: f64,
    ) -> Self {
        let t = crate::ops::compose_layers_gravity_compact_f64(
            master_key, num_layers, dim, min_lambda, max_lambda, mass_base, mass_step,
        );
        Self { t, dim }
    }

    fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> &'py PyArray2<f64> {
        let x_arr = x.as_array().to_owned();
        let out = crate::ops::apply_linear_f64(&self.t, &x_arr);
        PyArray2::from_owned_array(py, out)
    }

    fn matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        PyArray2::from_owned_array(py, self.t.clone())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }
}

#[pymethods]
impl CollapsedTransformF32 {
    #[new]
    fn new(t: PyReadonlyArray2<f32>) -> Self {
        let t_arr = t.as_array().to_owned();
        let dim = t_arr.dim().1;
        Self { t: t_arr, dim }
    }

    #[staticmethod]
    fn from_keys(
        keys: Vec<String>,
        masses: Vec<f32>,
        dim: usize,
        min_lambda: f32,
        max_lambda: f32,
    ) -> Self {
        let t = crate::ops::compose_layers_gravity(&keys, &masses, dim, min_lambda, max_lambda);
        Self { t, dim }
    }

    #[staticmethod]
    fn from_master_key_compact(
        master_key: &str,
        num_layers: usize,
        dim: usize,
        min_lambda: f64,
        max_lambda: f64,
        mass_base: f64,
        mass_step: f64,
    ) -> Self {
        // Use high-precision f64 path for composition, then cast to f32 for runtime apply
        let t64 = crate::ops::compose_layers_gravity_compact_f64(
            master_key, num_layers, dim, min_lambda, max_lambda, mass_base, mass_step,
        );
        let t = t64.mapv(|v| v as f32);
        Self { t, dim }
    }

    /// Apply the collapsed transform to a batch of row-vectors X: (batch, dim)
    fn apply<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>) -> &'py PyArray2<f32> {
        let x_arr = x.as_array().to_owned();
        let out = crate::ops::apply_linear(&self.t, &x_arr);
        PyArray2::from_owned_array(py, out)
    }

    /// Return the transform matrix (dim, dim)
    fn matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        PyArray2::from_owned_array(py, self.t.clone())
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }
}

#[pyfunction]
pub fn spd_metric_from_key<'py>(
    py: Python<'py>,
    key: &str,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> &'py PyArray2<f32> {
    let g = crate::ops::spd_metric_from_key(key, dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn metric_factor_cholesky<'py>(
    py: Python<'py>,
    g: PyReadonlyArray2<f32>,
) -> &'py PyArray2<f32> {
    let g = g.as_array().to_owned();
    let l = crate::ops::metric_factor_cholesky(&g);
    PyArray2::from_owned_array(py, l)
}

#[pyfunction]
pub fn mahalanobis_distance_sq_g(
    x: PyReadonlyArray1<f32>,
    y: PyReadonlyArray1<f32>,
    g: PyReadonlyArray2<f32>,
) -> f32 {
    let x = x.as_array().to_owned();
    let y = y.as_array().to_owned();
    let g = g.as_array().to_owned();
    crate::ops::mahalanobis_distance_sq_g(&x, &y, &g)
}

#[pyfunction]
pub fn mahalanobis_distance_sq_l(
    x: PyReadonlyArray1<f32>,
    y: PyReadonlyArray1<f32>,
    l: PyReadonlyArray2<f32>,
) -> f32 {
    let x = x.as_array().to_owned();
    let y = y.as_array().to_owned();
    let l = l.as_array().to_owned();
    crate::ops::mahalanobis_distance_sq_l(&x, &y, &l)
}

#[pyfunction]
pub fn block_orthogonal_from_key<'py>(
    py: Python<'py>,
    key: &str,
    global_dim: usize,
    dept_dim: usize,
) -> &'py PyArray2<f32> {
    let q = crate::ops::block_orthogonal_from_key(key, global_dim, dept_dim);
    PyArray2::from_owned_array(py, q)
}

#[pyfunction]
pub fn spd_block_metric_from_key<'py>(
    py: Python<'py>,
    key: &str,
    global_dim: usize,
    dept_dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> &'py PyArray2<f32> {
    let g =
        crate::ops::spd_block_metric_from_key(key, global_dim, dept_dim, min_lambda, max_lambda);
    PyArray2::from_owned_array(py, g)
}

#[pyfunction]
pub fn compose_layers_order_preserving<'py>(
    py: Python<'py>,
    layers: Vec<PyReadonlyArray2<f32>>,
) -> &'py PyArray2<f32> {
    let mut rust_layers = Vec::with_capacity(layers.len());
    for a in layers.into_iter() {
        rust_layers.push(a.as_array().to_owned());
    }
    let t = crate::ops::compose_layers_order_preserving(&rust_layers);
    PyArray2::from_owned_array(py, t)
}

#[pyfunction]
pub fn apply_linear<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f32>,
    vecs: PyReadonlyArray2<f32>,
) -> &'py PyArray2<f32> {
    let matrix = matrix.as_array().to_owned();
    let vecs = vecs.as_array().to_owned();
    let out = crate::ops::apply_linear(&matrix, &vecs);
    PyArray2::from_owned_array(py, out)
}

pub fn init_module(_py: Python, m: &PyModule) -> PyResult<()> {
    let sub = PyModule::new(_py, "metrikey")?;
    sub.add_function(wrap_pyfunction!(spd_metric_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(metric_factor_cholesky, sub)?)?;
    sub.add_function(wrap_pyfunction!(mahalanobis_distance_sq_g, sub)?)?;
    sub.add_function(wrap_pyfunction!(mahalanobis_distance_sq_l, sub)?)?;
    sub.add_function(wrap_pyfunction!(block_orthogonal_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(spd_block_metric_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(spd_metric_from_key_weighted, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_order_preserving, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_gravity, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_gravity_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(apply_linear, sub)?)?;
    sub.add_function(wrap_pyfunction!(apply_linear_f64, sub)?)?;
    // Exact ops exposure
    #[pyfunction]
    fn layer_norm_forward_exact_f32_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        gamma: PyReadonlyArray1<f32>,
        beta: PyReadonlyArray1<f32>,
        eps: f32,
    ) -> (&'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray1<f32>) {
        let x = x.as_array().to_owned();
        let gamma = gamma.as_array().to_owned();
        let beta = beta.as_array().to_owned();
        let (y, mu, rstd) = crate::ops::layer_norm_forward_exact_f32(&x, &gamma, &beta, eps);
        (
            PyArray2::from_owned_array(py, y),
            PyArray1::from_owned_array(py, mu),
            PyArray1::from_owned_array(py, rstd),
        )
    }
    #[pyfunction]
    fn gelu_new_f32_py<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>) -> &'py PyArray2<f32> {
        let x = x.as_array().to_owned();
        let y = crate::ops::gelu_new_f32(&x);
        PyArray2::from_owned_array(py, y)
    }
    #[pyfunction]
    fn softmax_lastdim_f32_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        let x = x.as_array().to_owned();
        let y = crate::ops::softmax_lastdim_f32(&x);
        PyArray2::from_owned_array(py, y)
    }

    sub.add_function(wrap_pyfunction!(layer_norm_forward_exact_f32_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(gelu_new_f32_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(softmax_lastdim_f32_py, sub)?)?;
    // f64 exact ops
    #[pyfunction]
    fn layer_norm_forward_exact_f64_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        gamma: PyReadonlyArray1<f64>,
        beta: PyReadonlyArray1<f64>,
        eps: f64,
    ) -> (&'py PyArray2<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>) {
        let x = x.as_array().to_owned();
        let gamma = gamma.as_array().to_owned();
        let beta = beta.as_array().to_owned();
        let (y, mu, rstd) = crate::ops::layer_norm_forward_exact_f64(&x, &gamma, &beta, eps);
        (
            PyArray2::from_owned_array(py, y),
            PyArray1::from_owned_array(py, mu),
            PyArray1::from_owned_array(py, rstd),
        )
    }
    #[pyfunction]
    fn gelu_new_f64_py<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        let x = x.as_array().to_owned();
        let y = crate::ops::gelu_new_f64(&x);
        PyArray2::from_owned_array(py, y)
    }
    #[pyfunction]
    fn softmax_lastdim_f64_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let x = x.as_array().to_owned();
        let y = crate::ops::softmax_lastdim_f64(&x);
        PyArray2::from_owned_array(py, y)
    }
    sub.add_function(wrap_pyfunction!(layer_norm_forward_exact_f64_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(gelu_new_f64_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(softmax_lastdim_f64_py, sub)?)?;
    sub.add_function(wrap_pyfunction!(effective_metric_from_transform_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(metric_factor_cholesky_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(compose_layers_gravity_compact_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(rotate_metric_factor_block, sub)?)?;
    // Implicit transforms
    sub.add_function(wrap_pyfunction!(householder_chain_apply_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        householder_chain_apply_transpose_from_key,
        sub
    )?)?;
    sub.add_function(wrap_pyfunction!(givens_chain_apply_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(lowrank_plus_diag_apply_from_key, sub)?)?;
    // Classes
    sub.add_class::<CollapsedTransformF32>()?;
    sub.add_class::<CollapsedTransformF64>()?;
    sub.add_class::<CollapsedRunnerF32>()?;
    sub.add_class::<CollapsedRunnerF64>()?;

    m.add_submodule(sub)?;
    Ok(())
}

// High-speed inference runner (CPU, f32): holds T_total, embedding matrix, and lm_head
#[pyclass]
pub struct CollapsedRunnerF32 {
    t: Array2<f32>,                     // (d, d)
    embed: Array2<f32>,                 // (vocab, d)
    lm_w: Array2<f32>,                  // (vocab, d)
    lm_b: Option<ndarray::Array1<f32>>, // (vocab)
}

// High-precision inference runner (CPU, f64)
#[pyclass]
pub struct CollapsedRunnerF64 {
    t: ndarray::Array2<f64>,            // (d, d)
    embed: ndarray::Array2<f64>,        // (vocab, d)
    lm_w: ndarray::Array2<f64>,         // (vocab, d)
    lm_b: Option<ndarray::Array1<f64>>, // (vocab)
}

#[pymethods]
impl CollapsedRunnerF64 {
    #[new]
    fn new(
        t: PyReadonlyArray2<f64>,
        embed: PyReadonlyArray2<f64>,
        lm_w: PyReadonlyArray2<f64>,
        lm_b: Option<PyReadonlyArray1<f64>>,
    ) -> Self {
        let t_arr = t.as_array().to_owned();
        let embed_arr = embed.as_array().to_owned();
        let lm_w_arr = lm_w.as_array().to_owned();
        let lm_b_arr = lm_b.map(|b| b.as_array().to_owned());
        Self {
            t: t_arr,
            embed: embed_arr,
            lm_w: lm_w_arr,
            lm_b: lm_b_arr,
        }
    }

    /// step: ids (batch,) -> logits (batch, vocab)
    fn step<'py>(&self, py: Python<'py>, ids: PyReadonlyArray1<'py, i64>) -> &'py PyArray2<f64> {
        use ndarray::Array2 as A2;
        let ids_arr = ids.as_array();
        let batch = ids_arr.len();
        let d = self.t.dim().1;
        // gather
        let mut x = A2::<f64>::zeros((batch, d));
        for (i, &tok) in ids_arr.iter().enumerate() {
            let idx = if tok < 0 { 0usize } else { tok as usize };
            let row = self.embed.row(idx);
            x.row_mut(i).assign(&row);
        }
        // x * T^T
        let xt = x.dot(&self.t.t());
        // logits = xt · W^T
        let mut logits = xt.dot(&self.lm_w.t());
        if let Some(bias) = &self.lm_b {
            for mut row in logits.rows_mut() {
                row += &bias.view();
            }
        }
        PyArray2::from_owned_array(py, logits)
    }
}

#[pymethods]
impl CollapsedRunnerF32 {
    #[new]
    fn new(
        t: PyReadonlyArray2<f32>,
        embed: PyReadonlyArray2<f32>,
        lm_w: PyReadonlyArray2<f32>,
        lm_b: Option<PyReadonlyArray1<f32>>,
    ) -> Self {
        let t_arr = t.as_array().to_owned();
        let embed_arr = embed.as_array().to_owned();
        let lm_w_arr = lm_w.as_array().to_owned();
        let lm_b_arr = lm_b.map(|b| b.as_array().to_owned());
        Self {
            t: t_arr,
            embed: embed_arr,
            lm_w: lm_w_arr,
            lm_b: lm_b_arr,
        }
    }

    /// step: ids (batch,) -> logits (batch, vocab)
    fn step<'py>(&self, py: Python<'py>, ids: PyReadonlyArray1<'py, i64>) -> &'py PyArray2<f32> {
        use ndarray::Array2;
        let ids_arr = ids.as_array();
        let batch = ids_arr.len();
        let d = self.t.dim().1;

        // Gather embeddings
        let mut x = Array2::<f32>::zeros((batch, d));
        for (i, &tok) in ids_arr.iter().enumerate() {
            let idx = if tok < 0 { 0usize } else { tok as usize };
            let row = self.embed.row(idx);
            x.row_mut(i).assign(&row);
        }
        // Apply T_total: x' = x * T^T  => (batch,d) = (batch,d) · (d,d)
        let xt = x.dot(&self.t.t());
        // Logits = x' · W^T  with W: (vocab,d) → W^T: (d,vocab)
        let mut logits = xt.dot(&self.lm_w.t());
        if let Some(bias) = &self.lm_b {
            // add bias per vocab
            for mut row in logits.rows_mut() {
                row += &bias.view();
            }
        }
        PyArray2::from_owned_array(py, logits)
    }
}
