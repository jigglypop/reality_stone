use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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
    sub.add_function(wrap_pyfunction!(compose_layers_gravity_compact_f64, sub)?)?;
    sub.add_function(wrap_pyfunction!(rotate_metric_factor_block, sub)?)?;
    // Implicit transforms
    sub.add_function(wrap_pyfunction!(householder_chain_apply_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(householder_chain_apply_transpose_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(givens_chain_apply_from_key, sub)?)?;
    sub.add_function(wrap_pyfunction!(lowrank_plus_diag_apply_from_key, sub)?)?;
    m.add_submodule(sub)?;
    Ok(())
}
