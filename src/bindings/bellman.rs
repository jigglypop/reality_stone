use crate::layers::bellman::{
    compute_diagonal_geodesic_backward, compute_diagonal_geodesic_update,
};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Applies the Bellman-Lagrangian Geodesic flow to the input features.
///
/// This layer deforms the input state based on a learnable metric (here approximated by Sigmoid)
/// to follow the geodesic path of the induced manifold.
///
/// Args:
///     x: Input tensor (Batch x Dim)
///     dt: Time step for the flow (default=0.1). Controls how far along the geodesic to move.
#[pyfunction]
#[pyo3(name = "bellman_geodesic_forward")]
pub fn bellman_geodesic_forward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    dt: Option<f64>,
) -> &'py PyArray2<f64> {
    let input = x.as_array();
    let step = dt.unwrap_or(0.1);

    let output = compute_diagonal_geodesic_update(&input, step);

    output.into_pyarray(py)
}

/// Backward pass for the Bellman-Lagrangian Geodesic flow.
#[pyfunction]
#[pyo3(name = "bellman_geodesic_backward")]
pub fn bellman_geodesic_backward<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f64>,
    input: PyReadonlyArray2<f64>,
    dt: Option<f64>,
) -> &'py PyArray2<f64> {
    let grad = grad_output.as_array();
    let inp = input.as_array();
    let step = dt.unwrap_or(0.1);

    let grad_input = compute_diagonal_geodesic_backward(&grad, &inp, step);

    grad_input.into_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bellman_geodesic_forward, m)?)?;
    m.add_function(wrap_pyfunction!(bellman_geodesic_backward, m)?)?;
    Ok(())
}
