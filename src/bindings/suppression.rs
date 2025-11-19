use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, ToPyArray};
use crate::layers::suppression::compute_dynamic_suppression;

#[pyfunction]
pub fn compute_suppression_field<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    base: f32,
    linear: f32,
    hyp: f32,
    scale: f32,
) -> &'py PyArray2<f32> {
    let x = x.as_array();
    let out = compute_dynamic_suppression(&x, base, linear, hyp, scale);
    out.to_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_suppression_field, m)?)?;
    Ok(())
}

