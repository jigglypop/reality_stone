use crate::ops::extraction;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "extract_metric_cuda")]
pub fn extract_metric_cuda_py<'py>(
    py: Python<'py>,
    w: PyReadonlyArray2<f32>,
    calibration_data: PyReadonlyArray2<f32>,
    target_dim: usize,
    num_steps: usize,
    curvature: f32,
    lr: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, &'py PyArray2<f32>) {
    let w_view = w.as_array();
    let calib_view = calibration_data.as_array();

    let (u, g, v) = py.allow_threads(move || {
        extraction::extract_metric_cuda(w_view, calib_view, target_dim, num_steps, curvature, lr)
    });

    (u.into_pyarray(py), g.into_pyarray(py), v.into_pyarray(py))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_metric_cuda_py, m)?)?;
    Ok(())
}
