use crate::ops::extraction;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Random-projection metric extraction on the CPU. This is the honest name for the
/// path that used to be served silently by `extract_metric_cuda` in non-CUDA builds;
/// `calibration_data`, `num_steps`, `curvature` and `lr` are accepted for signature
/// parity but ignored.
#[pyfunction]
#[pyo3(name = "extract_metric_random_projection")]
pub fn extract_metric_random_projection_py<'py>(
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
        extraction::extract_metric_random_projection(
            w_view, calib_view, target_dim, num_steps, curvature, lr,
        )
    });
    (u.into_pyarray(py), g.into_pyarray(py), v.into_pyarray(py))
}

#[cfg(feature = "cuda")]
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
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray2<f32>, &'py PyArray2<f32>)> {
    let w_view = w.as_array();
    let calib_view = calibration_data.as_array();
    let (u, g, v) = py.allow_threads(move || {
        extraction::extract_metric_cuda(w_view, calib_view, target_dim, num_steps, curvature, lr)
    });
    Ok((u.into_pyarray(py), g.into_pyarray(py), v.into_pyarray(py)))
}

/// Without the `cuda` feature this entry point fails closed instead of returning a
/// random projection that ignores its calibration arguments.
#[cfg(not(feature = "cuda"))]
#[pyfunction]
#[pyo3(name = "extract_metric_cuda")]
pub fn extract_metric_cuda_py<'py>(
    _py: Python<'py>,
    _w: PyReadonlyArray2<f32>,
    _calibration_data: PyReadonlyArray2<f32>,
    _target_dim: usize,
    _num_steps: usize,
    _curvature: f32,
    _lr: f32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray2<f32>, &'py PyArray2<f32>)> {
    Err(PyRuntimeError::new_err(
        "extract_metric_cuda requires reality_stone built with the `cuda` feature; \
         use extract_metric_random_projection for the CPU approximation",
    ))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_metric_cuda_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_metric_random_projection_py, m)?)?;
    Ok(())
}
