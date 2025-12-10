use crate::layers::spline::SplineLayer;
use pyo3::prelude::*;

#[cfg(feature = "cuda")]
use crate::layers::spline::cuda as spline_cuda;

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn spline_interpolate_cuda(
    control_points: usize,
    weights: usize,
    k: i32,
    in_features: i32,
    out_features: i32,
) -> PyResult<()> {
    spline_cuda::spline_interpolate_cuda(
        control_points as *const f32,
        weights as *mut f32,
        k,
        in_features,
        out_features,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn spline_forward_cuda(
    input: usize,
    control_points: usize,
    output: usize,
    batch_size: i32,
    k: i32,
    in_features: i32,
    out_features: i32,
) -> PyResult<()> {
    spline_cuda::spline_forward_cuda(
        input as *const f32,
        control_points as *const f32,
        output as *mut f32,
        batch_size,
        k,
        in_features,
        out_features,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn spline_backward_cuda(
    grad_output: usize,
    input: usize,
    grad_control_points: usize,
    batch_size: i32,
    k: i32,
    in_features: i32,
    out_features: i32,
) -> PyResult<()> {
    spline_cuda::spline_backward_cuda(
        grad_output as *const f32,
        input as *const f32,
        grad_control_points as *mut f32,
        batch_size,
        k,
        in_features,
        out_features,
    );
    Ok(())
}

/// Python 모듈에 SplineLayer를 등록합니다.
pub fn register_spline_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let spline_module = PyModule::new(py, "spline")?;
    spline_module.add_class::<SplineLayer>()?;

    #[cfg(feature = "cuda")]
    {
        spline_module.add_function(wrap_pyfunction!(spline_interpolate_cuda, spline_module)?)?;
        spline_module.add_function(wrap_pyfunction!(spline_forward_cuda, spline_module)?)?;
        spline_module.add_function(wrap_pyfunction!(spline_backward_cuda, spline_module)?)?;
    }

    parent_module.add_submodule(spline_module)?;
    Ok(())
}
