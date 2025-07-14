use crate::ops::mobius;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn mobius_add_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = mobius::mobius_add(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn mobius_add_cuda(
    _py: Python,
    u_ptr: usize,
    v_ptr: usize,
    out_ptr: usize,
    batch_size: i64,
    dim: i64,
    c: f32,
) -> PyResult<()> {
    let u_ptr_f32 = u_ptr as *const f32;
    let v_ptr_f32 = v_ptr as *const f32;
    let out_ptr_f32 = out_ptr as *mut f32;
    mobius::cuda::mobius_add_cuda(out_ptr_f32, u_ptr_f32, v_ptr_f32, c, batch_size, dim);
    Ok(())
}

#[pyfunction]
pub fn mobius_scalar_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let result = mobius::mobius_scalar(&u_arr, c, r);
    result.into_pyarray(py)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn mobius_scalar_cuda(
    _py: Python,
    u_ptr: usize,
    out_ptr: usize,
    batch_size: i64,
    dim: i64,
    r: f32,
    c: f32,
) -> PyResult<()> {
    let u_ptr_f32 = u_ptr as *const f32;
    let out_ptr_f32 = out_ptr as *mut f32;
    mobius::cuda::mobius_scalar_cuda(out_ptr_f32, u_ptr_f32, c, r, batch_size, dim);
    Ok(())
}

// 동적 곡률 Mobius 덧셈
#[pyfunction]
pub fn mobius_add_dynamic_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = mobius::DynamicCurvature::new(kappa, c_min, c_max);
    let (result, c) = mobius::mobius_add_dynamic(&u_arr, &v_arr, &dynamic_c);
    (result.into_pyarray(py), c)
}

// 동적 곡률 Mobius 덧셈의 backward pass
#[pyfunction]
pub fn mobius_add_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = mobius::DynamicCurvature::new(kappa, c_min, c_max);
    let (grad_u, grad_v, grad_kappa) =
        mobius::mobius_add_dynamic_backward(&grad_output_arr, &u_arr, &v_arr, &dynamic_c);
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py), grad_kappa)
}

#[pyfunction]
pub fn mobius_add_layerwise_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappas: Vec<f32>,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let layer_curvatures = mobius::LayerWiseDynamicCurvature::from_kappas(kappas, c_min, c_max);
    let (result, c) = mobius::mobius_add_layerwise(&u_arr, &v_arr, &layer_curvatures, layer_idx);
    (result.into_pyarray(py), c)
}

#[pyfunction]
pub fn mobius_add_layerwise_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappas: Vec<f32>,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let layer_curvatures = mobius::LayerWiseDynamicCurvature::from_kappas(kappas, c_min, c_max);
    let (grad_u, grad_v, grad_kappa) = mobius::mobius_add_layerwise_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &layer_curvatures,
        layer_idx,
    );
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py), grad_kappa)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mobius_add_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_scalar_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_dynamic_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_layerwise_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_layerwise_backward_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_add_cuda, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_scalar_cuda, m)?)?;
    Ok(())
}
