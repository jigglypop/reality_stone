use crate::layers::lorentz;
use crate::ops::{DynamicCurvature, LayerWiseDynamicCurvature};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
pub fn lorentz_add<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_add(&u_arr, &v_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_scalar<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let u_arr = u.as_array();
    let result = lorentz::lorentz_scalar(&u_arr, c, r);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_distance<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray1<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_distance(&u_arr, &v_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_inner<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray1<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_inner(&u_arr, &v_arr);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_to_poincare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_poincare(&x_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_to_klein<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> Bound<'py, PyArray2<f32>> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_klein(&x_arr, c);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_layer_forward<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> Bound<'py, PyArray2<f32>> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_layer_forward(&u_arr, &v_arr, c, t);
    result.into_pyarray_bound(py)
}

#[pyfunction]
pub fn lorentz_ball_layer_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let (grad_u, grad_v) = lorentz::lorentz_layer_backward(&grad_output_arr, &u_arr, &v_arr, c, t);
    (grad_u.into_pyarray_bound(py), grad_v.into_pyarray_bound(py))
}

#[pyfunction]
pub fn lorentz_layer_dynamic_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = DynamicCurvature::new(kappa, c_min, c_max);
    let (out, c) = lorentz::lorentz_layer_dynamic(&u_arr, &v_arr, &dynamic_c, t);
    (out.into_pyarray_bound(py), c)
}

#[pyfunction]
pub fn lorentz_layer_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = DynamicCurvature::new(kappa, c_min, c_max);
    let (gu, gv, gk) = lorentz::lorentz_layer_dynamic_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &dynamic_c,
        t,
    );
    (gu.into_pyarray_bound(py), gv.into_pyarray_bound(py), gk)
}

#[pyfunction]
pub fn lorentz_layer_layerwise_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (out, c) = lorentz::lorentz_layer_layerwise(&u_arr, &v_arr, &lw, layer_idx, t);
    (out.into_pyarray_bound(py), c)
}

#[pyfunction]
pub fn lorentz_layer_layerwise_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (gu, gv, gk) = lorentz::lorentz_layer_layerwise_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &lw,
        layer_idx,
        t,
    );
    (gu.into_pyarray_bound(py), gv.into_pyarray_bound(py), gk)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn lorentz_distance_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    lorentz::cuda::lorentz_distance_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn lorentz_layer_forward_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    lorentz::cuda::lorentz_layer_forward_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn lorentz_ball_layer_backward_cuda(
    grad_output: usize,
    u: usize,
    v: usize,
    grad_u: usize,
    grad_v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    lorentz::cuda::lorentz_layer_backward_cuda(
        grad_output as *const f32,
        u as *const f32,
        v as *const f32,
        grad_u as *mut f32,
        grad_v as *mut f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

#[pyfunction]
fn from_poincare_dynamic_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (Bound<'py, PyArray2<f32>>, f32) {
    let x_view = x.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();
    let result = lorentz::from_poincare(&x_view, c);
    (result.into_pyarray_bound(py), c)
}

#[pyfunction]
fn from_poincare_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<'py, f32>,
    x: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (Py<PyArray2<f32>>, f32) {
    let grad_output_view = grad_output.as_array();
    let x_view = x.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();

    let grad_x = Array2::zeros((x.shape()[0], x.shape()[1]));

    let grad_c_tensor = lorentz::from_poincare_grad_c(&x_view, c);
    let grad_c = (&grad_output_view * &grad_c_tensor).sum();
    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;

    (grad_x.into_pyarray_bound(py).unbind(), grad_kappa)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lorentz_add, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_distance, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_inner, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_klein, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_ball_layer_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_dynamic_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_layerwise_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_layer_layerwise_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_backward_cpu, m)?)?;

    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(lorentz_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(lorentz_layer_forward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(lorentz_ball_layer_backward_cuda, m)?)?;
    }

    Ok(())
}
