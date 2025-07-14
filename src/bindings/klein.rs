use crate::layers::klein;
use crate::ops::mobius;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn klein_add<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = klein::klein_add(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_scalar<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let result = klein::klein_scalar(&u_arr, c, r);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_distance<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray1<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = klein::klein_distance(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_to_poincare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = klein::klein_to_poincare(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_to_lorentz<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = klein::klein_to_lorentz(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_layer_forward<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = klein::klein_layer_forward(&u_arr, &v_arr, c, t);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn klein_ball_layer_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let (grad_u, grad_v) = klein::klein_layer_backward(&grad_output_arr, &u_arr, &v_arr, c, t);
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py))
}

#[pyfunction]
fn from_poincare_dynamic_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
) -> (&'py PyArray2<f32>, f32) {
    let x_view = x.as_array();
    let dynamic_c = mobius::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();
    let result = klein::from_poincare(&x_view, c);
    (result.into_pyarray(py), c)
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
    let dynamic_c = mobius::DynamicCurvature::new(kappa, c_min, c_max);
    let c = dynamic_c.compute_c();

    let grad_x = Array2::zeros(x.dims());

    let grad_c_tensor = klein::from_poincare_grad_c(&x_view, c);
    let grad_c = (&grad_output_view * &grad_c_tensor).sum();
    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;

    (grad_x.into_pyarray(py).to_owned(), grad_kappa)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn klein_distance_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    klein::cuda::klein_distance_cuda(
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
pub fn klein_layer_forward_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    klein::cuda::klein_layer_forward_cuda(
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
pub fn klein_ball_layer_backward_cuda(
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
    klein::cuda::klein_layer_backward_cuda(
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

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(klein_add, m)?)?;
    m.add_function(wrap_pyfunction!(klein_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(klein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(klein_layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(klein_ball_layer_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(from_poincare_dynamic_backward_cpu, m)?)?;

    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(klein_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(klein_layer_forward_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(klein_ball_layer_backward_cuda, m)?)?;
    }

    Ok(())
}
