use crate::{create_binding, layers::poincare, ops::project};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// --- 매크로를 사용한 바인딩 생성 ---

create_binding!(
    poincare_distance_cpu,
    poincare::poincare_distance,
    [u, v, c, eps],
    PyArray1
);
create_binding!(
    poincare_to_lorentz_cpu,
    poincare::poincare_to_lorentz,
    [x, c],
    PyArray2
);
create_binding!(
    poincare_to_klein_cpu,
    poincare::poincare_to_klein,
    [x, c],
    PyArray2
);

// --- 매크로로 처리하기 복잡한 함수들 ---

#[pyfunction]
pub fn poincare_ball_layer_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    poincare::poincare_ball_layer(&u_arr, &v_arr, c, t).into_pyarray(py)
}

/// Exponential map on the Poincaré ball at point x with tangent vector v.
#[pyfunction]
pub fn poincare_exp_at_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    eps: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let v_arr = v.as_array();
    poincare::poincare_exp_at(&x_arr, &v_arr, c, eps).into_pyarray(py)
}

/// Logarithmic map on the Poincaré ball at point x for point y.
#[pyfunction]
pub fn poincare_log_at_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    c: f32,
    eps: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    poincare::poincare_log_at(&x_arr, &y_arr, c, eps).into_pyarray(py)
}

#[pyfunction]
pub fn poincare_ball_layer_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let (grad_u, grad_v) = poincare::poincare_ball_layer_backward(
        &grad_output.as_array(),
        &u.as_array(),
        &v.as_array(),
        c,
        t,
    );
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py))
}

// --- Dynamic / Layerwise bindings ---

#[pyfunction]
pub fn poincare_ball_layer_dynamic_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let (out, c) = poincare::poincare_ball_layer_dynamic(&u_arr, &v_arr, &dynamic_c, t);
    (out.into_pyarray(py), c)
}

#[pyfunction]
pub fn poincare_ball_layer_dynamic_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let dynamic_c = crate::ops::DynamicCurvature::new(kappa, c_min, c_max);
    let (gu, gv, gk) = poincare::poincare_ball_layer_dynamic_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &dynamic_c,
        t,
    );
    (gu.into_pyarray(py), gv.into_pyarray(py), gk)
}

#[pyfunction]
pub fn poincare_ball_layer_layerwise_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, f32) {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = crate::ops::LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (out, c) = poincare::poincare_ball_layer_layerwise(&u_arr, &v_arr, &lw, layer_idx, t);
    (out.into_pyarray(py), c)
}

#[pyfunction]
pub fn poincare_ball_layer_layerwise_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    kappa: f32,
    layer_idx: usize,
    c_min: f32,
    c_max: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, f32) {
    let grad_output_arr = grad_output.as_array();
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let lw = crate::ops::LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (gu, gv, gk) = poincare::poincare_ball_layer_layerwise_backward(
        &grad_output_arr,
        &u_arr,
        &v_arr,
        &lw,
        layer_idx,
        t,
    );
    (gu.into_pyarray(py), gv.into_pyarray(py), gk)
}

#[pyfunction]
pub fn mobius_add_vjp_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    c: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let (grad_x, grad_y) = crate::ops::mobius::mobius_add_vjp(
        &grad_output.as_array(),
        &x.as_array(),
        &y.as_array(),
        c,
    );
    (grad_x.into_pyarray(py), grad_y.into_pyarray(py))
}

#[pyfunction]
pub fn mobius_scalar_vjp_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    x: PyReadonlyArray2<f32>,
    c: f32,
    r: f32,
) -> &'py PyArray2<f32> {
    crate::ops::mobius::mobius_scalar_vjp(&grad_output.as_array(), &x.as_array(), c, r)
        .into_pyarray(py)
}

#[pyfunction]
pub fn project_to_ball_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    epsilon: f32,
) -> &'py PyArray2<f32> {
    project::project_to_ball(&x.as_array(), epsilon).into_pyarray(py)
}

#[pyfunction]
pub fn poincare_riemannian_adam_step_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    grad: PyReadonlyArray2<'py, f32>,
    m: PyReadonlyArray2<'py, f32>,
    v: PyReadonlyArray2<'py, f32>,
    step: u64,
    c: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    max_norm_eps: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>, &'py PyArray2<f32>) {
    let x_arr = x.as_array();
    let grad_arr = grad.as_array();
    let mut m_arr = m.as_array().to_owned();
    let mut v_arr = v.as_array().to_owned();
    let x_new = poincare::poincare_riemannian_adam_step(
        &x_arr,
        &grad_arr,
        &mut m_arr,
        &mut v_arr,
        step,
        c,
        lr,
        beta1,
        beta2,
        eps,
        max_norm_eps,
    );
    (
        x_new.into_pyarray(py),
        m_arr.into_pyarray(py),
        v_arr.into_pyarray(py),
    )
}

// --- CUDA bindings ---

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_distance_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    boundary_eps: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    poincare::cuda::poincare_distance_cuda(
        out as *mut f32,
        u as *const f32,
        v as *const f32,
        c,
        boundary_eps,
        batch_size,
        dim,
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_ball_layer_cuda(
    out: usize,
    u: usize,
    v: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    poincare::cuda::poincare_ball_layer_cuda(
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
pub fn poincare_ball_layer_backward_cuda(
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
    poincare::cuda::poincare_ball_layer_backward_cuda(
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

// --- 모듈 등록 ---

pub fn register(m: &PyModule) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "poincare")?;
    sub.add_function(wrap_pyfunction!(poincare_distance_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_to_lorentz_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_to_klein_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_exp_at_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_log_at_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(mobius_add_vjp_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(mobius_scalar_vjp_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(project_to_ball_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(poincare_riemannian_adam_step_cpu, sub)?)?;

    // Dynamic / Layerwise
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_dynamic_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        poincare_ball_layer_dynamic_backward_cpu,
        sub
    )?)?;
    sub.add_function(wrap_pyfunction!(poincare_ball_layer_layerwise_cpu, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        poincare_ball_layer_layerwise_backward_cpu,
        sub
    )?)?;

    #[cfg(feature = "cuda")]
    {
        sub.add_function(wrap_pyfunction!(poincare_distance_cuda, sub)?)?;
        sub.add_function(wrap_pyfunction!(poincare_ball_layer_cuda, sub)?)?;
        sub.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cuda, sub)?)?;
        m.add_function(wrap_pyfunction!(poincare_distance_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(poincare_ball_layer_cuda, m)?)?;
        m.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cuda, m)?)?;
    }

    m.add_submodule(sub)?;
    Ok(())
}
