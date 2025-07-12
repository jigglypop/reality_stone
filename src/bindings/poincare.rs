use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::layers::{poincare, utils};

#[pyfunction]
pub fn poincare_ball_layer_backward_cpu<'py>(
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
    
    let (grad_u, grad_v) = poincare::poincare_ball_layer_backward(
        &grad_output_arr, &u_arr, &v_arr, c, t
    );

    (grad_u.into_pyarray(py), grad_v.into_pyarray(py))
}

#[pyfunction]
pub fn mobius_add_vjp_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    c: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let grad_output_arr = grad_output.as_array();
    let x_arr = x.as_array();
    let y_arr = y.as_array();
    
    let (grad_x, grad_y) = poincare::mobius_add_vjp(&grad_output_arr, &x_arr, &y_arr, c);

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
    let grad_output_arr = grad_output.as_array();
    let x_arr = x.as_array();
    
    let grad_x = poincare::mobius_scalar_vjp(&grad_output_arr, &x_arr, c, r);

    grad_x.into_pyarray(py)
}

#[pyfunction]
pub fn poincare_distance_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray1<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = poincare::poincare_distance(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_distance_cuda(
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
    unsafe {
        crate::layers::poincare::cuda::poincare_distance_cuda(out_ptr_f32, u_ptr_f32, v_ptr_f32, c, batch_size, dim);
    }
    Ok(())
}

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
    let result = poincare::poincare_ball_layer(&u_arr, &v_arr, c, t);
    result.into_pyarray(py)
}

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
    let dynamic_c = crate::layers::mobius::DynamicCurvature::new(kappa, c_min, c_max);
    let (result, c) = poincare::poincare_ball_layer_dynamic(&u_arr, &v_arr, &dynamic_c, t);
    (result.into_pyarray(py), c)
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
    let dynamic_c = crate::layers::mobius::DynamicCurvature::new(kappa, c_min, c_max);
    let (grad_u, grad_v, grad_kappa) = poincare::poincare_ball_layer_dynamic_backward(
        &grad_output_arr, &u_arr, &v_arr, &dynamic_c, t
    );
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py), grad_kappa)
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
    let layer_curvatures = crate::layers::mobius::LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (result, c) = poincare::poincare_ball_layer_layerwise(&u_arr, &v_arr, &layer_curvatures, 0, t);
    (result.into_pyarray(py), c)
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
    let layer_curvatures = crate::layers::mobius::LayerWiseDynamicCurvature::from_kappas(vec![kappa], c_min, c_max);
    let (grad_u, grad_v, grad_kappa) = poincare::poincare_ball_layer_layerwise_backward(
        &grad_output_arr, &u_arr, &v_arr, &layer_curvatures, 0, t
    );
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py), grad_kappa)
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_ball_layer_cuda(
    _py: Python,
    u_ptr: usize,
    v_ptr: usize,
    out_ptr: usize,
    batch_size: i64,
    dim: i64,
    c: f32,
    t: f32,
) -> PyResult<()> {
    unsafe {
        crate::layers::poincare::cuda::poincare_ball_layer_cuda(
            out_ptr as *mut f32,
            u_ptr as *const f32,
            v_ptr as *const f32,
            c,
            t,
            batch_size,
            dim,
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[pyfunction]
pub fn poincare_ball_layer_backward_cuda(
    _py: Python,
    grad_output_ptr: usize,
    u_ptr: usize,
    v_ptr: usize,
    grad_u_ptr: usize,
    grad_v_ptr: usize,
    c: f32,
    t: f32,
    batch_size: i64,
    dim: i64,
) -> PyResult<()> {
    unsafe {
        crate::layers::poincare::cuda::poincare_ball_layer_backward_cuda(
            grad_output_ptr as *const f32,
            u_ptr as *const f32,
            v_ptr as *const f32,
            grad_u_ptr as *mut f32,
            grad_v_ptr as *mut f32,
            c,
            t,
            batch_size,
            dim,
        );
    }
    Ok(())
}

#[pyfunction]
pub fn poincare_to_lorentz<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = poincare::poincare_to_lorentz(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn poincare_to_klein<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = poincare::poincare_to_klein(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn project_to_ball_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    epsilon: f32,
) -> &'py PyArray2<f32> {
    let x_view = x.as_array();
    let output = utils::project_to_ball(&x_view, epsilon);
    output.into_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(poincare_distance_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(poincare_distance_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_dynamic_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_dynamic_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_layerwise_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_layerwise_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_ball_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_vjp_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_scalar_vjp_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(poincare_ball_layer_cuda, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_to_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_to_klein, m)?)?;
    Ok(())
} 