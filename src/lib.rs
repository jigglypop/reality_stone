use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod ops;
pub mod layers;
pub mod utils;

use ops::{klein, lorentz, mobius, poincare};

#[pyfunction]
fn mobius_add_cpu<'py>(
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
fn mobius_add_cuda(
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

    ops::mobius::cuda::mobius_add_cuda(out_ptr_f32, u_ptr_f32, v_ptr_f32, c, batch_size, dim);
    Ok(())
}

#[pyfunction]
fn mobius_scalar_cpu<'py>(
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
fn mobius_scalar_cuda(
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

    ops::mobius::cuda::mobius_scalar_cuda(out_ptr_f32, u_ptr_f32, c, r, batch_size, dim);
    Ok(())
}

// Poincaré operations
#[pyfunction]
fn poincare_distance_cpu<'py>(
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
fn poincare_distance_cuda(
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

    ops::poincare::cuda::poincare_distance_cuda(out_ptr_f32, u_ptr_f32, v_ptr_f32, c, batch_size, dim);
    Ok(())
}

#[pyfunction]
fn poincare_ball_layer_cpu<'py>(
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

#[cfg(feature = "cuda")]
#[pyfunction]
fn poincare_ball_layer_cuda(
    _py: Python,
    u_ptr: usize,
    v_ptr: usize,
    out_ptr: usize,
    batch_size: i64,
    dim: i64,
    c: f32,
    t: f32,
) -> PyResult<()> {
    ops::poincare::cuda::poincare_ball_layer_cuda(
        out_ptr as *mut f32,
        u_ptr as *const f32,
        v_ptr as *const f32,
        c,
        t,
        batch_size,
        dim,
    );
    Ok(())
}

#[pyfunction]
fn poincare_to_lorentz<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = poincare::poincare_to_lorentz(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
fn poincare_to_klein<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = poincare::poincare_to_klein(&x_arr, c);
    result.into_pyarray(py)
}

// Lorentz operations
#[pyfunction]
fn lorentz_add<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_add(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
fn lorentz_scalar<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let result = lorentz::lorentz_scalar(&u_arr, c, r);
    result.into_pyarray(py)
}

#[pyfunction]
fn lorentz_distance<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray1<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_distance(&u_arr, &v_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
fn lorentz_inner<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
) -> &'py PyArray1<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    let result = lorentz::lorentz_inner(&u_arr, &v_arr);
    result.into_pyarray(py)
}

#[pyfunction]
fn lorentz_to_poincare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_poincare(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
fn lorentz_to_klein<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_klein(&x_arr, c);
    result.into_pyarray(py)
}

// Klein operations
#[pyfunction]
fn klein_add<'py>(
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
fn klein_scalar<'py>(
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
fn klein_distance<'py>(
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
fn klein_to_poincare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = klein::klein_to_poincare(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
fn klein_to_lorentz<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = klein::klein_to_lorentz(&x_arr, c);
    result.into_pyarray(py)
}

/// Reality Stone - High-performance hyperbolic neural networks in Rust
#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.2.0")?;
    
    // Mobius operations
    m.add_function(wrap_pyfunction!(mobius_add_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_scalar_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_add_cuda, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_scalar_cuda, m)?)?;
    
    // Poincaré ball operations
    m.add_function(wrap_pyfunction!(poincare_distance_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(poincare_distance_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(poincare_ball_layer_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_to_lorentz, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_to_klein, m)?)?;
    
    // Lorentz operations
    m.add_function(wrap_pyfunction!(lorentz_add, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_distance, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_inner, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_klein, m)?)?;
    
    // Klein operations
    m.add_function(wrap_pyfunction!(klein_add, m)?)?;
    m.add_function(wrap_pyfunction!(klein_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(klein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_lorentz, m)?)?;
    
    Ok(())
} 