use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::ops::mobius;

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

    crate::ops::mobius::cuda::mobius_add_cuda(out_ptr_f32, u_ptr_f32, v_ptr_f32, c, batch_size, dim);
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

    crate::ops::mobius::cuda::mobius_scalar_cuda(out_ptr_f32, u_ptr_f32, c, r, batch_size, dim);
    Ok(())
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mobius_add_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_scalar_cpu, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_add_cuda, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(mobius_scalar_cuda, m)?)?;
    Ok(())
} 