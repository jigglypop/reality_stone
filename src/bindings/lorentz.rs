use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::ops::lorentz;

#[pyfunction]
pub fn lorentz_add<'py>(
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
pub fn lorentz_scalar<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    r: f32,
    _c: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let result = lorentz::lorentz_scalar(&u_arr, _c, r);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn lorentz_distance<'py>(
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
pub fn lorentz_inner<'py>(
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
pub fn lorentz_to_poincare<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_poincare(&x_arr, c);
    result.into_pyarray(py)
}

#[pyfunction]
pub fn lorentz_to_klein<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    c: f32,
) -> &'py PyArray2<f32> {
    let x_arr = x.as_array();
    let result = lorentz::lorentz_to_klein(&x_arr, c);
    result.into_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lorentz_add, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_distance, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_inner, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(lorentz_to_klein, m)?)?;
    Ok(())
} 