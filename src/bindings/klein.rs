use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::ops::klein;

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

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(klein_add, m)?)?;
    m.add_function(wrap_pyfunction!(klein_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(klein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(klein_to_lorentz, m)?)?;
    Ok(())
} 