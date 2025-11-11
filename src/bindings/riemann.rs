use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn riemann_lowrank_forward_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    p: PyReadonlyArray2<f32>,
    sigma: PyReadonlyArray2<f32>,
    q: PyReadonlyArray2<f32>,
    b_tan: PyReadonlyArray1<f32>,
    c: f32,
    epsilon: f32,
) -> &'py PyArray2<f32> {
    let out = crate::layers::riemann::riemann_lowrank_forward(
        &x.as_array(),
        &p.as_array(),
        &sigma.as_array(),
        &q.as_array(),
        &b_tan.as_array(),
        c,
        epsilon,
    );
    out.into_pyarray(py)
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(riemann_lowrank_forward_cpu, m)?)?;
    Ok(())
}
