use crate::layers::spline::SplineLayer;
use pyo3::prelude::*;

/// Python 모듈에 SplineLayer를 등록합니다.
pub fn register_spline_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let spline_module = PyModule::new(py, "spline")?;
    spline_module.add_class::<SplineLayer>()?;
    parent_module.add_submodule(spline_module)?;
    Ok(())
}
