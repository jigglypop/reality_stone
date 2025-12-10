use crate::layers::spline_cache::SplineCache;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

#[pyclass(name = "SplineCache")]
pub struct PySplineCache {
    inner: SplineCache,
}

#[pymethods]
impl PySplineCache {
    #[new]
    pub fn new(curvature: f32, dimension: usize) -> Self {
        Self {
            inner: SplineCache::new(curvature, dimension),
        }
    }

    pub fn add_point(
        &mut self,
        time: f32,
        state: PyReadonlyArray1<f32>,
        velocity: PyReadonlyArray1<f32>,
    ) {
        self.inner
            .add_point(time, state.as_array(), velocity.as_array());
    }

    pub fn reconstruct<'py>(&self, py: Python<'py>, t: f32) -> Option<&'py PyArray1<f32>> {
        self.inner.reconstruct(t).map(|arr| arr.to_pyarray(py))
    }

    pub fn batch_reconstruct<'py>(
        &self,
        py: Python<'py>,
        timestamps: PyReadonlyArray1<f32>,
    ) -> &'py PyArray2<f32> {
        let arr = self.inner.batch_reconstruct(timestamps.as_array());
        arr.to_pyarray(py)
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

pub fn register_spline_cache_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "spline_cache")?;
    m.add_class::<PySplineCache>()?;
    parent_module.add_submodule(m)?;
    Ok(())
}
