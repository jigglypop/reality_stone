use crate::layers::memory::GeodesicMemory;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass]
pub struct PyGeodesicMemory {
    inner: GeodesicMemory,
}

#[pymethods]
impl PyGeodesicMemory {
    #[new]
    pub fn new(d_model: usize, threshold: f32) -> Self {
        Self {
            inner: GeodesicMemory::new(d_model, threshold),
        }
    }

    pub fn push(&mut self, t: usize, x: PyReadonlyArray1<f32>) -> bool {
        self.inner.push(t, x.as_array())
    }

    pub fn query<'py>(&self, py: Python<'py>, t: f32) -> &'py PyArray1<f32> {
        let result = self.inner.query(t);
        result.into_pyarray(py)
    }

    pub fn get_stats(&self) -> (usize, usize, f32) {
        self.inner.get_compression_stats()
    }

    pub fn reset(&mut self) {
        let d = self.inner.d_model;
        let th = self.inner.threshold;
        self.inner = GeodesicMemory::new(d, th);
    }
}
