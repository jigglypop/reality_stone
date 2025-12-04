use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use crate::layers::hyper_metric::{HyperMetric, TinyMLP};
use crate::layers::symplectic::{SymplecticLayer, SymplecticState};

#[pyclass]
pub struct PyHyperMetric {
    inner: HyperMetric,
}

#[pymethods]
impl PyHyperMetric {
    #[new]
    #[pyo3(signature = (u_global, v_global, w1, b1, w2, b2))]
    pub fn new(
        u_global: PyReadonlyArray2<f32>,
        v_global: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        b1: PyReadonlyArray1<f32>,
        w2: PyReadonlyArray2<f32>,
        b2: PyReadonlyArray1<f32>,
    ) -> Self {
        let mlp = TinyMLP::from_weights(
            w1.as_array().to_owned(),
            b1.as_array().to_owned(),
            w2.as_array().to_owned(),
            b2.as_array().to_owned(),
        );
        
        let inner = HyperMetric::from_components(
            u_global.as_array().to_owned(),
            v_global.as_array().to_owned(),
            mlp,
        );
        
        Self { inner }
    }

    pub fn generate_core<'py>(
        &self,
        py: Python<'py>,
        layer_emb: PyReadonlyArray1<f32>,
    ) -> &'py PyArray2<f32> {
        let core = self.inner.generate_core(&layer_emb.as_array().to_owned());
        core.into_pyarray(py)
    }

    pub fn project_forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        layer_emb: PyReadonlyArray1<f32>,
    ) -> &'py PyArray2<f32> {
        let out = self.inner.project_forward(
            &x.as_array().to_owned(),
            &layer_emb.as_array().to_owned(),
        );
        out.into_pyarray(py)
    }
}

#[pyclass]
pub struct PySymplecticLayer {
    inner: SymplecticLayer,
}

#[pymethods]
impl PySymplecticLayer {
    #[new]
    pub fn new(
        layer_idx: usize,
        layer_emb: PyReadonlyArray1<f32>,
        hyper_metric: &PyHyperMetric,
        dt: f32,
    ) -> Self {
        let inner = SymplecticLayer::new(
            layer_idx,
            layer_emb.as_array().to_owned(),
            hyper_metric.inner.clone(),
            dt,
        );
        Self { inner }
    }

    pub fn step<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<f32>,
        p: PyReadonlyArray2<f32>,
        x_input: PyReadonlyArray2<f32>,
    ) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
        let mut state = SymplecticState {
            q: q.as_array().to_owned(),
            p: p.as_array().to_owned(),
        };
        
        let _ = self.inner.step(&mut state, &x_input.as_array().to_owned());
        
        (state.q.into_pyarray(py), state.p.into_pyarray(py))
    }
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHyperMetric>()?;
    m.add_class::<PySymplecticLayer>()?;
    Ok(())
}
