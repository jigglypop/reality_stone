use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use crate::layers::rsulf::{RSULFLayer, RSULFConfig, RSULFComponents, fold_dimension_svd, fold_ffn_svd, create_causal_laplacian};

#[pyclass]
pub struct PyRSULFLayer {
    inner: RSULFLayer,
}

#[pymethods]
impl PyRSULFLayer {
    #[new]
    #[pyo3(signature = (wq, wk, w1, w2, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8))]
    pub fn new(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
    ) -> Self {
        let config = RSULFConfig {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
        };
        
        let inner = RSULFLayer::from_transformer(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
        );
        
        Self { inner }
    }
    
    pub fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
        v_mem: Option<PyReadonlyArray1<f32>>,
    ) -> (&'py PyArray2<f32>, &'py PyArray1<f32>) {
        let v_view = v_mem.as_ref().map(|v| v.as_array());
        let (output, v_new) = self.inner.forward(x.as_array(), v_view);
        (output.into_pyarray(py), v_new.into_pyarray(py))
    }
    
    pub fn param_count(&self) -> (usize, usize, f32) {
        self.inner.param_count()
    }
    
    #[getter]
    pub fn curvature(&self) -> f32 {
        self.inner.curvature
    }
    
    #[getter]
    pub fn d_model(&self) -> usize {
        self.inner.config.d_model
    }
    
    #[getter]
    pub fn r(&self) -> usize {
        self.inner.config.r
    }

    pub fn export_components<'py>(&self, py: Python<'py>) -> &'py PyDict {
        let comp = self.inner.export_components();
        let dict = PyDict::new(py);
        dict.set_item("d_model", comp.d_model).unwrap();
        dict.set_item("r", comp.r).unwrap();
        dict.set_item("eta", comp.eta).unwrap();
        dict.set_item("alpha", comp.alpha).unwrap();
        dict.set_item("beta", comp.beta).unwrap();
        dict.set_item("gamma", comp.gamma).unwrap();
        dict.set_item("seq_len", comp.seq_len).unwrap();
        dict.set_item("window", comp.window).unwrap();
        dict.set_item("g_diag", comp.g_diag.into_pyarray(py)).unwrap();
        dict.set_item("g_inv", comp.g_inv.into_pyarray(py)).unwrap();
        dict.set_item("u_metric", comp.u_metric.into_pyarray(py)).unwrap();
        dict.set_item("v_metric", comp.v_metric.into_pyarray(py)).unwrap();
        dict.set_item("curvature", comp.curvature).unwrap();
        dict.set_item("ffn_u1", comp.ffn_u1.into_pyarray(py)).unwrap();
        dict.set_item("ffn_s1", comp.ffn_s1.into_pyarray(py)).unwrap();
        dict.set_item("ffn_v1", comp.ffn_v1.into_pyarray(py)).unwrap();
        dict.set_item("ffn_u2", comp.ffn_u2.into_pyarray(py)).unwrap();
        dict.set_item("ffn_s2", comp.ffn_s2.into_pyarray(py)).unwrap();
        dict.set_item("ffn_v2", comp.ffn_v2.into_pyarray(py)).unwrap();
        dict
    }

    #[staticmethod]
    pub fn from_components(
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
        g_diag: PyReadonlyArray1<f32>,
        g_inv: PyReadonlyArray1<f32>,
        u_metric: PyReadonlyArray2<f32>,
        v_metric: PyReadonlyArray2<f32>,
        curvature: f32,
        ffn_u1: PyReadonlyArray2<f32>,
        ffn_s1: PyReadonlyArray1<f32>,
        ffn_v1: PyReadonlyArray2<f32>,
        ffn_u2: PyReadonlyArray2<f32>,
        ffn_s2: PyReadonlyArray1<f32>,
        ffn_v2: PyReadonlyArray2<f32>,
    ) -> Self {
        let comp = RSULFComponents {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
            g_diag: g_diag.as_array().to_owned(),
            g_inv: g_inv.as_array().to_owned(),
            u_metric: u_metric.as_array().to_owned(),
            v_metric: v_metric.as_array().to_owned(),
            curvature,
            ffn_u1: ffn_u1.as_array().to_owned(),
            ffn_s1: ffn_s1.as_array().to_owned(),
            ffn_v1: ffn_v1.as_array().to_owned(),
            ffn_u2: ffn_u2.as_array().to_owned(),
            ffn_s2: ffn_s2.as_array().to_owned(),
            ffn_v2: ffn_v2.as_array().to_owned(),
        };
        let inner = RSULFLayer::from_components(comp);
        Self { inner }
    }

    #[staticmethod]
    #[pyo3(signature = (wq, wk, w1, w2, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8))]
    pub fn new_fast(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        d_model: usize,
        r: usize,
        eta: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        seq_len: usize,
        window: usize,
    ) -> Self {
        let config = RSULFConfig {
            d_model,
            r,
            eta,
            alpha,
            beta,
            gamma,
            seq_len,
            window,
        };
        
        let inner = RSULFLayer::from_transformer_fast(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
        );
        
        Self { inner }
    }
}

#[pyfunction]
pub fn fold_metric_svd<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
) -> (&'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray2<f32>, f32) {
    let folded = fold_dimension_svd(wq.as_array(), wk.as_array(), target_dim);
    let curvature = crate::layers::rsulf::compute_curvature(&folded.s_residual);
    (
        folded.u.into_pyarray(py),
        folded.s.into_pyarray(py),
        folded.v.into_pyarray(py),
        curvature,
    )
}

#[pyfunction]
pub fn build_causal_laplacian<'py>(
    py: Python<'py>,
    seq_len: usize,
    window: usize,
) -> &'py PyArray2<f32> {
    let l = create_causal_laplacian(seq_len, window);
    l.into_pyarray(py)
}

#[pyfunction]
pub fn fold_ffn<'py>(
    py: Python<'py>,
    w1: PyReadonlyArray2<f32>,
    w2: PyReadonlyArray2<f32>,
    target_dim: usize,
) -> (
    &'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray2<f32>,
    &'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray2<f32>,
) {
    let folded = fold_ffn_svd(w1.as_array(), w2.as_array(), target_dim);
    (
        folded.u1.into_pyarray(py),
        folded.s1.into_pyarray(py),
        folded.v1.into_pyarray(py),
        folded.u2.into_pyarray(py),
        folded.s2.into_pyarray(py),
        folded.v2.into_pyarray(py),
    )
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRSULFLayer>()?;
    m.add_function(wrap_pyfunction!(fold_metric_svd, m)?)?;
    m.add_function(wrap_pyfunction!(fold_ffn, m)?)?;
    m.add_function(wrap_pyfunction!(build_causal_laplacian, m)?)?;
    Ok(())
}
