use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use crate::layers::rsulf::{
    RSULFLayer, RSULFConfig, RSULFComponents, 
    fold_dimension_svd, fold_ffn_svd, create_causal_laplacian,
    verify_fold_consistency, 
    block_lanczos_svd, nystrom_approximation, adaptive_rank_svd,
};

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

    #[staticmethod]
    #[pyo3(signature = (wq, wk, w1, w2, g_diag, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8))]
    pub fn new_with_metric(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        g_diag: PyReadonlyArray1<f32>,
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

        let inner = RSULFLayer::from_transformer_with_metric(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
            g_diag.as_array(),
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

#[pyfunction]
pub fn verify_metric_consistency<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
) -> &'py PyDict {
    let folded = fold_dimension_svd(wq.as_array(), wk.as_array(), target_dim);
    let result = verify_fold_consistency(wq.as_array(), wk.as_array(), &folded);
    
    let dict = PyDict::new(py);
    dict.set_item("symmetry_error", result.symmetry_error).unwrap();
    dict.set_item("reconstruction_error", result.reconstruction_error).unwrap();
    dict.set_item("fold_accuracy", result.fold_accuracy).unwrap();
    dict.set_item("min_eigenvalue", result.min_eigenvalue).unwrap();
    dict.set_item("condition_number", result.condition_number).unwrap();
    dict.set_item("is_valid", result.is_valid).unwrap();
    dict
}

#[pyfunction]
pub fn fold_metric_optimized<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
    method: &str,
) -> (&'py PyArray2<f32>, &'py PyArray1<f32>, &'py PyArray2<f32>, f32, &'py PyDict) {
    let d_q = wq.as_array().nrows();
    let d_k = wk.as_array().nrows();
    let d_in = wq.as_array().ncols();
    
    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = ndarray::Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded.slice_mut(ndarray::s![i*d_k..(i+1)*d_k, ..]).assign(&wk.as_array());
        }
        expanded
    } else {
        wk.as_array().to_owned()
    };
    
    let g = wq.as_array().t().dot(&wk_expanded);
    
    let (u, s, v) = match method {
        "block_lanczos" => block_lanczos_svd(&g, target_dim, 32, 10),
        "adaptive" => {
            let (u, s, v, _) = adaptive_rank_svd(&g, 0.95, target_dim);
            (u, s, v)
        },
        _ => crate::layers::rsulf::randomized_svd(&g, target_dim, 5, 2),
    };
    
    let frob_g: f32 = g.iter().map(|x| x * x).sum();
    let frob_approx: f32 = s.iter().map(|x| x * x).sum();
    let tail = frob_g - frob_approx;
    let curvature = if tail > 0.0 { tail.sqrt() } else { 0.0 };
    
    let folded = crate::layers::rsulf::FoldedMetric {
        u: u.clone(),
        s: s.clone(),
        v: v.clone(),
        s_residual: ndarray::Array1::from_elem(1, curvature),
    };
    let consistency = verify_fold_consistency(wq.as_array(), wk.as_array(), &folded);
    
    let info = PyDict::new(py);
    info.set_item("symmetry_error", consistency.symmetry_error).unwrap();
    info.set_item("reconstruction_error", consistency.reconstruction_error).unwrap();
    info.set_item("fold_accuracy", consistency.fold_accuracy).unwrap();
    info.set_item("min_eigenvalue", consistency.min_eigenvalue).unwrap();
    info.set_item("condition_number", consistency.condition_number).unwrap();
    info.set_item("is_valid", consistency.is_valid).unwrap();
    info.set_item("method", method).unwrap();
    
    (
        u.into_pyarray(py),
        s.into_pyarray(py),
        v.into_pyarray(py),
        curvature,
        info,
    )
}

#[pyfunction]
pub fn nystrom_metric<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    target_dim: usize,
    n_samples: usize,
) -> (&'py PyArray2<f32>, &'py PyArray1<f32>) {
    let d_q = wq.as_array().nrows();
    let d_k = wk.as_array().nrows();
    let d_in = wq.as_array().ncols();
    
    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = ndarray::Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded.slice_mut(ndarray::s![i*d_k..(i+1)*d_k, ..]).assign(&wk.as_array());
        }
        expanded
    } else {
        wk.as_array().to_owned()
    };
    
    let g = wq.as_array().t().dot(&wk_expanded);
    let (u, s) = nystrom_approximation(&g, target_dim, n_samples);
    
    (u.into_pyarray(py), s.into_pyarray(py))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRSULFLayer>()?;
    m.add_function(wrap_pyfunction!(fold_metric_svd, m)?)?;
    m.add_function(wrap_pyfunction!(fold_ffn, m)?)?;
    m.add_function(wrap_pyfunction!(build_causal_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(verify_metric_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(fold_metric_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(nystrom_metric, m)?)?;
    Ok(())
}
