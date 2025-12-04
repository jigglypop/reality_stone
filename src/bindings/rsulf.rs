use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use numpy::PyUntypedArrayMethods;
use ndarray::Array2;
use crate::layers::rsulf::{
    RSULFLayer, RSULFConfig, RSULFComponents, 
    fold_dimension_svd, fold_ffn_svd, create_causal_laplacian,
    verify_fold_consistency, 
    block_lanczos_svd, nystrom_approximation, adaptive_rank_svd,
    analyze_layer,
};

#[cfg(feature = "cuda")]
mod rsulf_cuda_ffi {
    use std::ffi::c_void;

    extern "C" {
        pub fn rsulf_forward_cuda(
            x: *const f32,
            v1: *const f32,
            s1: *const f32,
            u1: *const f32,
            v2: *const f32,
            s2: *const f32,
            u2: *const f32,
            g_inv: *const f32,
            v_mem: *const f32,
            eta: f32,
            alpha: f32,
            gamma_param: f32,
            batch: i32,
            d: i32,
            r: i32,
            ffn_dim: i32,
            x_out: *mut f32,
            v_out: *mut f32,
        );

        pub fn rsulf_batch_forward_cuda(
            x: *const f32,
            v1: *const f32,
            s1: *const f32,
            u1: *const f32,
            v2: *const f32,
            s2: *const f32,
            u2: *const f32,
            g_inv: *const f32,
            v_mem: *const f32,
            eta: f32,
            alpha: f32,
            gamma_param: f32,
            batch: i32,
            seq_len: i32,
            d: i32,
            r: i32,
            ffn_dim: i32,
            x_out: *mut f32,
            v_out: *mut f32,
        );

        pub fn rsulf_unified_forward_cuda(
            x: *const f32,
            v1: *const f32,
            s1: *const f32,
            u1: *const f32,
            v2: *const f32,
            s2: *const f32,
            u2: *const f32,
            g_inv: *const f32,
            laplacian: *const f32,
            v_mem: *const f32,
            eta: f32,
            alpha: f32,
            beta: f32,
            gamma_param: f32,
            curvature: f32,
            batch: i32,
            seq_len: i32,
            d: i32,
            r: i32,
            ffn_dim: i32,
            window: i32,
            x_out: *mut f32,
            v_out: *mut f32,
        );

        pub fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
        pub fn cudaFree(ptr: *mut c_void) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
    }
}

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
            calibration_samples: 1024,
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
            calibration_samples: 1024,
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
    
    #[staticmethod]
    #[pyo3(signature = (wq, wk, w1, w2, u_basis, basis_rank, d_model=4096, r=1024, eta=0.01, alpha=0.02, beta=0.01, gamma=0.99, seq_len=128, window=8))]
    pub fn new_with_basis(
        wq: PyReadonlyArray2<f32>,
        wk: PyReadonlyArray2<f32>,
        w1: PyReadonlyArray2<f32>,
        w2: PyReadonlyArray2<f32>,
        u_basis: PyReadonlyArray2<f32>,
        basis_rank: usize,
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
            calibration_samples: 1024,
        };
        
        let global_basis = crate::layers::rsulf::GlobalBasis {
            u: u_basis.as_array().to_owned(),
            rank: basis_rank,
        };

        let inner = RSULFLayer::from_transformer_with_basis(
            wq.as_array(),
            wk.as_array(),
            w1.as_array(),
            w2.as_array(),
            config,
            &global_basis,
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
    
    #[getter]
    pub fn eta(&self) -> f32 {
        self.inner.config.eta
    }
    
    #[getter]
    pub fn alpha(&self) -> f32 {
        self.inner.config.alpha
    }
    
    #[getter]
    pub fn beta(&self) -> f32 {
        self.inner.config.beta
    }
    
    #[getter]
    pub fn gamma(&self) -> f32 {
        self.inner.config.gamma
    }
    
    #[getter]
    pub fn g_inv<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.inner.g_inv.clone().into_pyarray(py)
    }
    
    #[getter]
    pub fn g_diag<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.inner.g_diag.clone().into_pyarray(py)
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
        dict.set_item("g_sym", comp.g_sym.into_pyarray(py)).unwrap();
        dict.set_item("a_antisym", comp.a_antisym.into_pyarray(py)).unwrap();
        dict.set_item("u_metric", comp.u_metric.into_pyarray(py)).unwrap();
        dict.set_item("v_metric", comp.v_metric.into_pyarray(py)).unwrap();
        dict.set_item("g_core", comp.g_core.into_pyarray(py)).unwrap();
        dict.set_item("a_core", comp.a_core.into_pyarray(py)).unwrap();
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
        g_sym: PyReadonlyArray2<f32>,
        a_antisym: PyReadonlyArray2<f32>,
        u_metric: PyReadonlyArray2<f32>,
        v_metric: PyReadonlyArray2<f32>,
        g_core: PyReadonlyArray2<f32>,
        a_core: PyReadonlyArray2<f32>,
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
            g_sym: g_sym.as_array().to_owned(),
            a_antisym: a_antisym.as_array().to_owned(),
            u_metric: u_metric.as_array().to_owned(),
            v_metric: v_metric.as_array().to_owned(),
            g_core: g_core.as_array().to_owned(),
            a_core: a_core.as_array().to_owned(),
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
            calibration_samples: 1024,
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

#[pyfunction(signature = (
    x, v1, s1, u1, v2, s2, u2, g_inv,
    v_mem=None,
    eta=0.01, alpha=0.02, gamma_param=0.99
))]
pub fn rsulf_forward_cuda_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v1: PyReadonlyArray2<f32>,
    s1: PyReadonlyArray1<f32>,
    u1: PyReadonlyArray2<f32>,
    v2: PyReadonlyArray2<f32>,
    s2: PyReadonlyArray1<f32>,
    u2: PyReadonlyArray2<f32>,
    g_inv: PyReadonlyArray1<f32>,
    v_mem: Option<PyReadonlyArray1<f32>>,
    eta: f32,
    alpha: f32,
    gamma_param: f32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (&py, &x, &v1, &s1, &u1, &v2, &s2, &u2, &g_inv, &v_mem, eta, alpha, gamma_param);
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::bindings::rsulf::rsulf_cuda_ffi::*;
        use numpy::{PyArray1};
        use pyo3::exceptions::PyRuntimeError;
        use pyo3::PyErr;
        use std::ffi::c_void;
        use std::ptr;
        use std::slice;

        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(PyRuntimeError::new_err("x must be 2D array"));
        }
        let batch = x_shape[0] as i32;
        let d = x_shape[1] as i32;
        let r = s1.shape()[0] as i32;
        let ffn_dim = v2.shape()[0] as i32;

        unsafe fn alloc_and_copy(src: &[f32]) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = src.len() * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            Ok(dst)
        }

        unsafe fn alloc_zeroed(len: usize) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = len * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            for i in 0..len {
                ptr::write(dst.add(i), 0.0);
            }
            Ok(dst)
        }

        let x_slice = x.as_slice()?;
        let v1_slice = v1.as_slice()?;
        let s1_slice = s1.as_slice()?;
        let u1_slice = u1.as_slice()?;
        let v2_slice = v2.as_slice()?;
        let s2_slice = s2.as_slice()?;
        let u2_slice = u2.as_slice()?;
        let g_inv_slice = g_inv.as_slice()?;
        let v_mem_slice = v_mem.as_ref().map(|v| v.as_slice().ok()).flatten();

        unsafe {
            let mut to_free: Vec<*mut c_void> = Vec::new();

            let x_dev = alloc_and_copy(x_slice)?;
            to_free.push(x_dev as *mut c_void);
            let v1_dev = alloc_and_copy(v1_slice)?;
            to_free.push(v1_dev as *mut c_void);
            let s1_dev = alloc_and_copy(s1_slice)?;
            to_free.push(s1_dev as *mut c_void);
            let u1_dev = alloc_and_copy(u1_slice)?;
            to_free.push(u1_dev as *mut c_void);
            let v2_dev = alloc_and_copy(v2_slice)?;
            to_free.push(v2_dev as *mut c_void);
            let s2_dev = alloc_and_copy(s2_slice)?;
            to_free.push(s2_dev as *mut c_void);
            let u2_dev = alloc_and_copy(u2_slice)?;
            to_free.push(u2_dev as *mut c_void);
            let g_inv_dev = alloc_and_copy(g_inv_slice)?;
            to_free.push(g_inv_dev as *mut c_void);

            let v_mem_dev = if let Some(slice_v) = v_mem_slice {
                let ptr_vm = alloc_and_copy(slice_v)?;
                to_free.push(ptr_vm as *mut c_void);
                ptr_vm
            } else {
                ptr::null_mut()
            };

            let total_x = (batch as usize) * (d as usize);
            let x_out_dev = alloc_zeroed(total_x)?;
            to_free.push(x_out_dev as *mut c_void);

            let v_out_dev = alloc_zeroed(batch as usize)?;
            to_free.push(v_out_dev as *mut c_void);

            rsulf_forward_cuda(
                x_dev,
                v1_dev,
                s1_dev,
                u1_dev,
                v2_dev,
                s2_dev,
                u2_dev,
                g_inv_dev,
                v_mem_dev,
                eta,
                alpha,
                gamma_param,
                batch,
                d,
                r,
                ffn_dim,
                x_out_dev,
                v_out_dev,
            );

            let sync_err = cudaDeviceSynchronize();
            if sync_err != 0 {
                for ptr_raw in to_free {
                    let _ = cudaFree(ptr_raw);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "cudaDeviceSynchronize failed: {}",
                    sync_err
                )));
            }

            let x_host = slice::from_raw_parts(x_out_dev, total_x).to_vec();
            let v_host = slice::from_raw_parts(v_out_dev, batch as usize).to_vec();

            for ptr_raw in to_free {
                let _ = cudaFree(ptr_raw);
            }

            let x_arr = Array2::from_shape_vec((batch as usize, d as usize), x_host)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let x_out = x_arr.into_pyarray(py);
            let v_out = PyArray1::from_vec(py, v_host);

            Ok((x_out, v_out))
        }
    }
}

#[pyfunction(signature = (
    x, v1, s1, u1, v2, s2, u2, g_inv,
    v_mem=None,
    eta=0.01, alpha=0.02, gamma_param=0.99,
    batch=1, seq_len=1
))]
pub fn rsulf_batch_forward_cuda_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v1: PyReadonlyArray2<f32>,
    s1: PyReadonlyArray1<f32>,
    u1: PyReadonlyArray2<f32>,
    v2: PyReadonlyArray2<f32>,
    s2: PyReadonlyArray1<f32>,
    u2: PyReadonlyArray2<f32>,
    g_inv: PyReadonlyArray1<f32>,
    v_mem: Option<PyReadonlyArray1<f32>>,
    eta: f32,
    alpha: f32,
    gamma_param: f32,
    batch: i32,
    seq_len: i32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            &py, &x, &v1, &s1, &u1, &v2, &s2, &u2, &g_inv, &v_mem, eta, alpha, gamma_param, batch,
            seq_len,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::bindings::rsulf::rsulf_cuda_ffi::*;
        use numpy::{PyArray1};
        use pyo3::exceptions::PyRuntimeError;
        use pyo3::PyErr;
        use std::ffi::c_void;
        use std::ptr;
        use std::slice;

        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(PyRuntimeError::new_err("x must be 2D array"));
        }
        let d = x_shape[1] as i32;
        let r = s1.shape()[0] as i32;
        let ffn_dim = v2.shape()[0] as i32;

        unsafe fn alloc_and_copy(src: &[f32]) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = src.len() * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            Ok(dst)
        }

        unsafe fn alloc_zeroed(len: usize) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = len * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            for i in 0..len {
                ptr::write(dst.add(i), 0.0);
            }
            Ok(dst)
        }

        let x_slice = x.as_slice()?;
        let v1_slice = v1.as_slice()?;
        let s1_slice = s1.as_slice()?;
        let u1_slice = u1.as_slice()?;
        let v2_slice = v2.as_slice()?;
        let s2_slice = s2.as_slice()?;
        let u2_slice = u2.as_slice()?;
        let g_inv_slice = g_inv.as_slice()?;
        let v_mem_slice = v_mem.as_ref().map(|v| v.as_slice().ok()).flatten();

        unsafe {
            let mut to_free: Vec<*mut c_void> = Vec::new();

            let x_dev = alloc_and_copy(x_slice)?;
            to_free.push(x_dev as *mut c_void);
            let v1_dev = alloc_and_copy(v1_slice)?;
            to_free.push(v1_dev as *mut c_void);
            let s1_dev = alloc_and_copy(s1_slice)?;
            to_free.push(s1_dev as *mut c_void);
            let u1_dev = alloc_and_copy(u1_slice)?;
            to_free.push(u1_dev as *mut c_void);
            let v2_dev = alloc_and_copy(v2_slice)?;
            to_free.push(v2_dev as *mut c_void);
            let s2_dev = alloc_and_copy(s2_slice)?;
            to_free.push(s2_dev as *mut c_void);
            let u2_dev = alloc_and_copy(u2_slice)?;
            to_free.push(u2_dev as *mut c_void);
            let g_inv_dev = alloc_and_copy(g_inv_slice)?;
            to_free.push(g_inv_dev as *mut c_void);

            let v_mem_dev = if let Some(slice_v) = v_mem_slice {
                let ptr_vm = alloc_and_copy(slice_v)?;
                to_free.push(ptr_vm as *mut c_void);
                ptr_vm
            } else {
                ptr::null_mut()
            };

            let total_tokens = (batch as usize) * (seq_len as usize);
            let total_x = total_tokens * (d as usize);
            let x_out_dev = alloc_zeroed(total_x)?;
            to_free.push(x_out_dev as *mut c_void);

            let v_out_dev = alloc_zeroed(total_tokens)?;
            to_free.push(v_out_dev as *mut c_void);

            rsulf_batch_forward_cuda(
                x_dev,
                v1_dev,
                s1_dev,
                u1_dev,
                v2_dev,
                s2_dev,
                u2_dev,
                g_inv_dev,
                v_mem_dev,
                eta,
                alpha,
                gamma_param,
                batch,
                seq_len,
                d,
                r,
                ffn_dim,
                x_out_dev,
                v_out_dev,
            );

            let sync_err = cudaDeviceSynchronize();
            if sync_err != 0 {
                for ptr_raw in to_free {
                    let _ = cudaFree(ptr_raw);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "cudaDeviceSynchronize failed: {}",
                    sync_err
                )));
            }

            let x_host = slice::from_raw_parts(x_out_dev, total_x).to_vec();
            let v_host = slice::from_raw_parts(v_out_dev, total_tokens).to_vec();

            for ptr_raw in to_free {
                let _ = cudaFree(ptr_raw);
            }

            let x_arr = Array2::from_shape_vec((total_tokens, d as usize), x_host)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let x_out = x_arr.into_pyarray(py);
            let v_out = PyArray1::from_vec(py, v_host);

            Ok((x_out, v_out))
        }
    }
}

#[pyfunction(signature = (
    x, v1, s1, u1, v2, s2, u2, g_inv, laplacian,
    v_mem=None,
    eta=0.01, alpha=0.02, beta=0.0, gamma_param=0.99, curvature=0.0,
    batch=1, seq_len=1, window=1
))]
pub fn rsulf_unified_forward_cuda_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    v1: PyReadonlyArray2<f32>,
    s1: PyReadonlyArray1<f32>,
    u1: PyReadonlyArray2<f32>,
    v2: PyReadonlyArray2<f32>,
    s2: PyReadonlyArray1<f32>,
    u2: PyReadonlyArray2<f32>,
    g_inv: PyReadonlyArray1<f32>,
    laplacian: PyReadonlyArray2<f32>,
    v_mem: Option<PyReadonlyArray1<f32>>,
    eta: f32,
    alpha: f32,
    beta: f32,
    gamma_param: f32,
    curvature: f32,
    batch: i32,
    seq_len: i32,
    window: i32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            &py, &x, &v1, &s1, &u1, &v2, &s2, &u2, &g_inv, &laplacian, &v_mem, eta, alpha, beta,
            gamma_param, curvature, batch, seq_len, window,
        );
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda",
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::bindings::rsulf::rsulf_cuda_ffi::*;
        use numpy::{PyArray1};
        use pyo3::exceptions::PyRuntimeError;
        use pyo3::PyErr;
        use std::ffi::c_void;
        use std::ptr;
        use std::slice;

        let x_shape = x.shape();
        if x_shape.len() != 2 {
            return Err(PyRuntimeError::new_err("x must be 2D array"));
        }
        let d = x_shape[1] as i32;
        let r = s1.shape()[0] as i32;
        let ffn_dim = v2.shape()[0] as i32;

        unsafe fn alloc_and_copy(src: &[f32]) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = src.len() * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
            Ok(dst)
        }

        unsafe fn alloc_zeroed(len: usize) -> Result<*mut f32, PyErr> {
            let mut ptr_raw: *mut c_void = ptr::null_mut();
            let size = len * std::mem::size_of::<f32>();
            let err = cudaMallocManaged(&mut ptr_raw as *mut *mut c_void, size, 1);
            if err != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "cudaMallocManaged failed: {}",
                    err
                )));
            }
            let dst = ptr_raw as *mut f32;
            for i in 0..len {
                ptr::write(dst.add(i), 0.0);
            }
            Ok(dst)
        }

        let x_slice = x.as_slice()?;
        let v1_slice = v1.as_slice()?;
        let s1_slice = s1.as_slice()?;
        let u1_slice = u1.as_slice()?;
        let v2_slice = v2.as_slice()?;
        let s2_slice = s2.as_slice()?;
        let u2_slice = u2.as_slice()?;
        let g_inv_slice = g_inv.as_slice()?;
        let lap_slice = laplacian.as_slice()?;
        let v_mem_slice = v_mem.as_ref().map(|v| v.as_slice().ok()).flatten();

        unsafe {
            let mut to_free: Vec<*mut c_void> = Vec::new();

            let x_dev = alloc_and_copy(x_slice)?;
            to_free.push(x_dev as *mut c_void);
            let v1_dev = alloc_and_copy(v1_slice)?;
            to_free.push(v1_dev as *mut c_void);
            let s1_dev = alloc_and_copy(s1_slice)?;
            to_free.push(s1_dev as *mut c_void);
            let u1_dev = alloc_and_copy(u1_slice)?;
            to_free.push(u1_dev as *mut c_void);
            let v2_dev = alloc_and_copy(v2_slice)?;
            to_free.push(v2_dev as *mut c_void);
            let s2_dev = alloc_and_copy(s2_slice)?;
            to_free.push(s2_dev as *mut c_void);
            let u2_dev = alloc_and_copy(u2_slice)?;
            to_free.push(u2_dev as *mut c_void);

            let d_usize = d as usize;
            let g_inv_host: Vec<f32> = if g_inv_slice.len() >= d_usize {
                g_inv_slice[..d_usize].to_vec()
            } else {
                let mut v = vec![1.0_f32; d_usize];
                for (i, val) in g_inv_slice.iter().enumerate() {
                    v[i] = *val;
                }
                v
            };
            let g_inv_dev = alloc_and_copy(&g_inv_host)?;
            to_free.push(g_inv_dev as *mut c_void);
            let lap_dev = alloc_and_copy(lap_slice)?;
            to_free.push(lap_dev as *mut c_void);

            let v_mem_dev = if let Some(slice_v) = v_mem_slice {
                let ptr_vm = alloc_and_copy(slice_v)?;
                to_free.push(ptr_vm as *mut c_void);
                ptr_vm
            } else {
                ptr::null_mut()
            };

            let total_tokens = (batch as usize) * (seq_len as usize);
            let total_x = total_tokens * (d as usize);
            let x_out_dev = alloc_zeroed(total_x)?;
            to_free.push(x_out_dev as *mut c_void);

            let v_out_dev = alloc_zeroed(total_tokens)?;
            to_free.push(v_out_dev as *mut c_void);

            rsulf_unified_forward_cuda(
                x_dev,
                v1_dev,
                s1_dev,
                u1_dev,
                v2_dev,
                s2_dev,
                u2_dev,
                g_inv_dev,
                lap_dev,
                v_mem_dev,
                eta,
                alpha,
                beta,
                gamma_param,
                curvature,
                batch,
                seq_len,
                d,
                r,
                ffn_dim,
                window,
                x_out_dev,
                v_out_dev,
            );

            let sync_err = cudaDeviceSynchronize();
            if sync_err != 0 {
                for ptr_raw in to_free {
                    let _ = cudaFree(ptr_raw);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "cudaDeviceSynchronize failed: {}",
                    sync_err
                )));
            }

            let total_tokens_usize = total_tokens;
            let x_host = slice::from_raw_parts(x_out_dev, total_x).to_vec();
            let v_host = slice::from_raw_parts(v_out_dev, total_tokens_usize).to_vec();

            for ptr_raw in to_free {
                let _ = cudaFree(ptr_raw);
            }

            let x_arr = Array2::from_shape_vec((total_tokens_usize, d as usize), x_host)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let x_out = x_arr.into_pyarray(py);
            let v_out = PyArray1::from_vec(py, v_host);

            Ok((x_out, v_out))
        }
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

#[pyfunction(name = "analyze_layer")]
pub fn analyze_layer_py<'py>(
    py: Python<'py>,
    wq: PyReadonlyArray2<f32>,
    wk: PyReadonlyArray2<f32>,
    w1: PyReadonlyArray2<f32>,
    w2: PyReadonlyArray2<f32>,
    layer_idx: usize,
    target_rank: usize,
) -> &'py PyDict {
    let analysis = analyze_layer(
        wq.as_array(),
        wk.as_array(),
        w1.as_array(),
        w2.as_array(),
        layer_idx,
        target_rank,
    );
    let dict = PyDict::new(py);
    dict.set_item("layer_idx", analysis.layer_idx).unwrap();
    dict.set_item("param_count", analysis.param_count).unwrap();
    dict.set_item("spectral_decay", analysis.spectral_decay).unwrap();
    dict.set_item("condition_number", analysis.condition_number).unwrap();
    dict.set_item("recommended_rank", analysis.recommended_rank).unwrap();
    dict.set_item("expected_accuracy", analysis.expected_accuracy).unwrap();
    dict
}

#[pyfunction(name = "extract_global_basis")]
pub fn extract_global_basis_py<'py>(
    py: Python<'py>,
    layers_wq: Vec<PyReadonlyArray2<f32>>,
    layers_wk: Vec<PyReadonlyArray2<f32>>,
    target_rank: usize,
) -> &'py PyDict {
    let wq_views: Vec<_> = layers_wq.iter().map(|x| x.as_array()).collect();
    let wk_views: Vec<_> = layers_wk.iter().map(|x| x.as_array()).collect();
    
    let basis = crate::layers::rsulf::extract_global_basis(&wq_views, &wk_views, target_rank);
    
    let dict = PyDict::new(py);
    dict.set_item("u", basis.u.into_pyarray(py)).unwrap();
    dict.set_item("rank", basis.rank).unwrap();
    dict
}

#[pyfunction(name = "create_compression_plan")]
pub fn create_compression_plan_py<'py>(
    py: Python<'py>,
    analyses: Vec<&PyDict>,
    compression_ratio: f32,
) -> &'py PyDict {
    let mut layer_analyses = Vec::new();
    
    for d in analyses {
        let layer_idx = d.get_item("layer_idx").unwrap().expect("layer_idx missing").extract::<usize>().unwrap_or(0);
        let param_count = d.get_item("param_count").unwrap().expect("param_count missing").extract::<usize>().unwrap_or(0);
        let spectral_decay = d.get_item("spectral_decay").unwrap().expect("spectral_decay missing").extract::<f32>().unwrap_or(0.0);
        let condition_number = d.get_item("condition_number").unwrap().expect("condition_number missing").extract::<f32>().unwrap_or(0.0);
        let recommended_rank = d.get_item("recommended_rank").unwrap().expect("recommended_rank missing").extract::<usize>().unwrap_or(1);
        let expected_accuracy = d.get_item("expected_accuracy").unwrap().expect("expected_accuracy missing").extract::<f32>().unwrap_or(0.0);
        
        use crate::layers::rsulf::{LayerAnalysis, LayerType, CompressionStrategy};
        
        let strategy = CompressionStrategy::MetricSVD { 
            target_rank: recommended_rank, 
            expected_accuracy 
        };
        
        layer_analyses.push(LayerAnalysis {
            layer_idx,
            layer_type: LayerType::Attention,
            input_shape: (0, 0),
            output_shape: (0, 0),
            param_count,
            spectral_decay,
            condition_number,
            recommended_rank,
            expected_accuracy,
            strategy
        });
    }
    
    let plan = crate::layers::rsulf::create_compression_plan(layer_analyses, compression_ratio);
    
    let dict = PyDict::new(py);
    dict.set_item("total_original_params", plan.total_original_params).unwrap();
    dict.set_item("total_compressed_params", plan.total_compressed_params).unwrap();
    dict.set_item("expected_compression_ratio", plan.expected_compression_ratio).unwrap();
    dict.set_item("min_expected_accuracy", plan.min_expected_accuracy).unwrap();
    
    dict
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRSULFLayer>()?;
    m.add_function(wrap_pyfunction!(fold_metric_svd, m)?)?;
    m.add_function(wrap_pyfunction!(fold_ffn, m)?)?;
    m.add_function(wrap_pyfunction!(build_causal_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(verify_metric_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(fold_metric_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(nystrom_metric, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_layer_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_global_basis_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_compression_plan_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(rsulf_forward_cuda_py, m)?)?;
        m.add_function(wrap_pyfunction!(rsulf_batch_forward_cuda_py, m)?)?;
        m.add_function(wrap_pyfunction!(rsulf_unified_forward_cuda_py, m)?)?;
    }
    Ok(())
}
