use pyo3::prelude::*;
use numpy::{PyArray4, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use std::os::raw::c_void;

#[cfg(feature = "cuda")]
extern "C" {
    fn geodesic_topk_attention_cuda(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        idx: *const i64,
        l: *const f32,
        c: f32,
        tau: f32,
        b: i32,
        h: i32,
        t: i32,
        s: i32,
        k_topk: i32,
        d_h: i32,
        d_v: i32,
        out: *mut f32,
    );
}

/// Fused Geodesic Top-k Attention (CUDA)
/// 
/// 한 커널에서 SPD metric 적용, geodesic distance 계산, softmax, aggregation을 모두 수행
/// 
/// # Arguments
/// * `q` - Query tensor [B, H, T, d_h]
/// * `k` - Key tensor [B, H, S, d_h]
/// * `v` - Value tensor [B, H, S, d_v]
/// * `idx` - Top-k indices [B, T, K]
/// * `l_factor` - SPD Cholesky factor [d_h, d_h]
/// * `c` - Curvature (default: 1.0)
/// * `tau` - Temperature (default: 1.0)
/// 
/// # Returns
/// * Output tensor [B, H, T, d_v]
#[pyfunction]
#[pyo3(signature = (q, k, v, idx, l_factor, c=1.0, tau=1.0))]
pub fn geodesic_topk_attention(
    py: Python,
    q: PyReadonlyArray4<f32>,
    k: PyReadonlyArray4<f32>,
    v: PyReadonlyArray4<f32>,
    idx: PyReadonlyArray3<i64>,
    l_factor: PyReadonlyArray2<f32>,
    c: f32,
    tau: f32,
) -> PyResult<Py<PyArray4<f32>>> {
    #[cfg(not(feature = "cuda"))]
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda"
        ));
    }

    #[cfg(feature = "cuda")]
    {
        // Get dimensions
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();
        let idx_shape = idx.shape();

        let b = q_shape[0] as i32;
        let h = q_shape[1] as i32;
        let t = q_shape[2] as i32;
        let d_h = q_shape[3] as i32;

        let s = k_shape[2] as i32;
        let d_v = v_shape[3] as i32;
        let k_topk = idx_shape[2] as i32;

        // Validate shapes
        if k_shape[0] != b as usize || k_shape[1] != h as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("K shape mismatch: expected [{}, {}, ?, {}], got {:?}", b, h, d_h, k_shape)
            ));
        }
        if v_shape[0] != b as usize || v_shape[1] != h as usize || v_shape[2] != s as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("V shape mismatch: expected [{}, {}, {}, {}], got {:?}", b, h, s, d_v, v_shape)
            ));
        }
        if idx_shape[0] != b as usize || idx_shape[1] != t as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("idx shape mismatch: expected [{}, {}, {}], got {:?}", b, t, k_topk, idx_shape)
            ));
        }

        // Get raw pointers
        let q_ptr = q.as_slice()?.as_ptr();
        let k_ptr = k.as_slice()?.as_ptr();
        let v_ptr = v.as_slice()?.as_ptr();
        let idx_ptr = idx.as_slice()?.as_ptr();
        let l_ptr = l_factor.as_slice()?.as_ptr();

        // Allocate output
        let out_shape = [b as usize, h as usize, t as usize, d_v as usize];
        let out_size = (b * h * t * d_v) as usize;
        let mut out_vec = vec![0.0f32; out_size];

        // Call CUDA kernel
        unsafe {
            geodesic_topk_attention_cuda(
                q_ptr,
                k_ptr,
                v_ptr,
                idx_ptr,
                l_ptr,
                c,
                tau,
                b,
                h,
                t,
                s,
                k_topk,
                d_h,
                d_v,
                out_vec.as_mut_ptr(),
            );
        }

        // Convert to numpy array
        Ok(PyArray4::from_vec(py, out_vec, out_shape).to_owned())
    }
}

/// Batched SPD Cholesky Decomposition (CUDA)
/// 
/// # Arguments
/// * `g` - SPD matrices [B, T, d, d]
/// 
/// # Returns
/// * Cholesky factors [B, T, d, d]
#[pyfunction]
pub fn batched_cholesky_cuda(
    py: Python,
    g: PyReadonlyArray4<f32>,
) -> PyResult<Py<PyArray4<f32>>> {
    #[cfg(not(feature = "cuda"))]
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support not enabled. Rebuild with --features cuda"
        ));
    }

    #[cfg(feature = "cuda")]
    {
        // TODO: Implement batched Cholesky CUDA kernel
        // For now, fallback to CPU
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "batched_cholesky_cuda not yet implemented"
        ))
    }
}

/// Register Python module
#[pymodule]
pub fn _rust_geodesic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(geodesic_topk_attention, m)?)?;
    m.add_function(wrap_pyfunction!(batched_cholesky_cuda, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_rust_geodesic").unwrap();
            let result = _rust_geodesic(py, module);
            assert!(result.is_ok());
        });
    }
}

