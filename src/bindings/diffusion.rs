use pyo3::prelude::*;
use crate::layers::diffusion::RiemannianDiffusion;

// CUDA FFI 선언
#[cfg(feature = "cuda")]
extern "C" {
    fn riemannian_diffusion_step_cuda(
        h: *const f32,
        flow: *const f32,
        output: *mut f32,
        alpha: f32,
        dt: f32,
        n: i32,
        d: i32,
        stream: *mut std::ffi::c_void,
    );
}

#[pyclass]
pub struct PyRiemannianDiffusion {
    inner: RiemannianDiffusion,
}

#[pymethods]
impl PyRiemannianDiffusion {
    #[new]
    pub fn new(dim: usize, alpha: f32, dt: f32) -> Self {
        Self {
            inner: RiemannianDiffusion::new(dim, alpha, dt),
        }
    }

    /// CUDA 전용 Step 함수 (Zero-copy)
    /// PyTorch Tensor의 data_ptr을 직접 받습니다.
    #[cfg(feature = "cuda")]
    pub fn step_cuda(
        &self,
        h_ptr: u64,
        flow_ptr: u64,
        out_ptr: u64,
        n: i32,
        d: i32,
    ) -> PyResult<()> {
        unsafe {
            riemannian_diffusion_step_cuda(
                h_ptr as *const f32,
                flow_ptr as *const f32,
                out_ptr as *mut f32,
                self.inner.alpha,
                self.inner.dt,
                n,
                d,
                std::ptr::null_mut(), // Default stream
            );
        }
        Ok(())
    }

    // CPU Fallback (기존 코드 유지)
    pub fn step_cpu<'py>(
        &self,
        py: Python<'py>,
        h: numpy::PyReadonlyArray2<f32>,
        flow_field: numpy::PyReadonlyArray2<f32>,
    ) -> &'py numpy::PyArray2<f32> {
        use numpy::IntoPyArray;
        let h_arr = h.as_array();
        let flow_arr = flow_field.as_array();
        let result = self.inner.step(&h_arr, &flow_arr);
        result.into_pyarray(py)
    }
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRiemannianDiffusion>()?;
    Ok(())
}
