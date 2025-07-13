// src/bindings/bitfield.rs

//! # BitfieldLinear를 위한 Python 바인딩
//!
//! 이 모듈은 `pyo3`를 사용하여 Rust로 구현된 `BitfieldLinear` 구조체를
//! Python에서 직접 사용할 수 있는 클래스로 노출합니다.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayDyn, PyReadonlyArrayDyn};
use ndarray::{Array, Array1, ArrayView, IxDyn};
use std::slice;

use crate::layers::bitfield::BitfieldLinear as RustBitfieldLinear;
use crate::layers::bitfield::kernel;

#[pyclass(name = "BitfieldLinear", module="reality_stone._rust")]
pub struct PyBitfieldLinear {
    inner: RustBitfieldLinear,
    bias: Option<Array1<f32>>,
}

#[pymethods]
impl PyBitfieldLinear {
    #[new]
    #[pyo3(signature = (m, n, b, r_max))]
    fn new(m: usize, n: usize, b: usize, r_max: f32) -> Self {
        PyBitfieldLinear {
            inner: RustBitfieldLinear::new(m, n, b, r_max),
            bias: None,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (weights, bias=None, basis_size=256, r_max=1.0, use_residual=true))]
    fn from_weights(
        weights: PyReadonlyArray2<f32>,
        bias: Option<PyReadonlyArray1<f32>>,
        basis_size: usize,
        r_max: f32,
        use_residual: bool
    ) -> PyResult<Self> {
        let weights_array = weights.as_array().to_owned();
        let bias_array = bias.map(|b| b.as_array().to_owned());
        
        let layer = RustBitfieldLinear::from_weights(
            &weights_array,
            basis_size,
            r_max,
            use_residual,
            false, // use_cuda는 나중에 설정
            false  // bitfield_residual - INT8 잔차 사용
        );
        
        Ok(PyBitfieldLinear {
            inner: layer,
            bias: bias_array,
        })
    }
    
    pub fn set_use_cuda(&mut self, use_cuda: bool) {
        self.inner.use_cuda = use_cuda;
    }

    pub fn use_cuda(&self) -> bool {
        self.inner.use_cuda
    }
    
    /// 압축된 코드와 기저 테이블 반환
    pub fn get_codes_and_basis<'py>(&self, py: Python<'py>) -> PyResult<(
        &'py PyArray1<u32>,      // codes
        &'py PyArray2<f32>,      // basis_table
        f32,                     // delta
    )> {
        // inner의 필드에 직접 접근하는 getter 메서드 필요
        // 일단 임시로 구현
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_codes_and_basis not implemented yet"
        ))
    }
    
    /// 잔차 데이터 반환 (있으면)
    pub fn get_residual_data<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        // 잔차 데이터를 딕셔너리로 반환
        // residual_codes 또는 residual_int8 + residual_scales
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "get_residual_data not implemented yet"
        ))
    }

    #[cfg(feature = "cuda")]
    #[pyo3(name = "init_gpu_memory")]
    pub fn init_gpu_memory(&mut self) -> PyResult<()> {
        self.inner.init_gpu_memory();
        Ok(())
    }

    /// CPU forward
    pub fn forward_cpu<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>) -> &'py PyArray2<f32> {
        let x_owned = x.as_array().to_owned();
        let result = self.inner.forward(&x_owned);
        PyArray2::from_array(py, &result)
    }
    
    /// CPU backward
    pub fn backward_cpu<'py>(&self, py: Python<'py>, grad_output: PyReadonlyArray2<'py, f32>) -> &'py PyArray2<f32> {
        let grad_output_owned = grad_output.as_array().to_owned();
        let grad_input = self.inner.backward(&grad_output_owned);
        PyArray2::from_array(py, &grad_input)
    }
    
    /// 통합 forward 메서드 (PyTorch 텐서 직접 처리)
    pub fn forward(&mut self, input: PyObject) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let device = input.getattr(py, "device")?;

            // PyTorch 텐서를 ndarray로 변환 (to_owned()로 복사하여 수명 문제 해결)
            let x_arr_owned = input
                .call_method0(py, "cpu")?
                .call_method0(py, "numpy")?
                .extract::<PyReadonlyArrayDyn<f32>>(py)?
                .to_owned_array();
            
            let x_arr = x_arr_owned.view();

            // 차원에 따라 다른 함수 호출
            let result_array_dyn: Array<f32, IxDyn> = if x_arr.ndim() == 2 {
                let x_2d = x_arr.to_owned().into_shape((x_arr.shape()[0], x_arr.shape()[1])).unwrap();
                self.inner.forward(&x_2d).into_dyn()
            } else {
                self.inner.forward_nd(&x_arr)
            };

            let result_numpy = PyArrayDyn::from_array(py, &result_array_dyn);
            
            // numpy를 PyTorch 텐서로 변환
            let torch = py.import("torch")?;
            let tensor = torch.getattr("from_numpy")?.call1((result_numpy,))?;
            
            // 원본과 같은 device로 이동
            let result = tensor.call_method1("to", (device,))?;
            Ok(result.into_py(py))
        })
    }
    
    /// 통합 backward 메서드
    pub fn backward(&self, grad_output: PyObject) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let device = grad_output.getattr(py, "device")?;

            // PyTorch 텐서를 ndarray로 변환 (to_owned()로 복사)
            let grad_arr_owned = grad_output
                .call_method0(py, "cpu")?
                .call_method0(py, "numpy")?
                .extract::<PyReadonlyArrayDyn<f32>>(py)?
                .to_owned_array();

            let grad_arr = grad_arr_owned.view();
            
            // 차원에 따라 다른 함수 호출
            let grad_input_dyn: Array<f32, IxDyn> = if grad_arr.ndim() == 2 {
                let grad_2d = grad_arr.to_owned().into_shape((grad_arr.shape()[0], grad_arr.shape()[1])).unwrap();
                self.inner.backward(&grad_2d).into_dyn()
            } else {
                self.inner.backward_nd(&grad_arr)
            };

            let grad_input_numpy = PyArrayDyn::from_array(py, &grad_input_dyn);
            
            // numpy를 PyTorch 텐서로 변환
            let torch = py.import("torch")?;
            let tensor = torch.getattr("from_numpy")?.call1((grad_input_numpy,))?;
            
            // 원본과 같은 device로 이동
            let result = tensor.call_method1("to", (device,))?;
            Ok(result.into_py(py))
        })
    }

    /// GPU forward (데이터 포인터 사용)
    #[cfg(feature = "cuda")]
    pub fn forward_cuda(
        &mut self,
        input_ptr: usize,
        shape: Vec<usize>,
        strides: Vec<i64>,
        _device: i32,
    ) -> PyResult<usize> {
        // GPU 메모리 초기화 (필요시)
        if self.inner.get_gpu_buffers().is_none() {
            self.inner.init_gpu_memory();
        }
        
        let buffers = self.inner.get_gpu_buffers().unwrap();
        let batch_size = shape[0];
        let num_elements = batch_size * self.inner.get_n();
        
        // 입력 포인터를 슬라이스로 변환
        let input_slice = unsafe {
            slice::from_raw_parts(input_ptr as *const f32, num_elements)
        };
        let x = ArrayView::from_shape((batch_size, self.inner.get_n()), input_slice).unwrap();
        
        // GPU 커널 호출
        let output = unsafe {
            kernel::gemm_hyper_bit_gpu_optimized(
                &x,
                buffers.codes_gpu,
                buffers.basis_table_gpu,
                buffers.residual_codes_gpu,
                self.inner.get_delta(),
                self.inner.get_residual_delta().unwrap_or(0.0),
                buffers.input_buffer_gpu,
                buffers.output_buffer_gpu,
                buffers.stream,
                self.inner.get_m(),
                self.inner.get_n(),
                self.inner.get_basis_table_shape().0,
            )
        };
        
        // 출력 포인터 반환
        Ok(output.as_ptr() as usize)
    }
    
    #[cfg(feature = "cuda")]
    #[pyo3(name = "backward_cuda")]
    pub fn backward_cuda(
        &self,
        grad_output_ptr: usize,
        shape: Vec<usize>,
        strides: Vec<i64>,
        _device: i32,
    ) -> PyResult<usize> {
        let buffers = self.inner.get_gpu_buffers()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("GPU not initialized"))?;
        
        let batch_size = shape[0];
        let grad_output_slice = unsafe { slice::from_raw_parts(grad_output_ptr as *const f32, batch_size * self.inner.get_m()) };
        let grad_output = ArrayView::from_shape((batch_size, self.inner.get_m()), grad_output_slice).unwrap();
        
        // 역전파 수행
        let grad_input = self.inner.backward(&grad_output.to_owned());
        
        // 계산된 grad_input을 GPU 입력 버퍼에 복사
        unsafe {
            kernel::cuda_memcpy_async_h2d(
                buffers.input_buffer_gpu,
                grad_input.as_ptr(),
                grad_input.len() * std::mem::size_of::<f32>(),
                buffers.stream,
            );
            kernel::cuda_stream_synchronize(buffers.stream);
        }
        
        // grad_input이 저장된 GPU 버퍼 포인터 반환
        Ok(buffers.input_buffer_gpu as usize)
    }
} 