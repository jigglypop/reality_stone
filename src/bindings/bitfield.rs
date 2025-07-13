// src/bindings/bitfield.rs

//! # BitfieldLinear를 위한 Python 바인딩
//!
//! 이 모듈은 `pyo3`를 사용하여 Rust로 구현된 `BitfieldLinear` 구조체를
//! Python에서 직접 사용할 수 있는 클래스로 노출합니다.

use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use ndarray::Array2;

use crate::layers::bitfield::BitfieldLinear as RustBitfieldLinear;

#[pyclass(name = "BitfieldLinear")]
pub struct PyBitfieldLinear {
    inner: RustBitfieldLinear,
}

#[pymethods]
impl PyBitfieldLinear {
    #[new]
    #[pyo3(signature = (m, n, b, r_max))]
    fn new(m: usize, n: usize, b: usize, r_max: f32) -> Self {
        let inner = RustBitfieldLinear::new(m, n, b, r_max);
        Self { inner }
    }

    #[staticmethod]
    fn from_weights(weights: &PyArray2<f32>, b: usize, r_max: f32) -> PyResult<Self> {
        let weights_array = unsafe { weights.as_array() };
        let weights_owned = weights_array.to_owned();
        
        Ok(Self {
            inner: RustBitfieldLinear::from_weights(&weights_owned, b, r_max),
        })
    }

    /// Python에서 순전파를 수행합니다.
    /// numpy 배열을 입력받아 계산 후 다시 numpy 배열로 반환합니다.
    ///
    /// # 인자
    /// * `x` - `[batch_size, n]` 형태의 numpy 배열.
    ///
    /// # 반환
    /// `[batch_size, m]` 형태의 numpy 배열.
    pub fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>) -> &'py PyArray2<f32> {
        // PyReadonlyArray2를 ndarray 뷰로 변환한 뒤, 소유권을 가진 배열로 복사합니다.
        // 이것이 컴파일 오류(mismatched types)를 해결하는 핵심입니다.
        let x_owned: Array2<f32> = x.as_array().to_owned();
        
        // 이제 소유권을 가진 배열의 참조를 전달합니다.
        let result: Array2<f32> = self.inner.forward(&x_owned);
        PyArray2::from_array(py, &result)
    }

    /// Python에서 역전파를 수행합니다.
    pub fn backward<'py>(&self, py: Python<'py>, grad_output: PyReadonlyArray2<'py, f32>) -> &'py PyArray2<f32> {
        let grad_output_owned: Array2<f32> = grad_output.as_array().to_owned();
        let grad_input: Array2<f32> = self.inner.backward(&grad_output_owned);
        PyArray2::from_array(py, &grad_input)
    }
} 