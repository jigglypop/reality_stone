use crate::rbe::types::{DecodedParams, Packed64, PoincareMatrix};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::{pymethods, PyResult, Python};

#[pyclass(name = "RBECompressor")]
#[derive(Clone)]
pub struct RBECompressorPy;

#[pymethods]
impl RBECompressorPy {
    #[new]
    pub fn new() -> Self {
        RBECompressorPy
    }

    #[pyo3(name = "compress")]
    fn compress(
        &self,
        py: Python,
        matrix: PyReadonlyArray2<f32>,
        _progress_callback: Option<PyObject>,
    ) -> PyResult<u64> {
        let array = matrix.as_array();
        let rows = array.shape()[0];
        let cols = array.shape()[1];
        let slice = array.as_slice().unwrap();

        // 블록 크기가 작을 때는 빠른 compress 사용
        let callback = |_progress: u32| {};

        let compressed_matrix = if rows <= 128 && cols <= 128 {
            // 작은 블록은 빠른 압축
            py.allow_threads(|| PoincareMatrix::compress(slice, rows, cols, &callback))
        } else {
            // 큰 블록은 품질 우선
            py.allow_threads(|| PoincareMatrix::deep_compress(slice, rows, cols, &callback))
        };

        Ok(compressed_matrix.seed.0)
    }

    #[pyo3(name = "decompress")]
    fn decompress(
        &self,
        py: Python,
        seed: i64,
        rows: usize,
        cols: usize,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let p_matrix = PoincareMatrix {
            seed: Packed64(seed as u64),
            rows,
            cols,
        };

        let decompressed = py.allow_threads(|| p_matrix.decompress());

        let py_array = PyArray1::from_vec(py, decompressed);
        Ok(py_array.to_owned())
    }

    #[pyo3(name = "decode_params")]
    fn decode_params(&self, seed: i64) -> PyResult<String> {
        let params: DecodedParams = Packed64(seed as u64).decode();
        Ok(format!("{:?}", params))
    }
}

pub fn init_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RBECompressorPy>()?;
    Ok(())
}
