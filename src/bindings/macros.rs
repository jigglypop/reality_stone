// src/bindings/macros.rs

/// PyFunction 바인딩 생성을 위한 매크로
///
/// 사용법:
/// `create_binding!(파이썬_함수명, Rust_함수_경로, [인자1, 인자2, ...], 반환_타입);`
///
/// 예시:
/// `create_binding!(poincare_add, crate::layers::poincare::poincare_add, [u, v, c], PyArray2);`

#[macro_export]
macro_rules! create_binding {
    // (u, v, c) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [u, v, c], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            u: numpy::PyReadonlyArray2<f32>,
            v: numpy::PyReadonlyArray2<f32>,
            c: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let u_arr = u.as_array();
            let v_arr = v.as_array();
            let result = $rust_fn(&u_arr, &v_arr, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (u, r, c) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [u, r, c], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            u: numpy::PyReadonlyArray2<f32>,
            r: f32,
            c: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let u_arr = u.as_array();
            let result = $rust_fn(&u_arr, r, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (u, v, c) -> Array1<f32>
    ($py_fn_name:ident, $rust_fn:path, [u, v, c], PyArray1) => {
        #[pyo3::prelude::pyfunction]
        fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            u: numpy::PyReadonlyArray2<f32>,
            v: numpy::PyReadonlyArray2<f32>,
            c: f32,
        ) -> &'py numpy::PyArray1<f32> {
            let u_arr = u.as_array();
            let v_arr = v.as_array();
            let result = $rust_fn(&u_arr, &v_arr, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };

    // (x, c) -> Array2<f32>
    ($py_fn_name:ident, $rust_fn:path, [x, c], PyArray2) => {
        #[pyo3::prelude::pyfunction]
        fn $py_fn_name<'py>(
            py: pyo3::prelude::Python<'py>,
            x: numpy::PyReadonlyArray2<f32>,
            c: f32,
        ) -> &'py numpy::PyArray2<f32> {
            let x_arr = x.as_array();
            let result = $rust_fn(&x_arr, c);
            numpy::IntoPyArray::into_pyarray(result, py)
        }
    };
}
