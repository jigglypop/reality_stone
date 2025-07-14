use crate::{create_binding, layers::poincare, ops::project};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// --- 매크로를 사용한 바인딩 생성 ---

create_binding!(
    poincare_distance_cpu,
    poincare::poincare_distance,
    [u, v, c],
    PyArray1
);
create_binding!(
    poincare_to_lorentz_cpu,
    poincare::poincare_to_lorentz,
    [x, c],
    PyArray2
);
create_binding!(
    poincare_to_klein_cpu,
    poincare::poincare_to_klein,
    [x, c],
    PyArray2
);

// --- 매크로로 처리하기 복잡한 함수들 ---

#[pyfunction]
pub fn poincare_ball_layer_cpu<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> &'py PyArray2<f32> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();
    poincare::poincare_ball_layer(&u_arr, &v_arr, c, t).into_pyarray(py)
}

#[pyfunction]
pub fn poincare_ball_layer_backward_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    u: PyReadonlyArray2<f32>,
    v: PyReadonlyArray2<f32>,
    c: f32,
    t: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let (grad_u, grad_v) = poincare::poincare_ball_layer_backward(
        &grad_output.as_array(),
        &u.as_array(),
        &v.as_array(),
        c,
        t,
    );
    (grad_u.into_pyarray(py), grad_v.into_pyarray(py))
}

#[pyfunction]
pub fn mobius_add_vjp_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    x: PyReadonlyArray2<f32>,
    y: PyReadonlyArray2<f32>,
    c: f32,
) -> (&'py PyArray2<f32>, &'py PyArray2<f32>) {
    let (grad_x, grad_y) =
        poincare::mobius_add_vjp(&grad_output.as_array(), &x.as_array(), &y.as_array(), c);
    (grad_x.into_pyarray(py), grad_y.into_pyarray(py))
}

#[pyfunction]
pub fn mobius_scalar_vjp_cpu<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<f32>,
    x: PyReadonlyArray2<f32>,
    c: f32,
    r: f32,
) -> &'py PyArray2<f32> {
    poincare::mobius_scalar_vjp(&grad_output.as_array(), &x.as_array(), c, r).into_pyarray(py)
}

#[pyfunction]
pub fn project_to_ball_cpu<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    epsilon: f32,
) -> &'py PyArray2<f32> {
    project::project_to_ball(&x.as_array(), epsilon).into_pyarray(py)
}

// ... (Dynamic/Layerwise/CUDA 함수들은 생략 - 필요시 동일한 패턴으로 추가) ...

// --- 모듈 등록 ---

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(poincare_distance_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_to_lorentz_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_to_klein_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(poincare_ball_layer_backward_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_add_vjp_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(mobius_scalar_vjp_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_ball_cpu, m)?)?;

    // ... (Dynamic/Layerwise/CUDA 함수 등록) ...

    Ok(())
}
