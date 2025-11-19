use ndarray::{arr2, s, Array2};
use _rust::bindings;
use _rust::layers::{klein, lorentz, poincare, spline};
use _rust::ops::mobius;

#[test]
fn poincare_exp_log_inverse_local() {
    let c = 0.3_f32;
    let x = arr2(&[[0.05_f32, -0.02]]);
    let v = arr2(&[[0.01_f32, 0.015]]);
    let y = poincare::poincare_exp_at(&x.view(), &v.view(), c);
    let v_rec = poincare::poincare_log_at(&x.view(), &y.view(), c);
    let diff = (&v - &v_rec).mapv(f32::abs).sum();
    assert!(
        diff < 5e-4,
        "Poincaré exp/log 국소 역함수 성질 실패: diff={} (허용 5e-4)",
        diff
    );
}

#[test]
fn poincare_exp_log_inverse_origin_multiple() {
    let c = 0.5_f32;
    let x = arr2(&[[0.0_f32, 0.0_f32]]);
    let vs = arr2(&[
        [0.01_f32, 0.02_f32],
        [-0.02_f32, 0.015_f32],
        [0.03_f32, -0.01_f32],
    ]);
    for i in 0..vs.nrows() {
        let v = vs.slice(s![i..i + 1, ..]).to_owned();
        let y = poincare::poincare_exp_at(&x.view(), &v.view(), c);
        let v_rec = poincare::poincare_log_at(&x.view(), &y.view(), c);
        let diff = (&v - &v_rec).mapv(f32::abs).sum();
        assert!(
            diff < 5e-4,
            "Poincaré exp/log 원점 다중 샘플 역함수 성질 실패 (row {}): diff={} (허용 5e-4)",
            i,
            diff
        );
    }
}

#[test]
fn poincare_exp_zero_step_is_identity() {
    let c = 0.7_f32;
    let x = arr2(&[[0.05_f32, -0.02_f32], [0.1_f32, 0.1_f32]]);
    let v = arr2(&[[0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32]]);
    let y = poincare::poincare_exp_at(&x.view(), &v.view(), c);
    let diff = (&x - &y).mapv(f32::abs).sum();
    assert!(
        diff < 1e-7,
        "Poincaré exp 0 스텝 항등 실패: diff={} (허용 1e-7)",
        diff
    );
}

#[test]
fn poincare_euclidean_limit_matches() {
    let c = 1e-9_f32;
    let x = arr2(&[[0.2_f32, -0.1]]);
    let v = arr2(&[[0.03_f32, 0.02]]);
    let y = poincare::poincare_exp_at(&x.view(), &v.view(), c);
    let approx = &x + &v;
    let diff_exp = (&y - &approx).mapv(f32::abs).sum();
    assert!(
        diff_exp < 1e-6,
        "Poincaré exp 유클리드 극한 실패: diff_exp={} (허용 1e-6)",
        diff_exp
    );
    let v_rec = poincare::poincare_log_at(&x.view(), &y.view(), c);
    let diff_log = (&v - &v_rec).mapv(f32::abs).sum();
    assert!(
        diff_log < 1e-6,
        "Poincaré log 유클리드 극한 실패: diff_log={} (허용 1e-6)",
        diff_log
    );
}

#[test]
fn poincare_distance_same_point() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let d = poincare::poincare_distance(&x.view(), &x.view(), c);
    assert!(
        d[0].abs() < 1e-5,
        "Poincaré 거리 자기 자신 검증 실패: d(x,x)={} (허용 1e-5)",
        d[0]
    );
}

#[test]
fn poincare_distance_symmetry() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let y = arr2(&[[0.3_f32, -0.1_f32]]);
    let d_xy = poincare::poincare_distance(&x.view(), &y.view(), c)[0];
    let d_yx = poincare::poincare_distance(&y.view(), &x.view(), c)[0];
    let diff = (d_xy - d_yx).abs();
    assert!(
        diff < 1e-6,
        "Poincaré 거리 대칭성 실패: d_xy={} d_yx={} diff={} (허용 1e-6)",
        d_xy,
        d_yx,
        diff
    );
}

#[test]
fn poincare_triangle_inequality_small() {
    let c = 1.0_f32;
    let x = arr2(&[[0.0_f32, 0.0_f32]]);
    let y = arr2(&[[0.1_f32, 0.1_f32]]);
    let z = arr2(&[[0.2_f32, -0.05_f32]]);
    let d_xy = poincare::poincare_distance(&x.view(), &y.view(), c)[0];
    let d_yz = poincare::poincare_distance(&y.view(), &z.view(), c)[0];
    let d_xz = poincare::poincare_distance(&x.view(), &z.view(), c)[0];
    assert!(
        d_xz <= d_xy + d_yz + 1e-4,
        "Poincaré 삼각부등식 실패: d_xz={} d_xy+d_yz+eps={} (eps=1e-4)",
        d_xz,
        d_xy + d_yz + 1e-4
    );
}

#[test]
fn poincare_ball_layer_interpolation() {
    let c = 1.0_f32;
    let u = arr2(&[[0.3_f32, 0.4_f32]]);
    let v = arr2(&[[-0.2_f32, 0.1_f32]]);
    let result_t0 = poincare::poincare_ball_layer(&u.view(), &v.view(), c, 0.0);
    let diff0 = (&result_t0 - &u).mapv(f32::abs).sum();
    assert!(
        diff0 < 1e-5,
        "Poincaré 지오데식 보간 t=0 끝점 불일치: diff0={} (허용 1e-5)",
        diff0
    );
    let result_t1 = poincare::poincare_ball_layer(&u.view(), &v.view(), c, 1.0);
    let diff1 = (&result_t1 - &v).mapv(f32::abs).sum();
    assert!(
        diff1 < 1e-5,
        "Poincaré 지오데식 보간 t=1 끝점 불일치: diff1={} (허용 1e-5)",
        diff1
    );
}

#[test]
fn mobius_add_identity_poincare() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let zero = arr2(&[[0.0_f32, 0.0_f32]]);
    let result = mobius::mobius_add(&x.view(), &zero.view(), c);
    let diff = (&result - &x).mapv(f32::abs).sum();
    assert!(
        diff < 1e-6,
        "모비우스 덧셈 항등원 실패 (x+0): diff={} (허용 1e-6)",
        diff
    );
}

#[test]
fn mobius_scalar_zero_poincare() {
    let c = 1.0_f32;
    let x = arr2(&[[0.3_f32, 0.4_f32]]);
    let r = 0.0_f32;
    let result = mobius::mobius_scalar(&x.view(), c, r);
    let zero = arr2(&[[0.0_f32, 0.0_f32]]);
    let diff = (&result - &zero).mapv(f32::abs).sum();
    assert!(
        diff < 1e-6,
        "모비우스 스칼라 r=0 실패: diff={} (허용 1e-6)",
        diff
    );
}

#[test]
fn mobius_scalar_identity_poincare() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let r = 1.0_f32;
    let result = mobius::mobius_scalar(&x.view(), c, r);
    let diff = (&result - &x).mapv(f32::abs).sum();
    assert!(
        diff < 1e-3,
        "모비우스 스칼라 r=1 항등 실패: diff={} (허용 1e-3)",
        diff
    );
}

#[test]
fn poincare_to_lorentz_constraint() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let lorentz_x = poincare::poincare_to_lorentz(&x.view(), c);
    let x0 = lorentz_x[[0, 0]];
    let space_norm_sq = lorentz_x[[0, 1]].powi(2) + lorentz_x[[0, 2]].powi(2);
    let constraint = x0 * x0 - space_norm_sq;
    let target = 1.0 / c;
    let diff = (constraint - target).abs();
    assert!(
        diff < 1e-5,
        "Poincaré→Lorentz 제약식 위반: x0^2-||x||^2={} target={} diff={} (허용 1e-5)",
        constraint,
        target,
        diff
    );
}

#[test]
fn poincare_klein_roundtrip() {
    let c = 0.7_f32;
    let x = arr2(&[[0.1_f32, -0.2_f32], [0.05_f32, 0.05_f32]]);
    let k = poincare::poincare_to_klein(&x.view(), c);
    let p = klein::klein_to_poincare(&k.view(), c);
    let diff = (&x - &p).mapv(f32::abs).sum();
    assert!(
        diff < 1e-4,
        "Poincaré↔Klein 왕복 변환 실패: diff={} (허용 1e-4)",
        diff
    );
}

#[test]
fn poincare_lorentz_distance_consistency() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let y = arr2(&[[0.2_f32, -0.1_f32]]);
    let d_p = poincare::poincare_distance(&x.view(), &y.view(), c)[0];
    let lx = poincare::poincare_to_lorentz(&x.view(), c);
    let ly = poincare::poincare_to_lorentz(&y.view(), c);
    let d_l = lorentz::lorentz_distance(&lx.view(), &ly.view(), c)[0];
    let diff = (d_p - d_l).abs();
    assert!(
        diff < 5e-2,
        "Poincaré/Lorentz 거리 불일치: d_p={} d_l={} diff={} (허용 5e-2)",
        d_p,
        d_l,
        diff
    );
}

// Lorentz layer tests (from src/layers/tests/lorentz.rs)

fn to_lorentz_coords(space: &Array2<f32>, c: f32) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((space.nrows(), space.ncols() + 1));
    for i in 0..space.nrows() {
        let mut norm_sq = 0.0f32;
        for j in 0..space.ncols() {
            let v = space[[i, j]];
            norm_sq += v * v;
            out[[i, j + 1]] = v;
        }
        out[[i, 0]] = (1.0 / c + norm_sq).sqrt();
    }
    out
}

fn sum_forward(u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> f32 {
    let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
    y.sum()
}

#[test]
fn lorentz_distance_self_zero() {
    let c = 1.0_f32;
    let sp = arr2(&[[0.1_f32, 0.2]]);
    let x = to_lorentz_coords(&sp, c);
    let d = lorentz::lorentz_distance(&x.view(), &x.view(), c);
    assert!(
        d[0] >= 0.0 && d[0] <= 1e-3,
        "Lorentz 거리 자기 자신 검증 실패: d(x,x)={} (허용 [0,1e-3])",
        d[0]
    );
}

#[test]
fn lorentz_layer_forward_shapes_and_finiteness() {
    let c = 0.8_f32;
    let t = 0.3_f32;
    let usp = arr2(&[[0.1_f32, 0.2], [-0.1, 0.05]]);
    let vsp = arr2(&[[-0.05_f32, 0.1], [0.02, -0.03]]);
    let u = to_lorentz_coords(&usp, c);
    let v = to_lorentz_coords(&vsp, c);
    let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
    assert_eq!(
        y.dim(),
        u.dim(),
        "Lorentz 레이어 forward 출력 shape 불일치: got={:?} expected={:?}",
        y.dim(),
        u.dim()
    );
    assert!(
        y.iter().all(|z| z.is_finite()),
        "Lorentz 레이어 forward 출력에 비유한/NaN 값 포함"
    );
}

#[test]
fn lorentz_layer_backward_matches_numeric_small() {
    let c = 1.0_f32;
    let t = 0.3_f32;
    let usp = arr2(&[[0.1_f32, 0.2]]);
    let vsp = arr2(&[[-0.05_f32, 0.1]]);
    let mut u = to_lorentz_coords(&usp, c);
    let mut v = to_lorentz_coords(&vsp, c);

    let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
    let grad_output = Array2::<f32>::ones(y.dim());
    let (gu, gv) =
        lorentz::lorentz_layer_backward(&grad_output.view(), &u.view(), &v.view(), c, t);

    let eps = 1e-4f32;
    let mut gu_num = Array2::<f32>::zeros(u.dim());
    for j in 0..u.ncols() {
        let orig = u[(0, j)];
        u[(0, j)] = orig + eps;
        let f_plus = sum_forward(&u, &v, c, t);
        u[(0, j)] = orig - eps;
        let f_minus = sum_forward(&u, &v, c, t);
        u[(0, j)] = orig;
        gu_num[(0, j)] = (f_plus - f_minus) / (2.0 * eps);
    }
    let mut gv_num = Array2::<f32>::zeros(v.dim());
    for j in 0..v.ncols() {
        let orig = v[(0, j)];
        v[(0, j)] = orig + eps;
        let f_plus = sum_forward(&u, &v, c, t);
        v[(0, j)] = orig - eps;
        let f_minus = sum_forward(&u, &v, c, t);
        v[(0, j)] = orig;
        gv_num[(0, j)] = (f_plus - f_minus) / (2.0 * eps);
    }

    let tol = 2e-1f32;
    for j in 0..u.ncols() {
        let diff_u = (gu[(0, j)] - gu_num[(0, j)]).abs();
        assert!(
            diff_u <= tol,
            "Lorentz 역전파 grad_u 불일치 (차원 {}): analytic={} numeric={} diff={} tol={}",
            j,
            gu[(0, j)],
            gu_num[(0, j)],
            diff_u,
            tol
        );
        let diff_v = (gv[(0, j)] - gv_num[(0, j)]).abs();
        assert!(
            diff_v <= tol,
            "Lorentz 역전파 grad_v 불일치 (차원 {}): analytic={} numeric={} diff={} tol={}",
            j,
            gv[(0, j)],
            gv_num[(0, j)],
            diff_v,
            tol
        );
    }
}

// Klein model tests (from src/layers/tests/klein.rs)

#[test]
fn klein_add_identity_like() {
    let c = 0.9_f32;
    let u = arr2(&[[0.1_f32, 0.2]]);
    let z = arr2(&[[0.0_f32, 0.0]]);
    let res = klein::klein_add(&u.view(), &z.view(), c);
    let u_norm_sq = u[[0, 0]].powi(2) + u[[0, 1]].powi(2);
    let eps = 1e-7_f32;
    let u_denom = (1.0 - c * u_norm_sq).max(eps).sqrt();
    let expected_scale = 1.0 / (u_denom + 1.0);
    let expected = u.mapv(|x| x * expected_scale);
    assert!(((res - expected).mapv(f32::abs)).sum() < 1e-6);
}

#[test]
fn klein_scalar_bounds() {
    let c = 0.5_f32;
    let x = arr2(&[[0.4_f32, 0.2]]);
    let y = klein::klein_scalar(&x.view(), c, 2.0);
    let norm = (y[[0, 0]].powi(2) + y[[0, 1]].powi(2)).sqrt();
    assert!(norm < 1.0 / c.sqrt());
}

#[test]
fn klein_to_poincare_and_back_shapes() {
    let c = 0.7_f32;
    let x = arr2(&[[0.1_f32, -0.2], [0.05, 0.05]]);
    let p = klein::klein_to_poincare(&x.view(), c);
    assert_eq!(p.dim(), x.dim());
    let l = klein::klein_to_lorentz(&x.view(), c);
    assert_eq!(l.ncols(), x.ncols() + 1);
}

// Spline layer tests (from src/layers/spline.rs)

#[test]
fn spline_interpolate_shape_and_ratio() {
    let k = 8;
    let in_features = 4;
    let out_features = 10;
    let layer = spline::SplineLayer::new(k, in_features, out_features);
    let weight = layer.interpolate_internal();
    assert_eq!(weight.dim(), (out_features, in_features));
    let ratio = layer.get_compression_ratio();
    assert!(ratio > 1.0);
}

#[test]
fn spline_forward_linearity_on_zero() {
    let k = 6;
    let in_features = 3;
    let out_features = 5;
    let layer = spline::SplineLayer::new(k, in_features, out_features);
    let input = Array2::<f32>::zeros((2, in_features));
    pyo3::prepare_freethreaded_python();
    let numpy_available = pyo3::Python::with_gil(|py| py.import("numpy").is_ok());
    if !numpy_available {
        // 환경에 numpy 모듈이 없으면 이 테스트는 스킵 (핵심 로직은 순수 Rust에서 이미 검증됨)
        return;
    }
    pyo3::Python::with_gil(|py| {
        use numpy::ToPyArray;
        let out = layer.forward(py, input.view().to_pyarray(py).readonly());
        let out_arr = unsafe { out.as_array() };
        let max_abs = out_arr.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        assert!(
            max_abs < 1e-6,
            "SplineLayer 0 입력 선형성 실패: max_abs={} (허용 1e-6)",
            max_abs
        );
    });
}

// ---------------------------------------------------------------------------
// Python bindings vs Rust core consistency tests
// ---------------------------------------------------------------------------

#[test]
fn poincare_distance_binding_matches_rust() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let y = arr2(&[[0.2_f32, -0.1_f32]]);
    let d_rust = poincare::poincare_distance(&x.view(), &y.view(), c)[0];

    pyo3::prepare_freethreaded_python();
    let numpy_available = pyo3::Python::with_gil(|py| py.import("numpy").is_ok());
    if !numpy_available {
        return;
    }

    pyo3::Python::with_gil(|py| {
        use numpy::ToPyArray;
        let x_py = x.to_pyarray(py).readonly();
        let y_py = y.to_pyarray(py).readonly();
        let out = bindings::poincare::poincare_distance_cpu(py, x_py, y_py, c);
        let d_py = unsafe { out.as_array() };
        let diff = (d_rust - d_py[0]).abs();
        assert!(
            diff < 1e-5,
            "poincare_distance binding mismatch: rust={} py={} diff={} (허용 1e-5)",
            d_rust,
            d_py[0],
            diff
        );
    });
}

#[test]
fn lorentz_distance_binding_matches_rust() {
    let c = 1.0_f32;
    let sp = arr2(&[[0.1_f32, 0.2_f32]]);
    let x = to_lorentz_coords(&sp, c);
    let d_rust = lorentz::lorentz_distance(&x.view(), &x.view(), c)[0];

    pyo3::prepare_freethreaded_python();
    let numpy_available = pyo3::Python::with_gil(|py| py.import("numpy").is_ok());
    if !numpy_available {
        return;
    }

    pyo3::Python::with_gil(|py| {
        use numpy::ToPyArray;
        let x_py1 = x.to_pyarray(py).readonly();
        let x_py2 = x.to_pyarray(py).readonly();
        let out = bindings::lorentz::lorentz_distance(py, x_py1, x_py2, c);
        let d_py = unsafe { out.as_array() };
        let diff = (d_rust - d_py[0]).abs();
        assert!(
            diff < 1e-5,
            "lorentz_distance binding mismatch: rust={} py={} diff={} (허용 1e-5)",
            d_rust,
            d_py[0],
            diff
        );
    });
}

#[test]
fn klein_distance_binding_matches_rust() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let y = arr2(&[[0.2_f32, -0.1_f32]]);
    let d_rust = klein::klein_distance(&x.view(), &y.view(), c)[0];

    pyo3::prepare_freethreaded_python();
    let numpy_available = pyo3::Python::with_gil(|py| py.import("numpy").is_ok());
    if !numpy_available {
        return;
    }

    pyo3::Python::with_gil(|py| {
        use numpy::ToPyArray;
        let x_py = x.to_pyarray(py).readonly();
        let y_py = y.to_pyarray(py).readonly();
        let out = bindings::klein::klein_distance(py, x_py, y_py, c);
        let d_py = unsafe { out.as_array() };
        let diff = (d_rust - d_py[0]).abs();
        assert!(
            diff < 1e-5,
            "klein_distance binding mismatch: rust={} py={} diff={} (허용 1e-5)",
            d_rust,
            d_py[0],
            diff
        );
    });
}


