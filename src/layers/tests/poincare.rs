use crate::layers::{klein, lorentz, poincare};
use crate::ops::mobius;
use approx::assert_relative_eq;
use ndarray::{arr2, Array2};

#[test]
fn poincare_exp_log_inverse_local() {
    let c = 0.3_f32;
    let x = arr2(&[[0.05_f32, -0.02]]);
    let v = arr2(&[[0.01_f32, 0.015]]);
    let y = poincare::poincare_exp_at(&x.view(), &v.view(), c);
    let v_rec = poincare::poincare_log_at(&x.view(), &y.view(), c);
    assert_relative_eq!(v, v_rec, epsilon = 5e-4);
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
        assert_relative_eq!(v, v_rec, epsilon = 5e-4);
    }
}

#[test]
fn poincare_exp_zero_step_is_identity() {
    let c = 0.7_f32;
    let x = arr2(&[[0.05_f32, -0.02_f32], [0.1_f32, 0.1_f32]]);
    let v = arr2(&[[0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32]]);
    let y = poincare::poincare_exp_at(&x.view(), &v.view(), c);
    assert_relative_eq!(x, y, epsilon = 1e-7);
}

#[test]
fn poincare_euclidean_limit_matches() {
    let c = 1e-9_f32;
    let x = arr2(&[[0.2_f32, -0.1]]);
    let v = arr2(&[[0.03_f32, 0.02]]);
    let y = poincare::poincare_exp_at(&x.view(), &v.view(), c);
    let approx = &x + &v;
    assert_relative_eq!(y, approx, epsilon = 1e-6);
    let v_rec = poincare::poincare_log_at(&x.view(), &y.view(), c);
    assert_relative_eq!(v, v_rec, epsilon = 1e-6);
}

#[test]
fn test_poincare_distance_same_point() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let d = poincare::poincare_distance(&x.view(), &x.view(), c);
    assert!(d[0].abs() < 1e-5);
}

#[test]
fn test_poincare_distance_symmetry() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let y = arr2(&[[0.3_f32, -0.1_f32]]);
    let d_xy = poincare::poincare_distance(&x.view(), &y.view(), c);
    let d_yx = poincare::poincare_distance(&y.view(), &x.view(), c);
    assert_relative_eq!(d_xy[0], d_yx[0], epsilon = 1e-6);
}

#[test]
fn test_poincare_triangle_inequality_small() {
    let c = 1.0_f32;
    let x = arr2(&[[0.0_f32, 0.0_f32]]);
    let y = arr2(&[[0.1_f32, 0.1_f32]]);
    let z = arr2(&[[0.2_f32, -0.05_f32]]);
    let d_xy = poincare::poincare_distance(&x.view(), &y.view(), c)[0];
    let d_yz = poincare::poincare_distance(&y.view(), &z.view(), c)[0];
    let d_xz = poincare::poincare_distance(&x.view(), &z.view(), c)[0];
    assert!(d_xz <= d_xy + d_yz + 1e-4);
}

#[test]
fn test_poincare_ball_layer_interpolation() {
    let c = 1.0_f32;
    let u = arr2(&[[0.3_f32, 0.4_f32]]);
    let v = arr2(&[[-0.2_f32, 0.1_f32]]);
    let result_t0 = poincare::poincare_ball_layer(&u.view(), &v.view(), c, 0.0);
    assert_relative_eq!(result_t0, u, epsilon = 1e-5);
    let result_t1 = poincare::poincare_ball_layer(&u.view(), &v.view(), c, 1.0);
    assert_relative_eq!(result_t1, v, epsilon = 1e-5);
}

#[test]
fn test_mobius_add_identity() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let zero = arr2(&[[0.0_f32, 0.0_f32]]);
    let result = mobius::mobius_add(&x.view(), &zero.view(), c);
    assert_relative_eq!(result, x, epsilon = 1e-6);
}

#[test]
fn test_mobius_scalar_zero() {
    let c = 1.0_f32;
    let x = arr2(&[[0.3_f32, 0.4_f32]]);
    let r = 0.0_f32;
    let result = mobius::mobius_scalar(&x.view(), c, r);
    let zero = arr2(&[[0.0_f32, 0.0_f32]]);
    assert_relative_eq!(result, zero, epsilon = 1e-6);
}

#[test]
fn test_mobius_scalar_identity() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let r = 1.0_f32;
    let result = mobius::mobius_scalar(&x.view(), c, r);
    assert_relative_eq!(result, x, epsilon = 1e-3);
}

#[test]
fn test_poincare_to_lorentz_constraint() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let lorentz_x = poincare::poincare_to_lorentz(&x.view(), c);
    let x0 = lorentz_x[[0, 0]];
    let space_norm_sq = lorentz_x[[0, 1]].powi(2) + lorentz_x[[0, 2]].powi(2);
    let constraint = x0 * x0 - space_norm_sq;
    assert_relative_eq!(constraint, 1.0 / c, epsilon = 1e-5);
}

#[test]
fn test_poincare_klein_roundtrip() {
    let c = 0.7_f32;
    let x = arr2(&[[0.1_f32, -0.2_f32], [0.05_f32, 0.05_f32]]);
    let k = poincare::poincare_to_klein(&x.view(), c);
    let p = klein::klein_to_poincare(&k.view(), c);
    assert_relative_eq!(x, p, epsilon = 1e-4);
}

#[test]
fn test_poincare_lorentz_distance_consistency() {
    let c = 1.0_f32;
    let x = arr2(&[[0.1_f32, 0.2_f32]]);
    let y = arr2(&[[0.2_f32, -0.1_f32]]);
    let d_p = poincare::poincare_distance(&x.view(), &y.view(), c)[0];
    let lx = poincare::poincare_to_lorentz(&x.view(), c);
    let ly = poincare::poincare_to_lorentz(&y.view(), c);
    let d_l = lorentz::lorentz_distance(&lx.view(), &ly.view(), c)[0];
    assert_relative_eq!(d_p, d_l, epsilon = 1e-4);
}
