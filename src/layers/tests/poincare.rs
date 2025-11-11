use crate::layers::poincare;
use approx::assert_relative_eq;
use ndarray::arr2;

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
