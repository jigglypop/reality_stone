use ndarray::{arr1, arr2, Array1, Array2};
use _rust::ops::mobius::{
    mobius_add, mobius_add_grad_c, mobius_scalar, mobius_scalar_grad_c,
};

fn numeric_grad_c<F>(f: F, c: f32, h: f32) -> Array2<f32>
where
    F: Fn(f32) -> Array2<f32>,
{
    let plus = f(c + h);
    let minus = f(c - h);
    (&plus - &minus) / (2.0 * h)
}

#[test]
fn mobius_scalar_grad_c_matches_numeric() {
    let u = arr2(&[[0.1_f32, -0.15_f32]]);
    let cases = [(0.3_f32, 1.2_f32), (-0.8_f32, 0.7_f32)];
    let h = 1e-3_f32;
    for &(c, r) in &cases {
        let numeric = numeric_grad_c(|cur| mobius_scalar(&u.view(), cur, r), c, h);
        let analytic = mobius_scalar_grad_c(&u.view(), c, r);
        let diff = (&numeric - &analytic).mapv(f32::abs).sum();
        assert!(
            diff < 5e-3,
            "mobius_scalar_grad_c f32 수치/해석 도함수 불일치: c={} r={} diff={} (허용 5e-3)",
            c,
            r,
            diff
        );
    }
}

#[test]
fn mobius_add_grad_c_matches_numeric() {
    let u = arr2(&[[0.05_f32, 0.07_f32]]);
    let v = arr2(&[[-0.03_f32, 0.02_f32]]);
    let h = 1e-3_f32;
    for &c in &[0.4_f32, -0.6_f32] {
        let numeric = numeric_grad_c(|cur| mobius_add(&u.view(), &v.view(), cur), c, h);
        let analytic = mobius_add_grad_c(&u.view(), &v.view(), c);
        let diff = (&numeric - &analytic).mapv(f32::abs).sum();
        assert!(
            diff < 5e-3,
            "mobius_add_grad_c f32 수치/해석 도함수 불일치: c={} diff={} (허용 5e-3)",
            c,
            diff
        );
    }
}

use _rust::layers::poincare::{poincare_exp_at_f64, poincare_log_at_f64};
use _rust::ops::mobius::{
    mobius_add_f64, mobius_add_grad_c_f64, mobius_scalar_f64, mobius_scalar_grad_c_f64,
};

fn numeric_grad_c_f64<F>(f: F, c: f64, h: f64) -> ndarray::Array2<f64>
where
    F: Fn(f64) -> ndarray::Array2<f64>,
{
    let plus = f(c + h);
    let minus = f(c - h);
    (&plus - &minus) / (2.0 * h)
}

#[test]
fn mobius_scalar_grad_c_f64_matches_numeric() {
    let u = ndarray::arr2(&[[0.1_f64, -0.15_f64]]);
    let cases = [(0.3_f64, 1.2_f64), (-0.8_f64, 0.7_f64)];
    let h = 1e-6_f64;
    for &(c, r) in &cases {
        let numeric = numeric_grad_c_f64(|cur| mobius_scalar_f64(&u.view(), cur, r), c, h);
        let analytic = mobius_scalar_grad_c_f64(&u.view(), c, r);
        let diff = (&numeric - &analytic).mapv(f64::abs).sum();
        assert!(
            diff < 5e-6,
            "mobius_scalar_grad_c f64 수치/해석 도함수 불일치: c={} r={} diff={} (허용 5e-6)",
            c,
            r,
            diff
        );
    }
}

#[test]
fn mobius_add_grad_c_f64_matches_numeric() {
    let u = ndarray::arr2(&[[0.05_f64, 0.07_f64]]);
    let v = ndarray::arr2(&[[-0.03_f64, 0.02_f64]]);
    let h = 1e-6_f64;
    for &c in &[0.4_f64, -0.6_f64] {
        let numeric = numeric_grad_c_f64(|cur| mobius_add_f64(&u.view(), &v.view(), cur), c, h);
        let analytic = mobius_add_grad_c_f64(&u.view(), &v.view(), c);
        let diff = (&numeric - &analytic).mapv(f64::abs).sum();
        assert!(
            diff < 5e-6,
            "mobius_add_grad_c f64 수치/해석 도함수 불일치: c={} diff={} (허용 5e-6)",
            c,
            diff
        );
    }
}

#[test]
fn poincare_exp_log_roundtrip_f64() {
    let c = 0.7_f64;
    let x = ndarray::arr2(&[[0.1_f64, -0.05_f64], [0.2_f64, 0.1_f64]]);
    let v = ndarray::arr2(&[[0.03_f64, 0.02_f64], [0.01_f64, -0.02_f64]]);
    let y = poincare_exp_at_f64(&x.view(), &v.view(), c);
    let v_rec = poincare_log_at_f64(&x.view(), &y.view(), c);
    let diff = (&v - &v_rec).mapv(f64::abs).sum();
    assert!(diff < 1e-8);
}

use _rust::ops::metrikey::{
    layer_norm_forward_exact_f32, layer_norm_forward_exact_f64, mahalanobis_distance_sq_g,
    spd_metric_from_key, spd_metric_from_key_f64,
};

#[test]
fn spd_metric_from_key_f64_properties() {
    let dim = 32;
    let g = spd_metric_from_key_f64("dept:AI:f64", dim, 0.1, 2.0);
    let gt = g.t().to_owned();
    let diff_sym = &g - &gt;
    let max_sym = diff_sym.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(max_sym < 1e-10);
    let v = Array1::from_shape_vec(dim, (0..dim).map(|i| (i as f64).cos()).collect()).unwrap();
    let s = v.dot(&g.dot(&v));
    assert!(s > 0.0);
}

#[test]
fn spd_metric_f32_f64_mahalanobis_close() {
    let dim = 16;
    let g32 = spd_metric_from_key("match", dim, 0.5, 1.5);
    let g64 = spd_metric_from_key_f64("match", dim, 0.5, 1.5);
    let g64_as_f32 =
        Array2::<f32>::from_shape_vec((dim, dim), g64.iter().map(|v| *v as f32).collect()).unwrap();
    let x = arr1(&(0..dim).map(|i| (i as f32).cos()).collect::<Vec<_>>());
    let y = arr1(&(0..dim).map(|i| (i as f32).sin()).collect::<Vec<_>>());
    let d_g32 = mahalanobis_distance_sq_g(&x, &y, &g32);
    let d_g64 = mahalanobis_distance_sq_g(&x, &y, &g64_as_f32);
    let diff = (d_g32 - d_g64).abs();
    assert!(diff < 1e-3);
}

#[test]
fn layer_norm_f32_f64_match() {
    let x32 = Array2::<f32>::from_shape_vec(
        (2, 3),
        vec![1.0_f32, -2.0_f32, 3.0_f32, 0.5_f32, 0.0_f32, -1.5_f32],
    )
    .unwrap();
    let gamma32 = arr1(&[1.0_f32, 0.5_f32, -1.0_f32]);
    let beta32 = arr1(&[0.0_f32, 1.0_f32, 0.5_f32]);
    let (y32, mu32, rstd32) =
        layer_norm_forward_exact_f32(&x32, &gamma32, &beta32, 1e-5_f32);

    let x64 = ndarray::Array2::<f64>::from_shape_vec(
        (2, 3),
        x32.iter().map(|v| *v as f64).collect(),
    )
    .unwrap();
    let gamma64 =
        ndarray::Array1::<f64>::from_vec(gamma32.iter().map(|v| *v as f64).collect());
    let beta64 =
        ndarray::Array1::<f64>::from_vec(beta32.iter().map(|v| *v as f64).collect());
    let (y64, mu64, rstd64) =
        layer_norm_forward_exact_f64(&x64, &gamma64, &beta64, 1e-8_f64);

    let y64_f32 = Array2::<f32>::from_shape_vec(
        (2, 3),
        y64.iter().map(|v| *v as f32).collect(),
    )
    .unwrap();
    let mu64_f32 = arr1(&mu64.iter().map(|v| *v as f32).collect::<Vec<_>>());
    let rstd64_f32 =
        arr1(&rstd64.iter().map(|v| *v as f32).collect::<Vec<_>>());

    let diff_y = (&y32 - &y64_f32).mapv(|v| v.abs()).sum();
    let diff_mu = (&mu32 - &mu64_f32).mapv(|v| v.abs()).sum();
    let diff_rstd = (&rstd32 - &rstd64_f32).mapv(|v| v.abs()).sum();

    assert!(
        diff_y < 1e-3,
        "LayerNorm f32/f64 출력 차이(diff_y)가 너무 큼: diff_y={}, diff_mu={}, diff_rstd={}",
        diff_y,
        diff_mu,
        diff_rstd
    );
    assert!(
        diff_mu < 1e-5,
        "LayerNorm f32/f64 평균(mu) 차이(diff_mu)가 너무 큼: diff_mu={}",
        diff_mu
    );
    assert!(
        diff_rstd < 1e-5,
        "LayerNorm f32/f64 rstd 차이(diff_rstd)가 너무 큼: diff_rstd={}",
        diff_rstd
    );
}
