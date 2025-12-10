use _rust::ops::{
    block_orthogonal_from_key, compose_layers_order_preserving, dot_batched,
    mahalanobis_distance_sq_g, mahalanobis_distance_sq_l, metric_factor_cholesky, norm_sq_batched,
    project_to_ball, rotate_metric_factor_block, spd_metric_from_key, DynamicCurvature,
    LayerWiseDynamicCurvature,
};
use ndarray::{arr1, arr2, Array};

#[test]
fn test_norm_sq_batched() {
    let x = arr2(&[[1.0_f32, 2.0, 2.0], [3.0, 4.0, 0.0]]);
    let norms = norm_sq_batched(&x.view());
    let diff0 = (norms[0] - 9.0).abs();
    let diff1 = (norms[1] - 25.0).abs();
    assert!(
        diff0 < 1e-6,
        "norm_sq_batched 첫 번째 배치 값 불일치: got={} expected=9.0 diff={}",
        norms[0],
        diff0
    );
    assert!(
        diff1 < 1e-6,
        "norm_sq_batched 두 번째 배치 값 불일치: got={} expected=25.0 diff={}",
        norms[1],
        diff1
    );
}

#[test]
fn test_dot_batched() {
    let x = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let y = arr2(&[[6.0_f32, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let dots = dot_batched(&x.view(), &y.view());
    let expected0 = 1.0 * 6.0 + 2.0 * 5.0 + 3.0 * 4.0;
    let expected1 = 4.0 * 3.0 + 5.0 * 2.0 + 6.0 * 1.0;
    let diff0 = (dots[0] - expected0).abs();
    let diff1 = (dots[1] - expected1).abs();
    assert!(
        diff0 < 1e-6,
        "dot_batched 첫 번째 배치 값 불일치: got={} expected={} diff={}",
        dots[0],
        expected0,
        diff0
    );
    assert!(
        diff1 < 1e-6,
        "dot_batched 두 번째 배치 값 불일치: got={} expected={} diff={}",
        dots[1],
        expected1,
        diff1
    );
}

#[test]
fn dynamic_curvature_limits() {
    let dc_min = DynamicCurvature::new(-20.0, 0.1, 1.0);
    let dc_max = DynamicCurvature::new(20.0, 0.1, 1.0);
    let c_min_val = dc_min.compute_c();
    let c_max_val = dc_max.compute_c();
    let diff_min = (c_min_val - 0.1).abs();
    let diff_max = (c_max_val - 1.0).abs();
    assert!(
        diff_min < 1e-3,
        "DynamicCurvature c_min 극한값 불일치: got={} expected=0.1 diff={}",
        c_min_val,
        diff_min
    );
    assert!(
        diff_max < 1e-3,
        "DynamicCurvature c_max 극한값 불일치: got={} expected=1.0 diff={}",
        c_max_val,
        diff_max
    );
}

#[test]
fn dynamic_curvature_derivative_matches_numeric() {
    let kappa = 0.5_f32;
    let c_min = 0.1_f32;
    let c_max = 1.0_f32;
    let base = DynamicCurvature::new(kappa, c_min, c_max);
    let h = 1e-3_f32;
    let plus = DynamicCurvature::new(kappa + h, c_min, c_max);
    let minus = DynamicCurvature::new(kappa - h, c_min, c_max);
    let num = (plus.compute_c() - minus.compute_c()) / (2.0 * h);
    let ana = base.compute_dc_dkappa();
    let diff = (num - ana).abs();
    assert!(
        diff < 1e-3,
        "DynamicCurvature 도함수 불일치: numeric={} analytic={} diff={}",
        num,
        ana,
        diff
    );
}

#[test]
fn layerwise_curvature_limits_and_derivative() {
    let kappas_lim = vec![-20.0_f32, 0.0_f32, 20.0_f32];
    let lw_lim = LayerWiseDynamicCurvature::from_kappas(kappas_lim, 0.2, 0.8);
    let c0 = lw_lim.compute_c(0);
    let c2 = lw_lim.compute_c(2);
    let diff_c0 = (c0 - 0.2).abs();
    let diff_c2 = (c2 - 0.8).abs();
    assert!(
        diff_c0 < 1e-3,
        "LayerWiseDynamicCurvature c[0] 극한값 불일치: got={} expected=0.2 diff={}",
        c0,
        diff_c0
    );
    assert!(
        diff_c2 < 1e-3,
        "LayerWiseDynamicCurvature c[2] 극한값 불일치: got={} expected=0.8 diff={}",
        c2,
        diff_c2
    );

    let kappa0 = 0.5_f32;
    let h = 1e-3_f32;
    let lw_base = LayerWiseDynamicCurvature::from_kappas(vec![kappa0, 0.0_f32, 0.0_f32], 0.2, 0.8);
    let lw_plus =
        LayerWiseDynamicCurvature::from_kappas(vec![kappa0 + h, 0.0_f32, 0.0_f32], 0.2, 0.8);
    let lw_minus =
        LayerWiseDynamicCurvature::from_kappas(vec![kappa0 - h, 0.0_f32, 0.0_f32], 0.2, 0.8);
    let num = (lw_plus.compute_c(0) - lw_minus.compute_c(0)) / (2.0 * h);
    let ana = lw_base.compute_dc_dkappa(0);
    let diff = (num - ana).abs();
    assert!(
        diff < 1e-3,
        "LayerWiseDynamicCurvature 도함수 불일치: numeric={} analytic={} diff={}",
        num,
        ana,
        diff
    );
}

#[test]
fn test_project_to_ball_inside() {
    let x = arr2(&[[0.1_f32, 0.2], [0.3, 0.2]]);
    let y = project_to_ball(&x.view(), 1e-5);
    let diff00 = (y[[0, 0]] - 0.1).abs();
    let diff11 = (y[[1, 1]] - 0.2).abs();
    assert!(
        diff00 < 1e-6,
        "project_to_ball 내부점 보존 실패 (0,0): got={} expected=0.1 diff={}",
        y[[0, 0]],
        diff00
    );
    assert!(
        diff11 < 1e-6,
        "project_to_ball 내부점 보존 실패 (1,1): got={} expected=0.2 diff={}",
        y[[1, 1]],
        diff11
    );
}

#[test]
fn test_project_to_ball_outside() {
    let x = arr2(&[[0.9_f32, 0.9]]);
    let y = project_to_ball(&x.view(), 1e-3);
    let norm = (y[[0, 0]].powi(2) + y[[0, 1]].powi(2)).sqrt();
    let target = 1.0 - 1e-3;
    assert!(
        norm <= target + 1e-5,
        "project_to_ball 외부점 클리핑 실패: norm={} target_max={} margin={}",
        norm,
        target,
        1e-5_f32
    );
    assert!(
        norm > 0.0,
        "project_to_ball 결과 노름이 0 이하: norm={}",
        norm
    );
}

fn is_symmetric(a: &ndarray::Array2<f32>, tol: f32) -> bool {
    let diff = a - &a.t();
    diff.iter().all(|v| v.abs() <= tol)
}

#[test]
fn test_spd_metric_from_key_properties_f32() {
    let dim = 64;
    let g = spd_metric_from_key("dept:AI", dim, 0.1, 2.0);
    assert!(is_symmetric(&g, 1e-4), "SPD 메트릭 대칭성 위반 (f32)");
    let v = Array::from_shape_vec((dim,), (0..dim).map(|i| (i as f32).sin()).collect()).unwrap();
    let s = v.dot(&g.dot(&v));
    assert!(s > 0.0, "SPD 메트릭 양의정부 위반 (v^T G v <= 0): {}", s);
}

#[test]
fn test_mahalanobis_consistency_g_vs_l_f32() {
    let dim = 32;
    let g = spd_metric_from_key("k", dim, 0.5, 1.5);
    let l = metric_factor_cholesky(&g);
    let x = arr1(&(0..dim).map(|i| (i as f32).cos()).collect::<Vec<_>>());
    let y = arr1(&(0..dim).map(|i| (i as f32).sin()).collect::<Vec<_>>());
    let d_g = mahalanobis_distance_sq_g(&x, &y, &g);
    let d_l = mahalanobis_distance_sq_l(&x, &y, &l);
    let diff = (d_g - d_l).abs();
    assert!(
        diff < 1e-3,
        "Mahalanobis 거리 일관성 실패: d_g={} d_l={} diff={}",
        d_g,
        d_l,
        diff
    );
}

#[test]
fn test_block_orthogonal_is_orthonormal_f32() {
    let total = 128;
    let q = block_orthogonal_from_key("dept:AI", 64, 64);
    assert_eq!(q.dim(), (total, total));
    let i = ndarray::Array2::<f32>::eye(total);
    let should_be_i = q.t().dot(&q);
    let diff = &should_be_i - &i;
    let max_abs = diff.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
    assert!(max_abs < 5e-4, "직교성 오차가 너무 큼: max_abs={}", max_abs);
}

#[test]
fn test_compose_layers_order_preserving_matches_sequential_f32() {
    let dim = 16;
    let t1 = spd_metric_from_key("l1", dim, 0.8, 1.2);
    let t2 = spd_metric_from_key("l2", dim, 0.8, 1.2);
    let t3 = spd_metric_from_key("l3", dim, 0.8, 1.2);
    let layers = vec![t1.clone(), t2.clone(), t3.clone()];
    let t_total = compose_layers_order_preserving(&layers);
    let x = arr1(&(0..dim).map(|i| i as f32).collect::<Vec<_>>());
    let x_seq = t3.dot(&t2.dot(&t1.dot(&x)));
    let x_total = t_total.dot(&x);
    let err = (&x_seq - &x_total).mapv(|v| v.abs()).sum();
    assert!(
        err < 1e-3,
        "레이어 합성 결과 불일치: err={} (허용 1e-3)",
        err
    );
    let t_rev = compose_layers_order_preserving(&[t3, t2, t1]);
    let x_rev = t_rev.dot(&x);
    let diff_rev = (&x_rev - &x_total).mapv(|v| v.abs()).sum();
    assert!(
        diff_rev > 1e-4,
        "레이어 순서 뒤집기 후에도 결과가 거의 동일함: diff_rev={}",
        diff_rev
    );
}

#[test]
fn test_rotate_metric_factor_preserves_g_f32() {
    let dim = 32;
    let g = spd_metric_from_key("key", dim, 0.5, 1.5);
    let l = metric_factor_cholesky(&g);
    let l_rot = rotate_metric_factor_block("session", &l, 16);
    let g_rot = l_rot.t().dot(&l_rot);
    let diff = (&g - &g_rot).mapv(|v| v * v).sum().sqrt();
    assert!(
        diff < 1e-2,
        "메트릭 회전 후 Frobenius 차이 너무 큼: diff={} (허용 1e-2)",
        diff
    );
}
