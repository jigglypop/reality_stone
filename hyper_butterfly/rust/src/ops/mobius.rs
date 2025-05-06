//! 뫼비우스 관련 연산 구현

use crate::config::constants::Constants;
use crate::utils::numeric::{atanh, batch_dot, l2_norm, safe_tanh};
use ndarray::{Array1, Array2, Axis};

/// 뫼비우스 덧셈: u ⊕_c v
pub fn mobius_add(u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array2<f32> {
    // u2 및 v2는 각 벡터의 제곱 노름
    let u2 = u
        .map_axis(Axis(1), |row| row.dot(&row))
        .insert_axis(Axis(1));
    let v2 = v
        .map_axis(Axis(1), |row| row.dot(&row))
        .insert_axis(Axis(1));

    // uv는 u와 v의 내적
    let uv = batch_dot(u, v).insert_axis(Axis(1));

    let c2 = c * c;

    // num_u = (1 + 2c<u,v> + c|v|²)u
    let num_u = u * &(1.0 + 2.0 * c * &uv + c * &v2);

    // num_v = (1 - c|u|²)v
    let num_v = v * &(1.0 - c * &u2);

    // denom = 1 + 2c<u,v> + c²|u|²|v|²
    let denom = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|x| x.max(Constants::MIN_DENOMINATOR));

    // 최종 결과: (num_u + num_v) / denom
    (num_u + num_v) / denom
}

/// 뫼비우스 스칼라 곱셈: r ⊗_c u
pub fn mobius_scalar(u: &Array2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = l2_norm(u).mapv(|x| x.max(Constants::EPS));
    let sqrt_c = c.sqrt();

    // factor = tanh(r * atanh(√c * norm)) / (√c * norm)
    let factor = norm.mapv(|n| {
        let scn = (sqrt_c * n).clamp(Constants::EPS, 1.0 - Constants::BOUNDARY_EPS);
        let alpha = atanh(scn);
        safe_tanh(r * alpha) / scn
    });

    // 각 행(벡터)에 스칼라 factor 적용
    let mut result = u.clone();
    for (mut row, f) in result.outer_iter_mut().zip(factor.iter()) {
        row.mapv_inplace(|u| u * f);
    }

    result
}
