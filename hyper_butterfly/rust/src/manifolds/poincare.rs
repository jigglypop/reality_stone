//! 포인카레 볼 모델 구현

use super::Manifold;
use crate::config::constants::Constants;
use crate::utils::numeric::{atanh, batch_dot, l2_norm, safe_tanh};
use ndarray::{Array1, Array2, Axis};

/// 포인카레 볼 모델 구현체
pub struct PoincareBall;

impl PoincareBall {
    /// 새로운 포인카레 볼 인스턴스 생성
    pub fn new() -> Self {
        PoincareBall
    }
}

impl Manifold for PoincareBall {
    fn exp_map(&self, v: &Array2<f32>, c: f32) -> Array2<f32> {
        let norm = l2_norm(v).mapv(|x| x.max(Constants::EPS));
        let sqrt_c = c.sqrt();

        // tanh(√c‖v‖)/(√c‖v‖) * v
        let factor = norm.mapv(|n| {
            let scn = (sqrt_c * n).clamp(Constants::EPS, 10.0);
            safe_tanh(scn) / scn
        });

        let mut result = v.clone();
        for (mut row, f) in result.outer_iter_mut().zip(factor.iter()) {
            row.mapv_inplace(|v| v * f);
        }

        result
    }

    fn log_map(&self, x: &Array2<f32>, c: f32) -> Array2<f32> {
        let norm = l2_norm(x).mapv(|x| x.max(Constants::EPS));
        let sqrt_c = c.sqrt();

        // atanh(√c‖x‖)/(√c‖x‖) * x
        let factor = norm.mapv(|n| {
            let scn = (sqrt_c * n).clamp(Constants::EPS, 1.0 - Constants::BOUNDARY_EPS);
            atanh(scn) / scn
        });

        let mut result = x.clone();
        for (mut row, f) in result.outer_iter_mut().zip(factor.iter()) {
            row.mapv_inplace(|x| x * f);
        }

        result
    }

    fn add(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array2<f32> {
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
        let denom =
            (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|x| x.max(Constants::MIN_DENOMINATOR));

        // 최종 결과: (num_u + num_v) / denom
        (num_u + num_v) / denom
    }

    fn scalar(&self, u: &Array2<f32>, c: f32, r: f32) -> Array2<f32> {
        let norm = l2_norm(u).mapv(|x| x.max(Constants::EPS));
        let sqrt_c = c.sqrt();

        // tanh(r * atanh(√c * norm)) / (√c * norm) * u
        let factor = norm.mapv(|n| {
            let scn = (sqrt_c * n).clamp(Constants::EPS, 1.0 - Constants::BOUNDARY_EPS);
            let alpha = atanh(scn);
            safe_tanh(r * alpha) / scn
        });

        let mut result = u.clone();
        for (mut row, f) in result.outer_iter_mut().zip(factor.iter()) {
            row.mapv_inplace(|u| u * f);
        }

        result
    }

    fn geodesic(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> Array2<f32> {
        // γ(u,v,t) = u ⊕ (t ⊗ (⊖u ⊕ v))
        let neg_u = self.scalar(u, c, -1.0);
        let delta = self.add(&neg_u, v, c);
        let delta_t = self.scalar(&delta, c, t);
        self.add(u, &delta_t, c)
    }

    fn dist(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array1<f32> {
        // d(u,v) = 2/√c * atanh(√c * |−u ⊕ v|)
        let neg_u = self.scalar(u, c, -1.0);
        let diff = self.add(&neg_u, v, c);
        let norm = l2_norm(&diff);
        let sqrt_c = c.sqrt();

        norm.mapv(|n| {
            let arg = (sqrt_c * n).clamp(0.0, 1.0 - Constants::BOUNDARY_EPS);
            (2.0 / sqrt_c) * atanh(arg)
        })
    }
}
