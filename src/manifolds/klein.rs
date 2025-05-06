//! 클라인 모델 구현
use super::Manifold;
use crate::config::constants::Constants;
use crate::utils::numeric::{batch_dot, l2_norm, safe_tanh};
use ndarray::{Array1, Array2, Axis};

/// 클라인 모델 구현체
pub struct KleinModel;

impl KleinModel {
    /// 새로운 클라인 모델 인스턴스 생성
    pub fn new() -> Self {
        KleinModel
    }

    /// 클라인 → 포인카레 변환
    fn klein_to_poincare(&self, x: &Array2<f32>, c: f32) -> Array2<f32> {
        let norm_sq = x
            .map_axis(Axis(1), |row| row.dot(&row))
            .insert_axis(Axis(1));
        let denominator = (1.0 + c * &norm_sq).mapv(|v| v.sqrt());
        x.clone() / denominator
    }

    /// 포인카레 → 클라인 변환
    fn poincare_to_klein(&self, x: &Array2<f32>, c: f32) -> Array2<f32> {
        let norm_sq = x
            .map_axis(Axis(1), |row| row.dot(&row))
            .insert_axis(Axis(1));
        let denominator = (1.0 + c * &norm_sq);
        2.0 * x.clone() / denominator
    }
}

impl Manifold for KleinModel {
    fn exp_map(&self, v: &Array2<f32>, c: f32) -> Array2<f32> {
        // 포인카레 모델로 변환하여 exp_map 수행
        let poincare_ball = super::PoincareBall::new();
        let result = poincare_ball.exp_map(v, c);

        // 결과를 클라인 모델로 변환
        self.poincare_to_klein(&result, c)
    }

    fn log_map(&self, x: &Array2<f32>, c: f32) -> Array2<f32> {
        // 클라인 모델을 포인카레 모델로 변환
        let poincare_x = self.klein_to_poincare(x, c);

        // 포인카레 모델에서 log_map 수행
        let poincare_ball = super::PoincareBall::new();
        poincare_ball.log_map(&poincare_x, c)
    }

    fn add(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array2<f32> {
        // 클라인 모델을 포인카레 모델로 변환
        let poincare_u = self.klein_to_poincare(u, c);
        let poincare_v = self.klein_to_poincare(v, c);

        // 포인카레 모델에서 덧셈 수행
        let poincare_ball = super::PoincareBall::new();
        let poincare_result = poincare_ball.add(&poincare_u, &poincare_v, c);

        // 결과를 클라인 모델로 변환
        self.poincare_to_klein(&poincare_result, c)
    }

    fn scalar(&self, u: &Array2<f32>, c: f32, r: f32) -> Array2<f32> {
        // 클라인 모델을 포인카레 모델로 변환
        let poincare_u = self.klein_to_poincare(u, c);

        // 포인카레 모델에서 스칼라 곱 수행
        let poincare_ball = super::PoincareBall::new();
        let poincare_result = poincare_ball.scalar(&poincare_u, c, r);

        // 결과를 클라인 모델로 변환
        self.poincare_to_klein(&poincare_result, c)
    }

    fn geodesic(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> Array2<f32> {
        // 클라인 모델에서는 직선이 측지선이므로 선형 보간 수행
        let u_scaled = u * (1.0 - t);
        let v_scaled = v * t;

        // 결과 정규화
        let result = &u_scaled + &v_scaled;
        let norm = l2_norm(&result);

        // 클라인 모델의 경계 내부로 제한
        let factor = norm.mapv(|n| {
            let max_allowed = 1.0 / c.sqrt() - Constants::BOUNDARY_EPS;
            if n > max_allowed {
                max_allowed / n
            } else {
                1.0
            }
        });

        let mut normalized = result.clone();
        for (mut row, f) in normalized.outer_iter_mut().zip(factor.iter()) {
            row.mapv_inplace(|x| x * f);
        }

        normalized
    }

    fn dist(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array1<f32> {
        // 클라인 모델에서의 거리 계산
        let sqrt_c = c.sqrt();
        let origin = Array2::zeros((u.shape()[0], u.shape()[1]));
        // 각 점에서 원점까지의 거리
        let d_u_o = self.dist_helper(u, &origin, c);
        let d_v_o = self.dist_helper(v, &origin, c);
        // u와 v 사이의 거리
        let d_u_v = self.dist_helper(u, v, c);
        d_u_v
    }

    // 헬퍼 메서드
    fn dist_helper(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array1<f32> {
        // 클라인 모델을 포인카레 모델로 변환
        let poincare_u = self.klein_to_poincare(u, c);
        let poincare_v = self.klein_to_poincare(v, c);
        // 포인카레 모델에서 거리 계산
        let poincare_ball = super::PoincareBall::new();
        poincare_ball.dist(&poincare_u, &poincare_v, c)
    }
}
