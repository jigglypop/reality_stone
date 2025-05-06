//! 로렌츠 모델(쌍곡면) 구현
use super::Manifold;
use crate::config::constants::Constants;
use ndarray::{Array1, Array2, Axis};
use ndarray::s;
/// 로렌츠 모델 구현체
pub struct LorentzModel;

impl LorentzModel {
    /// 새로운 로렌츠 모델 인스턴스 생성
    pub fn new() -> Self {
        LorentzModel
    }
    /// 로렌츠 → 포인카레 변환 (로렌츠에서 마지막 좌표가 시간 차원)
    fn lorentz_to_poincare(&self, x: &Array2<f32>, _: f32) -> Array2<f32> {
        let dim = x.shape()[1] - 1; // 시간 차원 제외
        let x_space = x.slice(s![.., 0..dim]).to_owned();
        let x_time = x.slice(s![.., dim..dim + 1]).to_owned();
        // 포인카레 좌표 계산: x_space / (x_time + 1)
        x_space / (x_time + 1.0)
    }
    /// 포인카레 → 로렌츠 변환
    fn poincare_to_lorentz(&self, x: &Array2<f32>, _: f32) -> Array2<f32> {
        let batch_size = x.shape()[0];
        let dim = x.shape()[1];
        // 노름 계산
        let norm_sq = x
            .map_axis(Axis(1), |row| row.dot(&row))
            .insert_axis(Axis(1));

        // 시간 좌표 계산
        let t = (1.0 + &norm_sq) / (1.0 - &norm_sq);

        // 공간 좌표 계산
        let space_coords = 2.0 * x.clone() / (1.0 - &norm_sq);

        // 로렌츠 좌표 (공간 + 시간)
        let mut result = Array2::zeros((batch_size, dim + 1));
        result.slice_mut(s![.., 0..dim]).assign(&space_coords);
        result.slice_mut(s![.., dim]).assign(&t.slice(s![.., 0]));

        result
    }

    /// 내적: 로렌츠 계량을 사용한 내적 계산
    fn lorentz_inner(&self, u: &Array2<f32>, v: &Array2<f32>) -> Array1<f32> {
        let batch_size = u.shape()[0];
        let dim = u.shape()[1] - 1; // 시간 차원 제외

        // 공간 부분의 내적 (음수)
        let mut result = Array1::zeros(batch_size);
        for i in 0..batch_size {
            for j in 0..dim {
                result[i] -= u[[i, j]] * v[[i, j]];
            }
            // 시간 부분의 내적 (양수)
            result[i] += u[[i, dim]] * v[[i, dim]];
        }

        result
    }
}

impl Manifold for LorentzModel {
    fn exp_map(&self, v: &Array2<f32>, c: f32) -> Array2<f32> {
        let batch_size = v.shape()[0];
        let dim = v.shape()[1];

        // 접공간 -> 로렌츠 모델로 매핑
        let mut result = Array2::zeros((batch_size, dim + 1));

        // 각 벡터에 대해 지수 매핑 수행
        for i in 0..batch_size {
            let tangent_vec = v.slice(s![i, ..]).to_owned();
            let norm = tangent_vec.dot(&tangent_vec).sqrt();

            if norm < Constants::EPS {
                // 영벡터인 경우, 원점으로 매핑
                result[[i, dim]] = 1.0; // 시간 좌표 = 1
                continue;
            }

            // 단위 벡터 계산
            let unit_vec = &tangent_vec / norm;

            // 하이퍼볼릭 함수 적용
            let sinh_norm = (norm / c.sqrt()).sinh();
            let cosh_norm = (norm / c.sqrt()).cosh();

            // 공간 좌표 계산
            for j in 0..dim {
                result[[i, j]] = sinh_norm * unit_vec[j];
            }

            // 시간 좌표 계산
            result[[i, dim]] = cosh_norm;
        }

        result
    }

    fn log_map(&self, x: &Array2<f32>, c: f32) -> Array2<f32> {
        // 포인카레 모델로 변환하여 log_map 수행
        let poincare_x = self.lorentz_to_poincare(x, c);
        let poincare_ball = super::PoincareBall::new();
        poincare_ball.log_map(&poincare_x, c)
    }

    fn add(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array2<f32> {
        // 로렌츠 모델에서는 병렬 이동이 복잡함
        // 포인카레 모델로 변환하여 덧셈 수행
        let poincare_u = self.lorentz_to_poincare(u, c);
        let poincare_v = self.lorentz_to_poincare(v, c);

        let poincare_ball = super::PoincareBall::new();
        let poincare_result = poincare_ball.add(&poincare_u, &poincare_v, c);

        // 결과를 로렌츠 모델로 변환
        self.poincare_to_lorentz(&poincare_result, c)
    }

    fn scalar(&self, u: &Array2<f32>, c: f32, r: f32) -> Array2<f32> {
        // 포인카레 모델로 변환하여 스칼라 곱 수행
        let poincare_u = self.lorentz_to_poincare(u, c);

        let poincare_ball = super::PoincareBall::new();
        let poincare_result = poincare_ball.scalar(&poincare_u, c, r);

        // 결과를 로렌츠 모델로 변환
        self.poincare_to_lorentz(&poincare_result, c)
    }

    fn geodesic(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> Array2<f32> {
        let batch_size = u.shape()[0];
        let dim = u.shape()[1];
        let mut result = Array2::zeros((batch_size, dim));

        for i in 0..batch_size {
            let u_i = u.slice(s![i, ..]).to_owned();
            let v_i = v.slice(s![i, ..]).to_owned();
            // 로렌츠 내적 계산
            let inner_uv =
                self.lorentz_inner(&u_i.insert_axis(Axis(0)), &v_i.insert_axis(Axis(0)))[0];
            let angle = (-inner_uv).acos() / c.sqrt();
            if angle.abs() < Constants::EPS {
                // u와 v가 거의 같은 경우
                result.slice_mut(s![i, ..]).assign(&u_i);
                continue;
            }

            // 측지선 보간
            let sin_angle = angle.sin();
            let u_coef = ((1.0 - t) * angle).sin() / sin_angle;
            let v_coef = (t * angle).sin() / sin_angle;

            for j in 0..dim {
                result[[i, j]] = u_coef * u_i[j] + v_coef * v_i[j];
            }
        }

        result
    }

    fn dist(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array1<f32> {
        // 로렌츠 내적 사용하여 거리 계산
        let inner_prod = self.lorentz_inner(u, v);
        // 로렌츠 거리 계산: d = arcosh(-<u,v>_L) / sqrt(c)
        let sqrt_c = c.sqrt();
        inner_prod.mapv(|x| {
            let x_clamped = x.max(-1e15); // 수치 안정성
            (x_clamped.acosh()) / sqrt_c
        })
    }
}
