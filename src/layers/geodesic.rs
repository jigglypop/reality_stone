
//! 지오데직 레이어 구현

use ndarray::{Array2};
use crate::ops::{mobius_add, mobius_scalar};

/// 지오데직 레이어 구조체
pub struct GeodesicLayer {
    /// 곡률 파라미터
    c: f32,
    /// 보간 파라미터 (0 <= t <= 1)
    t: f32,
    /// 입력 차원
    input_dim: usize,
    /// 출력 차원
    output_dim: usize,
    /// 가중치 매트릭스
    weights: Array2<f32>,
    /// 편향 벡터
    bias: Option<Array2<f32>>,
}

impl GeodesicLayer {
    /// 새 지오데직 레이어 생성
    pub fn new(input_dim: usize, output_dim: usize, c: f32, t: f32, use_bias: bool) -> Self {
        // Xavier 초기화 사용
        let scale = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let weights = Array2::zeros((input_dim, output_dim))
            .mapv(|_| rand::random::<f32>() * 2.0 * scale - scale);
        
        let bias = if use_bias {
            Some(Array2::zeros((1, output_dim))
                .mapv(|_| rand::random::<f32>() * 0.1))
        } else {
            None
        };
        
        GeodesicLayer {
            c,
            t,
            input_dim,
            output_dim,
            weights,
            bias,
        }
    }
    
    /// 순전파 (forward pass)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let batch_size = x.shape()[0];
        
        // 선형 변환 (u 계산)
        let u = x.dot(&self.weights);
        
        // 편향 추가 (있는 경우)
        let u_with_bias = match &self.bias {
            Some(b) => {
                let mut result = u.clone();
                for i in 0..batch_size {
                    for j in 0..self.output_dim {
                        result[[i, j]] += b[[0, j]];
                    }
                }
                result
            },
            None => u,
        };
        
        // v 생성 (여기서는 모델 자체가 파라미터가 있는 변환으로 구현)
        // 실제 애플리케이션에 맞게 조정 필요
        let v = x.clone();
        
        // 지오데직 보간 계산
        self.geodesic_interpolation(&u_with_bias, &v)
    }
    
    /// 지오데직 보간법 구현
    fn geodesic_interpolation(&self, u: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        // -u 계산
        let minus_u = mobius_scalar(u, self.c, -1.0);
        
        // -u ⊕ v 계산
        let delta = mobius_add(&minus_u, v, self.c);
        
        // t ⊗ (-u ⊕ v) 계산
        let delta_t = mobius_scalar(&delta, self.c, self.t);
        
        // u ⊕ (t ⊗ (-u ⊕ v)) 계산
        mobius_add(u, &delta_t, self.c)
    }
    
    /// 가중치 업데이트
    pub fn update_weights(&mut self, grad_weights: &Array2<f32>, learning_rate: f32) {
        // 단순 SGD 업데이트
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                self.weights[[i, j]] -= learning_rate * grad_weights[[i, j]];
            }
        }
    }
    
    /// 편향 업데이트
    pub fn update_bias(&mut self, grad_bias: &Array2<f32>, learning_rate: f32) {
        if let Some(bias) = &mut self.bias {
            for j in 0..self.output_dim {
                bias[[0, j]] -= learning_rate * grad_bias[[0, j]];
            }
        }
    }
}