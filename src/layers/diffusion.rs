// ============================================================================
// 파일: src/layers/diffusion.rs
// 목적: 리만 라그랑지안 디퓨전 (Riemannian Lagrangian Diffusion) 구현
// ============================================================================

use ndarray::{Array2, ArrayView2};
use crate::layers::metric::DiagonalMetric;
use crate::layers::geodesic;

/// 리만 라그랑지안 디퓨전 상태 관리
pub struct RiemannianDiffusion {
    pub metric: DiagonalMetric,
    pub alpha: f32, // 에너지 감쇠 계수 (0.0 ~ 1.0)
    pub dt: f32,    // 시간 간격
}

impl RiemannianDiffusion {
    pub fn new(dim: usize, alpha: f32, dt: f32) -> Self {
        Self {
            metric: DiagonalMetric::new(dim),
            alpha,
            dt,
        }
    }

    /// 디퓨전 스텝: h(t+1) = Exp_h(t) ( -∇E * dt )
    /// 여기서 에너지는 잠재 에너지(Potential)와 운동 에너지(Kinetic)의 상호작용으로 정의됩니다.
    /// 단순화된 모델: 흐름(Flow)을 접공간(Tangent Space)에서의 벡터장으로 해석하고,
    /// 지수 맵(Exponential Map)을 통해 다양체 위로 업데이트합니다.
    pub fn step(
        &self,
        h: &ArrayView2<f32>,       // 현재 상태 (Batch, Hidden)
        flow_field: &ArrayView2<f32>, // 흐름 벡터장 (Batch, Hidden) - 예를 들어 tanh(h @ W)
    ) -> Array2<f32> {
        // 1. 접공간에서의 업데이트 방향 계산
        // dH = -alpha * H + (1-alpha) * Flow
        // 여기서는 사용자 코드의 수식: h_new = alpha * h + (1-alpha) * tanh(flow) 를
        // 리만 관점에서 해석:
        // Tangent Vector v = (1-alpha) * (Flow - h)  (유클리드 근사)
        // 혹은 더 정확하게는, Flow가 목표 지점이라면 Geodesic 방향.
        
        // 사용자의 수식을 그대로 따르되, 리만 지수 맵을 사용하여 이동
        // h_next = h + (1-alpha) * (Flow - h) * dt  (유클리드 Euler)
        // -> v = (Flow - h) * (1-alpha)
        // -> h_next = Exp_h(v * dt)
        
        // Flow field는 이미 활성화 함수가 적용된 상태라고 가정 (외부에서 계산)
        let delta = flow_field - h;
        let tangent_vector = &delta * (1.0 - self.alpha);
        
        // 2. 지수 맵을 사용하여 업데이트 (Manifold 제약 조건 유지)
        // Diagonal Metric을 고려한 지수 맵 사용
        // 여기서는 간단히 유클리드에 가까운 근사를 사용하거나, 
        // 실제 geodesic 모듈을 활용.
        
        // MetricTensor trait을 통해 exponential map 호출
        // geodesic::exponential_map expects &MetricType enum wrapper
        let metric_enum = crate::layers::metric::MetricType::Diagonal(self.metric.clone());
        geodesic::exponential_map(&metric_enum, h, &tangent_vector.view(), self.dt)
    }
    
    /// 가중치 기반 에너지 흐름 계산 (Rust 내부에서 처리할 경우)
    pub fn compute_flow(
        &self,
        h: &ArrayView2<f32>,
        weights: &ArrayView2<f32>, // (Hidden, Hidden)
    ) -> Array2<f32> {
        let linear = h.dot(weights);
        linear.mapv(|x| x.tanh())
    }
}

