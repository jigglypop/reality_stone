//! 하이퍼볼릭 다양체(manifold) 구현 모듈

pub mod dynamic_curvature;
mod klein;
mod lorentz;
mod poincare;

pub use klein::KleinModel;
pub use lorentz::LorentzModel;
pub use poincare::PoincareBall;

use ndarray::{Array1, Array2};

/// 지원하는 다양체 유형
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ManifoldType {
    /// 포인카레 볼 모델
    Poincare,
    /// 로렌츠(쌍곡면) 모델
    Lorentz,
    /// 클라인 모델
    Klein,
}

/// 하이퍼볼릭 다양체의 핵심 트레이트
pub trait Manifold {
    /// 뫼비우스 덧셈: u ⊕_c v
    fn add(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array2<f32>;

    /// 뫼비우스 스칼라 곱셈: r ⊗_c u
    fn scalar(&self, u: &Array2<f32>, c: f32, r: f32) -> Array2<f32>;

    /// 측지선(geodesic) 보간법
    fn geodesic(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> Array2<f32>;

    /// 두 점 사이의 거리 계산
    fn dist(&self, u: &Array2<f32>, v: &Array2<f32>, c: f32) -> Array1<f32>;

    /// 지수 맵(exp map): 접공간 → 다양체
    fn exp_map(&self, v: &Array2<f32>, c: f32) -> Array2<f32>;

    /// 로그 맵(log map): 다양체 → 접공간
    fn log_map(&self, x: &Array2<f32>, c: f32) -> Array2<f32>;
}

/// 다양체 인스턴스 생성 팩토리 함수
pub fn create_manifold(manifold_type: ManifoldType) -> Box<dyn Manifold> {
    match manifold_type {
        ManifoldType::Poincare => Box::new(PoincareBall::new()),
        ManifoldType::Lorentz => Box::new(LorentzModel::new()),
        ManifoldType::Klein => Box::new(KleinModel::new()),
    }
}
