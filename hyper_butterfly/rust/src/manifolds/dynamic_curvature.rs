//! 동적 곡률 학습이 가능한 다양체 구현

use crate::manifolds::{create_manifold, Manifold, ManifoldType};
use ndarray::{Array1, Array2};
use std::cell::RefCell;

/// 학습 가능한 곡률 파라미터를 가진 다양체
pub struct LearningCurvatureManifold {
    /// 내부 다양체 인스턴스
    manifold: Box<dyn Manifold>,
    /// 현재 곡률 값 (RefCell로 내부 가변성 제공)
    curvature: RefCell<f32>,
}

impl LearningCurvatureManifold {
    /// 새로운 동적 곡률 다양체 생성
    pub fn new(manifold_type: ManifoldType, initial_c: f32) -> Self {
        let manifold = create_manifold(manifold_type);
        LearningCurvatureManifold {
            manifold,
            curvature: RefCell::new(initial_c),
        }
    }

    /// 현재 곡률 값 반환
    pub fn get_curvature(&self) -> f32 {
        *self.curvature.borrow()
    }

    /// 곡률 값 업데이트
    pub fn update_curvature(&self, new_c: f32) {
        *self.curvature.borrow_mut() = new_c;
    }
}

impl Manifold for LearningCurvatureManifold {
    fn add(&self, u: &Array2<f32>, v: &Array2<f32>, _c: f32) -> Array2<f32> {
        let c = self.get_curvature();
        self.manifold.add(u, v, c)
    }

    fn scalar(&self, u: &Array2<f32>, _c: f32, r: f32) -> Array2<f32> {
        let c = self.get_curvature();
        self.manifold.scalar(u, c, r)
    }

    fn geodesic(&self, u: &Array2<f32>, v: &Array2<f32>, _c: f32, t: f32) -> Array2<f32> {
        let c = self.get_curvature();
        self.manifold.geodesic(u, v, c, t)
    }

    fn dist(&self, u: &Array2<f32>, v: &Array2<f32>, _c: f32) -> Array1<f32> {
        let c = self.get_curvature();
        self.manifold.dist(u, v, c)
    }

    fn exp_map(&self, v: &Array2<f32>, _c: f32) -> Array2<f32> {
        let c = self.get_curvature();
        self.manifold.exp_map(v, c)
    }

    fn log_map(&self, x: &Array2<f32>, _c: f32) -> Array2<f32> {
        let c = self.get_curvature();
        self.manifold.log_map(x, c)
    }
}
