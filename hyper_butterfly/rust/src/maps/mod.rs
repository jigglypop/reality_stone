//! 하이퍼볼릭 맵 함수들 (log_map, exp_map 등)

use crate::manifolds::{Manifold, PoincareBall};
use ndarray::Array2;

/// 로그 맵 함수 (하이퍼볼릭 → 유클리드)
pub fn log_map(x: &Array2<f32>, c: f32) -> Array2<f32> {
    let manifold = PoincareBall::new();
    manifold.log_map(x, c)
}

/// 지수 맵 함수 (유클리드 → 하이퍼볼릭)
pub fn exp_map(v: &Array2<f32>, c: f32) -> Array2<f32> {
    let manifold = PoincareBall::new();
    manifold.exp_map(v, c)
}

/// 측지선 함수
pub fn geodesic(u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> Array2<f32> {
    let manifold = PoincareBall::new();
    manifold.geodesic(u, v, c, t)
}
