// ============================================================================
// 파일: src/layers/geodesic.rs
// 목적: 측지선 흐름 및 exponential/logarithmic map
// ============================================================================

use ndarray::{Array2, ArrayView2, Axis};
use super::metric::{MetricTensor, MetricType};

const EPS: f32 = 1e-7;
const MAX_GEODESIC_STEPS: usize = 100;

/// 지수 사상 (Exponential Map): Exp_x(v)
/// 점 x에서 tangent vector v 방향으로 이동
pub fn exponential_map(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    step_size: f32,
) -> Array2<f32> {
    match metric {
        MetricType::Poincare(m) => {
            crate::layers::poincare::poincare_exp_at(x, v, m.curvature, 1e-5)
        }
        MetricType::Lorentz(m) => {
            // Lorentz exp_0: v (tangent) → hyperboloid
            if is_at_origin(x) {
                crate::layers::lorentz::lorentz_exp0_space(v, m.curvature)
            } else {
                // General point: use geodesic flow
                exponential_map_generic(metric, x, v, step_size)
            }
        }
        MetricType::Klein(m) => {
            // Klein uses geodesic flow
            exponential_map_generic(metric, x, v, step_size)
        }
        MetricType::Diagonal(_) => {
            // Euclidean-like: Exp_x(v) ≈ x + v
            x + &(v * step_size)
        }
    }
}

/// 로그 사상 (Logarithmic Map): Log_x(y)
/// 두 점 x, y를 연결하는 tangent vector
pub fn logarithmic_map(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Array2<f32> {
    match metric {
        MetricType::Poincare(m) => {
            crate::layers::poincare::poincare_log_at(x, y, m.curvature, 1e-5)
        }
        MetricType::Lorentz(m) => {
            if is_at_origin(x) {
                crate::layers::lorentz::lorentz_log0_space(y, m.curvature)
            } else {
                logarithmic_map_generic(metric, x, y)
            }
        }
        MetricType::Klein(_) => {
            logarithmic_map_generic(metric, x, y)
        }
        MetricType::Diagonal(_) => {
            // Euclidean: Log_x(y) = y - x
            y - x
        }
    }
}

/// 일반적인 exponential map (측지선 방정식 수치 적분)
fn exponential_map_generic(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    step_size: f32,
) -> Array2<f32> {
    let metric_trait = metric.as_trait();
    let batch_size = x.nrows();
    let dim = x.ncols();
    
    let mut position = x.to_owned();
    let mut velocity = v * step_size;
    let dt = 0.01;  // 작은 시간 스텝
    let num_steps = (step_size / dt).ceil() as usize;
    
    for _ in 0..num_steps.min(MAX_GEODESIC_STEPS) {
        // 측지선 방정식: d²x^k/dt² + Γ^k_ij dx^i/dt dx^j/dt = 0
        let christoffel = metric_trait.christoffel_symbols(&position.view());
        
        let mut acceleration = Array2::zeros((batch_size, dim));
        for b in 0..batch_size {
            for k in 0..dim {
                let mut acc = 0.0;
                // 대각 근사: Γ^k_ii만 고려
                for i in 0..dim {
                    acc -= christoffel[b][[k, i]] * velocity[[b, i]] * velocity[[b, i]];
                }
                acceleration[[b, k]] = acc;
            }
        }
        
        // Velocity Verlet integration
        velocity = &velocity + &(&acceleration * (dt * 0.5));
        position = &position + &(&velocity * dt);
        velocity = &velocity + &(&acceleration * (dt * 0.5));
    }
    
    position
}

/// 일반적인 logarithmic map (역문제)
fn logarithmic_map_generic(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Array2<f32> {
    // 초기 추정: v_0 = (y - x)
    let mut v = y - x;
    
    // Newton 방법으로 Exp_x(v) = y를 만족하는 v 찾기
    for _ in 0..10 {
        let exp_v = exponential_map(metric, x, &v.view(), 1.0);
        let residual = &exp_v - y;
        let residual_norm = crate::ops::norm_sq_batched(&residual.view()).mapv(|n| n.sqrt());
        
        if residual_norm.mean().unwrap() < EPS {
            break;
        }
        
        // v 업데이트: v -= learning_rate * residual
        v = &v - &(&residual * 0.5);
    }
    
    v
}

/// 측지선 보간: γ(t) = Exp_x(t * Log_x(y))
pub fn geodesic_interpolation(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    t: f32,
) -> Array2<f32> {
    match metric {
        MetricType::Poincare(m) => {
            crate::layers::poincare::poincare_ball_layer(x, y, m.curvature, t)
        }
        MetricType::Lorentz(m) => {
            crate::layers::lorentz::lorentz_layer_forward(x, y, m.curvature, t)
        }
        MetricType::Klein(m) => {
            crate::layers::klein::klein_layer_forward(x, y, m.curvature, t)
        }
        MetricType::Diagonal(_) => {
            // Linear interpolation
            x * (1.0 - t) + y * t
        }
    }
}

/// 측지선 경로 생성: x → y를 num_steps개 점으로 나눔
pub fn geodesic_path(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    num_steps: usize,
) -> Vec<Array2<f32>> {
    let mut path = Vec::with_capacity(num_steps);
    
    for i in 0..num_steps {
        let t = i as f32 / (num_steps - 1).max(1) as f32;
        let point = geodesic_interpolation(metric, x, y, t);
        path.push(point);
    }
    
    path
}

/// 평행 이동 (Parallel Transport)
/// tangent vector v를 x에서 y로 이동
pub fn parallel_transport(
    metric: &MetricType,
    v: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> Array2<f32> {
    let metric_trait = metric.as_trait();
    
    // 측지선을 따라 v를 이동
    let path = geodesic_path(metric, x, y, 10);
    let mut transported_v = v.to_owned();
    
    for i in 0..(path.len() - 1) {
        let christoffel = metric_trait.christoffel_symbols(&path[i].view());
        let dx = &path[i + 1] - &path[i];
        
        // dv^k/dt = -Γ^k_ij v^i dx^j/dt (대각 근사)
        let batch_size = transported_v.nrows();
        let dim = transported_v.ncols();
        
        for b in 0..batch_size {
            for k in 0..dim {
                let mut correction = 0.0;
                for i in 0..dim {
                    correction -= christoffel[b][[k, i]] * transported_v[[b, i]] * dx[[b, i]];
                }
                transported_v[[b, k]] += correction;
            }
        }
    }
    
    transported_v
}

/// 측지선 거리 계산 (메트릭 기반)
pub fn geodesic_distance(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
) -> ndarray::Array1<f32> {
    metric.as_trait().distance(x, y)
}

// 유틸리티
fn is_at_origin(x: &ArrayView2<f32>) -> bool {
    crate::ops::norm_sq_batched(x).iter().all(|&n| n < EPS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use super::super::metric::*;

    #[test]
    fn test_geodesic_interpolation_euclidean() {
        let metric = MetricType::Diagonal(DiagonalMetric::new(2));
        let x = arr2(&[[0.0, 0.0]]);
        let y = arr2(&[[1.0, 1.0]]);
        
        let mid = geodesic_interpolation(&metric, &x.view(), &y.view(), 0.5);
        assert!((mid[[0, 0]] - 0.5).abs() < 1e-5);
        assert!((mid[[0, 1]] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_geodesic_path() {
        let metric = MetricType::Diagonal(DiagonalMetric::new(2));
        let x = arr2(&[[0.0, 0.0]]);
        let y = arr2(&[[1.0, 0.0]]);
        
        let path = geodesic_path(&metric, &x.view(), &y.view(), 5);
        assert_eq!(path.len(), 5);
        assert!((path[0][[0, 0]] - 0.0).abs() < 1e-5);
        assert!((path[4][[0, 0]] - 1.0).abs() < 1e-5);
    }
}

