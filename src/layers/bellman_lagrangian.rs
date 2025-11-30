// ============================================================================
// 파일: src/layers/bellman_lagrangian.rs
// 목적: 벨만 가치 함수 + 라그랑지안 에너지 시스템
// ============================================================================

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use super::metric::{MetricTensor, MetricType, DiagonalMetric};

/// 벨만 가치 함수 근사
pub struct ValueFunction {
    /// MLP 파라미터 (단순화: 선형 근사)
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

impl ValueFunction {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let weights = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            rng.gen::<f32>() * 0.1 - 0.05
        });
        let bias = Array1::from_shape_fn(hidden_dim, |_| rng.gen::<f32>() * 0.1 - 0.05);
        
        Self { weights, bias }
    }
    
    /// V(x) 계산
    pub fn compute(&self, x: &ArrayView2<f32>) -> Array1<f32> {
        let hidden = x.dot(&self.weights);
        let mut output = Array1::zeros(x.nrows());
        
        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (j, &w) in hidden.row(i).iter().enumerate() {
                sum += (w + self.bias[j]).tanh();  // activation
            }
            *out = sum;
        }
        
        output
    }
    
    /// ∇V(x) 계산
    pub fn gradient(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let batch_size = x.nrows();
        let dim = x.ncols();
        let hidden = x.dot(&self.weights);
        
        let mut grad = Array2::zeros(x.raw_dim());
        
        for i in 0..batch_size {
            for j in 0..dim {
                let mut g = 0.0;
                for k in 0..self.bias.len() {
                    let h = hidden[[i, k]] + self.bias[k];
                    let tanh_h = h.tanh();
                    let sech_sq = 1.0 - tanh_h * tanh_h;
                    g += self.weights[[j, k]] * sech_sq;
                }
                grad[[i, j]] = g;
            }
        }
        
        grad
    }
}

/// 라그랑지안 파라미터
#[derive(Clone)]
pub struct LagrangianParams {
    pub kinetic_weight: f32,      // T의 가중치
    pub potential_weight: f32,    // V의 가중치
    pub gamma: f32,               // 할인율
    pub regularization: RegularizationConfig,
}

#[derive(Clone)]
pub struct RegularizationConfig {
    pub attractor_weight: f32,    // 기억 attractor β
    pub curvature_weight: f32,    // 곡률 복잡도 γ
}

impl Default for LagrangianParams {
    fn default() -> Self {
        Self {
            kinetic_weight: 0.5,
            potential_weight: 1.0,
            gamma: 0.99,
            regularization: RegularizationConfig {
                attractor_weight: 0.01,
                curvature_weight: 0.001,
            },
        }
    }
}

/// 벨만 잠재 에너지 계산
/// V_Bell = (V(x) - (R + γV(x')))²
pub fn bellman_potential(
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    gamma: f32,
) -> Array1<f32> {
    let v_x = value_fn.compute(x);
    let v_x_next = value_fn.compute(x_next);
    
    // δ(x, x') = V(x) - (R + γ V(x'))
    let bellman_error = &v_x - &(reward + &(&v_x_next * gamma));
    
    // V_Bell = δ²
    bellman_error.mapv(|e| e * e)
}

/// 운동 에너지 계산: T = (1/2) g_ij v^i v^j
pub fn kinetic_energy(
    metric: &dyn MetricTensor,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
) -> Array1<f32> {
    let g = metric.compute_metric(x);
    
    // 대각 근사: T = (1/2) Σ g_ii v_i²
    let v_sq = v.mapv(|x| x * x);
    let weighted = &g * &v_sq;
    weighted.sum_axis(Axis(1)) * 0.5
}

/// 라그랑지안 계산: L = T - V
pub fn lagrangian(
    metric: &dyn MetricTensor,
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    params: &LagrangianParams,
) -> Array1<f32> {
    let kinetic = kinetic_energy(metric, x, v);
    let potential = bellman_potential(value_fn, x, x_next, reward, params.gamma);
    
    &kinetic * params.kinetic_weight - &potential * params.potential_weight
}

/// 표현 흐름 업데이트: x' = Exp_x(-η ∇_g V)
pub fn representation_flow(
    metric: &MetricType,
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    learning_rate: f32,
) -> Array2<f32> {
    // ∇V(x)
    let grad_v = value_fn.gradient(x);
    
    // ∇_g V = g^{-1} ∇V (리만 그래디언트)
    let g_inv = metric.as_trait().compute_inverse_metric(x);
    let riemannian_grad = &grad_v * &g_inv;
    
    // 자연 경사 방향으로 이동
    let direction = &riemannian_grad * (-learning_rate);
    
    // Exp_x(direction)
    crate::layers::geodesic::exponential_map(metric, x, &direction.view(), 1.0)
}

/// 메트릭 흐름 업데이트: g' = g + η ∂L/∂g
/// (대각 메트릭에 대해서만 구현)
pub fn metric_flow(
    metric: &mut DiagonalMetric,
    _: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    lagrangian_value: &ArrayView1<f32>,
    learning_rate: f32,
) {
    // ∂L/∂g_ii ≈ (1/2) v_i²  (T 항에서)
    let v_sq = v.mapv(|x| x * x);
    let grad_g = v_sq.mean_axis(Axis(0)).unwrap() * 0.5;
    
    // 가중치 업데이트
    let mean_lagrangian = lagrangian_value.mean().unwrap();
    for (i, &g) in grad_g.iter().enumerate() {
        metric.weights[i] += learning_rate * g * mean_lagrangian;
    }
}

/// 벨만 업데이트: V(x) ← V(x) + α [R + γV(x') - V(x)]
pub fn bellman_update(
    value_fn: &mut ValueFunction,
    x: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    gamma: f32,
    learning_rate: f32,
) {
    let v_x = value_fn.compute(x);
    let v_x_next = value_fn.compute(x_next);
    
    let td_error = reward + &(&v_x_next * gamma) - &v_x;
    
    // SGD 업데이트 (단순화)
    let batch_size = x.nrows() as f32;
    let td_expanded = Array2::from_shape_fn((x.nrows(), 1), |(i, _)| td_error[i]);
    let grad_w = x.t().dot(&td_expanded) * (learning_rate / batch_size);
    
    for i in 0..value_fn.weights.nrows() {
        for j in 0..value_fn.weights.ncols() {
            if j < grad_w.ncols() {
                value_fn.weights[[i, j]] += grad_w[[i, j]];
            }
        }
    }
}

/// 에너지 구성 요소
pub struct EnergyComponents {
    pub kinetic: Array1<f32>,
    pub potential: Array1<f32>,
    pub lagrangian: Array1<f32>,
    pub bellman_residual: Array1<f32>,
}

impl EnergyComponents {
    pub fn new(batch_size: usize) -> Self {
        Self {
            kinetic: Array1::zeros(batch_size),
            potential: Array1::zeros(batch_size),
            lagrangian: Array1::zeros(batch_size),
            bellman_residual: Array1::zeros(batch_size),
        }
    }
}

/// 전체 에너지 계산
pub fn compute_energy_components(
    metric: &dyn MetricTensor,
    value_fn: &ValueFunction,
    x: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    x_next: &ArrayView2<f32>,
    reward: &ArrayView1<f32>,
    params: &LagrangianParams,
) -> EnergyComponents {
    let kinetic = kinetic_energy(metric, x, v);
    let potential = bellman_potential(value_fn, x, x_next, reward, params.gamma);
    
    let v_x = value_fn.compute(x);
    let v_x_next = value_fn.compute(x_next);
    let bellman_residual = &v_x - &(reward + &(&v_x_next * params.gamma));
    
    let lagrangian = &kinetic * params.kinetic_weight - &potential * params.potential_weight;
    
    EnergyComponents {
        kinetic,
        potential,
        lagrangian,
        bellman_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_value_function() {
        let vf = ValueFunction::new(4, 8);
        let x = arr2(&[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
        
        let v = vf.compute(&x.view());
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|&x| x.is_finite()));
        
        let grad = vf.gradient(&x.view());
        assert_eq!(grad.shape(), x.shape());
        assert!(grad.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kinetic_energy() {
        use super::super::metric::{DiagonalMetric, MetricTensor};
        let metric = DiagonalMetric::new(3);
        let x = arr2(&[[0.1, 0.2, 0.3]]);
        let v = arr2(&[[1.0, 0.5, 0.2]]);
        
        let ke = kinetic_energy(&metric, &x.view(), &v.view());
        assert_eq!(ke.len(), 1);
        assert!(ke[0] > 0.0);
    }
}

