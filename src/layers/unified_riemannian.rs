use super::bellman_lagrangian::{
    bellman_update, compute_energy_components, metric_flow, representation_flow, EnergyComponents,
    LagrangianParams, ValueFunction,
};
use super::geodesic::{geodesic_interpolation, geodesic_path};
use super::metric::{DiagonalMetric, KleinMetric, LorentzMetric, MetricType, PoincareMetric};
use indicatif::ProgressBar;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// 통합 리만 레이어
pub struct UnifiedRiemannianLayer {
    pub metric: MetricType,
    pub value_function: Option<ValueFunction>,
    pub lagrangian_params: LagrangianParams,
    pub enable_bellman: bool,
    pub enable_metric_learning: bool,
}

impl UnifiedRiemannianLayer {
    /// 새로운 통합 리만 레이어 생성
    ///
    /// # Arguments
    /// * `metric_type` - "poincare", "lorentz", "klein", "diagonal"
    /// * `curvature` - 곡률 파라미터 (양수)
    /// * `input_dim` - 입력 차원
    /// * `enable_bellman` - 벨만 가치 함수 활성화
    pub fn new(metric_type: &str, curvature: f32, input_dim: usize, enable_bellman: bool) -> Self {
        let metric = match metric_type {
            "poincare" => MetricType::Poincare(PoincareMetric::new(curvature)),
            "lorentz" => MetricType::Lorentz(LorentzMetric::new(curvature)),
            "klein" => MetricType::Klein(KleinMetric::new(curvature)),
            "diagonal" => MetricType::Diagonal(DiagonalMetric::new(input_dim)),
            _ => panic!("Unknown metric type: {}", metric_type),
        };

        let value_function = if enable_bellman {
            Some(ValueFunction::new(input_dim, input_dim * 2))
        } else {
            None
        };

        let enable_metric_learning = matches!(metric, MetricType::Diagonal(_));

        Self {
            metric,
            value_function,
            lagrangian_params: LagrangianParams::default(),
            enable_bellman,
            enable_metric_learning,
        }
    }

    /// 순전파
    ///
    /// # Arguments
    /// * `x` - 입력 (batch, dim)
    /// * `target` - 목표점 (optional)
    ///
    /// # Returns
    /// LayerOutput - 출력 및 에너지 정보
    pub fn forward(&self, x: &ArrayView2<f32>, target: Option<&ArrayView2<f32>>) -> LayerOutput {
        let batch_size = x.nrows();

        // 1. 메트릭 계산
        let metric_values = self.metric.as_trait().compute_metric(x);

        // 2. 출력 계산
        let output = if let Some(y) = target {
            // 목표가 있으면 측지선 보간 (중간점)
            geodesic_interpolation(&self.metric, x, y, 0.5)
        } else if self.enable_bellman && self.value_function.is_some() {
            // 벨만 활성화: 표현 흐름
            let vf = self.value_function.as_ref().unwrap();
            representation_flow(&self.metric, vf, x, 0.01)
        } else {
            // 단순 항등 (메트릭만 적용)
            x.to_owned()
        };

        // 3. 에너지 계산 (벨만 활성화 시)
        let energy = if self.enable_bellman && self.value_function.is_some() {
            let vf = self.value_function.as_ref().unwrap();
            let dt = 0.1;
            let velocity = (&output - x) / dt;
            let reward = Array1::zeros(batch_size); // 기본 보상 0

            Some(compute_energy_components(
                self.metric.as_trait(),
                vf,
                x,
                &velocity.view(),
                &output.view(),
                &reward.view(),
                &self.lagrangian_params,
            ))
        } else {
            None
        };

        LayerOutput {
            output,
            energy,
            cache: LayerCache {
                input: x.to_owned(),
                velocity: None,
                metric_values,
            },
        }
    }

    /// 역전파
    ///
    /// # Arguments
    /// * `grad_output` - 출력에 대한 그래디언트
    /// * `x` - 입력
    /// * `cache` - 순전파 캐시
    ///
    /// # Returns
    /// LayerGradients - 입력 및 파라미터에 대한 그래디언트
    pub fn backward(
        &self,
        grad_output: &ArrayView2<f32>,
        _x: &ArrayView2<f32>,
        _cache: &LayerCache,
    ) -> LayerGradients {
        // 단순화: 그래디언트 pass-through
        LayerGradients {
            grad_input: grad_output.to_owned(),
            grad_metric: None,
            grad_value_fn: None,
        }
    }

    /// 메트릭 학습 업데이트
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `v` - 속도 (변화율)
    /// * `learning_rate` - 학습률
    pub fn update_metric(&mut self, x: &ArrayView2<f32>, v: &ArrayView2<f32>, learning_rate: f32) {
        if !self.enable_metric_learning {
            return;
        }

        if let MetricType::Diagonal(ref mut metric) = self.metric {
            if let Some(ref vf) = self.value_function {
                let batch_size = x.nrows();
                let dt = 0.1;
                let x_next = x + &(v * dt);
                let reward = Array1::zeros(batch_size);

                let energy = compute_energy_components(
                    metric,
                    vf,
                    x,
                    v,
                    &x_next.view(),
                    &reward.view(),
                    &self.lagrangian_params,
                );

                metric_flow(metric, x, v, &energy.lagrangian.view(), learning_rate);
            }
        }
    }

    /// 에너지 계산
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `v` - 속도
    /// * `x_next` - 다음 상태
    /// * `reward` - 보상
    ///
    /// # Returns
    /// EnergyComponents - 운동/잠재/라그랑지안 에너지
    pub fn compute_energy(
        &self,
        x: &ArrayView2<f32>,
        v: &ArrayView2<f32>,
        x_next: &ArrayView2<f32>,
        reward: &ArrayView1<f32>,
    ) -> EnergyComponents {
        if let Some(ref vf) = self.value_function {
            compute_energy_components(
                self.metric.as_trait(),
                vf,
                x,
                v,
                x_next,
                reward,
                &self.lagrangian_params,
            )
        } else {
            EnergyComponents::new(x.nrows())
        }
    }

    /// 측지선 경로 생성
    ///
    /// # Arguments
    /// * `start` - 시작점
    /// * `end` - 끝점
    /// * `num_steps` - 경로 점 개수
    ///
    /// # Returns
    /// 측지선 경로 (각 점은 batch x dim)
    pub fn geodesic_path(
        &self,
        start: &ArrayView2<f32>,
        end: &ArrayView2<f32>,
        num_steps: usize,
    ) -> Vec<Array2<f32>> {
        geodesic_path(&self.metric, start, end, num_steps)
    }

    /// 표현 흐름 스텝
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `num_steps` - 흐름 반복 횟수
    /// * `learning_rate` - 학습률
    ///
    /// # Returns
    /// 흐름 후 상태
    pub fn flow_step(
        &self,
        x: &ArrayView2<f32>,
        num_steps: usize,
        learning_rate: f32,
    ) -> Array2<f32> {
        if let Some(ref vf) = self.value_function {
            let mut current = x.to_owned();
            for _ in 0..num_steps {
                current = representation_flow(&self.metric, vf, &current.view(), learning_rate);
            }
            current
        } else {
            x.to_owned()
        }
    }

    /// 벨만 가치 함수 업데이트
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `x_next` - 다음 상태
    /// * `reward` - 보상
    /// * `learning_rate` - 학습률
    pub fn update_value_function(
        &mut self,
        x: &ArrayView2<f32>,
        x_next: &ArrayView2<f32>,
        reward: &ArrayView1<f32>,
        learning_rate: f32,
    ) {
        if let Some(ref mut vf) = self.value_function {
            bellman_update(
                vf,
                x,
                x_next,
                reward,
                self.lagrangian_params.gamma,
                learning_rate,
            );
        }
    }
}

pub fn laplace_beltrami_matrix(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    sigma: f32,
    eps: f32,
) -> Array2<f32> {
    use ndarray::s;
    let n = x.nrows();
    let dim = x.ncols();
    let metric_trait = metric.as_trait();
    let mut dist_sq = Array2::<f32>::zeros((n, n));
    let pb_dist = ProgressBar::new(n as u64);
    {
        let x_ref = x;
        dist_sq
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                pb_dist.inc(1);
                let xi = x_ref.slice(s![i..i + 1, 0..dim]);
                for j in (i + 1)..n {
                    let xj = x_ref.slice(s![j..j + 1, 0..dim]);
                    let d_arr = metric_trait.distance(&xi.view(), &xj.view());
                    let d = d_arr[0];
                    let v = d * d;
                    row[j] = v;
                }
            });
    }
    pb_dist.finish_and_clear();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = dist_sq[[i, j]];
            dist_sq[[j, i]] = v;
        }
    }
    let det = metric_trait.determinant(x);
    let mut vol = det.clone();
    for v in vol.iter_mut() {
        let a = v.abs().sqrt();
        if a < eps {
            *v = eps;
        } else {
            *v = a;
        }
    }
    let mut w = Array2::<f32>::zeros((n, n));
    let denom = 2.0 * sigma * sigma.max(eps);
    let pb_w = ProgressBar::new(n as u64);
    {
        let vol_ref = &vol;
        w.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                pb_w.inc(1);
                for j in (i + 1)..n {
                    let d2 = dist_sq[[i, j]];
                    let mut value = (-d2 / denom).exp();
                    let scale = 1.0 / (vol_ref[i] * vol_ref[j]);
                    value *= scale;
                    row[j] = value;
                }
            });
    }
    pb_w.finish_and_clear();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = w[[i, j]];
            w[[j, i]] = v;
        }
    }
    let mut l = Array2::<f32>::zeros((n, n));
    let pb_l = ProgressBar::new(n as u64);
    {
        l.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                pb_l.inc(1);
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += w[[i, j]];
                }
                row[i] = sum;
                for j in 0..n {
                    if i != j {
                        row[j] = -w[[i, j]];
                    }
                }
            });
    }
    pb_l.finish_and_clear();
    l
}

/// 레이어 출력
pub struct LayerOutput {
    pub output: Array2<f32>,
    pub energy: Option<EnergyComponents>,
    pub cache: LayerCache,
}

/// 레이어 캐시 (역전파용)
pub struct LayerCache {
    pub input: Array2<f32>,
    pub velocity: Option<Array2<f32>>,
    pub metric_values: Array2<f32>,
}

/// 레이어 그래디언트
pub struct LayerGradients {
    pub grad_input: Array2<f32>,
    pub grad_metric: Option<Array1<f32>>,
    pub grad_value_fn: Option<ValueFunctionGrad>,
}

/// 가치 함수 그래디언트
pub struct ValueFunctionGrad {
    pub grad_weights: Array2<f32>,
    pub grad_bias: Array1<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_unified_layer_creation() {
        let layer = UnifiedRiemannianLayer::new("poincare", 1.0, 32, false);
        assert!(!layer.enable_bellman);

        let layer_bellman = UnifiedRiemannianLayer::new("diagonal", 0.0, 64, true);
        assert!(layer_bellman.enable_bellman);
        assert!(layer_bellman.value_function.is_some());
    }

    #[test]
    fn test_forward_poincare() {
        let layer = UnifiedRiemannianLayer::new("poincare", 1.0, 4, false);
        let x = arr2(&[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]);

        let output = layer.forward(&x.view(), None);
        assert_eq!(output.output.shape(), x.shape());
        assert!(output.energy.is_none());
    }

    #[test]
    fn test_forward_with_target() {
        let layer = UnifiedRiemannianLayer::new("diagonal", 0.0, 3, false);
        let x = arr2(&[[0.0, 0.0, 0.0]]);
        let y = arr2(&[[1.0, 1.0, 1.0]]);

        let output = layer.forward(&x.view(), Some(&y.view()));
        // 중간점이어야 함
        assert!((output.output[[0, 0]] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_geodesic_path() {
        let layer = UnifiedRiemannianLayer::new("diagonal", 0.0, 2, false);
        let start = arr2(&[[0.0, 0.0]]);
        let end = arr2(&[[1.0, 0.0]]);

        let path = layer.geodesic_path(&start.view(), &end.view(), 5);
        assert_eq!(path.len(), 5);
        assert!((path[0][[0, 0]] - 0.0).abs() < 1e-4);
        assert!((path[4][[0, 0]] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_energy_computation() {
        let layer = UnifiedRiemannianLayer::new("diagonal", 0.0, 4, true);
        let x = arr2(&[[0.1, 0.2, 0.3, 0.4]]);
        let v = arr2(&[[0.01, 0.02, 0.03, 0.04]]);
        let x_next = &x + &v;
        let reward = ndarray::arr1(&[0.5]);

        let energy = layer.compute_energy(&x.view(), &v.view(), &x_next.view(), &reward.view());
        assert!(energy.kinetic[0] >= 0.0);
        assert!(energy.potential[0] >= 0.0);
        assert!(energy.lagrangian[0].is_finite());
    }
}
