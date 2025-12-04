use ndarray::{Array1, Array2};
use super::hyper_metric::HyperMetric;

#[derive(Debug, Clone)]
pub struct SymplecticState {
    pub q: Array2<f32>,
    pub p: Array2<f32>,
}

impl SymplecticState {
    pub fn new(batch_size: usize, d: usize) -> Self {
        Self {
            q: Array2::zeros((batch_size, d)),
            p: Array2::zeros((batch_size, d)),
        }
    }
}

pub struct SymplecticLayer {
    pub hyper_metric: HyperMetric,
    pub layer_idx: usize,
    pub layer_emb: Array1<f32>,
    pub dt: f32,
}

impl SymplecticLayer {
    pub fn new(
        layer_idx: usize, 
        layer_emb: Array1<f32>, 
        hyper_metric: HyperMetric,
        dt: f32
    ) -> Self {
        Self {
            hyper_metric,
            layer_idx,
            layer_emb,
            dt,
        }
    }

    pub fn step(&self, state: &mut SymplecticState, x_input: &Array2<f32>) -> Array2<f32> {
        let force_metric = self.hyper_metric.project_forward(&state.q, &self.layer_emb);
        let force_total = &force_metric + x_input;
        let dt = self.dt;

        state.p = &state.p + &(&force_total * dt);
        state.q = &state.q + &(&state.p * dt);
        
        state.q.clone()
    }
}
