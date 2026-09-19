//! Explicit field-engine types extracted from the old `ce_core` defaults.

use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryMode {
    Clamp,
    Periodic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldConfig {
    pub mu: f64,
    pub lam: f64,
    pub alpha2: f64,
    pub coupling_k: f64,
    pub dt: f64,
    pub damping: f64,
    pub boundary: BoundaryMode,
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            mu: 1.0,
            lam: 1.0,
            alpha2: 0.0,
            coupling_k: 50.0,
            dt: 0.01,
            damping: 0.1,
            boundary: BoundaryMode::Clamp,
        }
    }
}

impl FieldConfig {
    pub fn vacuum_vev(&self) -> f64 {
        self.mu / self.lam.sqrt()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldState {
    pub phi: Vec<f64>,
    pub dphi: Vec<f64>,
    pub source_j: Vec<f64>,
}

impl FieldState {
    pub fn new_uniform(size: usize, vacuum_vev: f64) -> Self {
        Self {
            phi: vec![vacuum_vev; size],
            dphi: vec![0.0; size],
            source_j: vec![0.0; size],
        }
    }

    pub fn with_localized_source(
        size: usize,
        vacuum_vev: f64,
        center: usize,
        radius: usize,
        amplitude: f64,
    ) -> Self {
        let mut state = Self::new_uniform(size, vacuum_vev);
        if size == 0 {
            return state;
        }
        let center = center.min(size - 1);
        let start = center.saturating_sub(radius);
        let end = (center + radius + 1).min(size);
        state.source_j[start..end].fill(amplitude);
        state
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.phi.len();
        if self.dphi.len() != n || self.source_j.len() != n {
            return Err("field state vectors must share the same length".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldStepOutput {
    pub center_value: f64,
    pub mean_abs_force: f64,
    pub max_abs_force: f64,
}

pub struct FieldEngine {
    pub phi: Array1<f64>,
    pub dphi: Array1<f64>,
    pub source_j: Array1<f64>,
    pub forces_buffer: Array1<f64>,
    pub config: FieldConfig,
}

impl FieldEngine {
    pub fn new(config: FieldConfig, state: FieldState) -> Result<Self, String> {
        state.validate()?;
        let size = state.phi.len();
        Ok(Self {
            phi: Array1::from_vec(state.phi),
            dphi: Array1::from_vec(state.dphi),
            source_j: Array1::from_vec(state.source_j),
            forces_buffer: Array1::zeros(size),
            config,
        })
    }

    pub fn with_size(size: usize, config: FieldConfig) -> Self {
        let state = FieldState::new_uniform(size, config.vacuum_vev());
        Self::new(config, state).expect("uniform state should be valid")
    }

    pub fn state(&self) -> FieldState {
        FieldState {
            phi: self.phi.to_vec(),
            dphi: self.dphi.to_vec(),
            source_j: self.source_j.to_vec(),
        }
    }

    #[inline(always)]
    fn potential_force(phi_val: f64, mu: f64, lam: f64) -> f64 {
        phi_val * (mu.powi(2) - lam * phi_val.powi(2))
    }

    #[inline(always)]
    fn sample(phi_slice: &[f64], idx: isize, boundary: BoundaryMode) -> f64 {
        let n = phi_slice.len() as isize;
        match boundary {
            BoundaryMode::Clamp => {
                let clamped = idx.clamp(0, n.saturating_sub(1)) as usize;
                phi_slice[clamped]
            }
            BoundaryMode::Periodic => {
                let wrapped = idx.rem_euclid(n) as usize;
                phi_slice[wrapped]
            }
        }
    }

    pub fn step(&mut self) -> FieldStepOutput {
        let n = self.phi.len();
        if n == 0 {
            return FieldStepOutput {
                center_value: 0.0,
                mean_abs_force: 0.0,
                max_abs_force: 0.0,
            };
        }
        let phi_slice = self.phi.as_slice().expect("contiguous phi");
        let dphi_slice = self.dphi.as_slice().expect("contiguous dphi");
        let source_slice = self.source_j.as_slice().expect("contiguous source");
        let forces_slice = self
            .forces_buffer
            .as_slice_mut()
            .expect("contiguous forces buffer");
        let cfg = self.config.clone();

        forces_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, force)| {
                let i = i as isize;
                let left = Self::sample(phi_slice, i - 1, cfg.boundary);
                let center = Self::sample(phi_slice, i, cfg.boundary);
                let right = Self::sample(phi_slice, i + 1, cfg.boundary);
                let laplacian = left + right - 2.0 * center;

                let biharmonic = if cfg.alpha2 != 0.0 {
                    let p_2l = Self::sample(phi_slice, i - 2, cfg.boundary);
                    let p_1l = left;
                    let p_1r = right;
                    let p_2r = Self::sample(phi_slice, i + 2, cfg.boundary);
                    p_2l - 4.0 * p_1l + 6.0 * center - 4.0 * p_1r + p_2r
                } else {
                    0.0
                };

                let idx = i as usize;
                let pot_f = Self::potential_force(center, cfg.mu, cfg.lam);
                let damping = -cfg.damping * dphi_slice[idx];

                *force = pot_f + cfg.coupling_k * laplacian - cfg.alpha2 * biharmonic
                    + source_slice[idx]
                    + damping;
            });

        let dphi_slice = self.dphi.as_slice_mut().expect("contiguous dphi");
        let phi_slice = self.phi.as_slice_mut().expect("contiguous phi");
        let forces_slice = self
            .forces_buffer
            .as_slice()
            .expect("contiguous forces buffer");

        dphi_slice
            .par_iter_mut()
            .zip(forces_slice.par_iter())
            .for_each(|(v, f)| {
                *v += f * cfg.dt;
            });

        phi_slice
            .par_iter_mut()
            .zip(dphi_slice.par_iter())
            .for_each(|(p, v)| {
                *p += v * cfg.dt;
            });

        let mean_abs_force = if n == 0 {
            0.0
        } else {
            forces_slice.iter().map(|f| f.abs()).sum::<f64>() / n as f64
        };
        let max_abs_force = forces_slice.iter().map(|f| f.abs()).fold(0.0_f64, f64::max);

        FieldStepOutput {
            center_value: self.get_center_value(),
            mean_abs_force,
            max_abs_force,
        }
    }

    pub fn get_center_value(&self) -> f64 {
        if self.phi.is_empty() {
            0.0
        } else {
            self.phi[self.phi.len() / 2]
        }
    }
}
