use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

pub struct FilterFunction {
    pub omega_grid: Vec<f64>,
    pub y_squared: Vec<f64>,
}

impl FilterFunction {
    pub fn compute(pulse_times: &[f64], duration: f64, n_omega: usize) -> Self {
        let steps = 8192;
        let mut y_time = vec![0.0; steps];

        let mut current_sign = 1.0;
        let mut pulse_idx = 0;

        for (t, y) in y_time.iter_mut().enumerate() {
            let t_norm = t as f64 / steps as f64;

            if pulse_idx < pulse_times.len() && t_norm >= pulse_times[pulse_idx] {
                current_sign *= -1.0;
                pulse_idx += 1;
            }

            *y = current_sign;
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(steps);

        let mut buffer: Vec<Complex<f64>> = y_time.iter().map(|&x| Complex::new(x, 0.0)).collect();

        fft.process(&mut buffer);

        let mut omega_grid = Vec::with_capacity(n_omega);
        let mut y_squared = Vec::with_capacity(n_omega);

        let dt = duration / steps as f64;
        let omega_max = PI / dt;

        for i in 0..n_omega {
            let omega = omega_max * (i as f64) / (n_omega as f64);
            omega_grid.push(omega);

            let idx = ((omega / (2.0 * PI)) * steps as f64) as usize;
            let idx = idx.min(steps / 2);

            let magnitude = buffer[idx].norm();
            let normalized = magnitude * dt;
            y_squared.push(normalized * normalized);
        }

        FilterFunction {
            omega_grid,
            y_squared,
        }
    }

    pub fn integrate_with_spectrum<F>(&self, spectrum: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut integral = 0.0;

        for i in 0..self.omega_grid.len().saturating_sub(1) {
            let omega = self.omega_grid[i];
            let d_omega = self.omega_grid[i + 1] - omega;

            if omega < 1e-10 {
                continue;
            }

            let s_omega = spectrum(omega);
            let y2 = self.y_squared[i];

            integral += s_omega * y2 * d_omega / (2.0 * PI);
        }

        integral
    }

    pub fn compute_moment(&self, order: usize) -> f64 {
        if self.omega_grid.is_empty() {
            return 0.0;
        }

        let mut moment = 0.0;

        for i in 0..self.omega_grid.len().saturating_sub(1) {
            let omega = self.omega_grid[i];
            let d_omega = self.omega_grid[i + 1] - omega;

            moment += omega.powi(order as i32) * self.y_squared[i] * d_omega;
        }

        moment
    }
}

pub fn compute_gain_function(
    cpmg_pulses: &[f64],
    ce_pulses: &[f64],
    duration: f64,
    n_omega: usize,
) -> Vec<(f64, f64)> {
    let ff_cpmg = FilterFunction::compute(cpmg_pulses, duration, n_omega);
    let ff_ce = FilterFunction::compute(ce_pulses, duration, n_omega);

    let mut gain = Vec::with_capacity(n_omega);

    for i in 0..n_omega
        .min(ff_cpmg.omega_grid.len())
        .min(ff_ce.omega_grid.len())
    {
        let omega = ff_cpmg.omega_grid[i];
        let g = ff_cpmg.y_squared[i] / (ff_ce.y_squared[i] + 1e-12);
        gain.push((omega, g));
    }

    gain
}

pub fn generate_cpmg_sequence(n_pulses: usize) -> Vec<f64> {
    (1..=n_pulses)
        .map(|j| (j as f64 - 0.5) / n_pulses as f64)
        .collect()
}

pub fn generate_udd_sequence(n_pulses: usize) -> Vec<f64> {
    (1..=n_pulses)
        .map(|j| {
            let arg = (j as f64 * PI) / (2.0 * n_pulses as f64 + 2.0);
            arg.sin().powi(2)
        })
        .collect()
}
