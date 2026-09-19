#[derive(Clone, Debug)]
pub struct NoiseConfig {
    pub alpha: f64,
    pub scale: f64,
    pub rho: f64,
    pub moment_order: usize,
    pub tls_omega: f64,
    pub tls_weight: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            alpha: 0.8,
            scale: 1.5,
            rho: 0.0,
            moment_order: 3,
            tls_omega: 0.0,
            tls_weight: 0.0,
        }
    }
}

impl NoiseConfig {
    pub fn from_env_with_noise(noise_amp: f64) -> Self {
        Self {
            alpha: env_f64("CE_NOISE_ALPHA", 0.8),
            scale: env_f64("CE_NOISE_SCALE", 1.5 * noise_amp.abs()),
            rho: env_f64("CE_NOISE_RHO", 0.0),
            moment_order: env_usize("CE_MOMENT_ORDER", 3).min(3),
            tls_omega: env_f64("CE_TLS_OMEGA", 0.0),
            tls_weight: env_f64("CE_TLS_WEIGHT", 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SuppressionConfig {
    pub omega: f64,
    pub amp: f64,
    pub omega2: f64,
    pub amp2: f64,
    pub anc_enabled: bool,
}

impl Default for SuppressionConfig {
    fn default() -> Self {
        Self {
            omega: 0.0,
            amp: 0.0,
            omega2: 0.0,
            amp2: 0.0,
            anc_enabled: false,
        }
    }
}

impl SuppressionConfig {
    pub fn from_env() -> Self {
        let omega = env_f64("CE_SUPPRESSON_OMEGA", 0.0);
        let amp = env_f64("CE_SUPPRESSON_AMP", 0.0);
        let omega2 = env_f64("CE_SUPPRESSON_OMEGA2", 0.0);
        let amp2 = env_f64("CE_SUPPRESSON_AMP2", 0.0);
        let anc_flag = env_i32("CE_SUPPRESSON_ANC", 0);
        let anc_enabled =
            anc_flag != 0 && ((omega != 0.0 && amp != 0.0) || (omega2 != 0.0 && amp2 != 0.0));

        Self {
            omega,
            amp,
            omega2,
            amp2,
            anc_enabled,
        }
    }

    pub fn has_any(&self) -> bool {
        (self.omega != 0.0 && self.amp != 0.0) || (self.omega2 != 0.0 && self.amp2 != 0.0)
    }

    pub fn apply_to_trace(&self, trace: &mut [f64]) {
        if self.omega != 0.0 && self.amp != 0.0 {
            for (t, v) in trace.iter_mut().enumerate() {
                *v += self.amp * (self.omega * t as f64).cos();
            }
        }
        if self.omega2 != 0.0 && self.amp2 != 0.0 {
            for (t, v) in trace.iter_mut().enumerate() {
                *v += self.amp2 * (self.omega2 * t as f64).cos();
            }
        }
    }

    pub fn cancel_from_sample(&self, val: f64, t_abs: usize) -> f64 {
        let mut result = val;
        if self.omega != 0.0 && self.amp != 0.0 {
            result -= self.amp * (self.omega * t_abs as f64).cos();
        }
        if self.omega2 != 0.0 && self.amp2 != 0.0 {
            result -= self.amp2 * (self.omega2 * t_abs as f64).cos();
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct QecConfig {
    pub t1_steps: f64,
    pub gate_error: f64,
    pub meas_error: f64,
}

impl Default for QecConfig {
    fn default() -> Self {
        Self {
            t1_steps: 1.0e5,
            gate_error: 1.0e-3,
            meas_error: 1.0e-3,
        }
    }
}

impl QecConfig {
    pub fn from_env() -> Self {
        Self {
            t1_steps: env_f64("CE_T1_STEPS", 1.0e5),
            gate_error: env_f64("CE_GATE_ERROR", 1.0e-3),
            meas_error: env_f64("CE_MEAS_ERROR", 1.0e-3),
        }
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_i32(key: &str, default: i32) -> i32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
