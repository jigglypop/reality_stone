//! Brain-runtime cell-step kernel.
//!
//! Mirrors the hot path of `runtime.py:BrainRuntime.step()`.
//! All state vectors are flat f32 slices of length `dim`.

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct ModeParams {
    pub activation_decay: f32,
    pub activation_gain: f32,
    pub refractory_decay: f32,
    pub refractory_gain: f32,
    pub adaptation_decay: f32,
    pub adaptation_gain: f32,
    pub adaptation_coupling: f32,
    pub memory_decay: f32,
    pub memory_gain: f32,
    pub replay_mix: f32,
    pub noise_sigma: f32,
}

impl ModeParams {
    /// 15_Equations.md C.3 / J.12 brain-grounded values.
    pub fn wake() -> Self {
        Self {
            activation_decay: 0.18, // J.13: tau_m_eff ~5ms
            activation_gain: 0.82,
            refractory_decay: 0.12,    // J.2: tau_rel ~5-10ms
            refractory_gain: 0.20,     // J.12: clamped to [0.05,0.2]
            adaptation_decay: 0.005,   // J.20: tau_w=200ms, dt=1ms
            adaptation_gain: 0.005,    // matched to decay -> w* = E[a^2]
            adaptation_coupling: 0.12, // beta_w: ~24% suppression at w=2
            memory_decay: 0.01,        // J.3: tau_NMDA=100ms -> 1/100
            memory_gain: 0.01,
            replay_mix: 0.002, // J.16: SWR ~2Hz * dt
            noise_sigma: 0.27, // J.15: sigma_V/delta_V
        }
    }
    pub fn nrem() -> Self {
        Self {
            activation_decay: 0.34,
            activation_gain: 0.52,
            refractory_decay: 0.26,
            refractory_gain: 0.12,
            adaptation_decay: 0.005,
            adaptation_gain: 0.005,
            adaptation_coupling: 0.12,
            memory_decay: 0.01,
            memory_gain: 0.01,
            replay_mix: 0.10,  // C.3: NREM strong replay
            noise_sigma: 0.07, // J.15: DOWN state low noise
        }
    }
    pub fn rem() -> Self {
        Self {
            activation_decay: 0.22,
            activation_gain: 0.68,
            refractory_decay: 0.18,
            refractory_gain: 0.18,
            adaptation_decay: 0.005,
            adaptation_gain: 0.005,
            adaptation_coupling: 0.12,
            memory_decay: 0.01,
            memory_gain: 0.01,
            replay_mix: 0.20,  // C.3: REM strongest replay
            noise_sigma: 0.27, // J.15: WAKE-like
        }
    }
    pub fn from_mode(mode: super::runtime_types::Mode) -> Self {
        match mode {
            super::runtime_types::Mode::Wake => Self::wake(),
            super::runtime_types::Mode::Nrem => Self::nrem(),
            super::runtime_types::Mode::Rem => Self::rem(),
        }
    }
}

/// Per-neuron STP (Tsodyks-Markram) parameters (15_Equations.md J.19).
#[derive(Clone, Debug)]
pub struct StpParams {
    pub tau_rec: f32,
    pub tau_fac: f32,
    pub u_base: f32,
}

impl Default for StpParams {
    fn default() -> Self {
        Self {
            tau_rec: 0.008,  // 1/tau_rec ~ 130ms, dt=1ms
            tau_fac: 0.0015, // 1/tau_fac ~ 670ms
            u_base: 0.5,     // baseline release probability
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepConfig {
    pub refractory_scale: f32,
    pub goal_gain: f32,
    pub external_gain: f32,
    pub active_threshold: f32,
    pub bit_lower: f32,
    pub bit_upper: f32,
    pub energy_budget: usize,
    pub stp: StpParams,
    /// E/I ratio: fraction of excitatory neurons (B.1 Dale's Law, 0.8 = 80:20)
    pub ei_ratio: f32,
    /// Inhibitory gain multiplier (w_I/w_E ~= 4)
    pub inh_gain: f32,
    pub adaptation_clamp: f32,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            refractory_scale: 0.35,
            goal_gain: 0.20,
            external_gain: 0.45,
            active_threshold: 0.22,
            bit_lower: 0.10,
            bit_upper: 0.30,
            energy_budget: 16,
            stp: StpParams::default(),
            ei_ratio: 0.80,
            inh_gain: 4.0,
            adaptation_clamp: 2.0,
        }
    }
}

/// Apply Dale's Law sign mask to CSR weight values.
/// Neurons 0..n_exc are excitatory (+), n_exc..dim are inhibitory (-).
/// Inhibitory weights are scaled by `inh_gain` (w_I/w_E).
pub fn apply_dale_sign(
    values: &mut [f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    dim: usize,
    ei_ratio: f32,
    inh_gain: f32,
) {
    let n_exc = (dim as f32 * ei_ratio) as usize;
    for j_neuron in 0..dim {
        let start = row_ptr[j_neuron] as usize;
        let end = row_ptr[j_neuron + 1] as usize;
        for idx in start..end {
            let pre = col_idx[idx] as usize;
            let abs_w = values[idx].abs();
            if pre < n_exc {
                values[idx] = abs_w;
            } else {
                values[idx] = -abs_w * inh_gain;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepOutput {
    pub active_count: usize,
    pub energy: f32,
}

/// Sparse CSR matvec: y = A @ x, only over rows where `mask[i]` is true.
fn spmv_masked(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    x: &[f32],
    mask: &[bool],
    out: &mut [f32],
) {
    let dim = out.len();
    out.par_iter_mut().enumerate().for_each(|(i, yi)| {
        if !mask[i] || i >= dim {
            *yi = 0.0;
            return;
        }
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut acc = 0.0_f32;
        for idx in start..end {
            let j = col_idx[idx] as usize;
            acc += values[idx] * x[j];
        }
        *yi = acc;
    });
}

/// Full brain-runtime cell step (one tick).
///
/// State: `activation` (a), `refractory` (r), `memory_trace` (m),
///        `adaptation` (w, J.20 AHP), `bitfield` (b, UP/DOWN).
/// Returns `StepOutput` with active count and energy estimate.
///
/// Equations: 15_Equations.md A.1--A.10, B.1, J.12--J.20.
pub fn brain_step(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    activation: &mut [f32],
    refractory: &mut [f32],
    memory_trace: &mut [f32],
    adaptation: &mut [f32],
    stp_u: &mut [f32],
    stp_x: &mut [f32],
    bitfield: &mut [u8],
    active_mask: &[u8],
    external: &[f32],
    goal: &[f32],
    replay: &[f32],
    noise: &[f32],
    mode_params: &ModeParams,
    cfg: &StepConfig,
) -> StepOutput {
    let dim = activation.len();
    if dim == 0 {
        return StepOutput {
            active_count: 0,
            energy: 0.0,
        };
    }

    // 0. STP update (Tsodyks-Markram, J.19): per-neuron approximation
    let stp_p = &cfg.stp;
    let prev_active: Vec<bool> = active_mask.iter().map(|&b| b > 0).collect();
    for i in 0..dim {
        let spike = if prev_active[i] { 1.0_f32 } else { 0.0 };
        let old_u = stp_u[i];
        stp_u[i] += -stp_p.tau_fac * old_u + stp_p.u_base * (1.0 - old_u) * spike;
        stp_x[i] += stp_p.tau_rec * (1.0 - stp_x[i]) - old_u * stp_x[i] * spike;
        stp_u[i] = stp_u[i].clamp(0.0, 1.0);
        stp_x[i] = stp_x[i].clamp(0.0, 1.0);
    }

    // 1. W_eff = u * x * a (STP-modulated presynaptic output)
    let masked_act: Vec<f32> = (0..dim)
        .map(|i| {
            if prev_active[i] {
                stp_u[i] * stp_x[i] * activation[i]
            } else {
                0.0
            }
        })
        .collect();
    let mut recurrent = vec![0.0_f32; dim];
    let all_true = vec![true; dim];
    spmv_masked(
        values,
        col_idx,
        row_ptr,
        &masked_act,
        &all_true,
        &mut recurrent,
    );

    // 3. drive = recurrent + ext + goal + replay - refractory - adaptation (A.2 + A.6)
    let ext_g = cfg.external_gain;
    let goal_g = cfg.goal_gain;
    let ref_s = cfg.refractory_scale;
    let rep_m = mode_params.replay_mix;
    let adapt_c = mode_params.adaptation_coupling;
    let drive: Vec<f32> = (0..dim)
        .map(|i| {
            recurrent[i] + ext_g * external[i] + goal_g * goal[i] + rep_m * replay[i]
                - ref_s * refractory[i]
                - adapt_c * adaptation[i].min(cfg.adaptation_clamp)
                + noise[i]
        })
        .collect();

    // 4. activation update: a' = clamp[(1-gamma_a)*a + kappa_a*tanh(drive), -1, 1] (A.3)
    let decay = mode_params.activation_decay;
    let gain = mode_params.activation_gain;
    let new_act: Vec<f32> = (0..dim)
        .map(|i| {
            let raw = (1.0 - decay) * activation[i] + gain * drive[i].tanh();
            raw.clamp(-1.0, 1.0)
        })
        .collect();

    // 5. refractory update: r' = (1-gamma_r)*r + kappa_r*a'^2 (A.4)
    let r_decay = mode_params.refractory_decay;
    let r_gain = mode_params.refractory_gain;
    let new_ref: Vec<f32> = (0..dim)
        .map(|i| (1.0 - r_decay) * refractory[i] + r_gain * new_act[i] * new_act[i])
        .collect();

    // 6. memory trace: m' = (1-gamma_m)*m + gamma_m*a' (A.5, J.3 NMDA tau=100ms)
    let m_decay = mode_params.memory_decay;
    let m_gain = mode_params.memory_gain;
    let new_mem: Vec<f32> = (0..dim)
        .map(|i| (1.0 - m_decay) * memory_trace[i] + m_gain * new_act[i])
        .collect();

    // 7. adaptation update: w' = (1-gamma_w)*w + kappa_w*a'^2 (A.6, J.20 AHP)
    // gamma_w = kappa_w = 0.005 ensures w* = E[a^2] at steady state, clamped to [0,2]
    let w_decay = mode_params.adaptation_decay;
    let w_gain = mode_params.adaptation_gain;
    let new_adapt: Vec<f32> = (0..dim)
        .map(|i| {
            let raw = (1.0 - w_decay) * adaptation[i] + w_gain * new_act[i] * new_act[i];
            raw.clamp(0.0, cfg.adaptation_clamp)
        })
        .collect();

    // 8. bitfield hysteresis (A.7, J.17 UP/DOWN)
    let new_bit: Vec<u8> = (0..dim)
        .map(|i| {
            if new_act[i] >= cfg.bit_upper {
                1
            } else if new_act[i] <= cfg.bit_lower {
                0
            } else {
                bitfield[i]
            }
        })
        .collect();

    // 9. salience for TopK selection
    let salience: Vec<f32> = (0..dim)
        .map(|i| {
            new_act[i].abs()
                + 0.35 * external[i].abs()
                + 0.25 * replay[i].abs()
                + 0.20 * goal[i].abs()
                - 0.15 * new_ref[i]
        })
        .collect();

    // 10. active selection: topk by salience
    let budget = cfg.energy_budget.min(dim);
    let mut scored: Vec<(f32, usize)> = salience
        .iter()
        .enumerate()
        .filter(|(_, &s)| s >= cfg.active_threshold)
        .map(|(i, &s)| (s, i))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(budget);
    let active_count = scored.len();

    // 11. energy estimate (B.3)
    let coupling_energy = {
        let dot: f32 = new_act
            .iter()
            .zip(recurrent.iter())
            .map(|(a, r)| a * r)
            .sum();
        0.5 * dot.abs()
    };
    let dimf = dim as f32;
    let local_energy: f32 = new_ref.iter().map(|r| r.abs()).sum::<f32>() / dimf
        + 0.25 * new_mem.iter().map(|m| m.abs()).sum::<f32>() / dimf
        + 0.10 * new_adapt.iter().map(|w| w.abs()).sum::<f32>() / dimf;
    let replay_energy = 0.1 * replay.iter().map(|r| r.abs()).sum::<f32>() / dimf;
    let energy = coupling_energy + local_energy + replay_energy;

    // commit state
    activation.copy_from_slice(&new_act);
    refractory.copy_from_slice(&new_ref);
    memory_trace.copy_from_slice(&new_mem);
    adaptation.copy_from_slice(&new_adapt);
    bitfield.copy_from_slice(&new_bit);

    StepOutput {
        active_count,
        energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_csr(dim: usize) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
        let mut values = Vec::with_capacity(dim);
        let mut col_idx = Vec::with_capacity(dim);
        let mut row_ptr = Vec::with_capacity(dim + 1);
        row_ptr.push(0);
        for i in 0..dim {
            values.push(0.1);
            col_idx.push(i as i32);
            row_ptr.push((i + 1) as i32);
        }
        (values, col_idx, row_ptr)
    }

    #[test]
    fn basic_step_runs() {
        let dim = 16;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.5_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.1_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::wake();
        let cfg = StepConfig {
            energy_budget: 4,
            ..Default::default()
        };
        let out = brain_step(
            &vals, &cols, &rows, &mut act, &mut refr, &mut mem, &mut adapt, &mut su, &mut sx,
            &mut bit, &active, &ext, &goal, &replay, &noise, &mp, &cfg,
        );
        assert!(out.active_count <= 4);
        assert!(out.energy >= 0.0);
        assert!(act.iter().all(|x| x.is_finite()));
        assert!(adapt.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn energy_decreases_nrem() {
        let dim = 32;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.3_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.0_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::nrem();
        let cfg = StepConfig {
            energy_budget: 8,
            ..Default::default()
        };
        let mut energies = Vec::new();
        for _ in 0..20 {
            let out = brain_step(
                &vals, &cols, &rows, &mut act, &mut refr, &mut mem, &mut adapt, &mut su, &mut sx,
                &mut bit, &active, &ext, &goal, &replay, &noise, &mp, &cfg,
            );
            energies.push(out.energy);
        }
        assert!(energies.last().unwrap() < energies.first().unwrap());
    }

    #[test]
    fn adaptation_accumulates() {
        let dim = 8;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.8_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.5_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::wake();
        let cfg = StepConfig::default();
        for _ in 0..50 {
            brain_step(
                &vals, &cols, &rows, &mut act, &mut refr, &mut mem, &mut adapt, &mut su, &mut sx,
                &mut bit, &active, &ext, &goal, &replay, &noise, &mp, &cfg,
            );
        }
        let max_adapt = adapt.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_adapt > 0.0,
            "adaptation should accumulate with sustained input"
        );
    }

    #[test]
    fn stp_depletes_with_activity() {
        let dim = 4;
        let (vals, cols, rows) = make_identity_csr(dim);
        let mut act = vec![0.9_f32; dim];
        let mut refr = vec![0.0_f32; dim];
        let mut mem = vec![0.0_f32; dim];
        let mut adapt = vec![0.0_f32; dim];
        let mut su = vec![0.5_f32; dim];
        let mut sx = vec![1.0_f32; dim];
        let mut bit = vec![1_u8; dim];
        let active = bit.clone();
        let noise = vec![0.0_f32; dim];
        let ext = vec![0.5_f32; dim];
        let goal = vec![0.0_f32; dim];
        let replay = vec![0.0_f32; dim];
        let mp = ModeParams::wake();
        let cfg = StepConfig::default();
        let x0 = sx[0];
        for _ in 0..10 {
            brain_step(
                &vals, &cols, &rows, &mut act, &mut refr, &mut mem, &mut adapt, &mut su, &mut sx,
                &mut bit, &active, &ext, &goal, &replay, &noise, &mp, &cfg,
            );
        }
        assert!(
            sx[0] < x0,
            "STP resource x should deplete with sustained spiking"
        );
    }
}
