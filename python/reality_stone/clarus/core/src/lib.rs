#![allow(non_local_definitions)]
//! Canonical Rust compute surface for the Clarus runtime.

pub mod engine;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use engine::ce_riemann::RelaxOutput;
pub use engine::field::{BoundaryMode, FieldConfig, FieldEngine, FieldState, FieldStepOutput};
pub use engine::kernel::{ModeParams, StepConfig, StepOutput, StpParams, apply_dale_sign, brain_step};
pub use engine::runtime_types::{CellState, Mode, RelaxInput, SnapshotMeta};

#[cfg(feature = "python")]
mod python_binding {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
    use crate::engine::nn_ops;
    use crate::engine::ce_riemann;
    use crate::engine::kernel;
    use crate::engine::llm_pre_eq;
    use crate::engine::runtime_types;

    // Every array crosses the boundary as a flat C-contiguous slice. `as_slice()?`
    // raises `ValueError` for non-contiguous input and the helpers below validate
    // shapes, so a bad call becomes a Python exception instead of a panic.

    fn expect_len(name: &str, actual: usize, expected: usize) -> PyResult<()> {
        if actual != expected {
            return Err(PyValueError::new_err(format!(
                "{name}: expected {expected} elements, got {actual}"
            )));
        }
        Ok(())
    }

    fn expect_multiple(name: &str, len: usize, width: usize) -> PyResult<()> {
        if width == 0 {
            return Err(PyValueError::new_err(format!("{name}: row width must be > 0")));
        }
        if len % width != 0 {
            return Err(PyValueError::new_err(format!(
                "{name}: length {len} is not a multiple of row width {width}"
            )));
        }
        Ok(())
    }

    fn expect_even(name: &str, value: usize) -> PyResult<()> {
        if value == 0 || value % 2 != 0 {
            return Err(PyValueError::new_err(format!("{name}: must be a positive even number, got {value}")));
        }
        Ok(())
    }

    #[pyfunction]
    fn topk_sparse(data: Vec<f64>, ratio: f64) -> (Vec<f64>, usize) {
        let n = data.len();
        let k = std::cmp::max(1, (ratio * n as f64).ceil() as usize).min(n);
        if k >= n {
            return (data, n);
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            data[b].abs().partial_cmp(&data[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out = vec![0.0; n];
        for &i in &indices[..k] {
            out[i] = data[i];
        }
        (out, k)
    }

    #[pyfunction]
    fn topk_sparse_batch(data: Vec<f64>, row_len: usize, ratio: f64) -> PyResult<Vec<f64>> {
        use rayon::prelude::*;
        expect_multiple("data", data.len(), row_len)?;
        let k = std::cmp::max(1, (ratio * row_len as f64).ceil() as usize).min(row_len);
        if k >= row_len {
            return Ok(data);
        }
        let mut out = vec![0.0; data.len()];
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(row, out_row)| {
                let src = &data[row * row_len..(row + 1) * row_len];
                let mut indices: Vec<usize> = (0..row_len).collect();
                indices.sort_unstable_by(|&a, &b| {
                    src[b].abs().partial_cmp(&src[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
                });
                for &i in &indices[..k] {
                    out_row[i] = src[i];
                }
            });
        Ok(out)
    }

    #[pyfunction]
    fn nn_topk_silu_fwd<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
        dim: usize,
        ratio: f32,
    ) -> PyResult<(&'py PyArray1<f32>, &'py PyArray1<u8>)> {
        let data = input.as_slice()?;
        expect_multiple("input", data.len(), dim)?;
        let (out, mask) = nn_ops::topk_silu_fwd(data, dim, ratio);
        Ok((out.into_pyarray(py), mask.into_pyarray(py)))
    }

    #[pyfunction]
    fn nn_topk_silu_bwd<'py>(
        py: Python<'py>,
        grad: PyReadonlyArray1<'py, f32>,
        input: PyReadonlyArray1<'py, f32>,
        mask: PyReadonlyArray1<'py, u8>,
        dim: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let g = grad.as_slice()?;
        let x = input.as_slice()?;
        let m = mask.as_slice()?;
        expect_multiple("grad", g.len(), dim)?;
        expect_len("input", x.len(), g.len())?;
        expect_len("mask", m.len(), g.len())?;
        Ok(nn_ops::topk_silu_bwd(g, x, m, dim).into_pyarray(py))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_lbo_fused_fwd<'py>(
        py: Python<'py>,
        normed: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        h: f32,
        scale: PyReadonlyArray1<'py, f32>,
        bias: PyReadonlyArray1<'py, f32>,
        alpha_conf: f32,
        dim: usize,
        rank: usize,
    ) -> PyResult<(&'py PyArray1<f32>, f32)> {
        let normed_s = normed.as_slice()?;
        let v_s = v.as_slice()?;
        let scale_s = scale.as_slice()?;
        let bias_s = bias.as_slice()?;
        expect_multiple("normed", normed_s.len(), dim)?;
        expect_len("v", v_s.len(), rank * dim)?;
        expect_len("scale", scale_s.len(), dim)?;
        expect_len("bias", bias_s.len(), dim)?;
        let (out, curv) = nn_ops::lbo_fused_fwd(
            normed_s, v_s, h, scale_s, bias_s, alpha_conf, dim, rank,
        );
        Ok((out.into_pyarray(py), curv))
    }

    #[pyfunction]
    fn nn_power_iter<'py>(
        py: Python<'py>,
        v_mat: PyReadonlyArray1<'py, f32>,
        spectral_v: PyReadonlyArray1<'py, f32>,
        dim: usize,
        rank: usize,
    ) -> PyResult<(&'py PyArray1<f32>, f32)> {
        let v_mat_s = v_mat.as_slice()?;
        let spectral_s = spectral_v.as_slice()?;
        expect_len("v_mat", v_mat_s.len(), rank * dim)?;
        let (new_v, sigma) = nn_ops::power_iter_step(v_mat_s, spectral_s, dim, rank);
        Ok((new_v.into_pyarray(py), sigma))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_gauge_lattice_fwd<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
        su3_up: PyReadonlyArray1<'py, f32>,
        su3_down: PyReadonlyArray1<'py, f32>,
        su2_up: PyReadonlyArray1<'py, f32>,
        su2_down: PyReadonlyArray1<'py, f32>,
        u1_up: PyReadonlyArray1<'py, f32>,
        u1_down: PyReadonlyArray1<'py, f32>,
        mix_down: PyReadonlyArray1<'py, f32>,
        mix_up: PyReadonlyArray1<'py, f32>,
        d3: usize, d2: usize, d1: usize,
        h3: usize, h2: usize, h1: usize,
        mix_rank: usize,
        ratio: f32,
        dim: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let input_s = input.as_slice()?;
        let su3_up_s = su3_up.as_slice()?;
        let su3_down_s = su3_down.as_slice()?;
        let su2_up_s = su2_up.as_slice()?;
        let su2_down_s = su2_down.as_slice()?;
        let u1_up_s = u1_up.as_slice()?;
        let u1_down_s = u1_down.as_slice()?;
        let mix_down_s = mix_down.as_slice()?;
        let mix_up_s = mix_up.as_slice()?;
        expect_multiple("input", input_s.len(), dim)?;
        expect_len("d3 + d2 + d1", d3 + d2 + d1, dim)?;
        expect_len("su3_up", su3_up_s.len(), d3 * h3)?;
        expect_len("su3_down", su3_down_s.len(), d3 * h3)?;
        expect_len("su2_up", su2_up_s.len(), d2 * h2)?;
        expect_len("su2_down", su2_down_s.len(), d2 * h2)?;
        expect_len("u1_up", u1_up_s.len(), d1 * h1)?;
        expect_len("u1_down", u1_down_s.len(), d1 * h1)?;
        expect_len("mix_down", mix_down_s.len(), dim * mix_rank)?;
        expect_len("mix_up", mix_up_s.len(), dim * mix_rank)?;
        Ok(nn_ops::gauge_lattice_fwd(
            input_s, su3_up_s, su3_down_s, su2_up_s, su2_down_s, u1_up_s, u1_down_s,
            mix_down_s, mix_up_s,
            d3, d2, d1, h3, h2, h1, mix_rank, ratio, dim,
        ).into_pyarray(py))
    }

    #[pyfunction]
    fn nn_ce_pack_sparse<'py>(
        py: Python<'py>,
        w: PyReadonlyArray1<'py, f32>,
        dim: usize,
        zero_tol: f32,
    ) -> PyResult<(&'py PyArray1<f32>, &'py PyArray1<i32>, &'py PyArray1<i32>)> {
        let data = w.as_slice()?;
        expect_len("w", data.len(), dim * dim)?;
        let (vals, cols, rows) = ce_riemann::pack_sparse_csr(data, dim, zero_tol);
        Ok((vals.into_pyarray(py), cols.into_pyarray(py), rows.into_pyarray(py)))
    }

    #[pyfunction]
    fn nn_ce_metric_basis_fwd<'py>(
        py: Python<'py>,
        codebook: PyReadonlyArray1<'py, f32>,
        m_ref: PyReadonlyArray1<'py, f32>,
        n_code: usize,
        dim: usize,
        rank: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let cb = codebook.as_slice()?;
        let mr = m_ref.as_slice()?;
        expect_len("codebook", cb.len(), n_code * dim)?;
        expect_len("m_ref", mr.len(), dim)?;
        Ok(ce_riemann::metric_basis_from_codebook(cb, mr, n_code, dim, rank).into_pyarray(py))
    }

    #[pyfunction]
    fn nn_ce_codebook_pull<'py>(
        py: Python<'py>,
        m: PyReadonlyArray1<'py, f32>,
        codebook: PyReadonlyArray1<'py, f32>,
        n_code: usize,
        dim: usize,
        beta: f32,
        cb_w: f32,
    ) -> PyResult<(&'py PyArray1<f32>, f32)> {
        let m_s = m.as_slice()?;
        let cb = codebook.as_slice()?;
        expect_len("m", m_s.len(), dim)?;
        expect_len("codebook", cb.len(), n_code * dim)?;
        let (grad, energy) = ce_riemann::codebook_pull(m_s, cb, n_code, dim, beta, cb_w);
        Ok((grad.into_pyarray(py), energy))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_relax_fwd<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f32>,
        col_idx: PyReadonlyArray1<'py, i32>,
        row_ptr: PyReadonlyArray1<'py, i32>,
        b: PyReadonlyArray1<'py, f32>,
        phi: PyReadonlyArray1<'py, f32>,
        m0: PyReadonlyArray1<'py, f32>,
        codebook: PyReadonlyArray1<'py, f32>,
        metric_basis: PyReadonlyArray1<'py, f32>,
        dim: usize,
        n_code: usize,
        rank: usize,
        portal: f32,
        bypass: f32,
        t_wake: f32,
        beta: f32,
        cb_w: f32,
        lambda0: f32,
        lambda_phi: f32,
        lambda_var: f32,
        tau: f32,
        dt: f32,
        max_steps: usize,
        tol: f32,
        anneal_ratio: f32,
        noise_scale: f32,
        seed: u64,
    ) -> PyResult<(
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        usize,
    )> {
        let values_s = values.as_slice()?;
        let col_idx_s = col_idx.as_slice()?;
        let row_ptr_s = row_ptr.as_slice()?;
        let b_s = b.as_slice()?;
        let phi_s = phi.as_slice()?;
        let m0_s = m0.as_slice()?;
        let codebook_s = codebook.as_slice()?;
        let basis_s = metric_basis.as_slice()?;
        expect_len("col_idx", col_idx_s.len(), values_s.len())?;
        expect_len("row_ptr", row_ptr_s.len(), dim + 1)?;
        expect_len("b", b_s.len(), dim)?;
        expect_len("phi", phi_s.len(), dim)?;
        expect_len("m0", m0_s.len(), dim)?;
        expect_len("codebook", codebook_s.len(), n_code * dim)?;
        expect_len("metric_basis", basis_s.len(), rank * dim)?;
        let out = ce_riemann::relax_forward(
            values_s, col_idx_s, row_ptr_s, b_s, phi_s, m0_s, codebook_s, basis_s,
            dim, n_code, rank,
            portal, bypass, t_wake, beta, cb_w,
            lambda0, lambda_phi, lambda_var,
            tau, dt, max_steps, tol, anneal_ratio, noise_scale, seed,
        );
        Ok((
            out.best_m.into_pyarray(py),
            out.energy.into_pyarray(py),
            out.delta.into_pyarray(py),
            out.e_hop.into_pyarray(py),
            out.e_bias.into_pyarray(py),
            out.e_portal.into_pyarray(py),
            out.e_cb.into_pyarray(py),
            out.bypass_hist.into_pyarray(py),
            out.steps,
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_brain_step<'py>(
        py: Python<'py>,
        w_values: PyReadonlyArray1<'py, f32>,
        w_col_idx: PyReadonlyArray1<'py, i32>,
        w_row_ptr: PyReadonlyArray1<'py, i32>,
        activation: PyReadonlyArray1<'py, f32>,
        refractory: PyReadonlyArray1<'py, f32>,
        memory_trace: PyReadonlyArray1<'py, f32>,
        adaptation: PyReadonlyArray1<'py, f32>,
        stp_u: PyReadonlyArray1<'py, f32>,
        stp_x: PyReadonlyArray1<'py, f32>,
        bitfield: PyReadonlyArray1<'py, u8>,
        active_mask: PyReadonlyArray1<'py, u8>,
        external: PyReadonlyArray1<'py, f32>,
        goal: PyReadonlyArray1<'py, f32>,
        replay: PyReadonlyArray1<'py, f32>,
        noise: PyReadonlyArray1<'py, f32>,
        mode: u8,
        energy_budget: usize,
        activation_decay: f32,
        activation_gain: f32,
        refractory_decay: f32,
        refractory_gain: f32,
        replay_mix: f32,
        refractory_scale: f32,
        goal_gain: f32,
        external_gain: f32,
        bit_lower: f32,
        bit_upper: f32,
        stp_tau_fac_inv: f32,
        stp_tau_rec: f32,
        stp_u_base: f32,
        adaptation_coupling: f32,
        adaptation_decay: f32,
        memory_decay: f32,
        adaptation_clamp: f32,
    ) -> PyResult<(
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<u8>,
        usize,
        f32,
    )> {
        let mode_enum = match mode {
            1 => runtime_types::Mode::Nrem,
            2 => runtime_types::Mode::Rem,
            _ => runtime_types::Mode::Wake,
        };
        let mut mp = kernel::ModeParams::from_mode(mode_enum);
        mp.activation_decay = activation_decay;
        mp.activation_gain = activation_gain;
        mp.refractory_decay = refractory_decay;
        mp.refractory_gain = refractory_gain;
        mp.replay_mix = replay_mix;
        mp.adaptation_coupling = adaptation_coupling;
        mp.adaptation_decay = adaptation_decay;
        mp.adaptation_gain = adaptation_decay;
        mp.memory_decay = memory_decay;
        mp.memory_gain = memory_decay;
        let cfg = kernel::StepConfig {
            energy_budget,
            refractory_scale,
            goal_gain,
            external_gain,
            active_threshold: 0.22,
            bit_lower,
            bit_upper,
            stp: kernel::StpParams {
                tau_fac: stp_tau_fac_inv,
                tau_rec: stp_tau_rec,
                u_base: stp_u_base,
            },
            adaptation_clamp,
            ..Default::default()
        };
        let mut act = activation.as_slice()?.to_vec();
        let n = act.len();
        let mut refr = refractory.as_slice()?.to_vec();
        let mut mem = memory_trace.as_slice()?.to_vec();
        let mut adapt = adaptation.as_slice()?.to_vec();
        let mut su = stp_u.as_slice()?.to_vec();
        let mut sx = stp_x.as_slice()?.to_vec();
        let mut bit = bitfield.as_slice()?.to_vec();
        let w_values_s = w_values.as_slice()?;
        let w_col_idx_s = w_col_idx.as_slice()?;
        let w_row_ptr_s = w_row_ptr.as_slice()?;
        let active_mask_s = active_mask.as_slice()?;
        let external_s = external.as_slice()?;
        let goal_s = goal.as_slice()?;
        let replay_s = replay.as_slice()?;
        let noise_s = noise.as_slice()?;
        expect_len("refractory", refr.len(), n)?;
        expect_len("memory_trace", mem.len(), n)?;
        expect_len("adaptation", adapt.len(), n)?;
        expect_len("stp_u", su.len(), n)?;
        expect_len("stp_x", sx.len(), n)?;
        expect_len("bitfield", bit.len(), n)?;
        expect_len("active_mask", active_mask_s.len(), n)?;
        expect_len("external", external_s.len(), n)?;
        expect_len("goal", goal_s.len(), n)?;
        expect_len("replay", replay_s.len(), n)?;
        expect_len("noise", noise_s.len(), n)?;
        expect_len("w_row_ptr", w_row_ptr_s.len(), n + 1)?;
        expect_len("w_col_idx", w_col_idx_s.len(), w_values_s.len())?;
        let out = kernel::brain_step(
            w_values_s,
            w_col_idx_s,
            w_row_ptr_s,
            &mut act,
            &mut refr,
            &mut mem,
            &mut adapt,
            &mut su,
            &mut sx,
            &mut bit,
            active_mask_s,
            external_s,
            goal_s,
            replay_s,
            noise_s,
            &mp,
            &cfg,
        );
        Ok((
            act.into_pyarray(py),
            refr.into_pyarray(py),
            mem.into_pyarray(py),
            adapt.into_pyarray(py),
            su.into_pyarray(py),
            sx.into_pyarray(py),
            bit.into_pyarray(py),
            out.active_count,
            out.energy,
        ))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_mfa_fwd<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        n: usize,
        d: usize,
        sigma_grav: f32,
        w_lang: f32,
        w_grav: f32,
        causal: bool,
    ) -> PyResult<(&'py PyArray1<f32>, &'py PyArray1<f32>)> {
        let q_s = q.as_slice()?;
        let k_s = k.as_slice()?;
        let v_s = v.as_slice()?;
        expect_len("q", q_s.len(), n * d)?;
        expect_len("k", k_s.len(), n * d)?;
        expect_len("v", v_s.len(), n * d)?;
        let (out, attn) = nn_ops::ce_mfa_fwd(q_s, k_s, v_s, n, d, sigma_grav, w_lang, w_grav, causal);
        Ok((out.into_pyarray(py), attn.into_pyarray(py)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_euler_fwd<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        pi_inv_freq: PyReadonlyArray1<'py, f32>,
        n: usize,
        d_head: usize,
        pi_gate: f32,
        e_gate: f32,
        xi: f32,
        causal: bool,
    ) -> PyResult<(&'py PyArray1<f32>, &'py PyArray1<f32>)> {
        let q_s = q.as_slice()?;
        let k_s = k.as_slice()?;
        let v_s = v.as_slice()?;
        let freq_s = pi_inv_freq.as_slice()?;
        expect_even("d_head", d_head)?;
        expect_len("q", q_s.len(), n * d_head)?;
        expect_len("k", k_s.len(), n * d_head)?;
        expect_len("v", v_s.len(), n * d_head)?;
        expect_len("pi_inv_freq", freq_s.len(), d_head / 2)?;
        let (out, attn) = nn_ops::ce_euler_fwd(
            q_s, k_s, v_s, freq_s, n, d_head, pi_gate, e_gate, xi, causal,
        );
        Ok((out.into_pyarray(py), attn.into_pyarray(py)))
    }

    #[allow(clippy::too_many_arguments)]
    fn check_riemann_shapes(
        q: usize, k: usize, v: usize, cos: usize, sin: usize, sheet_bias: usize,
        bh: usize, n: usize, d_head: usize,
    ) -> PyResult<()> {
        expect_even("d_head", d_head)?;
        let half = d_head / 2;
        expect_len("q", q, bh * n * d_head)?;
        expect_len("k", k, bh * n * d_head)?;
        expect_len("v", v, bh * n * d_head)?;
        expect_len("cos", cos, bh * n * half)?;
        expect_len("sin", sin, bh * n * half)?;
        expect_len("sheet_bias", sheet_bias, bh * n * n)?;
        Ok(())
    }

    /// Batched Riemann-surface attention (CPU). Inputs are flat row-major
    /// with leading dim `bh = batch * heads`. Returns the output tensor
    /// (bh * n * d_head); the attention matrix is not materialized.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_riemann_fwd<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        cos: PyReadonlyArray1<'py, f32>,
        sin: PyReadonlyArray1<'py, f32>,
        sheet_bias: PyReadonlyArray1<'py, f32>,
        bh: usize,
        n: usize,
        d_head: usize,
        causal: bool,
    ) -> PyResult<&'py PyArray1<f32>> {
        let q_s = q.as_slice()?;
        let k_s = k.as_slice()?;
        let v_s = v.as_slice()?;
        let cos_s = cos.as_slice()?;
        let sin_s = sin.as_slice()?;
        let sb_s = sheet_bias.as_slice()?;
        check_riemann_shapes(q_s.len(), k_s.len(), v_s.len(), cos_s.len(), sin_s.len(), sb_s.len(), bh, n, d_head)?;
        let out = nn_ops::ce_riemann_fwd(q_s, k_s, v_s, cos_s, sin_s, sb_s, bh, n, d_head, causal);
        Ok(out.into_pyarray(py))
    }

    /// Batched Riemann-surface attention (CUDA, host staging). Convenience
    /// path for CPU-resident tensors that should compute on GPU.
    #[cfg(feature = "cuda")]
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_riemann_fwd_cuda<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        cos: PyReadonlyArray1<'py, f32>,
        sin: PyReadonlyArray1<'py, f32>,
        sheet_bias: PyReadonlyArray1<'py, f32>,
        bh: usize,
        n: usize,
        d_head: usize,
        causal: bool,
    ) -> PyResult<&'py PyArray1<f32>> {
        use crate::cuda;
        let q_s = q.as_slice()?;
        let k_s = k.as_slice()?;
        let v_s = v.as_slice()?;
        let cos_s = cos.as_slice()?;
        let sin_s = sin.as_slice()?;
        let sb_s = sheet_bias.as_slice()?;
        check_riemann_shapes(q_s.len(), k_s.len(), v_s.len(), cos_s.len(), sin_s.len(), sb_s.len(), bh, n, d_head)?;
        let out = cuda::ce_riemann_fwd_cuda(q_s, k_s, v_s, cos_s, sin_s, sb_s, bh, n, d_head, causal)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Zero-copy CUDA entry. Inputs are raw CUDA device pointers
    /// (`tensor.data_ptr()` from PyTorch). The kernel writes the result
    /// directly into the buffer at `out_ptr`. Caller MUST ensure that
    /// PyTorch's current stream has been synchronized before this call.
    #[cfg(feature = "cuda")]
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_riemann_fwd_cuda_devptr(
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        cos_ptr: u64,
        sin_ptr: u64,
        sb_ptr: u64,
        out_ptr: u64,
        bh: usize,
        n: usize,
        d_head: usize,
        causal: bool,
    ) -> PyResult<()> {
        use crate::cuda;
        expect_even("d_head", d_head)?;
        unsafe {
            cuda::ce_riemann_fwd_cuda_devptr(
                q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, sb_ptr, out_ptr,
                bh, n, d_head, causal,
            )
        }
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_dual_attn_fwd<'py>(
        py: Python<'py>,
        z_l: PyReadonlyArray1<'py, f32>,
        z_g: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        n: usize,
        d_l: usize,
        d_g: usize,
        d_m: usize,
        sigma_grav: f32,
        w_lang: f32,
        w_grav: f32,
        causal: bool,
    ) -> PyResult<(&'py PyArray1<f32>, &'py PyArray1<f32>)> {
        let z_l_s = z_l.as_slice()?;
        let z_g_s = z_g.as_slice()?;
        let v_s = v.as_slice()?;
        expect_len("z_l", z_l_s.len(), n * d_l)?;
        expect_len("z_g", z_g_s.len(), n * d_g)?;
        expect_len("v", v_s.len(), n * d_m)?;
        let (out, k) = nn_ops::ce_dual_attn_fwd(
            z_l_s, z_g_s, v_s, n, d_l, d_g, d_m, sigma_grav, w_lang, w_grav, causal,
        );
        Ok((out.into_pyarray(py), k.into_pyarray(py)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_llm_pre_eq_fwd<'py>(
        py: Python<'py>,
        prior: PyReadonlyArray1<'py, f64>,
        supported: PyReadonlyArray1<'py, f64>,
        unsupported: PyReadonlyArray1<'py, f64>,
        contradicted: PyReadonlyArray1<'py, f64>,
        instruction: PyReadonlyArray1<'py, f64>,
        self_contradiction: PyReadonlyArray1<'py, f64>,
        uncertainty: PyReadonlyArray1<'py, f64>,
        beta: f64,
        w_contradicted: f64,
        w_unsupported: f64,
        w_no_evidence: f64,
        w_coverage: f64,
        w_instruction: f64,
        w_self_contradiction: f64,
        w_uncertainty: f64,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>)> {
        let weights = llm_pre_eq::LlmPreEqWeights {
            contradicted: w_contradicted,
            unsupported: w_unsupported,
            no_evidence: w_no_evidence,
            coverage: w_coverage,
            instruction: w_instruction,
            self_contradiction: w_self_contradiction,
            uncertainty: w_uncertainty,
        };
        let energy = llm_pre_eq::defect_energies(
            supported.as_slice()?,
            unsupported.as_slice()?,
            contradicted.as_slice()?,
            instruction.as_slice()?,
            self_contradiction.as_slice()?,
            uncertainty.as_slice()?,
            weights,
        )
        .map_err(PyValueError::new_err)?;
        let posterior = llm_pre_eq::gibbs_posterior(prior.as_slice()?, &energy, beta)
            .map_err(PyValueError::new_err)?;
        Ok((energy.into_pyarray(py), posterior.into_pyarray(py)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_llm_claim_pre_eq_fwd<'py>(
        py: Python<'py>,
        prior: PyReadonlyArray1<'py, f64>,
        residual: PyReadonlyArray1<'py, f64>,
        graph: PyReadonlyArray1<'py, f64>,
        tau: PyReadonlyArray1<'py, f64>,
        source_unreliability: PyReadonlyArray1<'py, f64>,
        independence: PyReadonlyArray1<'py, f64>,
        missing: PyReadonlyArray1<'py, f64>,
        instruction: PyReadonlyArray1<'py, f64>,
        schema: PyReadonlyArray1<'py, f64>,
        coverage: PyReadonlyArray1<'py, f64>,
        unsupported: PyReadonlyArray1<'py, f64>,
        ce_penalty: PyReadonlyArray1<'py, f64>,
        beta: f64,
        w_residual: f64,
        w_graph: f64,
        w_tau: f64,
        w_source: f64,
        w_independence: f64,
        w_missing: f64,
        w_instruction: f64,
        w_schema: f64,
        w_coverage: f64,
        w_unsupported: f64,
        w_ce_penalty: f64,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>)> {
        let weights = llm_pre_eq::ClaimActionWeights {
            residual: w_residual,
            graph: w_graph,
            tau: w_tau,
            source: w_source,
            independence: w_independence,
            missing: w_missing,
            instruction: w_instruction,
            schema: w_schema,
            coverage: w_coverage,
            unsupported: w_unsupported,
            ce_penalty: w_ce_penalty,
        };
        let actions = llm_pre_eq::claim_answer_actions(
            residual.as_slice()?,
            graph.as_slice()?,
            tau.as_slice()?,
            source_unreliability.as_slice()?,
            independence.as_slice()?,
            missing.as_slice()?,
            instruction.as_slice()?,
            schema.as_slice()?,
            coverage.as_slice()?,
            unsupported.as_slice()?,
            ce_penalty.as_slice()?,
            weights,
        )
        .map_err(PyValueError::new_err)?;
        let posterior = llm_pre_eq::gibbs_posterior(prior.as_slice()?, &actions, beta)
            .map_err(PyValueError::new_err)?;
        Ok((actions.into_pyarray(py), posterior.into_pyarray(py)))
    }

    #[pymodule]
    fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(topk_sparse, m)?)?;
        m.add_function(wrap_pyfunction!(topk_sparse_batch, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_bwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_lbo_fused_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_power_iter, m)?)?;
        m.add_function(wrap_pyfunction!(nn_gauge_lattice_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_pack_sparse, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_metric_basis_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_codebook_pull, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_relax_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_brain_step, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_mfa_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_dual_attn_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_euler_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_riemann_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_llm_pre_eq_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_llm_claim_pre_eq_fwd, m)?)?;
        #[cfg(feature = "cuda")]
        m.add_function(wrap_pyfunction!(nn_ce_riemann_fwd_cuda, m)?)?;
        #[cfg(feature = "cuda")]
        m.add_function(wrap_pyfunction!(nn_ce_riemann_fwd_cuda_devptr, m)?)?;
        Ok(())
    }
}
