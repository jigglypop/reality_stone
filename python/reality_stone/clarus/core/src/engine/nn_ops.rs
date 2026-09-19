//! Fused neural-network ops for the Clarus runtime.
//!
//! All functions operate on flat f32 slices laid out row-major.
//! Matrix multiplications use ndarray (matrixmultiply SIMD backend).
//! Row-parallel via rayon where beneficial.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::cmp;

// ---- helpers ---------------------------------------------------------------

#[inline(always)]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---- TopK SiLU -------------------------------------------------------------

/// Fused SiLU + TopK sparse masking (forward).
///
/// `input`: flat `[n_rows * dim]`, `dim`: row width, `ratio`: keep fraction.
/// Returns `(output, mask)`.
pub fn topk_silu_fwd(input: &[f32], dim: usize, ratio: f32) -> (Vec<f32>, Vec<u8>) {
    let k = cmp::max(1, (ratio * dim as f32).ceil() as usize).min(dim);
    let n = input.len();
    let mut output = vec![0.0f32; n];
    let mut mask = vec![0u8; n];

    if k >= dim {
        output
            .par_chunks_mut(dim)
            .zip(mask.par_chunks_mut(dim))
            .enumerate()
            .for_each(|(r, (out, msk))| {
                let src = &input[r * dim..(r + 1) * dim];
                for j in 0..dim {
                    out[j] = silu_f32(src[j]);
                    msk[j] = 1;
                }
            });
        return (output, mask);
    }

    output
        .par_chunks_mut(dim)
        .zip(mask.par_chunks_mut(dim))
        .enumerate()
        .for_each(|(r, (out, msk))| {
            let src = &input[r * dim..(r + 1) * dim];
            for j in 0..dim {
                out[j] = silu_f32(src[j]);
            }
            let mut abs_vals: Vec<f32> = out.iter().map(|x| x.abs()).collect();
            abs_vals.select_nth_unstable_by(dim - k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let thr = abs_vals[dim - k];
            for j in 0..dim {
                if out[j].abs() >= thr {
                    msk[j] = 1;
                } else {
                    out[j] = 0.0;
                }
            }
        });
    (output, mask)
}

/// TopK SiLU backward.
pub fn topk_silu_bwd(grad: &[f32], input: &[f32], mask: &[u8], dim: usize) -> Vec<f32> {
    let n = grad.len();
    let mut grad_in = vec![0.0f32; n];

    grad_in.par_chunks_mut(dim).enumerate().for_each(|(r, gi)| {
        let base = r * dim;
        for j in 0..dim {
            if mask[base + j] == 1 {
                let x = input[base + j];
                let s = sigmoid_f32(x);
                gi[j] = grad[base + j] * s * (1.0 + x * (1.0 - s));
            }
        }
    });
    grad_in
}

// ---- LBO Norm (ndarray-backed matmul) ---------------------------------------

/// Fused LBO normalization forward (post-LayerNorm).
///
/// Uses ndarray `dot()` (matrixmultiply SIMD) for projections.
pub fn lbo_fused_fwd(
    normed: &[f32],
    v: &[f32],
    h: f32,
    scale: &[f32],
    bias: &[f32],
    alpha_conf: f32,
    dim: usize,
    rank: usize,
) -> (Vec<f32>, f32) {
    let n_rows = normed.len() / dim;

    // conformal factor
    let phi_sq: f32 = normed.iter().map(|&x| x * x).sum::<f32>() / normed.len() as f32;
    let conformal = (-alpha_conf.abs() * phi_sq).exp();

    // V_eff = V * conformal  [rank, dim]
    let v_scaled: Vec<f32> = v.iter().map(|&x| x * conformal).collect();
    let x_mat = ArrayView2::from_shape((n_rows, dim), normed).unwrap();
    let v_mat = ArrayView2::from_shape((rank, dim), &v_scaled).unwrap();

    // proj = X @ V_eff^T  -> [n_rows, rank]   (ndarray SIMD dot)
    let proj = x_mat.dot(&v_mat.t());
    // xW = proj @ V_eff   -> [n_rows, dim]    (ndarray SIMD dot)
    let xw = proj.dot(&v_mat);

    // output + curvature
    let scale_v = ArrayView1::from(scale);
    let bias_v = ArrayView1::from(bias);
    let one_minus_h = 1.0 - h;
    let mut output = vec![0.0f32; normed.len()];
    let mut curv_sum = 0.0f64;

    for r in 0..n_rows {
        let base = r * dim;
        for j in 0..dim {
            let lx = x_mat[[r, j]] - xw[[r, j]];
            curv_sum += (lx as f64) * (lx as f64);
            output[base + j] =
                (one_minus_h * x_mat[[r, j]] + h * xw[[r, j]]) * scale_v[j] + bias_v[j];
        }
    }
    (output, (curv_sum / (n_rows * dim) as f64) as f32)
}

/// Power iteration: 1 step for sigma_max(V).
pub fn power_iter_step(
    v_mat: &[f32],
    spectral_v: &[f32],
    dim: usize,
    rank: usize,
) -> (Vec<f32>, f32) {
    let v_nd = ArrayView2::from_shape((rank, dim), v_mat).unwrap();
    let sv = ArrayView1::from_shape(dim, spectral_v).unwrap();

    // u = V @ sv  [rank]
    let u_raw = v_nd.dot(&sv);
    let u_norm = u_raw.mapv(|x| x * x).sum().sqrt().max(1e-12);
    let u = u_raw.mapv(|x| x / u_norm);

    // new_v = V^T @ u  [dim]
    let vt = v_nd.t();
    let nv_raw = vt.dot(&u);
    let nv_norm = nv_raw.mapv(|x| x * x).sum().sqrt().max(1e-12);
    let new_v = nv_raw.mapv(|x| x / nv_norm);

    // sigma = ||V @ new_v||
    let sigma = v_nd.dot(&new_v).mapv(|x| x * x).sum().sqrt();

    (new_v.to_vec(), sigma)
}

// ---- Gauge lattice (ndarray matmul per channel) ----------------------------

/// Single gauge channel: up -> SiLU -> TopK -> down.
fn channel_fwd(
    x: &ArrayView1<f32>,
    up_w: &ArrayView2<f32>,   // [hid, d_in]
    down_w: &ArrayView2<f32>, // [d_in, hid]
    k: usize,
) -> Array1<f32> {
    // hidden = x @ up^T  -> [hid]
    let hidden_raw = up_w.dot(x);
    let hid = hidden_raw.len();

    // SiLU + TopK
    let mut hidden: Vec<f32> = hidden_raw.iter().map(|&v| silu_f32(v)).collect();
    if k < hid {
        let mut abs_h: Vec<f32> = hidden.iter().map(|v| v.abs()).collect();
        abs_h.select_nth_unstable_by(hid - k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let thr = abs_h[hid - k];
        for v in hidden.iter_mut() {
            if v.abs() < thr {
                *v = 0.0;
            }
        }
    }

    // output = down @ hidden  -> [d_in]  (down is [d_in, hid])
    let h_arr = ArrayView1::from(&hidden);
    down_w.dot(&h_arr)
}

/// Gauge lattice 3-channel forward.
#[allow(clippy::too_many_arguments)]
pub fn gauge_lattice_fwd(
    input: &[f32],
    su3_up: &[f32],
    su3_down: &[f32],
    su2_up: &[f32],
    su2_down: &[f32],
    u1_up: &[f32],
    u1_down: &[f32],
    mix_down: &[f32],
    mix_up: &[f32],
    d3: usize,
    d2: usize,
    d1: usize,
    h3: usize,
    h2: usize,
    h1: usize,
    mix_rank: usize,
    ratio: f32,
    dim: usize,
) -> Vec<f32> {
    let _n_rows = input.len() / dim;
    let k3 = cmp::max(1, (ratio * h3 as f32).ceil() as usize).min(h3);
    let k2 = cmp::max(1, (ratio * h2 as f32).ceil() as usize).min(h2);
    let k1 = cmp::max(1, (ratio * h1 as f32).ceil() as usize).min(h1);
    let has_mix = mix_rank > 0 && !mix_down.is_empty() && !mix_up.is_empty();

    let su3_up_nd = ArrayView2::from_shape((h3, d3), su3_up).unwrap();
    let su3_dn_nd = ArrayView2::from_shape((d3, h3), su3_down).unwrap();
    let su2_up_nd = ArrayView2::from_shape((h2, d2), su2_up).unwrap();
    let su2_dn_nd = ArrayView2::from_shape((d2, h2), su2_down).unwrap();
    let u1_up_nd = ArrayView2::from_shape((h1, d1), u1_up).unwrap();
    let u1_dn_nd = ArrayView2::from_shape((d1, h1), u1_down).unwrap();

    let x_mat = ArrayView2::from_shape((_n_rows, dim), input).unwrap();
    let mut output = vec![0.0f32; input.len()];

    let s3 = d3;
    let s32 = d3 + d2;

    output.par_chunks_mut(dim).enumerate().for_each(|(r, out)| {
        let x_row = x_mat.row(r);
        let x3 = x_row.slice(ndarray::s![..s3]);
        let x2 = x_row.slice(ndarray::s![s3..s32]);
        let x1 = x_row.slice(ndarray::s![s32..]);

        let y3 = channel_fwd(&x3, &su3_up_nd, &su3_dn_nd, k3);
        let y2 = channel_fwd(&x2, &su2_up_nd, &su2_dn_nd, k2);
        let y1 = channel_fwd(&x1, &u1_up_nd, &u1_dn_nd, k1);

        out[..s3].copy_from_slice(y3.as_slice().unwrap());
        out[s3..s32].copy_from_slice(y2.as_slice().unwrap());
        out[s32..].copy_from_slice(y1.as_slice().unwrap());

        if has_mix {
            let md = ArrayView2::from_shape((mix_rank, dim), mix_down).unwrap();
            let mu = ArrayView2::from_shape((dim, mix_rank), mix_up).unwrap();
            let out_view = ArrayView1::from(&*out);
            let proj = md.dot(&out_view);
            let mix_result = mu.dot(&proj);
            for j in 0..dim {
                out[j] += mix_result[j];
            }
        }
    });
    output
}

// ---- CE Softmax / Metric-Family Attention (MFA) ----------------------------
//
// Equation 6.B.1 (applied compendium): compute
//   s_lang_ij = (q_i . k_j) / sqrt(d)
//   s_grav_ij = -||k_i - k_j||^2 / (2 sigma^2)     (identity metric)
//   s_ij      = w_lang * s_lang_ij + w_grav * s_grav_ij   (logit mixing)
//   A_ij      = softmax_j(s_ij)  (with optional causal mask)
//   out_i     = sum_j A_ij * v_j
//
// `q`, `k`, `v` are `(n, d)` row-major f32 slices for a single head.
// `causal = true` applies a lower-triangular mask.

#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline(always)]
fn sq_dist_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Fused CE MFA forward (single head, logit-mixing, identity gravity metric).
///
/// Returns (out `(n, d)`, attn `(n, n)`).
pub fn ce_mfa_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    sigma_grav: f32,
    w_lang: f32,
    w_grav: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let scale_lang = 1.0 / (d as f32).sqrt();
    let scale_grav = -1.0 / (2.0 * sigma_grav * sigma_grav);

    let mut attn = vec![0.0f32; n * n];
    let mut out = vec![0.0f32; n * d];

    // Row-parallel over queries.
    attn.par_chunks_mut(n)
        .zip(out.par_chunks_mut(d))
        .enumerate()
        .for_each(|(i, (attn_row, out_row))| {
            let q_i = &q[i * d..(i + 1) * d];
            let k_i = &k[i * d..(i + 1) * d];

            // compute raw scores for row i
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                if causal && j > i {
                    attn_row[j] = f32::NEG_INFINITY;
                    continue;
                }
                let k_j = &k[j * d..(j + 1) * d];
                let s_lang = dot_f32(q_i, k_j) * scale_lang;
                let s_grav = sq_dist_f32(k_i, k_j) * scale_grav;
                let s = w_lang * s_lang + w_grav * s_grav;
                attn_row[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }

            // softmax (numerically stable)
            let mut denom = 0.0f32;
            for j in 0..n {
                if attn_row[j].is_finite() {
                    let e = (attn_row[j] - max_s).exp();
                    attn_row[j] = e;
                    denom += e;
                } else {
                    attn_row[j] = 0.0;
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for j in 0..n {
                attn_row[j] *= inv;
            }

            // out_i = A_i @ V
            for t in 0..d {
                out_row[t] = 0.0;
            }
            for j in 0..n {
                let a = attn_row[j];
                if a == 0.0 {
                    continue;
                }
                let v_j = &v[j * d..(j + 1) * d];
                for t in 0..d {
                    out_row[t] += a * v_j[t];
                }
            }
        });

    (out, attn)
}

// ---- Dual-graph attention (precise mirror of ce_laplacian.DualLaplacianBlock) ----
//
// Inputs per forward (single head, already projected):
//   z_l: (n, d_l)  = P_lang(h)
//   z_g: (n, d_g)  = P_grav(h)
//   v:   (n, d_m)  = V(h)
//   sigma_grav: RBF bandwidth for the gravity graph
//   w_lang, w_grav: convex gate (expect w_lang + w_grav == 1)
//   causal: lower-triangular transition mask
//
// Computes:
//   A_l[i,j] = max(0, cos(z_l[i], z_l[j])), with diag = 0, symmetric
//   A_g[i,j] = exp(-||z_g[i] - z_g[j]||^2 / 2 sigma^2), diag = 0, symmetric
//   K_l = row_norm(apply_causal(A_l))
//   K_g = row_norm(apply_causal(A_g))
//   K   = w_lang * K_l + w_grav * K_g              (still row-stochastic)
//   out[i, t] = sum_j K[i,j] * v[j, t]
//
// Returns (out, K) for validation. K is flattened row-major (n * n).

pub fn ce_dual_attn_fwd(
    z_l: &[f32],
    z_g: &[f32],
    v: &[f32],
    n: usize,
    d_l: usize,
    d_g: usize,
    d_m: usize,
    sigma_grav: f32,
    w_lang: f32,
    w_grav: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(z_l.len(), n * d_l);
    debug_assert_eq!(z_g.len(), n * d_g);
    debug_assert_eq!(v.len(), n * d_m);

    let inv_2s2 = -1.0f32 / (2.0 * sigma_grav * sigma_grav);
    let eps = 1e-8f32;

    // Pre-compute row norms for cosine.
    let mut norm_l = vec![0.0f32; n];
    for i in 0..n {
        let row = &z_l[i * d_l..(i + 1) * d_l];
        let mut s = 0.0f32;
        for &x in row {
            s += x * x;
        }
        norm_l[i] = s.sqrt().max(eps);
    }

    let mut k_combined = vec![0.0f32; n * n];
    let mut out = vec![0.0f32; n * d_m];

    // Row-parallel over query positions.
    k_combined
        .par_chunks_mut(n)
        .zip(out.par_chunks_mut(d_m))
        .enumerate()
        .for_each(|(i, (k_row, out_row))| {
            let zi_l = &z_l[i * d_l..(i + 1) * d_l];
            let zi_g = &z_g[i * d_g..(i + 1) * d_g];
            let ni_l = norm_l[i];

            // First pass: raw unnormalized A_l, A_g for this row, applying causal.
            // We compute on-the-fly the two row sums for renormalization.
            let mut sum_l = 0.0f32;
            let mut sum_g = 0.0f32;
            // Scratch row stored as (lang, grav) in k_row in halves -- we will
            // overwrite twice: first with A_l, then combine with A_g.
            // To save a buffer we accumulate per-j into two scalars and then
            // pass through again; but that's two loops. Instead allocate a
            // small scratch here.
            let mut row_l = vec![0.0f32; n];
            let mut row_g = vec![0.0f32; n];
            for j in 0..n {
                if causal && j > i {
                    continue;
                }
                if j == i {
                    continue; // diagonal zero
                }
                let zj_l = &z_l[j * d_l..(j + 1) * d_l];
                let zj_g = &z_g[j * d_g..(j + 1) * d_g];

                // Cosine
                let mut dot = 0.0f32;
                for k in 0..d_l {
                    dot += zi_l[k] * zj_l[k];
                }
                let cos = (dot / (ni_l * norm_l[j])).max(0.0);
                row_l[j] = cos;
                sum_l += cos;

                // RBF
                let mut d2 = 0.0f32;
                for k in 0..d_g {
                    let diff = zi_g[k] - zj_g[k];
                    d2 += diff * diff;
                }
                let rbf = (d2 * inv_2s2).exp();
                row_g[j] = rbf;
                sum_g += rbf;
            }

            // Row-normalize each kernel separately, then convex combine.
            let inv_l = if sum_l > eps { 1.0 / sum_l } else { 0.0 };
            let inv_g = if sum_g > eps { 1.0 / sum_g } else { 0.0 };
            for j in 0..n {
                let kl = row_l[j] * inv_l;
                let kg = row_g[j] * inv_g;
                k_row[j] = w_lang * kl + w_grav * kg;
            }

            // out_i = K_i @ V
            for t in 0..d_m {
                out_row[t] = 0.0;
            }
            for j in 0..n {
                let a = k_row[j];
                if a == 0.0 {
                    continue;
                }
                let vj = &v[j * d_m..(j + 1) * d_m];
                for t in 0..d_m {
                    out_row[t] += a * vj[t];
                }
            }
        });

    (out, k_combined)
}

// ---- EulerCE attention (pi-phase rotary + e-decay) -------------------------
//
// Implements clarus::ce_euler::EulerCEAttention in native code for a
// single head (one call per (batch, head) element). Inputs are assumed
// pre-projected and reshaped to (n, d_head), d_head even.
//
//   Q', K' = rotate_pi(Q, K)   with theta = pi_gate * pos * pi^{1-k/(d/2)}
//   scores_ij = (Q'_i . K'_j) / sqrt(d_head)
//                + e_gate * (-|i-j| / xi)         [decay bias]
//   scores masked causally, softmax, out = A @ V
//
// Scalar gates (pi_gate, e_gate, xi) are per-head -- supplied by the
// caller for the relevant head. pi_inv_freq is the precomputed
// pi^{1-k/(d/2)} array of length d_head/2.

// ---- Riemann-surface PE attention -----------------------------------------
//
// Mirrors `clarus.ce_riemann_attn.RiemannRotaryAttention` per
// `docs/8_리만/riemann_pe_spec.md`.
//
// Batched layout — a single call processes (BH, N, D) at once. cos/sin are
// pre-broadcast to (BH, N, D/2); sheet_bias to (BH, N, N). All slices are
// row-major contiguous.
//
// Pipeline (per (bh, i)):
//   1. Rotate q[bh, i, :] (RoPE-style 2D rotation per pair) into q_rot.
//   2. score_ij = (q_rot[bh,i] · k_rot[bh,j]) / sqrt(D) + sheet_bias[bh,i,j]
//   3. causal mask + softmax + weighted sum over j → out[bh, i, :]
//
// Per-(bh, i) rotation of k_j is recomputed inside the j-loop to keep the
// hot path cache-resident; the cost is dominated by the dot product anyway.

#[allow(clippy::too_many_arguments)]
pub fn ce_riemann_fwd(
    q: &[f32],          // (bh * n * d_head)
    k: &[f32],          // (bh * n * d_head)
    v: &[f32],          // (bh * n * d_head)
    cos: &[f32],        // (bh * n * d_head/2)
    sin: &[f32],        // (bh * n * d_head/2)
    sheet_bias: &[f32], // (bh * n * n)
    bh: usize,
    n: usize,
    d_head: usize,
    causal: bool,
) -> Vec<f32> {
    debug_assert!(d_head % 2 == 0);
    let half = d_head / 2;
    debug_assert_eq!(q.len(), bh * n * d_head);
    debug_assert_eq!(k.len(), bh * n * d_head);
    debug_assert_eq!(v.len(), bh * n * d_head);
    debug_assert_eq!(cos.len(), bh * n * half);
    debug_assert_eq!(sin.len(), bh * n * half);
    debug_assert_eq!(sheet_bias.len(), bh * n * n);

    let scale = 1.0 / (d_head as f32).sqrt();
    let mut out = vec![0.0f32; bh * n * d_head];

    // Pre-rotate the entire q tensor once per (bh, n) row.
    let mut q_rot = vec![0.0f32; bh * n * d_head];
    let mut k_rot = vec![0.0f32; bh * n * d_head];
    q_rot
        .par_chunks_mut(d_head)
        .zip(k_rot.par_chunks_mut(d_head))
        .enumerate()
        .for_each(|(row, (qr, kr))| {
            let qi = &q[row * d_head..(row + 1) * d_head];
            let ki = &k[row * d_head..(row + 1) * d_head];
            let ci = &cos[row * half..(row + 1) * half];
            let si = &sin[row * half..(row + 1) * half];
            for p in 0..half {
                let c = ci[p];
                let s = si[p];
                let q0 = qi[2 * p];
                let q1 = qi[2 * p + 1];
                qr[2 * p] = q0 * c - q1 * s;
                qr[2 * p + 1] = q0 * s + q1 * c;
                let k0 = ki[2 * p];
                let k1 = ki[2 * p + 1];
                kr[2 * p] = k0 * c - k1 * s;
                kr[2 * p + 1] = k0 * s + k1 * c;
            }
        });

    // Outer-parallelize over (bh, i) rows.
    out.par_chunks_mut(d_head)
        .enumerate()
        .for_each(|(row, out_row)| {
            let bh_idx = row / n;
            let i = row % n;
            let q_rot_base = bh_idx * n * d_head;
            let v_base = bh_idx * n * d_head;
            let bias_base = bh_idx * n * n;

            let qi = &q_rot[q_rot_base + i * d_head..q_rot_base + (i + 1) * d_head];
            let bias_row = &sheet_bias[bias_base + i * n..bias_base + (i + 1) * n];

            // First pass: raw scores + max
            let mut scratch = vec![0.0f32; n];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                if causal && j > i {
                    scratch[j] = f32::NEG_INFINITY;
                    continue;
                }
                let kj = &k_rot[q_rot_base + j * d_head..q_rot_base + (j + 1) * d_head];
                let mut dot = 0.0f32;
                for t in 0..d_head {
                    dot += qi[t] * kj[t];
                }
                let s = dot * scale + bias_row[j];
                scratch[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }

            // Softmax (numerically stable)
            let mut denom = 0.0f32;
            for j in 0..n {
                if scratch[j].is_finite() {
                    let e = (scratch[j] - max_s).exp();
                    scratch[j] = e;
                    denom += e;
                } else {
                    scratch[j] = 0.0;
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };

            // out_i = sum_j (e_j * inv) * v_j
            for t in 0..d_head {
                out_row[t] = 0.0;
            }
            for j in 0..n {
                let w = scratch[j] * inv;
                if w == 0.0 {
                    continue;
                }
                let vj = &v[v_base + j * d_head..v_base + (j + 1) * d_head];
                for t in 0..d_head {
                    out_row[t] += w * vj[t];
                }
            }
        });

    out
}

pub fn ce_euler_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pi_inv_freq: &[f32],
    n: usize,
    d_head: usize,
    pi_gate: f32,
    e_gate: f32,
    xi: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(q.len(), n * d_head);
    debug_assert_eq!(k.len(), n * d_head);
    debug_assert_eq!(v.len(), n * d_head);
    debug_assert_eq!(pi_inv_freq.len(), d_head / 2);
    debug_assert!(d_head % 2 == 0);

    let scale = 1.0 / (d_head as f32).sqrt();
    let inv_xi = e_gate * (1.0 / xi.max(1e-6));

    // Build rotated Q, K once (n * d_head each).
    let mut q_rot = vec![0.0f32; n * d_head];
    let mut k_rot = vec![0.0f32; n * d_head];

    q_rot
        .par_chunks_mut(d_head)
        .zip(k_rot.par_chunks_mut(d_head))
        .enumerate()
        .for_each(|(i, (qr, kr))| {
            let qi = &q[i * d_head..(i + 1) * d_head];
            let ki = &k[i * d_head..(i + 1) * d_head];
            let pos = i as f32;
            for (pair, &inv_f) in pi_inv_freq.iter().enumerate() {
                let theta = pi_gate * pos * inv_f;
                let c = theta.cos();
                let s = theta.sin();
                let idx0 = 2 * pair;
                let idx1 = idx0 + 1;
                let q0 = qi[idx0];
                let q1 = qi[idx1];
                qr[idx0] = q0 * c - q1 * s;
                qr[idx1] = q0 * s + q1 * c;
                let k0 = ki[idx0];
                let k1 = ki[idx1];
                kr[idx0] = k0 * c - k1 * s;
                kr[idx1] = k0 * s + k1 * c;
            }
        });

    let mut attn = vec![0.0f32; n * n];
    let mut out = vec![0.0f32; n * d_head];

    attn.par_chunks_mut(n)
        .zip(out.par_chunks_mut(d_head))
        .enumerate()
        .for_each(|(i, (a_row, out_row))| {
            let qi = &q_rot[i * d_head..(i + 1) * d_head];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                if causal && j > i {
                    a_row[j] = f32::NEG_INFINITY;
                    continue;
                }
                let kj = &k_rot[j * d_head..(j + 1) * d_head];
                // dot
                let mut dot = 0.0f32;
                for t in 0..d_head {
                    dot += qi[t] * kj[t];
                }
                let decay = -((i as f32 - j as f32).abs()) * inv_xi;
                let s = dot * scale + decay;
                a_row[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            // softmax
            let mut denom = 0.0f32;
            for j in 0..n {
                if a_row[j].is_finite() {
                    let e = (a_row[j] - max_s).exp();
                    a_row[j] = e;
                    denom += e;
                } else {
                    a_row[j] = 0.0;
                }
            }
            let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for j in 0..n {
                a_row[j] *= inv_denom;
            }
            // out = a_row @ V
            for t in 0..d_head {
                out_row[t] = 0.0;
            }
            for j in 0..n {
                let w = a_row[j];
                if w == 0.0 {
                    continue;
                }
                let vj = &v[j * d_head..(j + 1) * d_head];
                for t in 0..d_head {
                    out_row[t] += w * vj[t];
                }
            }
        });

    (out, attn)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_silu_keeps_k_largest_per_row_and_mask_matches() {
        let input = vec![1.0, -3.0, 0.5, 2.0, -0.1, 0.2, 4.0, -4.0];
        let (out, mask) = topk_silu_fwd(&input, 4, 0.5);
        assert_eq!(out.len(), 8);
        for row in 0..2 {
            let kept: usize = mask[row * 4..(row + 1) * 4].iter().map(|&m| m as usize).sum();
            assert_eq!(kept, 2, "row {row} keeps ceil(0.5 * 4) = 2 entries");
            for j in 0..4 {
                let idx = row * 4 + j;
                if mask[idx] == 1 {
                    assert!((out[idx] - silu_f32(input[idx])).abs() < 1e-6);
                } else {
                    assert_eq!(out[idx], 0.0);
                }
            }
        }
    }

    #[test]
    fn topk_silu_full_ratio_keeps_everything() {
        let input = vec![0.3, -0.7, 1.5];
        let (out, mask) = topk_silu_fwd(&input, 3, 1.0);
        assert!(mask.iter().all(|&m| m == 1));
        for (o, x) in out.iter().zip(&input) {
            assert!((o - silu_f32(*x)).abs() < 1e-6);
        }
    }

    #[test]
    fn topk_silu_bwd_is_zero_off_mask_and_matches_finite_difference_on_mask() {
        let input = vec![0.8, -1.2, 2.5, 0.1];
        let (_, mask) = topk_silu_fwd(&input, 4, 0.5);
        let grad = vec![1.0; 4];
        let gi = topk_silu_bwd(&grad, &input, &mask, 4);
        let h = 1e-3f32;
        for j in 0..4 {
            if mask[j] == 0 {
                assert_eq!(gi[j], 0.0);
            } else {
                let fd = (silu_f32(input[j] + h) - silu_f32(input[j] - h)) / (2.0 * h);
                assert!((gi[j] - fd).abs() < 1e-3, "j={j}: analytic {} vs finite difference {fd}", gi[j]);
            }
        }
    }
}
