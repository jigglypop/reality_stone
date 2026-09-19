//! Riemannian CE relaxation -- CPU reference implementation.
//!
//! Mirrors `reality_stone.clarus.ce_ops` Python fallback with exact numerical parity.
//! All arrays are flat f32 slices (row-major).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

pub struct RelaxOutput {
    pub best_m: Vec<f32>,
    pub energy: Vec<f32>,
    pub delta: Vec<f32>,
    pub e_hop: Vec<f32>,
    pub e_bias: Vec<f32>,
    pub e_portal: Vec<f32>,
    pub e_cb: Vec<f32>,
    pub bypass_hist: Vec<f32>,
    pub steps: usize,
}

pub fn pack_sparse_csr(w: &[f32], dim: usize, zero_tol: f32) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let mut values = Vec::new();
    let mut col_idx = Vec::new();
    let mut row_ptr = vec![0i32; dim + 1];
    for r in 0..dim {
        for c in 0..dim {
            let v = w[r * dim + c];
            if v.abs() > zero_tol {
                values.push(v);
                col_idx.push(c as i32);
            }
        }
        row_ptr[r + 1] = values.len() as i32;
    }
    (values, col_idx, row_ptr)
}

fn csr_spmv(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    x: &Array1<f32>,
    dim: usize,
) -> Array1<f32> {
    let mut out = Array1::zeros(dim);
    for r in 0..dim {
        let start = row_ptr[r] as usize;
        let end = row_ptr[r + 1] as usize;
        let mut acc = 0.0f32;
        for idx in start..end {
            acc += values[idx] * x[col_idx[idx] as usize];
        }
        out[r] = acc;
    }
    out
}

pub fn codebook_pull(
    m: &[f32],
    codebook: &[f32],
    n_code: usize,
    dim: usize,
    beta: f32,
    cb_w: f32,
) -> (Vec<f32>, f32) {
    if n_code == 0 {
        return (vec![0.0; dim], 0.0);
    }
    let m_arr = ArrayView1::from(m);
    let cb = ArrayView2::from_shape((n_code, dim), codebook).unwrap();

    let logits: Vec<f32> = (0..n_code).map(|i| beta * cb.row(i).dot(&m_arr)).collect();
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits
        .iter()
        .map(|&l| (l - max_l).exp())
        .collect::<Vec<_>>()
        .iter()
        .sum();
    let weights: Vec<f32> = logits
        .iter()
        .map(|&l| (l - max_l).exp() / exp_sum)
        .collect();

    let mut grad = vec![0.0f32; dim];
    for i in 0..n_code {
        for j in 0..dim {
            grad[j] -= cb_w * weights[i] * cb[[i, j]];
        }
    }
    let lse = max_l + exp_sum.ln();
    let energy = -(cb_w / beta.max(1e-6)) * lse;
    (grad, energy)
}

pub fn metric_basis_from_codebook(
    codebook: &[f32],
    m_ref: &[f32],
    n_code: usize,
    dim: usize,
    rank: usize,
) -> Vec<f32> {
    if rank == 0 || n_code == 0 {
        return Vec::new();
    }
    let m_arr = ArrayView1::from(m_ref);
    let cb = ArrayView2::from_shape((n_code, dim), codebook).unwrap();

    let logits: Vec<f32> = (0..n_code).map(|i| cb.row(i).dot(&m_arr)).collect();
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|&l| (l - max_l).exp() / exp_sum)
        .collect();

    let mut mean = vec![0.0f32; dim];
    for i in 0..n_code {
        for j in 0..dim {
            mean[j] += probs[i] * cb[[i, j]];
        }
    }

    let mut indices: Vec<usize> = (0..n_code).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let take = (rank * 4).min(n_code);
    let mut basis: Vec<Array1<f32>> = Vec::new();

    for &i in &indices[..take] {
        let mut v = Array1::zeros(dim);
        let sqrt_p = probs[i].sqrt();
        for j in 0..dim {
            v[j] = (cb[[i, j]] - mean[j]) * sqrt_p;
        }
        for b in &basis {
            let dot: f32 = v.dot(b);
            v = &v - &(b * dot);
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-6 {
            v /= norm;
            basis.push(v);
        }
        if basis.len() >= rank {
            break;
        }
    }

    let mut out = vec![0.0f32; basis.len() * dim];
    for (i, b) in basis.iter().enumerate() {
        for j in 0..dim {
            out[i * dim + j] = b[j];
        }
    }
    out
}

fn natural_direction(
    grad: &Array1<f32>,
    phi: &Array1<f32>,
    recent_var: &Array1<f32>,
    basis: &Array2<f32>,
    lambda0: f32,
    lambda_phi: f32,
    lambda_var: f32,
) -> (Array1<f32>, Array1<f32>) {
    let dim = grad.len();
    let mut diag = Array1::zeros(dim);
    for j in 0..dim {
        let d = lambda0 + lambda_phi * phi[j] * phi[j] + lambda_var * recent_var[j];
        diag[j] = d.max(1e-4);
    }
    let inv_diag: Array1<f32> = diag.mapv(|d| 1.0 / d);
    let inv_diag_grad: Array1<f32> = grad * &inv_diag;

    let r = basis.nrows();
    if r == 0 {
        return (inv_diag_grad, diag);
    }

    let weighted_basis: Array2<f32> = {
        let mut wb = basis.clone();
        for i in 0..r {
            for j in 0..dim {
                wb[[i, j]] *= inv_diag[j];
            }
        }
        wb
    };

    let mut small = Array2::<f32>::eye(r);
    small = small + basis.dot(&weighted_basis.t());

    let rhs: Array1<f32> = basis.dot(&inv_diag_grad);

    let tmp = solve_small_system(&small, &rhs);
    let correction = basis.t().dot(&tmp);
    let result = &inv_diag_grad - &(&correction * &inv_diag);
    (result, diag)
}

fn fdt_noise(
    z: &Array1<f32>,
    phi: &Array1<f32>,
    recent_var: &Array1<f32>,
    basis: &Array2<f32>,
    lambda0: f32,
    lambda_phi: f32,
    lambda_var: f32,
) -> Array1<f32> {
    let dim = z.len();
    let mut diag = Array1::zeros(dim);
    for j in 0..dim {
        let d = lambda0 + lambda_phi * phi[j] * phi[j] + lambda_var * recent_var[j];
        diag[j] = d.max(1e-4);
    }
    let inv_sqrt_diag: Array1<f32> = diag.mapv(|d| 1.0 / d.sqrt());

    let r = basis.nrows();
    if r == 0 {
        return z * &inv_sqrt_diag;
    }

    let mut q = basis.clone();
    for i in 0..r {
        for j in 0..dim {
            q[[i, j]] *= inv_sqrt_diag[j];
        }
    }

    let qqt = q.dot(&q.t());
    let (eigenvalues, eigenvectors) = symmetric_eigen(&qqt);

    let q_proj = q.t().dot(&eigenvectors);

    let mut corrected = z.clone();
    for k in 0..r {
        let factor = 1.0 - 1.0 / (1.0 + eigenvalues[k]).sqrt();
        let proj_k = q_proj.column(k).dot(z);
        for j in 0..dim {
            corrected[j] -= factor * proj_k * q_proj[[j, k]];
        }
    }

    &corrected * &inv_sqrt_diag
}

fn symmetric_eigen(a: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let n = a.nrows();
    let mut mat = a.clone();
    let mut vecs = Array2::<f32>::eye(n);

    let diag_norm: f32 = (0..n)
        .map(|i| mat[[i, i]] * mat[[i, i]])
        .sum::<f32>()
        .sqrt()
        .max(1e-30);
    let rel_tol = 1e-7 * diag_norm;
    let max_sweeps = n.max(30) * 5;

    for sweep in 0..max_sweeps {
        let mut off_diag_sq = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sq += 2.0 * mat[[i, j]] * mat[[i, j]];
            }
        }
        if off_diag_sq.sqrt() < rel_tol {
            break;
        }

        // Adaptive threshold: classical Jacobi with threshold decay
        let threshold = if sweep < 4 {
            0.2 * off_diag_sq.sqrt() / (n * n) as f32
        } else {
            0.0
        };

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[[p, q]];
                if apq.abs() < threshold {
                    continue;
                }
                let diff = mat[[q, q]] - mat[[p, p]];
                let t = if diff.abs() < 1e-30 * apq.abs() {
                    1.0_f32.copysign(apq / diff.abs().max(1e-30))
                } else {
                    let tau = diff / (2.0 * apq);
                    if tau.abs() > 1e12 {
                        1.0 / (2.0 * tau)
                    } else {
                        let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                        sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                    }
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let rho = s / (1.0 + c);

                mat[[p, p]] -= t * apq;
                mat[[q, q]] += t * apq;
                mat[[p, q]] = 0.0;
                mat[[q, p]] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let rp = mat[[r, p]];
                    let rq = mat[[r, q]];
                    mat[[r, p]] = rp - s * (rq + rho * rp);
                    mat[[p, r]] = mat[[r, p]];
                    mat[[r, q]] = rq + s * (rp - rho * rq);
                    mat[[q, r]] = mat[[r, q]];
                }

                for r in 0..n {
                    let vp = vecs[[r, p]];
                    let vq = vecs[[r, q]];
                    vecs[[r, p]] = vp - s * (vq + rho * vp);
                    vecs[[r, q]] = vq + s * (vp - rho * vq);
                }
            }
        }
    }

    let eigenvalues = mat.diag().to_owned();
    (eigenvalues, vecs)
}

fn solve_small_system(a: &Array2<f32>, b: &Array1<f32>) -> Array1<f32> {
    let n = a.nrows();
    let mut aug = Array2::<f32>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > aug[[pivot, col]].abs() {
                pivot = row;
            }
        }
        if pivot != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot, j]];
                aug[[pivot, j]] = tmp;
            }
        }
        let diag = aug[[col, col]];
        if diag.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / diag;
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }

    let mut x = Array1::<f32>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        x[i] = if diag.abs() > 1e-12 { sum / diag } else { 0.0 };
    }
    x
}

fn normalize(v: &Array1<f32>) -> Array1<f32> {
    let n = v.dot(v).sqrt();
    if n < 1e-8 {
        v.clone()
    } else {
        v / n
    }
}

fn norm(v: &Array1<f32>) -> f32 {
    v.dot(v).sqrt()
}

pub fn relax_forward(
    values: &[f32],
    col_idx: &[i32],
    row_ptr: &[i32],
    b: &[f32],
    phi: &[f32],
    m0: &[f32],
    codebook: &[f32],
    metric_basis: &[f32],
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
) -> RelaxOutput {
    let m0_arr = Array1::from(m0.to_vec());
    let scale = norm(&m0_arr).max(1.0);
    let inv_scale = 1.0 / scale;
    let mut m = &m0_arr * inv_scale;
    let b_n = Array1::from(b.to_vec()) * inv_scale;
    let phi_hat = normalize(&Array1::from(phi.to_vec()));
    let cb_n: Array2<f32> = if n_code > 0 {
        let mut c = Array2::from_shape_vec((n_code, dim), codebook.to_vec()).unwrap();
        c *= inv_scale;
        c
    } else {
        Array2::zeros((0, dim))
    };
    let basis_n: Array2<f32> = if rank > 0 {
        Array2::from_shape_vec((rank, dim), metric_basis.to_vec()).unwrap()
    } else {
        Array2::zeros((0, dim))
    };

    let tau = tau.max(1e-6);
    let dt_eff = dt.min(0.9 * tau);
    let anneal_end = (anneal_ratio * max_steps as f32).round().max(1.0) as usize;
    let t_eff = t_wake / (dim as f32).max(1.0);

    let mut m1 = m.clone();
    let mut m2 = m.clone();

    let mut rng = StdRng::seed_from_u64(seed);

    let mut hist_e = Vec::with_capacity(max_steps);
    let mut hist_delta = Vec::with_capacity(max_steps);
    let mut hist_e_hop = Vec::with_capacity(max_steps);
    let mut hist_e_bias = Vec::with_capacity(max_steps);
    let mut hist_e_portal = Vec::with_capacity(max_steps);
    let mut hist_e_cb = Vec::with_capacity(max_steps);
    let mut hist_bypass = Vec::with_capacity(max_steps);

    let mut best_m = m.clone();
    let mut best_e = f32::INFINITY;
    let mut steps_done = 0;

    for k in 0..max_steps {
        let diff1 = &m - &(2.0 * &m1) + &m2;
        let c_k = norm(&diff1);

        let w_m = csr_spmv(values, col_idx, row_ptr, &m, dim);
        let mut grad = &w_m + &b_n + &(&phi_hat * (portal + c_k * bypass));

        if n_code > 0 {
            let (cb_grad, _) = codebook_pull(
                m.as_slice().unwrap(),
                cb_n.as_slice().unwrap(),
                n_code,
                dim,
                beta,
                cb_w,
            );
            let cb_g = Array1::from(cb_grad);
            grad = &grad + &cb_g;
        }

        let diff_m_m1 = &m - &m1;
        let diff_m1_m2 = &m1 - &m2;
        let recent_var = 0.5 * (&diff_m_m1.mapv(|x| x * x) + &diff_m1_m2.mapv(|x| x * x));

        let (nat_grad, _diag) = natural_direction(
            &grad,
            &phi_hat,
            &recent_var,
            &basis_n,
            lambda0,
            lambda_phi,
            lambda_var,
        );

        let t_k = t_eff * (1.0 - k as f32 / anneal_end as f32).max(0.0);
        let noise_var = (2.0 * t_k * dt_eff / tau).max(0.0);
        let noise_std = noise_var.sqrt() * noise_scale.max(0.0);

        let noise: Array1<f32> = if noise_std > 0.0 {
            let z: Array1<f32> =
                Array1::from_iter((0..dim).map(|_| rng.sample::<f32, _>(StandardNormal)));
            let transformed = fdt_noise(
                &z,
                &phi_hat,
                &recent_var,
                &basis_n,
                lambda0,
                lambda_phi,
                lambda_var,
            );
            transformed * noise_std
        } else {
            Array1::zeros(dim)
        };

        m2 = m1.clone();
        m1 = m.clone();
        let dm = &nat_grad * (dt_eff / tau) + &noise;
        m = &m + &dm;

        let w_m_new = csr_spmv(values, col_idx, row_ptr, &m, dim);
        let e_hop = -0.5 * m.dot(&w_m_new);
        let e_bias_v = -m.dot(&b_n);
        let e_portal_v = -portal * m.dot(&phi_hat);
        let e_bypass_v = -bypass * c_k * m.dot(&phi_hat);
        let e_cb_v = if n_code > 0 {
            let logits: Vec<f32> = (0..n_code).map(|i| beta * cb_n.row(i).dot(&m)).collect();
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max_l + logits.iter().map(|&l| (l - max_l).exp()).sum::<f32>().ln();
            -(cb_w / beta.max(1e-6)) * lse
        } else {
            0.0
        };
        let e_total = e_hop + e_bias_v + e_portal_v + e_cb_v + e_bypass_v;
        let delta_v = norm(&dm);

        hist_e.push(e_total);
        hist_delta.push(delta_v);
        hist_e_hop.push(e_hop);
        hist_e_bias.push(e_bias_v);
        hist_e_portal.push(e_portal_v);
        hist_e_cb.push(e_cb_v);
        hist_bypass.push(c_k);

        if e_total < best_e {
            best_e = e_total;
            best_m = m.clone();
        }

        steps_done = k + 1;
        if k > 30 && delta_v < tol {
            break;
        }
    }

    best_m *= scale;
    RelaxOutput {
        best_m: best_m.to_vec(),
        energy: hist_e,
        delta: hist_delta,
        e_hop: hist_e_hop,
        e_bias: hist_e_bias,
        e_portal: hist_e_portal,
        e_cb: hist_e_cb,
        bypass_hist: hist_bypass,
        steps: steps_done,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_sparse_csr_keeps_only_entries_above_tolerance() {
        let w = vec![0.0, 0.5, 0.0, 1e-9, 0.0, -2.0, 3.0, 0.0, 0.0];
        let (values, col_idx, row_ptr) = pack_sparse_csr(&w, 3, 1e-6);
        assert_eq!(row_ptr, vec![0, 1, 2, 3]);
        assert_eq!(col_idx, vec![1, 2, 0]);
        assert_eq!(values, vec![0.5, -2.0, 3.0]);
    }

    #[test]
    fn pack_sparse_csr_dense_matrix_roundtrip() {
        let dim = 4;
        let w: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.25 + 0.5).collect();
        let (values, col_idx, row_ptr) = pack_sparse_csr(&w, dim, 0.0);
        assert_eq!(values.len(), dim * dim);
        let mut dense = vec![0.0f32; dim * dim];
        for r in 0..dim {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                dense[r * dim + col_idx[k] as usize] = values[k];
            }
        }
        assert_eq!(dense, w);
    }
}
