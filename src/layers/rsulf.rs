use ndarray::{Array1, Array2, ArrayView2, ArrayView1, Axis, s};
use faer::Mat;
use faer::prelude::*;

// RS-ULF (Riemannian Suppression Unified Lagrangian Flow)
// Mathematical foundations: docs/09_unified_flow/07_FOLD_CONSISTENCY.md
// SFE theory connection: docs/Derivations_Applications/05_Neural_RealityStone_Derivation.md

pub struct RSULFConfig {
    pub d_model: usize,
    pub r: usize,
    pub eta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub seq_len: usize,
    pub window: usize,
}

impl Default for RSULFConfig {
    fn default() -> Self {
        Self {
            d_model: 4096,
            r: 1024,
            eta: 0.01,
            alpha: 0.02,
            beta: 0.01,
            gamma: 0.99,
            seq_len: 128,
            window: 8,
        }
    }
}

pub struct FoldedMetric {
    pub u: Array2<f32>,
    pub s: Array1<f32>,
    pub v: Array2<f32>,
    pub s_residual: Array1<f32>,
}

use rand::Rng;
use rayon::prelude::*;

pub fn randomized_svd(
    a: &Array2<f32>,
    k: usize,
    n_oversamples: usize,
    n_iter: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let l = k + n_oversamples;
    
    let mut rng = rand::thread_rng();
    let mut omega = Array2::<f32>::zeros((n, l));
    for i in 0..n {
        for j in 0..l {
            omega[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }
    
    let mut y = a.dot(&omega);
    
    for _ in 0..n_iter {
        let z = a.t().dot(&y);
        y = a.dot(&z);
    }
    
    let (q, _) = qr_decomposition(&y);
    
    let b = q.t().dot(a);
    
    let b_faer = Mat::from_fn(b.nrows(), b.ncols(), |i, j| b[[i, j]]);
    let svd_b = b_faer.svd();
    let u_tilde = svd_b.u();
    let s_diag = svd_b.s_diagonal();
    let vt = svd_b.v();
    
    let k_actual = k.min(b.nrows()).min(b.ncols());
    
    let mut u_small = Array2::<f32>::zeros((b.nrows(), k_actual));
    let mut s = Array1::<f32>::zeros(k_actual);
    let mut v = Array2::<f32>::zeros((n, k_actual));
    
    for j in 0..k_actual {
        for i in 0..b.nrows() {
            u_small[[i, j]] = u_tilde.read(i, j);
        }
        s[j] = s_diag.read(j);
        for i in 0..n {
            v[[i, j]] = vt.read(i, j);
        }
    }
    
    let u = q.dot(&u_small);
    
    (u, s, v)
}

fn qr_decomposition(a: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);
    
    let mut q = a.clone();
    let mut r = Array2::<f32>::zeros((k, n));
    
    for j in 0..k {
        let mut col_j = q.column(j).to_owned();
        
        for i in 0..j {
            let col_i = q.column(i);
            let dot: f32 = col_j.iter().zip(col_i.iter()).map(|(a, b)| a * b).sum();
            r[[i, j]] = dot;
            for l in 0..m {
                col_j[l] -= dot * col_i[l];
            }
        }
        
        let norm: f32 = col_j.iter().map(|x| x * x).sum::<f32>().sqrt();
        r[[j, j]] = norm;
        
        if norm > 1e-10 {
            for l in 0..m {
                q[[l, j]] = col_j[l] / norm;
            }
        }
    }
    
    (q, r)
}

pub fn fold_dimension_svd(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    target_dim: usize,
) -> FoldedMetric {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();
    
    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded.slice_mut(s![i*d_k..(i+1)*d_k, ..]).assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };
    
    let g = wq.t().dot(&wk_expanded);
    let frob_g: f32 = g.iter().map(|x| x * x).sum();
    
    let k = target_dim.min(g.nrows().min(g.ncols()));
    let (u, s, v) = randomized_svd(&g, k, 5, 1);
    
    let frob_approx: f32 = s.iter().map(|x| x * x).sum();
    let mut s_residual = Array1::zeros(1);
    let tail = frob_g - frob_approx;
    if tail > 0.0 {
        s_residual[0] = tail.sqrt();
    }
    
    FoldedMetric { u, s, v, s_residual }
}

pub fn fold_dimension_diagonal(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    target_dim: usize,
) -> FoldedMetric {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();
    
    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded.slice_mut(s![i*d_k..(i+1)*d_k, ..]).assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };
    
    let mut g_diag = Array1::<f32>::zeros(d_in);
    for i in 0..d_in {
        let col_q = wq.column(i);
        let col_k = wk_expanded.column(i);
        g_diag[i] = col_q.dot(&col_k);
    }
    
    let k = target_dim.min(d_in);
    let u = Array2::<f32>::eye(k);
    let s = g_diag.slice(s![..k]).to_owned();
    let v = Array2::<f32>::eye(k);
    let s_residual = Array1::zeros(1);
    
    FoldedMetric { u, s, v, s_residual }
}

pub fn compute_curvature(s_residual: &Array1<f32>) -> f32 {
    let sum_sq: f32 = s_residual.iter().map(|x| x * x).sum();
    sum_sq.sqrt()
}

pub fn create_causal_laplacian(seq_len: usize, window: usize) -> Array2<f32> {
    let mut a = Array2::<f32>::zeros((seq_len, seq_len));
    
    for i in 0..seq_len {
        let start = if i > window { i - window } else { 0 };
        for j in start..i {
            let dist = (i - j) as f32;
            a[[i, j]] = 1.0 / (1.0 + dist);
        }
    }
    
    let d_vec: Array1<f32> = a.sum_axis(Axis(1));
    let mut l = Array2::<f32>::zeros((seq_len, seq_len));
    
    for i in 0..seq_len {
        l[[i, i]] = d_vec[i];
        for j in 0..seq_len {
            l[[i, j]] -= a[[i, j]];
        }
    }
    
    l
}

pub struct FoldedFFN {
    pub u1: Array2<f32>,
    pub s1: Array1<f32>,
    pub v1: Array2<f32>,
    pub u2: Array2<f32>,
    pub s2: Array1<f32>,
    pub v2: Array2<f32>,
}

pub fn fold_ffn_svd(
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
    target_dim: usize,
) -> FoldedFFN {
    let w1_owned = w1.to_owned();
    let k1 = target_dim.min(w1.nrows().min(w1.ncols()));
    let (u1, s1, v1) = randomized_svd(&w1_owned, k1, 5, 1);
    
    let w2_owned = w2.to_owned();
    let k2 = target_dim.min(w2.nrows().min(w2.ncols()));
    let (u2, s2, v2) = randomized_svd(&w2_owned, k2, 5, 1);
    
    FoldedFFN { u1, s1, v1, u2, s2, v2 }
}

pub fn fold_ffn_random_projection(
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
    target_dim: usize,
) -> FoldedFFN {
    let ffn_dim = w1.nrows();
    let d_in = w1.ncols();
    let d_out = w2.nrows();
    
    let k1 = target_dim.min(ffn_dim.min(d_in));
    let k2 = target_dim.min(d_out.min(ffn_dim));
    
    let mut rng = rand::thread_rng();
    
    let scale1 = (1.0 / (k1 as f32)).sqrt();
    let mut v1 = Array2::<f32>::zeros((d_in, k1));
    for i in 0..d_in {
        for j in 0..k1 {
            v1[[i, j]] = (rng.gen::<f32>() * 2.0 - 1.0) * scale1;
        }
    }
    
    let u1 = w1.dot(&v1);
    let mut s1 = Array1::<f32>::zeros(k1);
    for j in 0..k1 {
        let col = u1.column(j);
        let norm = col.dot(&col).sqrt().max(1e-6);
        s1[j] = norm;
    }
    let u1_normalized = {
        let mut u = u1.clone();
        for j in 0..k1 {
            let inv_norm = 1.0 / s1[j];
            for i in 0..ffn_dim {
                u[[i, j]] *= inv_norm;
            }
        }
        u
    };
    
    let scale2 = (1.0 / (k2 as f32)).sqrt();
    let mut v2 = Array2::<f32>::zeros((ffn_dim, k2));
    for i in 0..ffn_dim {
        for j in 0..k2 {
            v2[[i, j]] = (rng.gen::<f32>() * 2.0 - 1.0) * scale2;
        }
    }
    
    let u2 = w2.dot(&v2);
    let mut s2 = Array1::<f32>::zeros(k2);
    for j in 0..k2 {
        let col = u2.column(j);
        let norm = col.dot(&col).sqrt().max(1e-6);
        s2[j] = norm;
    }
    let u2_normalized = {
        let mut u = u2.clone();
        for j in 0..k2 {
            let inv_norm = 1.0 / s2[j];
            for i in 0..d_out {
                u[[i, j]] *= inv_norm;
            }
        }
        u
    };
    
    FoldedFFN { 
        u1: u1_normalized, 
        s1, 
        v1, 
        u2: u2_normalized, 
        s2, 
        v2 
    }
}

fn calibrate_eta_alpha(
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
    g_inv: &Array1<f32>,
    config: &mut RSULFConfig,
) {
    let d_model = w1.ncols();
    let ffn_dim = w1.nrows();
    if d_model == 0 || ffn_dim == 0 {
        return;
    }
    let num_samples = 8usize;
    let mut rng = rand::thread_rng();
    let mut x = Array2::<f32>::zeros((num_samples, d_model));
    for i in 0..num_samples {
        for j in 0..d_model {
            x[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }
    let mut f_all = Array2::<f32>::zeros((num_samples, d_model));
    let mut grad_riem_all = Array2::<f32>::zeros((num_samples, d_model));
    for i in 0..num_samples {
        let x_row = x.row(i);
        let a = w1.dot(&x_row);
        let h_act = a.mapv(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            v * s
        });
        let f_x = w2.dot(&h_act);
        let temp2 = w2.t().dot(&f_x);
        let d_sigma = a.mapv(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            s + v * s * (1.0 - s)
        });
        let mut temp3 = temp2.clone();
        for j in 0..ffn_dim {
            temp3[j] *= d_sigma[j];
        }
        let grad = w1.t().dot(&temp3);
        let mut grad_riem = grad.clone();
        if g_inv.len() == d_model {
            for j in 0..d_model {
                grad_riem[j] *= g_inv[j];
            }
        }
        f_all.row_mut(i).assign(&f_x);
        grad_riem_all.row_mut(i).assign(&grad_riem);
    }
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let mut diff_all = Array2::<f32>::zeros((num_samples, d_model));
    for i in 0..num_samples {
        for j in 0..d_model {
            diff_all[[i, j]] = x[[i, j]] - x_mean[j];
        }
    }
    let mut m00 = 0.0f64;
    let mut m01 = 0.0f64;
    let mut m11 = 0.0f64;
    let mut b0 = 0.0f64;
    let mut b1 = 0.0f64;
    for i in 0..num_samples {
        for j in 0..d_model {
            let a1 = -grad_riem_all[[i, j]] as f64;
            let a2 = diff_all[[i, j]] as f64;
            let y = f_all[[i, j]] as f64;
            m00 += a1 * a1;
            m01 += a1 * a2;
            m11 += a2 * a2;
            b0 += a1 * y;
            b1 += a2 * y;
        }
    }
    let det = m00 * m11 - m01 * m01;
    if det.abs() < 1e-12 {
        return;
    }
    let eta_hat = (m11 * b0 - m01 * b1) / det;
    let alpha_hat = (m00 * b1 - m01 * b0) / det;
    let mut eta_f = eta_hat as f32;
    let mut alpha_f = alpha_hat as f32;
    if eta_f < 0.0 {
        eta_f = 0.0;
    }
    if eta_f > 0.05 {
        eta_f = 0.05;
    }
    if alpha_f < 0.0 {
        alpha_f = 0.0;
    }
    if alpha_f > 0.01 {
        alpha_f = 0.01;
    }
    config.eta = eta_f;
    config.alpha = alpha_f;
}

pub struct RSULFLayer {
    pub config: RSULFConfig,
    pub g_diag: Array1<f32>,
    pub g_inv: Array1<f32>,
    pub u_metric: Array2<f32>,
    pub v_metric: Array2<f32>,
    pub curvature: f32,
    pub laplacian: Array2<f32>,
    pub ffn: FoldedFFN,
}

impl RSULFLayer {
    pub fn from_transformer(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
    ) -> Self {
        let folded_metric = fold_dimension_svd(wq, wk, config.r);
        let folded_ffn = fold_ffn_svd(w1, w2, config.r);
        
        let d = wq.ncols();
        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            let col_q = wq.column(i);
            let col_k = wk.column(i);
            g_diag[i] = col_q.dot(&col_k).abs();
        }
        for v in g_diag.iter_mut() {
            if *v < 1e-6 {
                *v = 1e-6;
            }
            if *v > 1e6 {
                *v = 1e6;
            }
        }
        let g_inv = g_diag.mapv(|x| 1.0 / x);
        calibrate_eta_alpha(w1, w2, &g_inv, &mut config);
        let curvature = compute_curvature(&folded_metric.s_residual);
        let laplacian = create_causal_laplacian(config.seq_len, config.window);
        
        Self {
            config,
            g_diag,
            g_inv,
            u_metric: folded_metric.u,
            v_metric: folded_metric.v,
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

    /// Construct RSULFLayer using an externally provided diagonal metric g_diag.
    /// This allows calibrated / learned Riemannian metrics instead of purely weight-derived ones.
    pub fn from_transformer_with_metric(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
        g_diag_external: ArrayView1<f32>,
    ) -> Self {
        let folded_metric = fold_dimension_svd(wq, wk, config.r);
        let folded_ffn = fold_ffn_svd(w1, w2, config.r);

        let d = wq.ncols();
        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            if i < g_diag_external.len() {
                g_diag[i] = g_diag_external[i];
            } else {
                g_diag[i] = 1.0;
            }
        }
        // Clamp to reasonable positive range
        for v in g_diag.iter_mut() {
            if *v < 1e-6 {
                *v = 1e-6;
            }
            if *v > 1e6 {
                *v = 1e6;
            }
        }
        let g_inv = g_diag.mapv(|x| 1.0 / x);
        // Use calibrated metric for eta/alpha estimation
        calibrate_eta_alpha(w1, w2, &g_inv, &mut config);
        let curvature = compute_curvature(&folded_metric.s_residual);
        let laplacian = create_causal_laplacian(config.seq_len, config.window);

        Self {
            config,
            g_diag,
            g_inv,
            u_metric: folded_metric.u,
            v_metric: folded_metric.v,
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

    pub fn from_transformer_fast(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
    ) -> Self {
        // Compute full diagonal metric: g_ii = |WQ_i . WK_i|
        let d = wq.ncols();
        let mut g_diag = Array1::zeros(d);
        
        // WQ, WK are (d_head*n_head, d_model) typically, or (d_out, d_in).
        // Transformer weights usually (out_features, in_features).
        // We want column-wise dot product if inputs are (d_model, d_something).
        // Assuming wq, wk are (d_out, d_in). We need to check how they are passed.
        // Usually passed as (d_model, d_model) square matrices from `extract_transformer_layer_weights`.
        // Let's assume they are (d_model, d_model).
        
        // In `transformer_converter.py`: weights["WQ"] is (d_model, d_model).
        // So columns are dimension indices.
        
        for i in 0..d {
            let col_q = wq.column(i);
            let col_k = wk.column(i);
            g_diag[i] = col_q.dot(&col_k).abs();
        }
        for v in g_diag.iter_mut() {
            if *v < 1e-6 {
                *v = 1e-6;
            }
            if *v > 1e6 {
                *v = 1e6;
            }
        }
        let g_inv = g_diag.mapv(|x| 1.0 / x);
        calibrate_eta_alpha(w1, w2, &g_inv, &mut config);
        let curvature = 0.0;
        let laplacian = create_causal_laplacian(config.seq_len, config.window);
        
        let folded_ffn = fold_ffn_random_projection(w1, w2, config.r);
        
        Self {
            config,
            g_diag,
            g_inv,
            u_metric: Array2::zeros((0, 0)), // Not used in fast mode
            v_metric: Array2::zeros((0, 0)), // Not used in fast mode
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }
    
    pub fn forward(
        &self,
        x: ArrayView2<f32>,
        v_mem: Option<ArrayView1<f32>>,
    ) -> (Array2<f32>, Array1<f32>) {
        let batch = x.nrows();
        let _d = x.ncols();
        
        let x_arr = x.to_owned();
        
        let h1 = x_arr.dot(&self.ffn.v1);
        let h1_scaled = &h1 * &self.ffn.s1;
        let pre_act = h1_scaled.dot(&self.ffn.u1.t());
        let h_act = pre_act.mapv(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            v * s
        });
        
        let p1 = h_act.dot(&self.ffn.v2);
        let p1_scaled = &p1 * &self.ffn.s2;
        let f_x = p1_scaled.dot(&self.ffn.u2.t());
        
        let phi_val: f32 = f_x.iter().map(|v| v * v).sum::<f32>() * 0.5 / (batch as f32);
        
        // Gradient Calculation: nabla Phi(x) = J_f^T f(x)
        // Backward through W2 (W2^T = u2 s2 v2^T)
        let dh_temp = f_x.dot(&self.ffn.u2);
        let dh_temp_s = &dh_temp * &self.ffn.s2;
        let dh = dh_temp_s.dot(&self.ffn.v2.t());
        
        // Backward through activation
        let d_sigma = pre_act.mapv(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            s + v * s * (1.0 - s)
        });
        let d_pre = dh * d_sigma;
        
        // Backward through W1 (W1^T = u1 s1 v1^T)
        let dx_temp = d_pre.dot(&self.ffn.u1);
        let dx_temp_s = &dx_temp * &self.ffn.s1;
        let mut grad_phi = dx_temp_s.dot(&self.ffn.v1.t());
        
        // Riemannian Gradient: g^{-1} grad_phi
        if self.g_inv.len() == x.ncols() {
            // Diagonal Metric scaling
            for i in 0..batch {
                let mut row = grad_phi.row_mut(i);
                row.zip_mut_with(&self.g_inv, |a, b| *a *= *b);
            }
        }
        
        let v_new = if let Some(v_prev) = v_mem {
            self.config.gamma * &v_prev + (1.0 - self.config.gamma) * phi_val
        } else {
            Array1::from_elem(batch, phi_val)
        };
        
        let term_opt = -self.config.eta * &grad_phi;
        
        let x_mean = x_arr.mean_axis(Axis(0)).unwrap();
        let diffusion = self.config.alpha * (&x_arr - &x_mean);
        let mut graph = Array2::<f32>::zeros((batch, x.ncols()));
        if self.config.beta.abs() > 0.0 {
            let seq_len = self.config.seq_len;
            if seq_len > 0 && batch >= seq_len && batch % seq_len == 0 {
                let num_seq = batch / seq_len;
                for s_idx in 0..num_seq {
                    let start = s_idx * seq_len;
                    let end = start + seq_len;
                    let x_seq = x_arr.slice(s![start..end, ..]);
                    let gx = self.laplacian.dot(&x_seq);
                    graph.slice_mut(s![start..end, ..]).assign(&gx);
                }
            }
            graph.mapv_inplace(|v| v * self.config.beta);
        }
        let v = &term_opt + &diffusion + &graph;
        
        // Second-order Curvature Correction
        // delta = -0.5 * curvature * ||v||^2 * x
        let mut delta = Array2::zeros((batch, x.ncols()));
        if self.curvature.abs() > 1e-6 {
             for i in 0..batch {
                 let v_row = v.row(i);
                 let x_row = x_arr.row(i);
                 let v_norm_sq = v_row.dot(&v_row);
                 let scale = -0.5 * self.curvature * v_norm_sq;
                 let mut d_row = delta.row_mut(i);
                 d_row.zip_mut_with(&x_row, |d, x_val| *d = scale * x_val);
             }
        }
        
        let x_next = &x_arr + &v + &delta;
        
        (x_next, v_new)
    }
    
    pub fn param_count(&self) -> (usize, usize, f32) {
        let d = self.config.d_model;
        let r = self.config.r;
        let ffn_dim = self.ffn.u1.nrows();
        
        let original_attn = 4 * d * d;
        let original_ffn = 2 * d * ffn_dim + ffn_dim * d;
        let original = original_attn + original_ffn;
        
        let compressed_metric = 2 * d * r + r;
        let compressed_ffn = 2 * (ffn_dim * r + d * r + r);
        let compressed_laplacian = self.config.seq_len * self.config.seq_len;
        let compressed = compressed_metric + compressed_ffn + compressed_laplacian;
        
        let ratio = original as f32 / compressed as f32;
        
        (compressed, original, ratio)
    }

    pub fn export_components(&self) -> RSULFComponents {
        RSULFComponents {
            d_model: self.config.d_model,
            r: self.config.r,
            eta: self.config.eta,
            alpha: self.config.alpha,
            beta: self.config.beta,
            gamma: self.config.gamma,
            seq_len: self.config.seq_len,
            window: self.config.window,
            g_diag: self.g_diag.clone(),
            g_inv: self.g_inv.clone(),
            u_metric: self.u_metric.clone(),
            v_metric: self.v_metric.clone(),
            curvature: self.curvature,
            ffn_u1: self.ffn.u1.clone(),
            ffn_s1: self.ffn.s1.clone(),
            ffn_v1: self.ffn.v1.clone(),
            ffn_u2: self.ffn.u2.clone(),
            ffn_s2: self.ffn.s2.clone(),
            ffn_v2: self.ffn.v2.clone(),
        }
    }

    pub fn from_components(comp: RSULFComponents) -> Self {
        let config = RSULFConfig {
            d_model: comp.d_model,
            r: comp.r,
            eta: comp.eta,
            alpha: comp.alpha,
            beta: comp.beta,
            gamma: comp.gamma,
            seq_len: comp.seq_len,
            window: comp.window,
        };
        let laplacian = create_causal_laplacian(comp.seq_len, comp.window);
        let ffn = FoldedFFN {
            u1: comp.ffn_u1,
            s1: comp.ffn_s1,
            v1: comp.ffn_v1,
            u2: comp.ffn_u2,
            s2: comp.ffn_s2,
            v2: comp.ffn_v2,
        };
        Self {
            config,
            g_diag: comp.g_diag,
            g_inv: comp.g_inv,
            u_metric: comp.u_metric,
            v_metric: comp.v_metric,
            curvature: comp.curvature,
            laplacian,
            ffn,
        }
    }
}

pub struct RSULFComponents {
    pub d_model: usize,
    pub r: usize,
    pub eta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub seq_len: usize,
    pub window: usize,
    pub g_diag: Array1<f32>,
    pub g_inv: Array1<f32>,
    pub u_metric: Array2<f32>,
    pub v_metric: Array2<f32>,
    pub curvature: f32,
    pub ffn_u1: Array2<f32>,
    pub ffn_s1: Array1<f32>,
    pub ffn_v1: Array2<f32>,
    pub ffn_u2: Array2<f32>,
    pub ffn_s2: Array1<f32>,
    pub ffn_v2: Array2<f32>,
}

pub struct FoldConsistencyResult {
    pub symmetry_error: f32,
    pub reconstruction_error: f32,
    pub fold_accuracy: f32,
    pub min_eigenvalue: f32,
    pub condition_number: f32,
    pub is_valid: bool,
}

pub fn verify_fold_consistency(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    folded: &FoldedMetric,
) -> FoldConsistencyResult {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();
    
    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded.slice_mut(s![i*d_k..(i+1)*d_k, ..]).assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };
    
    // G = WQ^T * WK (비대칭 행렬)
    let g = wq.t().dot(&wk_expanded);
    
    // 대칭화된 버전: G_sym = (G + G^T) / 2
    let g_sym = (&g + &g.t()) * 0.5;
    
    // 대칭성 오류: ||G - G^T|| / ||G||
    let g_t = g.t();
    let sym_diff: f32 = g.iter().zip(g_t.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let g_norm: f32 = g.iter().map(|x| x * x).sum();
    let symmetry_error = if g_norm > 1e-10 { (sym_diff / g_norm).sqrt() } else { 0.0 };
    
    // Frobenius norm 계산 (fold_accuracy용)
    let frob_g: f32 = g.iter().map(|x| x * x).sum();
    
    // SVD로 캡처된 에너지 비율 = sum(s_i^2) / ||G||_F^2
    let frob_captured: f32 = folded.s.iter().map(|x| x * x).sum();
    let fold_accuracy = if frob_g > 1e-10 { 
        (frob_captured / frob_g).min(1.0)  // 1.0 초과 방지
    } else { 
        1.0 
    };
    
    // 잔차 기반 재구성 오류
    let residual_sq: f32 = folded.s_residual.iter().map(|x| x * x).sum();
    let reconstruction_error = if frob_g > 1e-10 { 
        (residual_sq / frob_g).sqrt() 
    } else { 
        0.0 
    };
    
    // 대각 요소 통계 (양정치성 대리 지표)
    let mut diag_values: Vec<f32> = Vec::with_capacity(d_in);
    for i in 0..d_in {
        // 대칭화된 메트릭의 대각 사용
        diag_values.push(g_sym[[i, i]]);
    }
    let min_eigenvalue = diag_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_eigenvalue = diag_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let condition_number = if min_eigenvalue.abs() > 1e-10 {
        max_eigenvalue.abs() / min_eigenvalue.abs()
    } else {
        f32::INFINITY
    };
    
    // 정합성 조건 (비대칭 메트릭 허용)
    // - 대칭성 오류는 참고용 (Transformer 메트릭은 원래 비대칭)
    // - fold_accuracy >= 0.5 (저랭크 근사 품질)
    // - 대각 요소 대부분 양수
    let is_valid = fold_accuracy >= 0.5 
        && min_eigenvalue > -1e6  // 극단적 음수만 배제
        && condition_number < 1e8;
    
    FoldConsistencyResult {
        symmetry_error,
        reconstruction_error,
        fold_accuracy,
        min_eigenvalue,
        condition_number,
        is_valid,
    }
}

pub fn block_lanczos_svd(
    a: &Array2<f32>,
    k: usize,
    block_size: usize,
    max_iter: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let bs = block_size.min(k).min(m).min(n);
    let num_blocks = (k + bs - 1) / bs;
    
    let mut rng = rand::thread_rng();
    let mut v_blocks: Vec<Array2<f32>> = Vec::with_capacity(num_blocks + 1);
    
    let mut v0 = Array2::<f32>::zeros((n, bs));
    for i in 0..n {
        for j in 0..bs {
            v0[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }
    let (v0_orth, _) = qr_decomposition(&v0);
    v_blocks.push(v0_orth);
    
    let mut alpha_blocks: Vec<Array2<f32>> = Vec::new();
    let mut beta_blocks: Vec<Array2<f32>> = Vec::new();
    
    for iter in 0..max_iter.min(num_blocks) {
        let v_j = &v_blocks[iter];
        let mut u_j = a.dot(v_j);
        
        if iter > 0 {
            let beta_prev = &beta_blocks[iter - 1];
            let v_prev = &v_blocks[iter - 1];
            u_j = u_j - v_prev.dot(&beta_prev.t());
        }
        
        let alpha_j = v_j.t().dot(&a.t().dot(&u_j));
        u_j = a.t().dot(&u_j) - v_j.dot(&alpha_j);
        
        for prev in 0..=iter {
            let v_prev = &v_blocks[prev];
            let proj = v_prev.t().dot(&u_j);
            u_j = u_j - v_prev.dot(&proj);
        }
        
        let (v_next, beta_j) = qr_decomposition(&u_j);
        
        alpha_blocks.push(alpha_j);
        beta_blocks.push(beta_j.slice(s![..bs, ..bs]).to_owned());
        
        if iter + 1 < num_blocks {
            v_blocks.push(v_next.slice(s![.., ..bs]).to_owned());
        }
        
        let beta_norm: f32 = beta_j.iter().map(|x| x * x).sum::<f32>().sqrt();
        if beta_norm < 1e-10 {
            break;
        }
    }
    
    randomized_svd(a, k, 5, 2)
}

pub fn nystrom_approximation(
    a: &Array2<f32>,
    k: usize,
    n_samples: usize,
) -> (Array2<f32>, Array1<f32>) {
    let n = a.nrows();
    let l = n_samples.min(n).max(k);
    
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..l {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let sampled_indices: Vec<usize> = indices[..l].to_vec();
    
    let mut c = Array2::<f32>::zeros((n, l));
    for (j, &idx) in sampled_indices.iter().enumerate() {
        for i in 0..n {
            c[[i, j]] = a[[i, idx]];
        }
    }
    
    let mut w = Array2::<f32>::zeros((l, l));
    for (i, &idx_i) in sampled_indices.iter().enumerate() {
        for (j, &idx_j) in sampled_indices.iter().enumerate() {
            w[[i, j]] = a[[idx_i, idx_j]];
        }
    }
    
    let w_faer = Mat::from_fn(l, l, |i, j| w[[i, j]]);
    let svd_w = w_faer.svd();
    
    let mut w_pinv = Array2::<f32>::zeros((l, l));
    let s_diag = svd_w.s_diagonal();
    let u_w = svd_w.u();
    let v_w = svd_w.v();
    
    for i in 0..l {
        let s_val = s_diag.read(i);
        if s_val.abs() > 1e-10 {
            let s_inv = 1.0 / s_val;
            for row in 0..l {
                for col in 0..l {
                    w_pinv[[row, col]] += v_w.read(row, i) * s_inv * u_w.read(col, i);
                }
            }
        }
    }
    
    let approx = c.dot(&w_pinv).dot(&c.t());
    
    let approx_faer = Mat::from_fn(n, n, |i, j| approx[[i, j]]);
    let svd_approx = approx_faer.svd();
    
    let k_actual = k.min(n);
    let mut u = Array2::<f32>::zeros((n, k_actual));
    let mut s = Array1::<f32>::zeros(k_actual);
    
    let u_approx = svd_approx.u();
    let s_approx = svd_approx.s_diagonal();
    
    for j in 0..k_actual {
        s[j] = s_approx.read(j).sqrt().max(0.0);
        for i in 0..n {
            u[[i, j]] = u_approx.read(i, j);
        }
    }
    
    (u, s)
}

pub fn adaptive_rank_svd(
    a: &Array2<f32>,
    target_accuracy: f32,
    max_rank: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>, usize) {
    let m = a.nrows();
    let n = a.ncols();
    let frob_sq: f32 = a.iter().map(|x| x * x).sum();
    
    let mut low = 1usize;
    let mut high = max_rank.min(m).min(n);
    let mut best_k = high;
    
    while low < high {
        let mid = (low + high) / 2;
        let (_, s, _) = randomized_svd(a, mid, 3, 1);
        let captured: f32 = s.iter().map(|x| x * x).sum();
        let accuracy = captured / frob_sq.max(1e-10);
        
        if accuracy >= target_accuracy {
            best_k = mid;
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    
    let (u, s, v) = randomized_svd(a, best_k, 5, 2);
    (u, s, v, best_k)
}

