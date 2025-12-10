use faer::Mat;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

pub struct RSULFConfig {
    pub d_model: usize,
    pub r: usize,
    pub eta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub seq_len: usize,
    pub window: usize,
    pub calibration_samples: usize,
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
            calibration_samples: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalBasis {
    pub u: Array2<f32>,
    pub rank: usize,
}

pub fn extract_global_basis(
    layers_wq: &[ArrayView2<f32>],
    layers_wk: &[ArrayView2<f32>],
    target_rank: usize,
) -> GlobalBasis {
    let num_layers = layers_wq.len();
    if num_layers == 0 {
        return GlobalBasis {
            u: Array2::zeros((0, 0)),
            rank: 0,
        };
    }

    let d_model = layers_wq[0].ncols();
    // Reservoir sampling or aggregate covariance
    // For simplicity and memory efficiency, we accumulate covariance matrix
    // G_total = sum_l (W_Q^l)^T (W_K^l)

    let mut g_acc = Array2::<f32>::zeros((d_model, d_model));

    for (wq, wk) in layers_wq.iter().zip(layers_wk.iter()) {
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d_model));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        // G = WQ^T * WK
        // We want the basis that explains the interaction.
        // Approximate by summing G * G^T or just G.
        // Let's use the sum of singular vectors logic:
        // Or simpler: Aggregate G and find its SVD.
        // But G is d_model x d_model. Summing them is valid.

        let g = wq.t().dot(&wk_expanded);
        // Symmetrize contribution
        let g_sym = (&g + &g.t()) * 0.5;
        g_acc = g_acc + g_sym;
    }

    // Perform Randomized SVD on the accumulated Metric
    // This extracts the "Shared Global Basis" U
    let k = target_rank.min(d_model);
    let (u, _, _) = randomized_svd(&g_acc, k, 20, 5);

    GlobalBasis { u, rank: k }
}

pub struct FoldedMetric {
    pub u: Array2<f32>,
    pub s: Array1<f32>,
    pub v: Array2<f32>,
    pub s_residual: Array1<f32>,
}

use rand::Rng;

pub fn randomized_svd(
    a: &Array2<f32>,
    k: usize,
    _n_oversamples: usize,
    _n_iter: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();

    // Use faer for high-performance SVD
    let mat = Mat::from_fn(m, n, |i, j| a[[i, j]]);
    let svd = mat.svd();

    let u_faer = svd.u();
    let s_faer = svd.s_diagonal();
    let v_faer = svd.v();

    let k_actual = k.min(m).min(n).min(s_faer.nrows());

    let mut u = Array2::<f32>::zeros((m, k_actual));
    let mut s = Array1::<f32>::zeros(k_actual);
    let mut v = Array2::<f32>::zeros((n, k_actual));

    for j in 0..k_actual {
        s[j] = s_faer.read(j);
        for i in 0..m {
            u[[i, j]] = u_faer.read(i, j);
        }
        for i in 0..n {
            v[[i, j]] = v_faer.read(i, j);
        }
    }

    (u, s, v)
}

fn qr_decomposition(a: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let mat = Mat::from_fn(m, n, |i, j| a[[i, j]]);
    let qr = mat.qr();

    let q_faer = qr.compute_q();
    let r_faer = qr.compute_r();

    let mut q = Array2::<f32>::zeros((m, m.min(n)));
    let mut r = Array2::<f32>::zeros((m.min(n), n));

    let k = m.min(n);

    for j in 0..k {
        for i in 0..m {
            q[[i, j]] = q_faer.read(i, j);
        }
    }

    for j in 0..n {
        for i in 0..k {
            if i <= j {
                r[[i, j]] = r_faer.read(i, j);
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
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };

    let g = wq.t().dot(&wk_expanded);
    let frob_g: f32 = g.iter().map(|x| x * x).sum();

    let k = target_dim.min(g.nrows().min(g.ncols()));
    let oversamples = k.min(20);
    let n_iter = if k < 32 { 3 } else { 2 };
    let (u, s, v) = randomized_svd(&g, k, oversamples, n_iter);

    let frob_approx: f32 = s.iter().map(|x| x * x).sum();
    let mut s_residual = Array1::zeros(1);
    let tail = frob_g - frob_approx;
    if tail > 0.0 {
        s_residual[0] = tail.sqrt();
    }

    FoldedMetric {
        u,
        s,
        v,
        s_residual,
    }
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
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
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

    FoldedMetric {
        u,
        s,
        v,
        s_residual,
    }
}

pub fn fold_with_global_basis(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    global_basis: &GlobalBasis,
) -> FoldedMetric {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();

    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };

    let g = wq.t().dot(&wk_expanded);
    let u = global_basis.u.clone();
    let k = global_basis.rank;
    let g_sym = (&g + &g.t()) * 0.5;
    let g_core = u.t().dot(&g_sym).dot(&u);
    let g_approx = u.dot(&g_core).dot(&u.t());
    let diff = &g_sym - &g_approx;
    let residual_energy: f32 = diff.iter().map(|x| x * x).sum();
    let mut s_residual = Array1::zeros(1);
    s_residual[0] = residual_energy.sqrt();
    let g_core_faer = Mat::from_fn(k, k, |i, j| g_core[[i, j]]);
    let svd_core = g_core_faer.svd();
    let s_diag = svd_core.s_diagonal();
    let mut s = Array1::<f32>::zeros(k);
    for i in 0..k {
        s[i] = s_diag.read(i);
    }

    FoldedMetric {
        u: u.clone(),
        s,
        v: u,
        s_residual,
    }
}

pub fn compute_curvature(s_residual: &Array1<f32>) -> f32 {
    let sum_sq: f32 = s_residual.iter().map(|x| x * x).sum();
    sum_sq.sqrt()
}

#[derive(Debug, Clone)]
pub enum LayerType {
    Attention,
    FFN,
    Embedding,
    LMHead,
    LayerNorm,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    MetricSVD {
        target_rank: usize,
        expected_accuracy: f32,
    },
    DiagonalMetric,
    FFNFold {
        target_rank: usize,
    },
    NoCompression,
    Skip,
}

#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    pub layer_idx: usize,
    pub layer_type: LayerType,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize),
    pub param_count: usize,
    pub spectral_decay: f32,
    pub condition_number: f32,
    pub recommended_rank: usize,
    pub expected_accuracy: f32,
    pub strategy: CompressionStrategy,
}

#[derive(Debug, Clone)]
pub struct CompressionPlan {
    pub layers: Vec<LayerAnalysis>,
    pub total_original_params: usize,
    pub total_compressed_params: usize,
    pub expected_compression_ratio: f32,
    pub min_expected_accuracy: f32,
}

pub fn analyze_weight_matrix(w: ArrayView2<f32>, max_rank: usize) -> (f32, f32, usize, f32) {
    let m = w.nrows();
    let n = w.ncols();
    let k = max_rank.min(m.min(n));

    let w_faer = Mat::from_fn(m, n, |i, j| w[[i, j]]);
    let svd = w_faer.svd();
    let s_diag = svd.s_diagonal();

    let s_len = s_diag.nrows().min(k);
    let mut singular_values = Vec::with_capacity(s_len);
    for i in 0..s_len {
        singular_values.push(s_diag.read(i));
    }

    let s_max = singular_values.first().copied().unwrap_or(1.0).max(1e-10);
    let s_min = singular_values.last().copied().unwrap_or(1e-10).max(1e-10);
    let condition_number = s_max / s_min;

    let total_energy: f32 = singular_values.iter().map(|x| x * x).sum();
    let mut cumulative = 0.0_f32;
    let mut recommended_rank = s_len;
    let threshold = 0.95;

    for (i, &s) in singular_values.iter().enumerate() {
        cumulative += s * s;
        if cumulative / total_energy.max(1e-10) >= threshold {
            recommended_rank = i + 1;
            break;
        }
    }

    let spectral_decay = if singular_values.len() > 1 {
        let first_half: f32 = singular_values[..s_len / 2].iter().map(|x| x * x).sum();
        let second_half: f32 = singular_values[s_len / 2..].iter().map(|x| x * x).sum();
        first_half / (first_half + second_half).max(1e-10)
    } else {
        1.0
    };

    let mut approx_energy = 0.0_f32;
    for i in 0..recommended_rank.min(singular_values.len()) {
        approx_energy += singular_values[i] * singular_values[i];
    }
    let expected_accuracy = (approx_energy / total_energy.max(1e-10)).sqrt();

    (
        spectral_decay,
        condition_number,
        recommended_rank,
        expected_accuracy,
    )
}

pub fn analyze_layer(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    w1: ArrayView2<f32>,
    _: ArrayView2<f32>,
    layer_idx: usize,
    target_rank: usize,
) -> LayerAnalysis {
    let d_model = wq.ncols();
    let d_head = wq.nrows();
    let d_ff = w1.nrows();

    let g = wq.t().dot(&wk);
    let (spectral_decay, condition_number, rec_rank_metric, acc_metric) =
        analyze_weight_matrix(g.view(), target_rank);

    let (_, _, rec_rank_ffn, acc_ffn) = analyze_weight_matrix(w1, target_rank);

    let recommended_rank = rec_rank_metric.max(rec_rank_ffn).min(target_rank);
    let expected_accuracy = acc_metric.min(acc_ffn);

    let strategy = if spectral_decay > 0.9 && condition_number < 1e4 {
        CompressionStrategy::MetricSVD {
            target_rank: recommended_rank,
            expected_accuracy,
        }
    } else if spectral_decay > 0.7 {
        CompressionStrategy::DiagonalMetric
    } else {
        CompressionStrategy::MetricSVD {
            target_rank: (recommended_rank * 2).min(d_model),
            expected_accuracy,
        }
    };

    let original_params = d_head * d_model * 2 + d_ff * d_model * 2;

    LayerAnalysis {
        layer_idx,
        layer_type: LayerType::Attention,
        input_shape: (d_model, d_model),
        output_shape: (d_model, d_model),
        param_count: original_params,
        spectral_decay,
        condition_number,
        recommended_rank,
        expected_accuracy,
        strategy,
    }
}

pub fn create_compression_plan(layer_analyses: Vec<LayerAnalysis>, _: f32) -> CompressionPlan {
    let total_original: usize = layer_analyses.iter().map(|a| a.param_count).sum();

    let mut total_compressed = 0_usize;
    let mut min_accuracy = 1.0_f32;

    for analysis in &layer_analyses {
        let compressed = match &analysis.strategy {
            CompressionStrategy::MetricSVD { target_rank, .. } => {
                let d = analysis.input_shape.0;
                target_rank * d * 2 + target_rank
            }
            CompressionStrategy::DiagonalMetric => analysis.input_shape.0,
            CompressionStrategy::FFNFold { target_rank } => {
                let d = analysis.input_shape.0;
                target_rank * d * 2 + target_rank
            }
            CompressionStrategy::NoCompression => analysis.param_count,
            CompressionStrategy::Skip => 0,
        };
        total_compressed += compressed;
        if analysis.expected_accuracy < min_accuracy {
            min_accuracy = analysis.expected_accuracy;
        }
    }

    let ratio = total_original as f32 / total_compressed.max(1) as f32;

    CompressionPlan {
        layers: layer_analyses,
        total_original_params: total_original,
        total_compressed_params: total_compressed,
        expected_compression_ratio: ratio,
        min_expected_accuracy: min_accuracy,
    }
}

pub fn verify_compression_plan(plan: &CompressionPlan, min_accuracy: f32) -> Result<(), String> {
    if plan.min_expected_accuracy < min_accuracy {
        return Err(format!(
            "expected_accuracy {} < threshold {}",
            plan.min_expected_accuracy, min_accuracy
        ));
    }

    for layer in &plan.layers {
        if layer.expected_accuracy < min_accuracy {
            return Err(format!(
                "layer {} expected_accuracy {} < threshold {}",
                layer.layer_idx, layer.expected_accuracy, min_accuracy
            ));
        }

        if layer.condition_number > 1e8 {
            return Err(format!(
                "layer {} condition_number {} too high",
                layer.layer_idx, layer.condition_number
            ));
        }
    }

    Ok(())
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

pub fn ffn_force_and_grad_row(
    x: Array1<f32>,
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let a = w1.dot(&x);
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
    for j in 0..d_sigma.len() {
        temp3[j] *= d_sigma[j];
    }
    let grad = w1.t().dot(&temp3);
    (f_x, grad)
}

pub fn fold_ffn_svd(w1: ArrayView2<f32>, w2: ArrayView2<f32>, target_dim: usize) -> FoldedFFN {
    let w1_owned = w1.to_owned();
    let k1 = target_dim.min(w1.nrows().min(w1.ncols()));
    let (u1, s1, v1) = randomized_svd(&w1_owned, k1, 5, 1);

    let w2_owned = w2.to_owned();
    let k2 = target_dim.min(w2.nrows().min(w2.ncols()));
    let (u2, s2, v2) = randomized_svd(&w2_owned, k2, 5, 1);

    FoldedFFN {
        u1,
        s1,
        v1,
        u2,
        s2,
        v2,
    }
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
        v2,
    }
}

fn calibrate_eta_alpha(
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
    g_inv: &Array1<f32>,
    config: &mut RSULFConfig,
) {
    // Use d_model from config, but verify against weights
    let d_model = config.d_model;

    // Check W1 dimensions: should be (ffn_dim, d_model)
    if w1.ncols() != d_model {
        // If mismatch, try to detect if transposed (d_model, ffn_dim)
        if w1.nrows() == d_model {
            // Warn or handle? For now, we stick to the contract that W1 is (ffn_dim, d_model).
            // But to avoid panic, we should return or panic with clear message.
            panic!("RS-ULF W1 shape mismatch: expected ncols={} (d_model), got ncols={}. Ensure W1 is (hidden_dim, d_model).", d_model, w1.ncols());
        } else {
            panic!(
                "RS-ULF W1 shape mismatch: expected ncols={} (d_model), got ncols={}.",
                d_model,
                w1.ncols()
            );
        }
    }

    let ffn_dim = w1.nrows();
    if ffn_dim == 0 {
        return;
    }
    let num_samples = config.calibration_samples.max(64).min(256);
    let mut rng = rand::thread_rng();
    let mut x = Array2::<f32>::zeros((num_samples, d_model));
    for i in 0..num_samples {
        for j in 0..d_model {
            x[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }

    let results: Vec<_> = (0..num_samples)
        .into_par_iter()
        .map(|i| {
            let x_row = x.row(i);
            let (f_x, grad) = ffn_force_and_grad_row(x_row.to_owned(), w1.view(), w2.view());
            let mut grad_riem = grad.clone();
            if g_inv.len() == d_model {
                for j in 0..d_model {
                    grad_riem[j] *= g_inv[j];
                }
            }
            (f_x, grad_riem)
        })
        .collect();

    let mut f_all = Array2::<f32>::zeros((num_samples, d_model));
    let mut grad_riem_all = Array2::<f32>::zeros((num_samples, d_model));

    for (i, (f, g)) in results.into_iter().enumerate() {
        f_all.row_mut(i).assign(&f);
        grad_riem_all.row_mut(i).assign(&g);
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
    if eta_f < 0.5 {
        eta_f = 1.0;
    }
    if eta_f > 2.0 {
        eta_f = 1.0;
    }
    if alpha_f < 0.0 {
        alpha_f = 0.0;
    }
    if alpha_f > 0.1 {
        alpha_f = 0.1;
    }
    config.eta = eta_f;
    config.alpha = alpha_f;
}

pub struct RSULFLayer {
    pub config: RSULFConfig,
    pub g_diag: Array1<f32>,
    pub g_inv: Array1<f32>,
    pub g_sym: Array2<f32>,
    pub a_antisym: Array2<f32>,
    pub u_metric: Array2<f32>,
    pub v_metric: Array2<f32>,
    pub g_core: Array2<f32>,
    pub a_core: Array2<f32>,
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

        let d_q = wq.nrows();
        let d_k = wk.nrows();
        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        let u_metric = folded_metric.u.clone();
        let g_core = u_metric.t().dot(&g_sym).dot(&u_metric);
        let a_core = u_metric.t().dot(&a_antisym).dot(&u_metric);

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            g_diag[i] = g_sym[[i, i]].abs();
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
            g_sym,
            a_antisym,
            u_metric,
            v_metric: folded_metric.v,
            g_core,
            a_core,
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

    pub fn from_transformer_with_basis(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
        global_basis: &GlobalBasis,
    ) -> Self {
        // Use Global Basis for folding
        let folded_metric = fold_with_global_basis(wq, wk, global_basis);

        // FFN folding can also be optimized, but for now we keep local SVD or implement Global FFN Basis later.
        // The blueprint focuses on Metric Basis sharing.
        let folded_ffn = fold_ffn_svd(w1, w2, config.r);

        let d = wq.ncols();
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        // Use the Global U
        let u_metric = folded_metric.u.clone();
        let g_core = u_metric.t().dot(&g_sym).dot(&u_metric);
        let a_core = u_metric.t().dot(&a_antisym).dot(&u_metric);

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            g_diag[i] = g_sym[[i, i]].abs();
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
            g_sym,
            a_antisym,
            u_metric,
            v_metric: folded_metric.v, // Same as u_metric in this mode
            g_core,
            a_core,
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

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
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        let u_metric = folded_metric.u.clone();
        let g_core = u_metric.t().dot(&g_sym).dot(&u_metric);
        let a_core = u_metric.t().dot(&a_antisym).dot(&u_metric);

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            if i < g_diag_external.len() {
                g_diag[i] = g_diag_external[i];
            } else {
                g_diag[i] = 1.0;
            }
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
            g_sym,
            a_antisym,
            u_metric,
            v_metric: folded_metric.v,
            g_core,
            a_core,
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
        let d = wq.ncols();
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            g_diag[i] = g_sym[[i, i]].abs();
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
            g_sym,
            a_antisym,
            u_metric: Array2::zeros((0, 0)),
            v_metric: Array2::zeros((0, 0)),
            g_core: Array2::zeros((0, 0)),
            a_core: Array2::zeros((0, 0)),
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
        let batch_total = x.nrows();
        let d = x.ncols();

        let x_arr = x.to_owned();

        // 1. Attention (Metric) Step
        let mut attn_out = Array2::<f32>::zeros((batch_total, d));

        // Only apply if metric matrices are valid
        if self.g_sym.nrows() == d && self.a_antisym.nrows() == d {
            // In RS-ULF, Attention is modeled as Geodesic flow on the manifold defined by G.
            // The original code implemented a full quadratic attention.
            // Ideally, this should use the folded core for efficiency, but for exactness (blueprint),
            // it uses the reconstructed G (or G_sym) in the expanded space.

            // Note: O(N^2) naive attention implementation.
            // For production, this should be block-wise or linear attention.

            let scale = 1.0 / (d as f32).sqrt();

            // We need to handle batch/sequence structure for Attention masking.
            // Assumption: Input is flattened [Batch * SeqLen, D].
            // Attention should only happen within each sequence.

            let mut seq_len_cfg = self.config.seq_len;
            if seq_len_cfg == 0 || seq_len_cfg > batch_total {
                seq_len_cfg = batch_total;
            }
            let num_seq = if seq_len_cfg > 0 {
                (batch_total + seq_len_cfg - 1) / seq_len_cfg
            } else {
                1
            };

            for s_idx in 0..num_seq {
                let start_row = s_idx * seq_len_cfg;
                let end_row = (start_row + seq_len_cfg).min(batch_total);
                let current_len = end_row.saturating_sub(start_row);
                if current_len == 0 {
                    continue;
                }
                let mut attn_weights = Array2::<f32>::zeros((current_len, current_len));
                for i in 0..current_len {
                    let global_i = start_row + i;
                    let q_i = x_arr.row(global_i);
                    let mut max_val = f32::NEG_INFINITY;

                    for j in 0..=i {
                        let global_j = start_row + j;
                        let k_j = x_arr.row(global_j);
                        let mut score = 0.0_f32;

                        // score = x_i^T * G * x_j
                        // Optimized: pre-calculate G*x_j could be faster but O(N^2) dominates.
                        for m in 0..d {
                            // Using g_sym which captures the metric
                            for n in 0..d {
                                score += q_i[m] * self.g_sym[[m, n]] * k_j[n];
                            }
                        }
                        score *= scale;
                        if score > max_val {
                            max_val = score;
                        }
                        attn_weights[[i, j]] = score;
                    }

                    let mut sum_exp = 0.0_f32;
                    for j in 0..=i {
                        let w = (attn_weights[[i, j]] - max_val).exp();
                        attn_weights[[i, j]] = w;
                        sum_exp += w;
                    }

                    if sum_exp > 1e-10 {
                        let inv_sum = 1.0 / sum_exp;
                        for j in 0..=i {
                            attn_weights[[i, j]] *= inv_sum;
                        }
                    }
                }

                for i in 0..current_len {
                    let global_i = start_row + i;
                    for j in 0..=i {
                        let w = attn_weights[[i, j]];
                        if w.abs() > 1e-10 {
                            let global_j = start_row + j;
                            let x_j = x_arr.row(global_j);
                            for k in 0..d {
                                attn_out[[global_i, k]] += w * x_j[k];
                            }
                        }
                    }
                }
            }

            // Magnetic effect (Gauge field)
            let a_norm: f32 = self.a_antisym.iter().map(|v| v * v).sum::<f32>().sqrt();
            if a_norm > 1e-6 {
                let magnetic = x_arr.dot(&self.a_antisym);
                // Physical scaling for gauge force
                let mag_scale = self.config.alpha / a_norm.max(1.0);
                attn_out = &attn_out + &magnetic * mag_scale;
            }
        } else {
            attn_out = x_arr.clone();
        }

        // v_attn = Attention_Output - Input (Residual velocity)
        let v_attn = &attn_out - &x_arr;

        // 2. FFN (Potential) Step
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

        let mut v_ffn = f_x.clone();

        // Apply Riemannian Gradient correction: G^-1 * grad(Phi)
        if self.g_inv.len() == d {
            let g_inv_mean: f32 = self.g_inv.iter().sum::<f32>() / d as f32;
            // Clip extreme metric scaling to avoid instability
            let g_inv_scale = if g_inv_mean > 10.0 {
                1.0 / g_inv_mean
            } else {
                1.0
            };
            for i in 0..batch_total {
                let mut row = v_ffn.row_mut(i);
                row.zip_mut_with(&self.g_inv, |a, b| *a *= *b * g_inv_scale);
            }
        }

        // Potential Energy monitoring
        let phi_val: f32 = f_x.iter().map(|v| v * v).sum::<f32>() * 0.5 / (batch_total as f32);
        let v_new = if let Some(v_prev) = v_mem {
            self.config.gamma * &v_prev + (1.0 - self.config.gamma) * phi_val
        } else {
            Array1::from_elem(batch_total, phi_val)
        };

        // 3. Graph Diffusion Step
        let mut graph = Array2::<f32>::zeros((batch_total, d));
        if self.config.beta.abs() > 0.0 {
            let seq_len = self.config.seq_len;
            // Only apply if dimensions match sequence structure
            if seq_len > 0 && batch_total >= seq_len && batch_total % seq_len == 0 {
                let num_seq = batch_total / seq_len;
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

        let mut v_total = &v_attn + self.config.eta * &v_ffn + &graph;
        let mut max_vel = 0.0_f32;
        for val in v_total.iter() {
            let a = val.abs();
            if a > max_vel {
                max_vel = a;
            }
        }
        if max_vel > 5.0 {
            let scale = 5.0 / max_vel;
            v_total.mapv_inplace(|val| val * scale);
        }

        let mut v_norm_global: f32 = 0.0;
        if batch_total > 0 {
            v_norm_global =
                v_total.iter().map(|v| v * v).sum::<f32>().sqrt() / (batch_total as f32).sqrt();
        }
        let curvature_norm = self.curvature.abs();
        let mut step_scale = 1.0_f32;
        if curvature_norm > 0.0 && v_norm_global > 0.0 {
            let denom = 1.0 + curvature_norm * v_norm_global;
            if denom.is_finite() && denom > 0.0 {
                step_scale = 1.0 / denom;
            }
        }
        if step_scale < 1.0 {
            v_total.mapv_inplace(|val| val * step_scale);
        }

        let mut christoffel = Array2::zeros((batch_total, d));
        let curv_scale = curvature_norm.min(1.0);

        if curv_scale > 1e-8 {
            let mut v_norm_global_scaled: f32 = 0.0;
            if batch_total > 0 {
                v_norm_global_scaled =
                    v_total.iter().map(|v| v * v).sum::<f32>().sqrt() / (batch_total as f32).sqrt();
            }
            let stability_scale = if v_norm_global_scaled > 1.0 {
                1.0 / v_norm_global_scaled
            } else {
                1.0
            };

            for i in 0..batch_total {
                let v_row = v_total.row(i);
                let x_row = x_arr.row(i);

                let v_norm_sq = v_row.dot(&v_row) * stability_scale * stability_scale;
                let scale = -0.5 * curv_scale * v_norm_sq;

                for k in 0..d {
                    christoffel[[i, k]] = scale * x_row[k];
                }
            }
        }

        // x_next = x + v_total + correction
        let x_next = &x_arr + &v_total + &christoffel;

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
            g_sym: self.g_sym.clone(),
            a_antisym: self.a_antisym.clone(),
            u_metric: self.u_metric.clone(),
            v_metric: self.v_metric.clone(),
            g_core: self.g_core.clone(),
            a_core: self.a_core.clone(),
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
            calibration_samples: 1024,
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
            g_sym: comp.g_sym,
            a_antisym: comp.a_antisym,
            u_metric: comp.u_metric,
            v_metric: comp.v_metric,
            g_core: comp.g_core,
            a_core: comp.a_core,
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
    pub g_sym: Array2<f32>,
    pub a_antisym: Array2<f32>,
    pub u_metric: Array2<f32>,
    pub v_metric: Array2<f32>,
    pub g_core: Array2<f32>,
    pub a_core: Array2<f32>,
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
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
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
    let symmetry_error = if g_norm > 1e-10 {
        (sym_diff / g_norm).sqrt()
    } else {
        0.0
    };

    // Frobenius norm 계산 (fold_accuracy용)
    let frob_g: f32 = g.iter().map(|x| x * x).sum();

    // SVD로 캡처된 에너지 비율 = sum(s_i^2) / ||G||_F^2
    let frob_captured: f32 = folded.s.iter().map(|x| x * x).sum();
    let fold_accuracy = if frob_g > 1e-10 {
        (frob_captured / frob_g).min(1.0) // 1.0 초과 방지
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
    let max_eigenvalue = diag_values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let condition_number = if min_eigenvalue.abs() > 1e-10 {
        max_eigenvalue.abs() / min_eigenvalue.abs()
    } else {
        f32::INFINITY
    };

    let is_valid = fold_accuracy >= 0.5 && min_eigenvalue > -1e6 && condition_number < 1e8;

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
