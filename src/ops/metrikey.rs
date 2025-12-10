use ndarray::{s, Array1, Array2};
use rand::prelude::*;
use rand::rngs::SmallRng;

const EPS: f32 = 1e-6;
const EPS64: f64 = 1e-12;

fn seed_from_key(key: &str) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x00000100000001B3;
    for &b in key.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(prime);
    }
    hash
}

fn box_muller_pair<R: Rng>(rng: &mut R) -> (f32, f32) {
    // Generate two independent standard normals
    let u1 = rng.gen::<f32>().max(EPS);
    let u2 = rng.gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn box_muller_pair64<R: Rng>(rng: &mut R) -> (f64, f64) {
    let u1 = rng.gen::<f64>().max(EPS64);
    let u2 = rng.gen::<f64>();
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = 2.0_f64 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn random_normal_matrix(dim: usize, rng: &mut SmallRng) -> Array2<f32> {
    let mut m = Array2::<f32>::zeros((dim, dim));
    let mut i = 0;
    while i < dim * dim {
        let (z0, z1) = box_muller_pair(rng);
        let r = i / dim;
        let c = i % dim;
        m[(r, c)] = z0;
        i += 1;
        if i < dim * dim {
            let r2 = i / dim;
            let c2 = i % dim;
            m[(r2, c2)] = z1;
            i += 1;
        }
    }
    m
}

fn random_normal_matrix64(dim: usize, rng: &mut SmallRng) -> ndarray::Array2<f64> {
    let mut m = ndarray::Array2::<f64>::zeros((dim, dim));
    let mut i = 0;
    while i < dim * dim {
        let (z0, z1) = box_muller_pair64(rng);
        let r = i / dim;
        let c = i % dim;
        m[(r, c)] = z0;
        i += 1;
        if i < dim * dim {
            let r2 = i / dim;
            let c2 = i % dim;
            m[(r2, c2)] = z1;
            i += 1;
        }
    }
    m
}

fn modified_gram_schmidt(a: &Array2<f32>, reorth_passes: usize) -> Array2<f32> {
    let (rows, cols) = a.dim();
    assert_eq!(rows, cols, "Expected square matrix");
    let n = rows;
    let mut q = Array2::<f32>::zeros((n, n));

    // First pass: Modified Gram-Schmidt
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for k in 0..j {
            let qk = q.column(k);
            let r = v.dot(&qk);
            v -= &(qk.to_owned() * r);
        }
        let norm = v.dot(&v).sqrt().max(EPS);
        v /= norm;
        q.slice_mut(s![.., j]).assign(&v);
    }

    // Optional re-orthogonalization passes to improve numerical stability
    for _ in 0..reorth_passes {
        for j in 0..n {
            let mut v = q.column(j).to_owned();
            for k in 0..j {
                let qk = q.column(k);
                let r = v.dot(&qk);
                v -= &(qk.to_owned() * r);
            }
            let norm = v.dot(&v).sqrt().max(EPS);
            v /= norm;
            q.slice_mut(s![.., j]).assign(&v);
        }
    }

    q
}

fn modified_gram_schmidt64(a: &ndarray::Array2<f64>, reorth_passes: usize) -> ndarray::Array2<f64> {
    let (rows, cols) = a.dim();
    assert_eq!(rows, cols, "Expected square matrix");
    let n = rows;
    let mut q = ndarray::Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for k in 0..j {
            let qk = q.column(k);
            let r = v.dot(&qk);
            v -= &(qk.to_owned() * r);
        }
        let norm = v.dot(&v).sqrt().max(EPS64);
        v /= norm;
        q.slice_mut(s![.., j]).assign(&v);
    }
    for _ in 0..reorth_passes {
        for j in 0..n {
            let mut v = q.column(j).to_owned();
            for k in 0..j {
                let qk = q.column(k);
                let r = v.dot(&qk);
                v -= &(qk.to_owned() * r);
            }
            let norm = v.dot(&v).sqrt().max(EPS64);
            v /= norm;
            q.slice_mut(s![.., j]).assign(&v);
        }
    }
    q
}

pub fn deterministic_orthogonal_from_key(key: &str, dim: usize) -> Array2<f32> {
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed);
    let a = random_normal_matrix(dim, &mut rng);
    // Two re-orthogonalization passes provide good orthogonality in f32 for typical dims
    modified_gram_schmidt(&a, 1)
}

pub fn deterministic_orthogonal_from_key_f64(key: &str, dim: usize) -> ndarray::Array2<f64> {
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed);
    let a = random_normal_matrix64(dim, &mut rng);
    modified_gram_schmidt64(&a, 2)
}

pub fn spd_metric_from_key(key: &str, dim: usize, min_lambda: f32, max_lambda: f32) -> Array2<f32> {
    assert!(dim > 0);
    assert!(min_lambda > 0.0 && max_lambda > min_lambda);
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
    let q = deterministic_orthogonal_from_key(key, dim);
    // Build diagonal spectrum D
    let mut d = Array2::<f32>::zeros((dim, dim));
    for i in 0..dim {
        let u: f32 = rng.gen();
        let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
        d[(i, i)] = lam;
    }
    // G = Q^T D Q is symmetric SPD regardless of Q orthonormality accuracy
    let dq = d.dot(&q);
    q.t().dot(&dq)
}

pub fn spd_metric_from_key_f64(
    key: &str,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
) -> ndarray::Array2<f64> {
    assert!(dim > 0);
    assert!(min_lambda > 0.0 && max_lambda > min_lambda);
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
    let q = deterministic_orthogonal_from_key_f64(key, dim);
    let mut d = ndarray::Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        let u: f64 = rng.gen();
        let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
        d[(i, i)] = lam;
    }
    let dq = d.dot(&q);
    q.t().dot(&dq)
}

/// Weighted SPD metric where eigenvalues are exponentiated by a mass factor.
/// Interpreted as curvature/strength control: lam' = lam^{mass}.
pub fn spd_metric_from_key_weighted(
    key: &str,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
    mass: f32,
) -> Array2<f32> {
    assert!(mass > 0.0);
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
    let q = deterministic_orthogonal_from_key(key, dim);
    let mut d = Array2::<f32>::zeros((dim, dim));
    for i in 0..dim {
        let u: f32 = rng.gen();
        let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
        d[(i, i)] = lam.powf(mass);
    }
    let dq = d.dot(&q);
    q.t().dot(&dq)
}

/// Gravity composition: Order-preserving product of weighted layer factors.
/// Each layer l uses T_l = (G_l(mass_l))^{1/2}. Here we compute via spectrum exponent (mass/2).
pub fn compose_layers_gravity(
    keys: &[String],
    masses: &[f32],
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> Array2<f32> {
    assert!(!keys.is_empty());
    assert_eq!(keys.len(), masses.len());
    let mut acc = Array2::<f32>::eye(dim);
    for (key, &mass) in keys.iter().zip(masses.iter()) {
        assert!(mass > 0.0);
        // Build Q and D^{mass/2}
        let q = deterministic_orthogonal_from_key(key, dim);
        let seed = seed_from_key(key);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
        let mut d_sqrt = Array2::<f32>::zeros((dim, dim));
        for i in 0..dim {
            let u: f32 = rng.gen();
            let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
            d_sqrt[(i, i)] = lam.powf(0.5 * mass);
        }
        let t_l = q.t().dot(&d_sqrt.dot(&q));
        acc = t_l.dot(&acc);
    }
    acc
}

pub fn compose_layers_gravity_f64(
    keys: &[String],
    masses: &[f64],
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
) -> ndarray::Array2<f64> {
    assert!(!keys.is_empty());
    assert_eq!(keys.len(), masses.len());
    let mut acc = ndarray::Array2::<f64>::eye(dim);
    for (key, &mass) in keys.iter().zip(masses.iter()) {
        assert!(mass > 0.0);
        let q = deterministic_orthogonal_from_key_f64(key, dim);
        let seed = seed_from_key(key);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
        let mut d_sqrt = ndarray::Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            let u: f64 = rng.gen();
            let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
            d_sqrt[(i, i)] = lam.powf(0.5 * mass);
        }
        let t_l = q.t().dot(&d_sqrt.dot(&q));
        acc = t_l.dot(&acc);
    }
    acc
}

pub fn apply_linear_f64(
    matrix: &ndarray::Array2<f64>,
    vecs: &ndarray::Array2<f64>,
) -> ndarray::Array2<f64> {
    let (_, in_dim) = matrix.dim();
    let (_batch, in_dim_vec) = vecs.dim();
    assert_eq!(in_dim, in_dim_vec);
    vecs.dot(&matrix.t())
}

/// Compact composition using a single master key and a simple mass schedule.
/// keys: key_i = format!("{}#{}", master_key, i)
/// masses: mass_i = mass_base + i * mass_step
pub fn compose_layers_gravity_compact_f64(
    master_key: &str,
    num_layers: usize,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
    mass_base: f64,
    mass_step: f64,
) -> ndarray::Array2<f64> {
    assert!(num_layers > 0);
    let mut acc = ndarray::Array2::<f64>::eye(dim);
    for i in 0..num_layers {
        let key_i = format!("{}#{}", master_key, i);
        let q = deterministic_orthogonal_from_key_f64(&key_i, dim);
        let seed = seed_from_key(&key_i);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
        let mass = mass_base + (i as f64) * mass_step;
        assert!(mass > 0.0);
        let mut d_sqrt = ndarray::Array2::<f64>::zeros((dim, dim));
        for j in 0..dim {
            let u: f64 = rng.gen();
            let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
            d_sqrt[(j, j)] = lam.powf(0.5 * mass);
        }
        let t_l = q.t().dot(&d_sqrt.dot(&q));
        acc = t_l.dot(&acc);
    }
    acc
}

pub fn metric_factor_cholesky(g: &Array2<f32>) -> Array2<f32> {
    let (n, m) = g.dim();
    assert_eq!(n, m, "G must be square");
    let mut l = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = g[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, j)] = (sum.max(EPS)).sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)].max(EPS);
            }
        }
    }
    // Return upper-triangular factor U = L_lower^T so that G = U^T U holds
    l.t().to_owned()
}

pub fn mahalanobis_distance_sq_g(x: &Array1<f32>, y: &Array1<f32>, g: &Array2<f32>) -> f32 {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(g.dim(), (n, n));
    let diff = x - y;
    let tmp = g.dot(&diff);
    diff.dot(&tmp)
}

pub fn mahalanobis_distance_sq_l(x: &Array1<f32>, y: &Array1<f32>, l: &Array2<f32>) -> f32 {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(l.dim(), (n, n));
    let diff = x - y;
    // l is defined as upper-triangular factor such that G = l^T l
    let z = l.dot(&diff);
    z.dot(&z)
}

pub fn block_orthogonal_from_key(key: &str, global_dim: usize, dept_dim: usize) -> Array2<f32> {
    let total = global_dim + dept_dim;
    let mut q = Array2::<f32>::eye(total);
    if dept_dim > 0 {
        let r = deterministic_orthogonal_from_key(key, dept_dim);
        q.slice_mut(s![global_dim.., global_dim..]).assign(&r);
    }
    q
}

pub fn spd_block_metric_from_key(
    key: &str,
    global_dim: usize,
    dept_dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> Array2<f32> {
    let total = global_dim + dept_dim;
    let mut g = Array2::<f32>::eye(total);
    if dept_dim > 0 {
        let g_dept = spd_metric_from_key(key, dept_dim, min_lambda, max_lambda);
        g.slice_mut(s![global_dim.., global_dim..]).assign(&g_dept);
    }
    g
}

pub fn compose_layers_order_preserving(layers: &[Array2<f32>]) -> Array2<f32> {
    assert!(!layers.is_empty(), "layers must be non-empty");
    let (n, m) = layers[0].dim();
    assert_eq!(n, m, "only square layers supported");
    let mut acc = Array2::<f32>::eye(n);
    for l in layers {
        assert_eq!(l.dim(), (n, n));
        acc = l.dot(&acc);
    }
    acc
}

/// f64 variant: Order-preserving composition of square layers
pub fn compose_layers_order_preserving_f64(
    layers: &[ndarray::Array2<f64>],
) -> ndarray::Array2<f64> {
    assert!(!layers.is_empty(), "layers must be non-empty");
    let (n, m) = layers[0].dim();
    assert_eq!(n, m, "only square layers supported");
    let mut acc = ndarray::Array2::<f64>::eye(n);
    for l in layers {
        assert_eq!(l.dim(), (n, n));
        acc = l.dot(&acc);
    }
    acc
}

pub fn apply_linear(matrix: &Array2<f32>, vecs: &Array2<f32>) -> Array2<f32> {
    // matrix: (out, in), vecs: (batch, in) -> (batch, out)
    let (_, in_dim) = matrix.dim();
    let (_batch, in_dim_vec) = vecs.dim();
    assert_eq!(in_dim, in_dim_vec);
    vecs.dot(&matrix.t())
}

// ===== Exact inference ops (f32) for reversible path =====

/// LayerNorm forward (per-row) with epsilon, returns (y, mu, rstd)
pub fn layer_norm_forward_exact_f32(
    x: &Array2<f32>,
    gamma: &ndarray::Array1<f32>,
    beta: &ndarray::Array1<f32>,
    eps: f32,
) -> (Array2<f32>, ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let (batch, dim) = x.dim();
    assert_eq!(gamma.len(), dim);
    assert_eq!(beta.len(), dim);
    let mut y = Array2::<f32>::zeros((batch, dim));
    let mut mu = ndarray::Array1::<f32>::zeros(batch);
    let mut rstd = ndarray::Array1::<f32>::zeros(batch);
    for i in 0..batch {
        let xi = x.row(i);
        let m = xi.sum() / (dim as f32);
        mu[i] = m;
        // variance
        let mut var = 0.0f32;
        for j in 0..dim {
            let d = xi[j] - m;
            var += d * d;
        }
        var /= dim as f32;
        let rs = 1.0f32 / (var + eps).sqrt();
        rstd[i] = rs;
        for j in 0..dim {
            let norm = (xi[j] - m) * rs;
            y[(i, j)] = norm * gamma[j] + beta[j];
        }
    }
    (y, mu, rstd)
}

/// GPT-2 gelu_new activation (tanh-based) applied elementwise to (batch, dim)
pub fn gelu_new_f32(x: &Array2<f32>) -> Array2<f32> {
    let (batch, dim) = x.dim();
    let mut y = Array2::<f32>::zeros((batch, dim));
    // constants
    let k: f32 = std::f32::consts::FRAC_2_SQRT_PI * 0.5f32; // 0.5*sqrt(2/pi)
    for i in 0..batch {
        for j in 0..dim {
            let v = x[(i, j)];
            let v3 = v * v * v;
            let t = (k * (v + 0.044715f32 * v3)).tanh();
            y[(i, j)] = 0.5f32 * v * (1.0f32 + t);
        }
    }
    y
}

/// Stable softmax along last dimension of a 2D tensor (batch, dim)
pub fn softmax_lastdim_f32(x: &Array2<f32>) -> Array2<f32> {
    let (batch, dim) = x.dim();
    let mut y = Array2::<f32>::zeros((batch, dim));
    for i in 0..batch {
        // subtract max for stability
        let mut max_v = std::f32::NEG_INFINITY;
        for j in 0..dim {
            let v = x[(i, j)];
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..dim {
            let e = (x[(i, j)] - max_v).exp();
            y[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0f32 / sum.max(EPS);
        for j in 0..dim {
            y[(i, j)] *= inv;
        }
    }
    y
}

/// Apply causal mask in-place to 2D scores (seq, seq): set j>i to large negative
pub fn apply_causal_mask_inplace_f32(scores: &mut Array2<f32>, neg_large: f32) {
    let (n, m) = scores.dim();
    assert_eq!(n, m);
    for i in 0..n {
        for j in (i + 1)..n {
            scores[(i, j)] = neg_large;
        }
    }
}

// f64 counterparts
pub fn layer_norm_forward_exact_f64(
    x: &ndarray::Array2<f64>,
    gamma: &ndarray::Array1<f64>,
    beta: &ndarray::Array1<f64>,
    eps: f64,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
) {
    let (batch, dim) = x.dim();
    assert_eq!(gamma.len(), dim);
    assert_eq!(beta.len(), dim);
    let mut y = ndarray::Array2::<f64>::zeros((batch, dim));
    let mut mu = ndarray::Array1::<f64>::zeros(batch);
    let mut rstd = ndarray::Array1::<f64>::zeros(batch);
    for i in 0..batch {
        let xi = x.row(i);
        let m = xi.sum() / (dim as f64);
        mu[i] = m;
        let mut var = 0.0f64;
        for j in 0..dim {
            let d = xi[j] - m;
            var += d * d;
        }
        var /= dim as f64;
        let rs = 1.0f64 / (var + (eps as f64)).sqrt();
        rstd[i] = rs;
        for j in 0..dim {
            let norm = (xi[j] - m) * rs;
            y[(i, j)] = norm * gamma[j] + beta[j];
        }
    }
    (y, mu, rstd)
}

pub fn gelu_new_f64(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let (batch, dim) = x.dim();
    let mut y = ndarray::Array2::<f64>::zeros((batch, dim));
    let k: f64 = std::f64::consts::FRAC_2_SQRT_PI * 0.5f64;
    for i in 0..batch {
        for j in 0..dim {
            let v = x[(i, j)];
            let v3 = v * v * v;
            let t = (k * (v + 0.044715f64 * v3)).tanh();
            y[(i, j)] = 0.5f64 * v * (1.0f64 + t);
        }
    }
    y
}

pub fn softmax_lastdim_f64(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let (batch, dim) = x.dim();
    let mut y = ndarray::Array2::<f64>::zeros((batch, dim));
    for i in 0..batch {
        let mut max_v = std::f64::NEG_INFINITY;
        for j in 0..dim {
            let v = x[(i, j)];
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0f64;
        for j in 0..dim {
            let e = (x[(i, j)] - max_v).exp();
            y[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0f64 / sum.max(EPS64);
        for j in 0..dim {
            y[(i, j)] *= inv;
        }
    }
    y
}

pub fn apply_causal_mask_inplace_f64(scores: &mut ndarray::Array2<f64>, neg_large: f64) {
    let (n, m) = scores.dim();
    assert_eq!(n, m);
    for i in 0..n {
        for j in (i + 1)..n {
            scores[(i, j)] = neg_large;
        }
    }
}

// ===== f64 linear/attention/ffn exact forwards (GPT-2 style) =====

pub fn linear_f64(
    x: &ndarray::Array2<f64>,         // (batch, in)
    w: &ndarray::Array2<f64>,         // (out, in)
    b: Option<&ndarray::Array1<f64>>, // (out)
) -> ndarray::Array2<f64> {
    let y = x.dot(&w.t());
    if let Some(bias) = b {
        let mut out = y;
        for mut row in out.rows_mut() {
            row += &bias.view();
        }
        out
    } else {
        y
    }
}

pub fn attention_forward_f64(
    x: &ndarray::Array2<f64>, // (seq, d_model)
    wq: &ndarray::Array2<f64>,
    wk: &ndarray::Array2<f64>,
    wv: &ndarray::Array2<f64>,
    wo: &ndarray::Array2<f64>,
    bq: Option<&ndarray::Array1<f64>>,
    bk: Option<&ndarray::Array1<f64>>,
    bv: Option<&ndarray::Array1<f64>>,
    bo: Option<&ndarray::Array1<f64>>,
    n_heads: usize,
    causal: bool,
) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    // (y, attn_probs_flat)
    let (seq, d_model) = x.dim();
    assert_eq!(wq.dim().1, d_model);
    let d_q = wq.dim().0; // out dim for Q
    assert_eq!(d_q % n_heads, 0);
    let dh = d_q / n_heads;

    let q = linear_f64(x, wq, bq); // (seq, d_q)
    let k = linear_f64(x, wk, bk); // (seq, d_q)
    let v = linear_f64(x, wv, bv); // (seq, d_q)

    // reshape to heads: (n_heads, seq, dh)
    let mut y_heads = ndarray::Array3::<f64>::zeros((n_heads, seq, dh));
    let mut probs_all = ndarray::Array3::<f64>::zeros((n_heads, seq, seq));
    let scale = 1.0f64 / (dh as f64).sqrt();
    for h in 0..n_heads {
        // slices
        let qs = q.slice(ndarray::s![.., h * dh..(h + 1) * dh]).to_owned(); // (seq, dh)
        let ks = k.slice(ndarray::s![.., h * dh..(h + 1) * dh]).to_owned();
        let vs = v.slice(ndarray::s![.., h * dh..(h + 1) * dh]).to_owned();
        // scores = Q K^T * scale
        let mut scores = qs.dot(&ks.t()); // (seq, seq)
        scores.mapv_inplace(|z| z * scale);
        if causal {
            apply_causal_mask_inplace_f64(&mut scores, -1e9f64);
        }
        // softmax
        let probs = softmax_lastdim_f64(&scores);
        // out_h = probs V
        let out_h = probs.dot(&vs);
        y_heads.slice_mut(ndarray::s![h, .., ..]).assign(&out_h);
        probs_all.slice_mut(ndarray::s![h, .., ..]).assign(&probs);
    }
    // merge heads -> (seq, d_q)
    let mut yh = ndarray::Array2::<f64>::zeros((seq, d_q));
    for h in 0..n_heads {
        let s = y_heads.slice(ndarray::s![h, .., ..]);
        yh.slice_mut(ndarray::s![.., h * dh..(h + 1) * dh])
            .assign(&s);
    }
    let y = linear_f64(&yh, wo, bo); // (seq, d_model)
                                     // Flatten heads: (n_heads, seq, seq) -> (n_heads*seq, seq)
    let probs_flat = probs_all.into_shape((n_heads * seq, seq)).unwrap();
    (y, probs_flat)
}

pub fn ffn_gelu_forward_f64(
    x: &ndarray::Array2<f64>,
    w1: &ndarray::Array2<f64>,
    b1: Option<&ndarray::Array1<f64>>,
    w2: &ndarray::Array2<f64>,
    b2: Option<&ndarray::Array1<f64>>,
) -> ndarray::Array2<f64> {
    let h = linear_f64(x, w1, b1);
    let a = gelu_new_f64(&h);
    linear_f64(&a, w2, b2)
}

pub fn transformer_block_forward_f64(
    x: &ndarray::Array2<f64>,
    // LN1
    ln1_g: &ndarray::Array1<f64>,
    ln1_b: &ndarray::Array1<f64>,
    eps1: f64,
    // Attn
    wq: &ndarray::Array2<f64>,
    wk: &ndarray::Array2<f64>,
    wv: &ndarray::Array2<f64>,
    wo: &ndarray::Array2<f64>,
    bq: Option<&ndarray::Array1<f64>>,
    bk: Option<&ndarray::Array1<f64>>,
    bv: Option<&ndarray::Array1<f64>>,
    bo: Option<&ndarray::Array1<f64>>,
    n_heads: usize,
    // LN2
    ln2_g: &ndarray::Array1<f64>,
    ln2_b: &ndarray::Array1<f64>,
    eps2: f64,
    // FFN
    w1: &ndarray::Array2<f64>,
    b1: Option<&ndarray::Array1<f64>>,
    w2: &ndarray::Array2<f64>,
    b2: Option<&ndarray::Array1<f64>>,
    causal: bool,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
) {
    // LN1
    let (x1, mu1, rstd1) = layer_norm_forward_exact_f64(x, ln1_g, ln1_b, eps1);
    // Attn
    let (attn_out, _probs) =
        attention_forward_f64(&x1, wq, wk, wv, wo, bq, bk, bv, bo, n_heads, causal);
    let x_res1 = x + &attn_out;
    // LN2
    let (x2, mu2, rstd2) = layer_norm_forward_exact_f64(&x_res1, ln2_g, ln2_b, eps2);
    // FFN
    let ffn_out = ffn_gelu_forward_f64(&x2, w1, b1, w2, b2);
    let y = x_res1 + &ffn_out;
    (y, mu1, rstd1, mu2, rstd2)
}

/// Compute effective SPD metric G = T^T T for a given transform T (f64)
pub fn effective_metric_from_transform_f64(t: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    t.t().dot(t)
}

/// Simple Cholesky factorization (upper-triangular) in f64: returns U with G = U^T U
pub fn metric_factor_cholesky_f64(g: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let (n, m) = g.dim();
    assert_eq!(n, m, "G must be square");
    let mut l = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = g[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, j)] = (sum.max(EPS64)).sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)].max(EPS64);
            }
        }
    }
    // Return upper-triangular factor U = L_lower^T so that G = U^T U holds
    l.t().to_owned()
}

/// Session-rotation of the metric factor. Given an SPD factor L (G = L^T L),
/// apply an orthogonal rotation R_s on the left to preserve G: L' = R_s L.
/// A deterministic block-orthogonal R_s is generated from the key.
pub fn rotate_metric_factor_block(key: &str, l: &Array2<f32>, global_dim: usize) -> Array2<f32> {
    let (n, m) = l.dim();
    assert_eq!(n, m, "L must be square");
    assert!(global_dim <= n);
    let dept_dim = n - global_dim;
    let r_s = block_orthogonal_from_key(key, global_dim, dept_dim);
    r_s.dot(l)
}

// === Implicit transforms: Householder chain / Givens chain / Low-rank + Diagonal ===

fn random_unit_vector_f32(dim: usize, rng: &mut SmallRng) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(dim);
    for i in 0..dim {
        v[i] = rng.gen::<f32>() * 2.0 - 1.0;
    }
    let n = v.dot(&v).sqrt().max(EPS);
    v / n
}

fn householder_vectors_from_key(key: &str, dim: usize, num: usize) -> Vec<Array1<f32>> {
    let mut vecs = Vec::with_capacity(num);
    let mut rng = SmallRng::seed_from_u64(seed_from_key(key));
    for _ in 0..num {
        vecs.push(random_unit_vector_f32(dim, &mut rng));
    }
    vecs
}

fn apply_householder_chain(vecs: &[Array1<f32>], x: &Array1<f32>, reverse: bool) -> Array1<f32> {
    let mut y = x.clone();
    if reverse {
        for v in vecs.iter().rev() {
            let alpha = 2.0 * y.dot(v);
            y -= &(v * alpha);
        }
    } else {
        for v in vecs.iter() {
            let alpha = 2.0 * y.dot(v);
            y -= &(v * alpha);
        }
    }
    y
}

pub fn householder_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let vecs = householder_vectors_from_key(key, dim, num);
    apply_householder_chain(&vecs, x, false)
}

pub fn householder_chain_apply_transpose_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let vecs = householder_vectors_from_key(key, dim, num);
    // For Householder, H is symmetric, so Q^T = H_1 ... H_k (reverse order)
    apply_householder_chain(&vecs, x, true)
}

pub fn lowrank_plus_diag_apply_from_key(
    key_u: &str,
    key_v: &str,
    s_diag: &Array1<f32>,
    rank: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let dim = x.len();
    assert_eq!(s_diag.len(), dim);
    let mut rng_u = SmallRng::seed_from_u64(seed_from_key(key_u));
    let mut rng_v = SmallRng::seed_from_u64(seed_from_key(key_v));
    let mut y = s_diag * x;
    for _ in 0..rank {
        let a = random_unit_vector_f32(dim, &mut rng_u);
        let b = random_unit_vector_f32(dim, &mut rng_v);
        let coeff = b.dot(x);
        y += &(a * coeff);
    }
    y
}

pub fn givens_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let mut rng = SmallRng::seed_from_u64(seed_from_key(key) ^ 0xABCDEF0123456789);
    let mut y = x.clone();
    for _ in 0..num {
        let i = (rng.gen::<u32>() as usize) % dim;
        let mut j = (rng.gen::<u32>() as usize) % dim;
        if j == i {
            j = (j + 1) % dim;
        }
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        let c = theta.cos();
        let s = theta.sin();
        let yi = y[i];
        let yj = y[j];
        y[i] = c * yi - s * yj;
        y[j] = s * yi + c * yj;
    }
    y
}
