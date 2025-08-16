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

pub fn apply_linear(matrix: &Array2<f32>, vecs: &Array2<f32>) -> Array2<f32> {
    // matrix: (out, in), vecs: (batch, in) -> (batch, out)
    let (_, in_dim) = matrix.dim();
    let (_batch, in_dim_vec) = vecs.dim();
    assert_eq!(in_dim, in_dim_vec);
    vecs.dot(&matrix.t())
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array};

    fn is_symmetric(a: &Array2<f32>, tol: f32) -> bool {
        let diff = a - &a.t();
        diff.iter().all(|v| v.abs() <= tol)
    }

    #[test]
    fn test_spd_metric_from_key_properties() {
        let dim = 64;
        let g = spd_metric_from_key("dept:AI", dim, 0.1, 2.0);
        assert!(is_symmetric(&g, 1e-4));
        // Positive definite: v^T G v > 0 for random v
        let v =
            Array::from_shape_vec((dim,), (0..dim).map(|i| (i as f32).sin()).collect()).unwrap();
        let s = v.dot(&g.dot(&v));
        assert!(s > 0.0);
    }

    #[test]
    fn test_mahalanobis_consistency_g_vs_l() {
        let dim = 32;
        let g = spd_metric_from_key("k", dim, 0.5, 1.5);
        let l = metric_factor_cholesky(&g);
        let x = arr1(&(0..dim).map(|i| (i as f32).cos()).collect::<Vec<_>>());
        let y = arr1(&(0..dim).map(|i| (i as f32).sin()).collect::<Vec<_>>());
        let d_g = mahalanobis_distance_sq_g(&x, &y, &g);
        let d_l = mahalanobis_distance_sq_l(&x, &y, &l);
        assert!((d_g - d_l).abs() < 1e-3);
    }

    #[test]
    fn test_block_orthogonal_is_orthonormal() {
        let total = 128;
        let q = block_orthogonal_from_key("dept:AI", 64, 64);
        assert_eq!(q.dim(), (total, total));
        let i = Array2::<f32>::eye(total);
        let should_be_i = q.t().dot(&q);
        let diff = &should_be_i - &i;
        let max_abs = diff.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        assert!(max_abs < 5e-4, "orthogonality error too large: {}", max_abs);
    }

    #[test]
    fn test_compose_layers_order_preserving_matches_sequential() {
        let dim = 16;
        let t1 = spd_metric_from_key("l1", dim, 0.8, 1.2);
        let t2 = spd_metric_from_key("l2", dim, 0.8, 1.2);
        let t3 = spd_metric_from_key("l3", dim, 0.8, 1.2);
        let layers = vec![t1.clone(), t2.clone(), t3.clone()];
        let t_total = compose_layers_order_preserving(&layers);
        // Sequential application to a vector
        let x = arr1(&(0..dim).map(|i| i as f32).collect::<Vec<_>>());
        let x_seq = t3.dot(&t2.dot(&t1.dot(&x)));
        let x_total = t_total.dot(&x);
        let err = (&x_seq - &x_total).mapv(|v| v.abs()).scalar_sum();
        assert!(err < 1e-3);
        // Order dependence: reverse should differ
        let t_rev = compose_layers_order_preserving(&[t3, t2, t1]);
        let x_rev = t_rev.dot(&x);
        let diff_rev = (&x_rev - &x_total).mapv(|v| v.abs()).scalar_sum();
        assert!(diff_rev > 1e-4);
    }

    #[test]
    fn test_rotate_metric_factor_preserves_g() {
        let dim = 32;
        let g = spd_metric_from_key("key", dim, 0.5, 1.5);
        let l = metric_factor_cholesky(&g);
        let l_rot = rotate_metric_factor_block("session", &l, 16);
        let g_rot = l_rot.t().dot(&l_rot);
        // Use Frobenius norm for a robust metric and a looser tolerance for f32
        let diff = (&g - &g_rot).mapv(|v| v * v).sum().sqrt();
        assert!(diff < 1e-2, "frobenius diff too large: {}", diff);
    }
}
