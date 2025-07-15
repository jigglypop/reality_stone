use super::types::Packed64;
use std::f32::consts::PI;

/// 회전 각도 계산
pub fn get_rotation_angle(rot_code: u8) -> f32 {
    match rot_code {
        0 => 0.0,
        1 => PI / 8.0,
        2 => PI / 6.0,
        3 => PI / 4.0,
        4 => PI / 3.0,
        5 => PI / 2.0,
        6 => 2.0 * PI / 3.0,
        7 => 3.0 * PI / 4.0,
        8 => 5.0 * PI / 6.0,
        9 => 7.0 * PI / 8.0,
        _ => 0.0,
    }
}

/// 각도 미분 적용
pub fn apply_angular_derivative(theta: f32, d_theta: u8, basis_id: u8) -> f32 {
    let is_sin_based = (basis_id & 0x1) == 0;

    match (is_sin_based, d_theta % 4) {
        (true, 0) => theta.sin(),
        (true, 1) => theta.cos(),
        (true, 2) => -theta.sin(),
        (true, 3) => -theta.cos(),
        (false, 0) => theta.cos(),
        (false, 1) => -theta.sin(),
        (false, 2) => -theta.cos(),
        (false, 3) => theta.sin(),
        _ => unreachable!(),
    }
}

/// 반지름 미분 적용
pub fn apply_radial_derivative(r: f32, d_r: bool, basis_id: u8) -> f32 {
    let is_sinh_based = (basis_id & 0x2) == 0;

    match (is_sinh_based, d_r) {
        (true, false) => r.sinh(),
        (true, true) => r.cosh(),
        (false, false) => r.cosh(),
        (false, true) => r.sinh(),
    }
}

// 베셀 함수들
pub fn bessel_j0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0
            + y * (-13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0
            + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let ans1 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

pub fn bessel_i0(x: f32) -> f32 {
    if x.abs() < 3.75 {
        let t = (x / 3.75).powi(2);
        let mut result = 1.0;
        let mut term = 1.0;

        for k in 1..=10 {
            term *= t / (k * k) as f32;
            result += term;
        }
        result
    } else {
        let t = 3.75 / x.abs();
        let mut result = 0.39894228;
        result += 0.01328592 * t;
        result += 0.00225319 * t * t;
        result -= 0.00157565 * t.powi(3);
        result * x.exp() / x.sqrt()
    }
}

pub fn bessel_k0(x: f32) -> f32 {
    if x < 2.0 {
        let i0 = bessel_i0(x);
        -x.ln() * i0 + 0.5772156649
    } else {
        let mut result = 1.2533141;
        result -= 0.07832358 * (2.0 / x);
        result += 0.02189568 * (2.0 / x).powi(2);
        result * (-x).exp() / x.sqrt()
    }
}

pub fn bessel_y0(x: f32) -> f32 {
    let j0 = bessel_j0(x);
    2.0 / PI * (x.ln() * j0 + 0.07832358)
}

pub fn sech(x: f32) -> f32 {
    2.0 / (x.exp() + (-x).exp())
}

pub fn triangle_wave(x: f32) -> f32 {
    let phase = x / PI;
    let t = phase - phase.floor();
    if t < 0.5 {
        4.0 * t - 1.0
    } else {
        3.0 - 4.0 * t
    }
}

pub fn morlet_wavelet(r: f32, theta: f32, freq: f32) -> f32 {
    let sigma = 1.0 / freq.sqrt();
    let gaussian = (-0.5 * (r / sigma).powi(2)).exp();
    let oscillation = (freq * theta).cos();
    gaussian * oscillation
}

use rustfft::{num_complex::Complex, FftPlanner};

pub fn analyze_global_pattern(matrix: &[f32]) -> Vec<f32> {
    let mean = matrix.iter().sum::<f32>() / matrix.len() as f32;
    let variance = matrix.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / matrix.len() as f32;
    vec![mean, variance]
}

pub fn suggest_basis_functions(features: &Vec<f32>) -> Vec<u8> {
    let variance = features[1];
    if variance > 0.5 {
        (0..4).collect()
    } else {
        (4..12).collect()
    }
}

pub fn optimize_for_periodic(
    matrix: &[f32],
    _basis_id: u8,
    rows: usize,
    cols: usize,
) -> (f32, f32) {
    let mut complex_matrix: Vec<Vec<Complex<f32>>> = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| Complex {
                    re: matrix[i * cols + j],
                    im: 0.0,
                })
                .collect()
        })
        .collect();
    let fft = FftPlanner::new().plan_fft_forward(cols);
    for row in complex_matrix.iter_mut() {
        fft.process(row);
    }
    let fft_col = FftPlanner::new().plan_fft_forward(rows);
    for j in 0..cols {
        let mut col: Vec<Complex<f32>> = (0..rows).map(|i| complex_matrix[i][j]).collect();
        fft_col.process(&mut col);
        for i in 0..rows {
            complex_matrix[i][j] = col[i];
        }
    }
    let mut max_mag = 0.0;
    let mut peak_kx = 0.0;
    let mut peak_ky = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let mag = complex_matrix[i][j].norm();
            if mag > max_mag {
                max_mag = mag;
                peak_ky = i as f32;
                peak_kx = j as f32;
            }
        }
    }
    let r = (peak_kx.powi(2) + peak_ky.powi(2)).sqrt() / (rows.max(cols) as f32);
    let theta = peak_ky.atan2(peak_kx);
    (r.min(0.99), theta)
}

pub fn optimize_for_bessel(matrix: &[f32], _basis_id: u8, rows: usize, cols: usize) -> (f32, f32) {
    let mut radial_sum = vec![0.0; (rows / 2) as usize];
    let mut counts = vec![0; (rows / 2) as usize];
    for i in 0..rows {
        for j in 0..cols {
            let x = j as f32 - cols as f32 / 2.0;
            let y = i as f32 - rows as f32 / 2.0;
            let dist = (x * x + y * y).sqrt() as usize;
            if dist < radial_sum.len() {
                radial_sum[dist] += matrix[i * cols + j];
                counts[dist] += 1;
            }
        }
    }
    let radial_profile: Vec<f32> = radial_sum
        .iter()
        .zip(counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
        .collect();
    let mut first_zero = 1.0;
    for (i, &val) in radial_profile.iter().enumerate().skip(1) {
        if val * radial_profile[i - 1] < 0.0 {
            first_zero = i as f32 / radial_profile.len() as f32;
            break;
        }
    }
    (first_zero.min(0.99), 0.0)
}

pub fn optimize_for_special(matrix: &[f32], basis_id: u8, rows: usize, cols: usize) -> (f32, f32) {
    optimize_for_periodic(matrix, basis_id, rows, cols)
}

pub fn compute_sampled_rmse(matrix: &[f32], seed: Packed64, rows: usize, cols: usize) -> f32 {
    let mut error = 0.0;
    let samples = 100;
    for _ in 0..samples {
        let i = rand::random::<usize>() % rows;
        let j = rand::random::<usize>() % cols;
        let original = matrix[i * cols + j];
        let reconstructed = seed.compute_weight(i, j, rows, cols);
        error += (original - reconstructed).powi(2);
    }
    (error / samples as f32).sqrt()
}

pub fn compute_full_rmse(matrix: &[f32], seed: Packed64, rows: usize, cols: usize) -> f32 {
    let mut error = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let original = matrix[i * cols + j];
            let reconstructed = seed.compute_weight(i, j, rows, cols);
            error += (original - reconstructed).powi(2);
        }
    }
    (error / (rows * cols) as f32).sqrt()
}

pub fn local_search_exhaustive(
    seed: Packed64,
    matrix: &[f32],
    rows: usize,
    cols: usize,
) -> Packed64 {
    let mut best_rmse = compute_full_rmse(matrix, seed, rows, cols);
    let mut best_seed = seed;
    let deltas = [-0.01, 0.01];
    for _ in 0..100 {
        let mut params = best_seed.decode();
        params.r = (params.r + deltas[rand::random::<usize>() % 2] as f32).clamp(0.0, 0.999);
        params.theta = (params.theta + deltas[rand::random::<usize>() % 2] as f32)
            .rem_euclid(2.0 * std::f32::consts::PI);
        let new_seed = Packed64::new(
            params.r,
            params.theta,
            params.basis_id,
            params.d_theta,
            params.d_r,
            params.rot_code,
            params.log2_c,
            params.reserved,
        );
        let new_rmse = compute_full_rmse(matrix, new_seed, rows, cols);
        if new_rmse < best_rmse {
            best_rmse = new_rmse;
            best_seed = new_seed;
        }
    }
    best_seed
}
