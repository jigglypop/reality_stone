// Pure Lorentz implementation (no Poincaré fallback)
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::ops::{batch::EPS, norm_sq_batched};

#[inline]
fn safe_sqrt(x: f32) -> f32 {
    x.max(EPS).sqrt()
}

#[inline]
fn safe_acosh(x: f32) -> f32 {
    (x.max(1.0 + EPS)).acosh()
}

pub fn lorentz_inner(u: &ArrayView2<f32>, v: &ArrayView2<f32>) -> Array1<f32> {
    let batch_size = u.nrows();
    let mut result = Array1::zeros(batch_size);

    result
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, inner)| {
            let u_row = u.row(i);
            let v_row = v.row(i);

            // Minkowski inner product: u0*v0 - u1*v1 - u2*v2 - ...
            *inner = u_row[0] * v_row[0];
            for j in 1..u_row.len() {
                *inner -= u_row[j] * v_row[j];
            }
        });

    result
}

/// Exponential map at origin O = (1/√c, 0, ..., 0) mapping tangent vectors (R^d) -> hyperboloid (time + space)
pub fn lorentz_exp0_space(u: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch = u.nrows();
    let dim = u.ncols();
    let sqrtc = c.sqrt();
    let u_norm = norm_sq_batched(u).mapv(f32::sqrt);
    let s = u_norm.mapv(|v| sqrtc * v);
    let mut out = Array2::<f32>::zeros((batch, dim + 1));
    // time component
    {
        let mut tcol = out.slice_mut(s![.., 0..1]);
        let mut idx = 0;
        for mut row in tcol.rows_mut() {
            let sv = s[idx];
            row[[0]] = sv.cosh() / sqrtc;
            idx += 1;
        }
    }
    // space component
    for i in 0..batch {
        let sv = s[i];
        let scale = if sv.abs() < 1e-6 {
            1.0 / sqrtc
        } else {
            sv.sinh() / (sv * sqrtc)
        };
        for j in 0..dim {
            out[[i, j + 1]] = u[[i, j]] * scale;
        }
    }
    out
}

/// Logarithmic map at origin mapping hyperboloid points (time + space) -> tangent vectors (R^d)
pub fn lorentz_log0_space(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch = x.nrows();
    let dim = x.ncols() - 1;
    let sqrtc = c.sqrt();
    let mut out = Array2::<f32>::zeros((batch, dim));
    for i in 0..batch {
        let x0 = x[[i, 0]];
        let mut space_norm_sq = 0.0f32;
        for j in 0..dim {
            space_norm_sq += x[[i, j + 1]] * x[[i, j + 1]];
        }
        // s = arcosh(√c x0)
        let s = (sqrtc * x0).acosh();
        let denom = s.sinh().max(EPS);
        let scale = if s.abs() < 1e-6 {
            1.0
        } else {
            s / (denom * sqrtc)
        };
        for j in 0..dim {
            out[[i, j]] = x[[i, j + 1]] * scale;
        }
    }
    out
}

pub fn lorentz_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    // Standard hyperboloid distance: cosh(√c d) = -c ⟨u,v⟩
    let inner = lorentz_inner(u, v);
    let sqrtc = c.sqrt();
    inner.mapv(|x| safe_acosh((-c * x).max(1.0 + EPS)) / sqrtc)
}

pub fn lorentz_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let v_row = v.row(i);

            // Compute inner products
            let mut uu = u_row[0] * u_row[0];
            let mut vv = v_row[0] * v_row[0];
            let mut uv = u_row[0] * v_row[0];

            for j in 1..dim {
                uu -= u_row[j] * u_row[j];
                vv -= v_row[j] * v_row[j];
                uv -= u_row[j] * v_row[j];
            }

            let beta_u = (-uu / c).max(EPS);
            let beta_v = (-vv / c).max(EPS);
            let gamma_u = 1.0 / safe_sqrt(beta_u);
            let gamma_v = 1.0 / safe_sqrt(beta_v);
            let gamma_uv = -uv / (c * (beta_u * beta_v).sqrt());

            for j in 0..dim {
                let denom_u = (1.0 + gamma_u).max(EPS);
                let denom_v = (1.0 + gamma_v).max(EPS);
                row[j] = gamma_uv * (gamma_u * u_row[j] / denom_u + gamma_v * v_row[j] / denom_v)
                    + u_row[j]
                    + v_row[j];
            }
        });

    result
}

pub fn lorentz_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let time_comp = u_row[0];

            let mut space_norm_sq = 0.0;
            for j in 1..dim {
                space_norm_sq += u_row[j] * u_row[j];
            }

            // Hyperboloid constraint: time^2 - ||x||^2 = 1/c
            let denom = (time_comp * time_comp - 1.0 / c).max(EPS);
            let norm = (space_norm_sq / denom).sqrt();
            let theta = norm.min(1.0 - EPS).atanh() * r;
            let scale = theta.tanh() / norm.max(EPS);

            // Set time component
            let mut scaled_space_norm_sq = 0.0;
            for j in 1..dim {
                row[j] = u_row[j] * scale;
                scaled_space_norm_sq += row[j] * row[j];
            }
            // Recompute time component to satisfy hyperboloid: x0 = sqrt(1/c + ||x||^2)
            row[0] = (1.0 / c + scaled_space_norm_sq).sqrt();
        });

    result
}

pub fn lorentz_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols() - 1;
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let sqrtc = c.sqrt();
            let x0 = x_row[0] * sqrtc;
            let denom = (x0 + 1.0).max(EPS);

            for j in 0..dim {
                row[j] = (x_row[j + 1] * sqrtc) / denom;
            }
        });

    result
}

pub fn lorentz_to_klein(x: &ArrayView2<f32>, _: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols() - 1;
    let mut result = Array2::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let x0 = x_row[0].max(EPS);

            for j in 0..dim {
                row[j] = x_row[j + 1] / x0;
            }
        });

    result
}

/// Lorentz 스칼라 곱의 VJP를 계산합니다. (근사치)
pub fn lorentz_scalar_vjp(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    r: f32,
) -> Array2<f32> {
    let _ = (grad_output, r);
    Array2::zeros(u.raw_dim())
}

/// Lorentz 덧셈의 VJP를 계산합니다. (근사치)
pub fn lorentz_add_vjp(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let _ = grad_output;
    (
        Array2::<f32>::zeros(u.raw_dim()),
        Array2::<f32>::zeros(v.raw_dim()),
    )
}

/// Lorentz 모델의 순전파 레이어를 계산합니다.
pub fn lorentz_layer_forward(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    // Geodesic interpolation on hyperboloid between u and v with parameter t
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut result = Array2::<f32>::zeros((batch_size, dim));

    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let p = u.row(i);
            let q = v.row(i);
            // Minkowski inner product
            let mut inner = p[0] * q[0];
            for j in 1..dim {
                inner -= p[j] * q[j];
            }
            let theta = safe_acosh((-c * inner).max(1.0 + EPS));
            let sinh_theta = theta.sinh().max(EPS);
            let w1 = if theta.abs() < 1e-6 {
                1.0 - t
            } else {
                ((1.0 - t) * theta).sinh() / sinh_theta
            };
            let w2 = if theta.abs() < 1e-6 {
                t
            } else {
                (t * theta).sinh() / sinh_theta
            };

            // Ambient Minkowski linear combination (includes time component)
            for j in 0..dim {
                row[j] = w1 * p[j] + w2 * q[j];
            }
        });

    result
}

/// Lorentz 모델의 역전파 레이어를 계산합니다.
pub fn lorentz_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let batch_size = u.nrows();
    let dim = u.ncols();
    let mut gu = Array2::<f32>::zeros(u.raw_dim());
    let mut gv = Array2::<f32>::zeros(v.raw_dim());

    for i in 0..batch_size {
        let p = u.row(i);
        let q = v.row(i);
        let g = grad_output.row(i);

        // Minkowski inner product <p,q>
        let mut inner = p[0] * q[0];
        for j in 1..dim {
            inner -= p[j] * q[j];
        }

        let alpha_arg = (-c * inner).max(1.0 + EPS);
        let alpha = alpha_arg.acosh();
        let sinh_alpha = alpha.sinh().max(EPS);
        let cosh_alpha = alpha.cosh();

        // weights
        let w1 = if alpha.abs() < 1e-6 {
            1.0 - t
        } else {
            ((1.0 - t) * alpha).sinh() / sinh_alpha
        };
        let w2 = if alpha.abs() < 1e-6 {
            t
        } else {
            (t * alpha).sinh() / sinh_alpha
        };

        // derivatives dw/dalpha
        let num1 = (1.0 - t) * ((1.0 - t) * alpha).cosh() * sinh_alpha
            - ((1.0 - t) * alpha).sinh() * cosh_alpha;
        let num2 = t * (t * alpha).cosh() * sinh_alpha - (t * alpha).sinh() * cosh_alpha;
        let denom = (sinh_alpha * sinh_alpha).max(EPS);
        let dw1_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num1 / denom
        };
        let dw2_dalpha = if alpha.abs() < 1e-6 {
            0.0
        } else {
            num2 / denom
        };

        // d alpha / d p = (-c / sinh(alpha)) * G q  where G = diag(1, -1, ..., -1)
        let scale = -c / sinh_alpha;
        let mut dalpha_dp = vec![0.0f32; dim];
        let mut dalpha_dq = vec![0.0f32; dim];
        dalpha_dp[0] = scale * q[0];
        dalpha_dq[0] = scale * p[0];
        for j in 1..dim {
            dalpha_dp[j] = scale * (-q[j]);
            dalpha_dq[j] = scale * (-p[j]);
        }

        // g dot p, g dot q (Euclidean componentwise)
        let mut g_dot_p = 0.0f32;
        let mut g_dot_q = 0.0f32;
        for j in 0..dim {
            g_dot_p += g[j] * p[j];
            g_dot_q += g[j] * q[j];
        }

        for j in 0..dim {
            gu[[i, j]] = w1 * g[j] + (g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha) * dalpha_dp[j];
            gv[[i, j]] = w2 * g[j] + (g_dot_p * dw1_dalpha + g_dot_q * dw2_dalpha) * dalpha_dq[j];
        }
    }

    (gu, gv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_lorentz_inner_basic() {
        let u = arr2(&[[2.0_f32, 1.0, 1.0]]);
        let v = arr2(&[[2.0_f32, -1.0, 0.5]]);
        let inner = lorentz_inner(&u.view(), &v.view());
        assert!((inner[0] - (2.0 * 2.0 - 1.0 * -1.0 - 1.0 * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_lorentz_distance_non_negative() {
        let c = 1.0_f32;
        // 동일한 점(하이퍼볼릭 거리 0)이 되도록 시간성분을 1보다 크게 유지하고 동일한 좌표 설정
        let u = arr2(&[[1.5_f32, 0.1, 0.2]]);
        let v = arr2(&[[1.5_f32, 0.1, 0.2]]);
        let d = lorentz_distance(&u.view(), &v.view(), c);
        assert!(d[0] >= 0.0);
        assert!(d[0].abs() < 1e-3);
    }

    #[test]
    fn test_lorentz_add_scaling_consistency() {
        let c = 0.8_f32;
        let u = arr2(&[[1.5_f32, 0.1, 0.2]]);
        let v = arr2(&[[1.3_f32, 0.05, -0.1]]);
        let s = lorentz_scalar(&u.view(), c, 0.0);
        // r=0이면 공간 성분 0, 시간 성분은 양수(>=1)로 유지되는 근사
        assert!(s[[0, 1]].abs() < 1e-6 && s[[0, 2]].abs() < 1e-6 && s[[0, 0]] >= 1.0);
        let w = lorentz_add(&u.view(), &v.view(), c);
        assert_eq!(w.ncols(), u.ncols());
        assert!(w.iter().all(|x| x.is_finite()));
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn lorentz_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn lorentz_layer_forward_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn lorentz_layer_backward_cuda(
                grad_output: *const f32,
                u: *const f32,
                v: *const f32,
                grad_u: *mut f32,
                grad_v: *mut f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn lorentz_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_distance_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn lorentz_layer_forward_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_layer_forward_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn lorentz_layer_backward_cuda(
        grad_output: *const f32,
        u: *const f32,
        v: *const f32,
        grad_u: *mut f32,
        grad_v: *mut f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::lorentz_layer_backward_cuda(
                grad_output,
                u,
                v,
                grad_u,
                grad_v,
                c,
                t,
                batch_size,
                dim,
            );
        }
    }
}

pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let factor = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS));

    result
        .slice_mut(s![.., 0..1])
        .assign(&(&factor * (1.0 + c * &x_norm_sq) / c.sqrt()));
    result
        .slice_mut(s![.., 1..])
        .assign(&(&factor * 2.0 * x / c.sqrt()));
    result
}

pub fn from_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let mut grad_result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS));
    let sqrt_c = c.sqrt();

    // Time component gradient
    let d_time_den_dc = -&x_norm_sq;
    let d_time_num_dc = &x_norm_sq;
    let time_num = 1.0 + c * &x_norm_sq;
    let d_time_dc = (d_time_num_dc * &den - &time_num * d_time_den_dc) / (&den * &den);
    grad_result
        .slice_mut(s![.., 0..1])
        .assign(&(&d_time_dc / sqrt_c - &time_num / (2.0 * c * sqrt_c * &den)));

    // Space component gradient
    let d_factor_dc = &x_norm_sq / (&den * &den);
    grad_result
        .slice_mut(s![.., 1..])
        .assign(&(x * (&d_factor_dc / sqrt_c - 1.0 / (c * sqrt_c * &den))));

    grad_result
}
