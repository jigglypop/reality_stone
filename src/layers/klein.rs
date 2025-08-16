use crate::ops::{batch::EPS, dot_batched, norm_sq_batched};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

#[inline]
fn safe_sqrt(x: f32) -> f32 {
    x.max(EPS).sqrt()
}

#[inline]
fn safe_acosh(x: f32) -> f32 {
    (x.max(1.0 + EPS)).acosh()
}

const BOUNDARY_EPS: f32 = 1e-5;

pub fn klein_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    // Stable lambda per Klein distance formulation
    let num = 2.0 * (&u2 * &v2 - &uv * &uv);
    let den = ((1.0 - c * &u2) * (1.0 - c * &v2)).mapv(|z| z.max(EPS));
    let lambda = (num / den).mapv(|z| z.max(0.0).sqrt());
    let ratio = ((2.0 + &lambda) / (2.0 - &lambda).mapv(|z| z.max(EPS)));
    ratio.mapv(|r| safe_acosh(r) / sqrtc)
}

pub fn klein_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let v_norm_sq = norm_sq_batched(v).insert_axis(Axis(1));

    let u_denom = (1.0 - c * &u_norm_sq).mapv_into(|v| safe_sqrt(v));
    let v_denom = (1.0 - c * &v_norm_sq).mapv_into(|v| safe_sqrt(v));

    let temp = u / &u_denom + v / &v_denom;
    let temp_norm_sq = norm_sq_batched(&temp.view()).insert_axis(Axis(1));

    let result_denom = (1.0 + (1.0 + c * temp_norm_sq).mapv(|z| safe_sqrt(z))).mapv(|v| v.max(EPS));
    temp / result_denom
}

pub fn klein_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    let scaled_norm = (&norm_clamped * r).mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
    let scale = scaled_norm / &norm_clamped;

    u * scale
}

pub fn klein_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + (1.0 - c * x_norm_sq).mapv(|v| v.max(0.0).sqrt())).mapv(|v| v.max(EPS));
    x / &den
}

pub fn klein_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let x0 = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS).sqrt());
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    result.slice_mut(s![.., 0..1]).assign(&x0);
    result.slice_mut(s![.., 1..]).assign(&(x * &x0));
    result
}

/// Klein 스칼라 곱의 VJP(Vector-Jacobian Product)를 계산합니다.
pub fn klein_scalar_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let norm = norm_sq_batched(x).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    let scaled_norm = (&norm_clamped * r).mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
    let scale = scaled_norm / &norm_clamped;

    let boundary = 1.0 / c.sqrt() - BOUNDARY_EPS;
    let d_scale_d_norm = (&norm_clamped).mapv(|n| {
        let rn = r * n;
        if rn < boundary {
            0.0
        } else {
            -1.0 / (n * n).max(EPS)
        }
    });

    let grad_norm_component = (grad_output * x).sum_axis(Axis(1)).insert_axis(Axis(1));
    let grad_x = grad_output * &scale + (grad_norm_component * d_scale_d_norm / &norm_clamped) * x;
    grad_x
}

/// Klein 덧셈의 VJP(Vector-Jacobian Product)를 계산합니다.
pub fn klein_add_vjp(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let v_norm_sq = norm_sq_batched(v).insert_axis(Axis(1));
    let u_denom = (1.0 - c * &u_norm_sq).mapv_into(|val| val.max(EPS).sqrt());
    let v_denom = (1.0 - c * &v_norm_sq).mapv_into(|val| val.max(EPS).sqrt());
    let temp = u / &u_denom + v / &v_denom;
    let temp_norm_sq = norm_sq_batched(&temp.view()).insert_axis(Axis(1));
    let result_denom_inner_sqrt = (1.0 + c * &temp_norm_sq).mapv(f32::sqrt);
    let result_denom = (1.0 + &result_denom_inner_sqrt).mapv(|val| val.max(EPS));

    let grad_temp_part1 = grad_output / &result_denom;
    let grad_result_denom = -(grad_output * &temp / (&result_denom * &result_denom))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let grad_temp_norm_sq = grad_result_denom * c / (2.0 * &result_denom_inner_sqrt);
    let grad_temp = grad_temp_part1 + 2.0 * &grad_temp_norm_sq * &temp;

    let grad_u_from_temp = &grad_temp / &u_denom;
    let grad_v_from_temp = &grad_temp / &v_denom;

    let grad_u_denom = -(&grad_temp * u / (&u_denom * &u_denom))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let grad_v_denom = -(&grad_temp * v / (&v_denom * &v_denom))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));

    let grad_u_norm_sq = grad_u_denom * (-c / (2.0 * &u_denom));
    let grad_v_norm_sq = grad_v_denom * (-c / (2.0 * &v_denom));

    let grad_u = grad_u_from_temp + 2.0 * &grad_u_norm_sq * u;
    let grad_v = grad_v_from_temp + 2.0 * &grad_v_norm_sq * v;

    (grad_u, grad_v)
}

/// Klein 모델의 순전파 레이어를 계산합니다.
pub fn klein_layer_forward(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let u_prime = klein_scalar(u, c, 1.0 - t);
    let v_prime = klein_scalar(v, c, t);
    klein_add(&u_prime.view(), &v_prime.view(), c)
}

/// Klein 모델의 역전파 레이어를 계산합니다.
pub fn klein_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_prime = klein_scalar(u, c, 1.0 - t);
    let v_prime = klein_scalar(v, c, t);
    let (grad_u_prime, grad_v_prime) =
        klein_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);
    let grad_u = klein_scalar_vjp(&grad_u_prime.view(), &u.view(), c, 1.0 - t);
    let grad_v = klein_scalar_vjp(&grad_v_prime.view(), &v.view(), c, t);
    (grad_u, grad_v)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn klein_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn klein_layer_forward_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn klein_layer_backward_cuda(
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

    pub fn klein_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::klein_distance_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn klein_layer_forward_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::klein_layer_forward_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn klein_layer_backward_cuda(
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
            ffi::klein_layer_backward_cuda(
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

pub fn to_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = 1.0 + c * &x_norm_sq;
    let den_clamped = den.mapv_into(|v| v.max(EPS));

    let numerator = -2.0 * x * &x_norm_sq;
    let denominator = &den_clamped * &den_clamped;

    numerator / denominator
}

pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + (1.0 - c * x_norm_sq).mapv(|v| safe_sqrt(v))).mapv(|v| v.max(EPS));
    x / &den
}

pub fn from_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let sqrt_expr = (1.0 - c * &x_norm_sq)
        .mapv_into(|v| v.max(EPS))
        .mapv(f32::sqrt);
    let den = 1.0 + &sqrt_expr;
    let den_clamped = den.mapv_into(|v| v.max(EPS));

    let d_sqrt_expr_dc = -0.5 * &x_norm_sq / &sqrt_expr;
    let d_den_dc = &d_sqrt_expr_dc;

    let numerator = -x * d_den_dc;
    let denominator = &den_clamped * &den_clamped;

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_klein_add_identity_like() {
        let c = 0.9_f32;
        let u = arr2(&[[0.1_f32, 0.2]]);
        let z = arr2(&[[0.0_f32, 0.0]]);
        let res = klein_add(&u.view(), &z.view(), c);
        // 이 구현에서 v=0일 때 결과는 u / (sqrt(1 - c||u||^2) + 1)
        let u_norm_sq = (u[[0, 0]].powi(2) + u[[0, 1]].powi(2));
        let u_denom = (1.0 - c * u_norm_sq).max(EPS).sqrt();
        let expected_scale = 1.0 / (u_denom + 1.0);
        let expected = u.mapv(|x| x * expected_scale);
        assert!(((res - expected).mapv(f32::abs)).sum() < 1e-6);
    }

    #[test]
    fn test_klein_scalar_bounds() {
        let c = 0.5_f32;
        let x = arr2(&[[0.4_f32, 0.2]]);
        let y = klein_scalar(&x.view(), c, 2.0);
        // 경계 밖으로 나가지 않도록 norm이 제한됨
        let norm = (y[[0, 0]].powi(2) + y[[0, 1]].powi(2)).sqrt();
        assert!(norm < 1.0 / c.sqrt());
    }

    #[test]
    fn test_klein_to_poincare_and_back_shapes() {
        let c = 0.7_f32;
        let x = arr2(&[[0.1_f32, -0.2], [0.05, 0.05]]);
        let p = klein_to_poincare(&x.view(), c);
        assert_eq!(p.dim(), x.dim());
        let l = klein_to_lorentz(&x.view(), c);
        assert_eq!(l.ncols(), x.ncols() + 1);
    }
}
