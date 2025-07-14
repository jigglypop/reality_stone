use ndarray::{s, Array1, Array2, ArrayView2, Axis};

use crate::ops::{batch::EPS, dot_batched, norm_sq_batched};

const BOUNDARY_EPS: f32 = 1e-5;

pub fn klein_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u_norm_sq = norm_sq_batched(u);
    let v_norm_sq = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    let numerator = 2.0 * (&u_norm_sq * &v_norm_sq - &uv * &uv);
    let denominator = ((1.0 - c * &u_norm_sq) * (1.0 - c * &v_norm_sq)).mapv(|v| v.max(EPS));
    let lambda = (numerator / denominator).mapv(f32::sqrt);
    let two_minus_lambda_sq = (2.0 - &lambda).mapv(|v| v.max(EPS));

    ((2.0 + lambda) / two_minus_lambda_sq).mapv(|v| v.acosh() / sqrtc)
}

pub fn klein_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let v_norm_sq = norm_sq_batched(v).insert_axis(Axis(1));

    let u_denom = (1.0 - c * &u_norm_sq).mapv_into(|v| v.max(EPS).sqrt());
    let v_denom = (1.0 - c * &v_norm_sq).mapv_into(|v| v.max(EPS).sqrt());

    let temp = u / &u_denom + v / &v_denom;
    let temp_norm_sq = norm_sq_batched(&temp.view()).insert_axis(Axis(1));

    let result_denom = (1.0 + (1.0 + c * temp_norm_sq).mapv(f32::sqrt)).mapv(|v| v.max(EPS));
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

    let d_scale_d_norm = (&norm_clamped * r).mapv(|val| {
        if val < 1.0 / c.sqrt() - BOUNDARY_EPS {
            0.0
        } else {
            -1.0 / (norm_clamped[[0, 0]] * norm_clamped[[0, 0]])
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
    let den = (1.0 + (1.0 - c * x_norm_sq).mapv(|v| v.max(0.0).sqrt())).mapv(|v| v.max(EPS));
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
