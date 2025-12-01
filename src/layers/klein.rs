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

/// 클라인 거리 (Klein Distance)
///
/// d_K(u,v) = (1/√c) * acosh((1 - c⟨u,v⟩) / √((1-c||u||²)(1-c||v||²)))
pub fn klein_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    let numerator = 1.0 - c * &uv;
    let denominator = ((1.0 - c * &u2) * (1.0 - c * &v2)).mapv(|z| z.max(EPS).sqrt());
    let arg = (&numerator / &denominator).mapv(|z| z.max(1.0 + EPS));
    arg.mapv(|r| safe_acosh(r) / sqrtc)
}

/// 클라인 덧셈 (Einstein Addition)
///
/// u (+) v = (1 / (1 + c<u,v>)) * (u + v/gamma_u + (c*gamma_u / (1+gamma_u)) * <u,v> * u)
/// where gamma_u = 1 / sqrt(1 - c|u|^2)
pub fn klein_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    
    let gamma_u = (1.0 - c * &u_norm_sq).mapv(|val| 1.0 / safe_sqrt(val));
    let denom = 1.0 + c * &uv;
    let denom_inv = denom.mapv(|val| 1.0 / val.max(EPS));
    
    let coeff_v = &gamma_u.mapv(|g| 1.0 / g); // 1/gamma_u = sqrt(1-c|u|^2)
    let coeff_u_part = (c * &gamma_u * &uv) / (1.0 + &gamma_u);
    let coeff_u = 1.0 + &coeff_u_part;
    
    // Result = denom_inv * (coeff_u * u + coeff_v * v)
    let mut result = u * &coeff_u;
    result = &result + &(v * coeff_v);
    result * &denom_inv
}

/// 클라인 덧셈의 역전파 (VJP)
pub fn klein_add_vjp(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
) -> (Array2<f32>, Array2<f32>) {
    // Forward 중간값 재계산
    let u_norm_sq = norm_sq_batched(u).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    
    let gamma_u = (1.0 - c * &u_norm_sq).mapv(|val| 1.0 / safe_sqrt(val));
    let denom = 1.0 + c * &uv;
    let denom_inv = denom.mapv(|val| 1.0 / val.max(EPS));
    
    let inv_gamma_u = gamma_u.mapv(|g| 1.0 / g); // 1/gamma_u
    let coeff_u_part = (c * &gamma_u * &uv) / (1.0 + &gamma_u);
    let coeff_u = 1.0 + &coeff_u_part;
    
    let num = u * &coeff_u + v * &inv_gamma_u; // 분자
    // output = num / denom
    
    // 그라디언트 계산
    // dL/dNum = grad_output / denom
    let grad_num = grad_output * &denom_inv;
    
    // dL/dDenom = - <grad_output, num> / denom^2 = - <grad_output, output> / denom
    let output = &num * &denom_inv;
    let grad_denom = -(grad_output * &output).sum_axis(Axis(1)).insert_axis(Axis(1)) * &denom_inv;
    
    // Denom = 1 + c <u,v>
    // dL/du += c * grad_denom * v
    // dL/dv += c * grad_denom * u
    let mut grad_u = v * (&grad_denom * c);
    let mut grad_v = u * (&grad_denom * c);
    
    // Num = coeff_u * u + inv_gamma_u * v
    // dL/du += coeff_u * grad_num
    // dL/dv += inv_gamma_u * grad_num
    grad_u = &grad_u + &(&grad_num * &coeff_u);
    grad_v = &grad_v + &(&grad_num * &inv_gamma_u);
    
    // dL/d_coeff_u = <grad_num, u>
    let grad_coeff_u = (&grad_num * u).sum_axis(Axis(1)).insert_axis(Axis(1));
    // dL/d_inv_gamma_u = <grad_num, v>
    let grad_inv_gamma_u = (&grad_num * v).sum_axis(Axis(1)).insert_axis(Axis(1));
    
    // inv_gamma_u = sqrt(1 - c|u|^2)
    // d_inv_gamma_u / du = -c * gamma_u * u
    grad_u = &grad_u - &(u * (&grad_inv_gamma_u * c * &gamma_u));
    
    // coeff_u 미분
    let d_coeff_u_d_uv = c * &gamma_u / (1.0 + &gamma_u);
    let d_coeff_u_d_gamma_u = (c * &uv) / ((1.0 + &gamma_u) * (1.0 + &gamma_u));
    
    let grad_uv = &grad_coeff_u * &d_coeff_u_d_uv;
    grad_u = &grad_u + &(v * &grad_uv);
    grad_v = &grad_v + &(u * &grad_uv);
    
    let grad_gamma_u = &grad_coeff_u * &d_coeff_u_d_gamma_u;
    grad_u = &grad_u + &(u * (&grad_gamma_u * c * &gamma_u * &gamma_u * &gamma_u));
    
    (grad_u, grad_v)
}

/// 클라인 스칼라 곱 (Klein Scalar Multiplication)
pub fn klein_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    let scaled_norm = (&norm_clamped * r).mapv(|v| v.min(1.0 / c.sqrt() - BOUNDARY_EPS));
    let scale = scaled_norm / &norm_clamped;

    u * scale
}

/// 클라인 -> 푸앵카레 변환
pub fn klein_to_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + (1.0 - c * x_norm_sq).mapv(|v| v.max(0.0).sqrt())).mapv(|v| v.max(EPS));
    x / &den
}

/// 클라인 -> 로렌츠 변환
pub fn klein_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let x0 = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS).sqrt());
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    result.slice_mut(s![.., 0..1]).assign(&x0);
    result.slice_mut(s![.., 1..]).assign(&(x * &x0));
    result
}

/// 클라인 스칼라 곱의 VJP
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


/// 클라인 모델의 순전파 레이어
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

/// 클라인 모델의 역전파 레이어
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

/// 클라인 -> 푸앵카레 곡률 그라디언트
pub fn to_poincare_grad_c(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = 1.0 + c * &x_norm_sq;
    let den_clamped = den.mapv_into(|v| v.max(EPS));

    let numerator = -2.0 * x * &x_norm_sq;
    let denominator = &den_clamped * &den_clamped;

    numerator / denominator
}

/// 푸앵카레 -> 클라인 변환
pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    // Poincaré -> Klein: 2x / (1 + c||x||^2)
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 + c * &x_norm_sq).mapv(|v| v.max(EPS));
    (2.0 * x) / &den
}

/// 푸앵카레 -> 클라인 곡률 그라디언트
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
