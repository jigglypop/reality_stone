use crate::ops::utils::{dot_batched, norm_sq_batched, EPS};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

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