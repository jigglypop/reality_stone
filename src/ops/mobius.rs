use crate::{
    ops::{
        batch::EPS,
        batch::{dot_batched_f64, norm_sq_batched_f64, EPS64},
        dot_batched, norm_sq_batched, DynamicCurvature, LayerWiseDynamicCurvature,
    },
};
use ndarray::{Array2, ArrayView2, Axis};

const BOUNDARY_EPS: f32 = 1e-5;
const MIN_DENOMINATOR: f32 = 1e-6;
const BOUNDARY_EPS64: f64 = 1e-12;
const MIN_DENOMINATOR64: f64 = 1e-12;

pub fn mobius_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u2 = norm_sq_batched(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched(v).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    let c2 = c * c;

    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR));
    let coeff_u = (1.0 + 2.0 * c * &uv + c * &v2) / &den;
    let coeff_v = (1.0 - c * &u2) / &den;

    coeff_u * u + coeff_v * v
}

pub fn mobius_add_f64(u: &ArrayView2<f64>, v: &ArrayView2<f64>, c: f64) -> Array2<f64> {
    let u2 = norm_sq_batched_f64(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched_f64(v).insert_axis(Axis(1));
    let uv = dot_batched_f64(u, v).insert_axis(Axis(1));
    let c2 = c * c;
    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR64));
    let coeff_u = (1.0 + 2.0 * c * &uv + c * &v2) / &den;
    let coeff_v = (1.0 - c * &u2) / &den;
    coeff_u * u + coeff_v * v
}

pub fn mobius_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));

    if c.abs() < EPS {
        // c = 0: 유클리드 경우
        return Array2::from_elem(u.dim(), r) * u;
    }
    // 음수 곡률도 처리 가능하도록 수정
    // sqrt(c) * norm이 1보다 작아야 atanh가 정의됨
    let sqrt_c_norm = if c > 0.0 {
        (c.sqrt() * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS))
    } else {
        (-c).sqrt() * &norm_clamped
    };
    let scale = if c > 0.0 {
        // 양수 곡률: 원래 공식
        let alpha = sqrt_c_norm.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        beta / (c.sqrt() * &norm_clamped)
    } else {
        // 음수 곡률: atanh(i*x) = i*atan(x), tanh(i*x) = i*tan(x)
        let alpha = sqrt_c_norm.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        beta / ((-c).sqrt() * &norm_clamped)
    };
    scale * u
}

pub fn mobius_scalar_f64(u: &ArrayView2<f64>, c: f64, r: f64) -> Array2<f64> {
    let norm = norm_sq_batched_f64(u).mapv(f64::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS64));
    if c.abs() < EPS64 {
        return Array2::from_elem(u.dim(), r) * u;
    }
    let sqrt_c_norm = if c > 0.0 {
        (c.sqrt() * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS64))
    } else {
        (-c).sqrt() * &norm_clamped
    };
    let scale = if c > 0.0 {
        let alpha = sqrt_c_norm.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        beta / (c.sqrt() * &norm_clamped)
    } else {
        let alpha = sqrt_c_norm.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        beta / ((-c).sqrt() * &norm_clamped)
    };
    scale * u
}

pub fn mobius_scalar_grad_c(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    if c.abs() < EPS {
        // c = 0: gradient is 0
        return Array2::zeros(u.dim());
    }

    if c > 0.0 {
        // 양수 곡률
        let sqrt_c = c.sqrt();
        let scn = (sqrt_c * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS));
        let alpha = scn.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());

        // d(sqrt(c))/dc = 0.5/sqrt(c)
        let d_sqrt_c_dc = 0.5 / sqrt_c;

        // d(alpha)/d(scn) = 1/(1 - scn^2)
        let d_alpha_dscn = 1.0 / (1.0 - &scn * &scn).mapv(|v| v.max(EPS));

        // d(beta)/d(alpha) = r * (1 - tanh^2(r*alpha))
        let tanh_r_alpha = (r * &alpha).mapv(|v| v.tanh());
        let d_beta_dalpha = r * (1.0 - &tanh_r_alpha * &tanh_r_alpha);
        // Chain rule
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_c_dc;
        let d_scale_dc = (&d_beta_dc * sqrt_c - &beta * d_sqrt_c_dc) / (c * &norm_clamped);
        &d_scale_dc * u
    } else {
        // 음수 곡률
        let sqrt_abs_c = (-c).sqrt();
        let scn = sqrt_abs_c * &norm_clamped;
        let alpha = scn.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        // d(sqrt(|c|))/dc = -0.5/sqrt(|c|) (c가 음수이므로)
        let d_sqrt_abs_c_dc = -0.5 / sqrt_abs_c;
        // d(alpha)/d(scn) = 1/(1 + scn^2)
        let d_alpha_dscn = 1.0 / (1.0 + &scn * &scn);
        // d(beta)/d(alpha) = r * (1 + tan^2(r*alpha))
        let tan_r_alpha = (r * &alpha).mapv(|v| v.tan());
        let d_beta_dalpha = r * (1.0 + &tan_r_alpha * &tan_r_alpha);
        // Chain rule
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_abs_c_dc;
        // d(scale)/dc
        let d_scale_dc =
            (&d_beta_dc * sqrt_abs_c - &beta * d_sqrt_abs_c_dc) / ((-c) * &norm_clamped);
        &d_scale_dc * u
    }
}

pub fn mobius_scalar_grad_c_f64(u: &ArrayView2<f64>, c: f64, r: f64) -> Array2<f64> {
    let norm = norm_sq_batched_f64(u).mapv(f64::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS64));
    if c.abs() < EPS64 {
        return Array2::zeros(u.dim());
    }
    if c > 0.0 {
        let sqrt_c = c.sqrt();
        let scn = (sqrt_c * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS64));
        let alpha = scn.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        let d_sqrt_c_dc = 0.5 / sqrt_c;
        let d_alpha_dscn = 1.0 / (1.0 - &scn * &scn).mapv(|v| v.max(EPS64));
        let tanh_r_alpha = (r * &alpha).mapv(|v| v.tanh());
        let d_beta_dalpha = r * (1.0 - &tanh_r_alpha * &tanh_r_alpha);
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_c_dc;
        let d_scale_dc = (&d_beta_dc * sqrt_c - &beta * d_sqrt_c_dc) / (c * &norm_clamped);
        &d_scale_dc * u
    } else {
        let sqrt_abs_c = (-c).sqrt();
        let scn = sqrt_abs_c * &norm_clamped;
        let alpha = scn.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        let d_sqrt_abs_c_dc = -0.5 / sqrt_abs_c;
        let d_alpha_dscn = 1.0 / (1.0 + &scn * &scn);
        let tan_r_alpha = (r * &alpha).mapv(|v| v.tan());
        let d_beta_dalpha = r * (1.0 + &tan_r_alpha * &tan_r_alpha);
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_abs_c_dc;
        let d_scale_dc =
            (&d_beta_dc * sqrt_abs_c - &beta * d_sqrt_abs_c_dc) / ((-c) * &norm_clamped);
        &d_scale_dc * u
    }
}

pub fn mobius_add_grad_c(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u2 = norm_sq_batched(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched(v).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    let c2 = c * c;
    let num = (1.0 + 2.0 * c * &uv + c * &v2) * u + (1.0 - c * &u2) * v;
    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR));
    let dnum_dc = (2.0 * &uv + &v2) * u - &u2 * v;
    let dden_dc = 2.0 * &uv + 2.0 * c * &u2 * &v2;
    let result = (dnum_dc * &den - &num * &dden_dc) / (&den * &den);
    result
}

pub fn mobius_add_grad_c_f64(u: &ArrayView2<f64>, v: &ArrayView2<f64>, c: f64) -> Array2<f64> {
    let u2 = norm_sq_batched_f64(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched_f64(v).insert_axis(Axis(1));
    let uv = dot_batched_f64(u, v).insert_axis(Axis(1));
    let c2 = c * c;
    let num = (1.0 + 2.0 * c * &uv + c * &v2) * u + (1.0 - c * &u2) * v;
    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR64));
    let dnum_dc = (2.0 * &uv + &v2) * u - &u2 * v;
    let dden_dc = 2.0 * &uv + 2.0 * c * &u2 * &v2;
    let result = (dnum_dc * &den - &num * &dden_dc) / (&den * &den);
    result
}

// --- VJP Implementations (Moved from layers/poincare.rs) ---

pub fn mobius_scalar_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let x_norm = norm_sq_batched(&x).mapv(f32::sqrt).insert_axis(Axis(1));
    let x_norm_clamp = x_norm.mapv(|v| v.max(EPS));
    if c.abs() < EPS {
        // c = 0: 유클리드 경우
        return grad_output * r;
    }

    if c > 0.0 {
        // 양수 곡률
        let sqrt_c = c.sqrt();
        let scn = (sqrt_c * &x_norm_clamp).mapv(|v| v.min(1.0 - EPS));
        let alpha = scn.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        let scale = &beta / (sqrt_c * &x_norm_clamp);
        let grad_scale = (grad_output * x).sum_axis(Axis(1)).insert_axis(Axis(1));
        let inner_deriv_atanh = r * (1.0 - &beta * &beta);
        let inner_deriv_norm =
            (1.0 / (1.0 - &scn * &scn).mapv(|v| v.max(EPS))) * (sqrt_c / &x_norm_clamp);
        let grad_scale_b = &grad_scale * (&inner_deriv_atanh * &inner_deriv_norm - &scale * sqrt_c);
        grad_output * &scale + x * &grad_scale_b / (sqrt_c * &x_norm_clamp)
    } else {
        // 음수 곡률
        let sqrt_abs_c = (-c).sqrt();
        let scn = sqrt_abs_c * &x_norm_clamp;
        let alpha = scn.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        let scale = &beta / (sqrt_abs_c * &x_norm_clamp);

        let grad_scale = (grad_output * x).sum_axis(Axis(1)).insert_axis(Axis(1));
        let inner_deriv_atan = r * (1.0 + &beta * &beta);
        let inner_deriv_norm = (1.0 / (1.0 + &scn * &scn)) * (sqrt_abs_c / &x_norm_clamp);

        let grad_scale_b =
            &grad_scale * (&inner_deriv_atan * &inner_deriv_norm - &scale * sqrt_abs_c);

        grad_output * &scale + x * &grad_scale_b / (sqrt_abs_c * &x_norm_clamp)
    }
}

pub fn mobius_add_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    c: f32,
) -> (Array2<f32>, Array2<f32>) {
    let x2 = norm_sq_batched(&x).insert_axis(Axis(1));
    let y2 = norm_sq_batched(&y).insert_axis(Axis(1));
    let xy = dot_batched(&x, &y).insert_axis(Axis(1));

    let den = 1.0 + 2.0 * c * &xy + c * c * &x2 * &y2;
    let den_clamp = den.mapv(|v| v.max(EPS));

    let u = (1.0 + 2.0 * c * &xy + c * &y2) * x + (1.0 - c * &x2) * y;
    let output = &u / &den_clamp;

    let grad_u = grad_output / &den_clamp;
    let grad_den = -(grad_output * &output / &den_clamp)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));

    let grad_x_from_u = &grad_u * (1.0 + 2.0 * c * &xy + c * &y2);
    let grad_y_from_u = &grad_u * (1.0 - c * &x2);

    let grad_xy_from_u = (2.0 * c * (&grad_u * x))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let grad_x2_from_u = (-c * (&grad_u * y)).sum_axis(Axis(1)).insert_axis(Axis(1));

    let grad_xy_from_den = 2.0 * c * &grad_den;
    let grad_x2_from_den = c * c * &y2 * &grad_den;
    let grad_y2_from_den = c * c * &x2 * &grad_den;

    let grad_xy = grad_xy_from_u + grad_xy_from_den;
    let grad_x2 = grad_x2_from_u + grad_x2_from_den;
    let grad_y2 = grad_y2_from_den;

    let grad_x = grad_x_from_u + 2.0 * &grad_x2 * x + &grad_xy * y;
    let grad_y = grad_y_from_u + 2.0 * &grad_y2 * y + &grad_xy * x;

    (grad_x, grad_y)
}

// 동적 곡률을 사용한 Mobius 덧셈
pub fn mobius_add_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &DynamicCurvature,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let result = mobius_add(u, v, c);
    (result, c)
}

// 동적 곡률 Mobius 덧셈의 backward pass
pub fn mobius_add_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &DynamicCurvature,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let grad_c_tensor = mobius_add_grad_c(u, v, c);
    let grad_c = (grad_output * &grad_c_tensor).sum();
    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;
    let (grad_u, grad_v) = mobius_add_vjp(grad_output, u, v, c);
    (grad_u, grad_v, grad_kappa)
}

pub fn mobius_add_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let result = mobius_add(u, v, c);
    (result, c)
}

pub fn mobius_add_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let grad_c_tensor = mobius_add_grad_c(u, v, c);
    let grad_c = (grad_output * &grad_c_tensor).sum();
    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c * dc_dkappa;
    let (grad_u, grad_v) = mobius_add_vjp(grad_output, u, v, c);
    (grad_u, grad_v, grad_kappa)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn mobius_add_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn mobius_scalar_cuda(
                out: *mut f32,
                u: *const f32,
                c: f32,
                r: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn mobius_add_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::mobius_add_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn mobius_scalar_cuda(
        out: *mut f32,
        u: *const f32,
        c: f32,
        r: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::mobius_scalar_cuda(out, u, c, r, batch_size, dim);
        }
    }
}
