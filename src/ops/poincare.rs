use crate::ops::mobius;
use crate::ops::utils::{dot_batched, norm_sq_batched, EPS};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

fn mobius_scalar_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let sqrt_c = c.sqrt();
    let x_norm = norm_sq_batched(&x).mapv(f32::sqrt).insert_axis(Axis(1));
    let x_norm_clamp = x_norm.mapv(|v| v.max(EPS));
    let atanh_arg = (sqrt_c * &x_norm_clamp).mapv(|v| v.min(1.0 - EPS));
    let atanh_val = atanh_arg.mapv(|v| v.atanh());
    let tanh_val = (r * &atanh_val).mapv(|v| v.tanh());
    let scale = &tanh_val / (sqrt_c * &x_norm_clamp);
    let grad_scale = (grad_output * x).sum_axis(Axis(1)).insert_axis(Axis(1));
    let inner_deriv_atanh = r * (1.0 - &tanh_val * &tanh_val);
    let inner_deriv_norm = (1.0 / (1.0 - &atanh_arg * &atanh_arg)) * (sqrt_c / &x_norm_clamp);
    let inner_deriv = inner_deriv_atanh * inner_deriv_norm;
    let grad_scale_b = grad_scale * (inner_deriv - &scale / &x_norm_clamp);
    let grad_x = grad_output * &scale + (grad_scale_b / &x_norm_clamp) * x;
    grad_x
}

fn mobius_add_vjp(
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
    let grad_den = -(grad_output * &output / &den_clamp).sum_axis(Axis(1)).insert_axis(Axis(1));

    let grad_x_from_u = &grad_u * (1.0 + 2.0 * c * &xy + c * &y2);
    let grad_y_from_u = &grad_u * (1.0 - c * &x2);
    
    let grad_xy_from_u = (2.0 * c * (&grad_u * x)).sum_axis(Axis(1)).insert_axis(Axis(1));
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

pub fn poincare_ball_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    // 1. Recompute u_prime and v_prime
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    // 2. Compute VJPs for mobius_add
    let (grad_u_prime, grad_v_prime) = mobius_add_vjp(
        grad_output, &u_prime.view(), &v_prime.view(), c
    );

    // 3. Compute VJPs for mobius_scalar
    let grad_u = mobius_scalar_vjp(
        &grad_u_prime.view(), &u.view(), c, 1.0 - t
    );
    let grad_v = mobius_scalar_vjp(
        &grad_v_prime.view(), &v.view(), c, t
    );
    
    (grad_u, grad_v)
}

pub fn poincare_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let sqrtc = c.sqrt();
    let u2 = norm_sq_batched(u);
    let v2 = norm_sq_batched(v);
    let uv = dot_batched(u, v);

    let norm_sq_diff = (&u2 + &v2 - 2.0 * &uv).mapv_into(|val| val.max(EPS));
    let den = (1.0 - c * &u2) * (1.0 - c * &v2);
    let den_clamped = den.mapv_into(|val| val.max(EPS));

    let frac = norm_sq_diff / den_clamped;
    frac.mapv_into(|val| (2.0 / sqrtc) * (c * val).sqrt().atanh())
}

pub fn poincare_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = (1.0 - c * &x_norm_sq).mapv_into(|v| v.max(EPS));
    let sqrtc = c.sqrt();

    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let time_component = (1.0 + c * &x_norm_sq) / (&den * sqrtc);
    let space_components = (2.0 * x) / (&den * sqrtc);

    result.slice_mut(s![.., 0..1]).assign(&time_component);
    result.slice_mut(s![.., 1..]).assign(&space_components);
    result
}

pub fn poincare_to_klein(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let den = 1.0 + c * &x_norm_sq;
    let den_clamped = den.mapv_into(|v| v.max(EPS));
    (2.0 * x) / &den_clamped
}

pub fn poincare_ball_layer(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, t: f32) -> Array2<f32> {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    mobius::mobius_add(&u_prime.view(), &v_prime.view(), c)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        #[link(name = "kernel_poincare", kind="static")]
        extern "C" {
            pub fn poincare_distance_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn poincare_ball_layer_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                t: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn poincare_ball_layer_backward_cuda(
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

    pub fn poincare_distance_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_distance_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn poincare_ball_layer_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        t: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::poincare_ball_layer_cuda(out, u, v, c, t, batch_size, dim);
        }
    }

    pub fn poincare_ball_layer_backward_cuda(
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
            ffi::poincare_ball_layer_backward_cuda(
                grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim
            );
        }
    }
} 