use crate::ops::{batch::EPS, dot_batched, mobius, norm_sq_batched};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

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

pub fn poincare_ball_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    let (grad_u_prime, grad_v_prime) =
        mobius_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);

    let grad_u = mobius_scalar_vjp(&grad_u_prime.view(), &u.view(), c, 1.0 - t);
    let grad_v = mobius_scalar_vjp(&grad_v_prime.view(), &v.view(), c, t);

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

pub fn poincare_ball_layer(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> Array2<f32> {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    mobius::mobius_add(&u_prime.view(), &v_prime.view(), c)
}

pub fn poincare_ball_layer_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &mobius::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (result, _) = mobius::mobius_add_dynamic(&u_prime.view(), &v_prime.view(), dynamic_c);
    (result, c)
}

pub fn poincare_ball_layer_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &mobius::DynamicCurvature,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);

    let (grad_u_prime, grad_v_prime, grad_kappa) = mobius::mobius_add_dynamic_backward(
        grad_output,
        &u_prime.view(),
        &v_prime.view(),
        dynamic_c,
    );

    let grad_u = mobius_scalar_vjp(&grad_u_prime.view(), u, c, 1.0 - t);
    let grad_v = mobius_scalar_vjp(&grad_v_prime.view(), v, c, t);

    (grad_u, grad_v, grad_kappa)
}

pub fn poincare_ball_layer_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &mobius::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (result, _) = mobius::mobius_add_layerwise(
        &u_prime.view(),
        &v_prime.view(),
        layer_curvatures,
        layer_idx,
    );
    (result, c)
}

pub fn poincare_ball_layer_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &mobius::LayerWiseDynamicCurvature,
    layer_idx: usize,
    t: f32,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    let (grad_u_prime, grad_v_prime) =
        mobius_add_vjp(grad_output, &u_prime.view(), &v_prime.view(), c);
    let grad_u = mobius_scalar_vjp(&grad_u_prime.view(), u, c, 1.0 - t);
    let grad_v = mobius_scalar_vjp(&grad_v_prime.view(), v, c, t);
    let grad_c_from_add_tensor = mobius::mobius_add_grad_c(&u_prime.view(), &v_prime.view(), c);
    let grad_c_add = (grad_output * &grad_c_from_add_tensor).sum();
    let grad_c_from_scalar_u_tensor = mobius::mobius_scalar_grad_c(u, c, 1.0 - t);
    let grad_c_scalar_u = (&grad_u_prime * &grad_c_from_scalar_u_tensor).sum();
    let grad_c_from_scalar_v_tensor = mobius::mobius_scalar_grad_c(v, c, t);
    let grad_c_scalar_v = (&grad_v_prime * &grad_c_from_scalar_v_tensor).sum();
    let grad_c_total = grad_c_add + grad_c_scalar_u + grad_c_scalar_v;
    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c_total * dc_dkappa;
    (grad_u, grad_v, grad_kappa)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::lorentz; // Lorentz 모듈 import
    use approx::assert_relative_eq;
    use ndarray::arr2;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_mobius_add_identity() {
        let c = 1.0;
        let x = arr2(&[[0.1, 0.2]]);
        let z = arr2(&[[0.0, 0.0]]);
        let result = mobius::mobius_add(&x.view(), &z.view(), c);
        assert_relative_eq!(result, x, epsilon = EPSILON);
    }

    #[test]
    fn test_poincare_to_lorentz_and_back() {
        let c = 1.0;
        let x_poincare = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        let x_lorentz = poincare_to_lorentz(&x_poincare.view(), c);
        let x_poincare_restored = lorentz::lorentz_to_poincare(&x_lorentz.view(), c);

        assert_relative_eq!(x_poincare, x_poincare_restored, epsilon = EPSILON);
    }

    #[test]
    fn test_poincare_ball_layer_interpolation() {
        let c = 1.0;
        let u = arr2(&[[0.5, 0.5]]);
        let v = arr2(&[[-0.5, -0.5]]);

        // t=0 이면 u와 같아야 함
        let result_t0 = poincare_ball_layer(&u.view(), &v.view(), c, 0.0);
        assert_relative_eq!(result_t0, u, epsilon = EPSILON);

        // t=1 이면 v와 같아야 함
        let result_t1 = poincare_ball_layer(&u.view(), &v.view(), c, 1.0);
        assert_relative_eq!(result_t1, v, epsilon = EPSILON);
    }

    #[test]
    fn test_distance_is_zero_for_same_point() {
        let c = 1.0;
        let x = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        let dist = poincare_distance(&x.view(), &x.view(), c);

        for val in dist.iter() {
            assert_relative_eq!(*val, 0.0, epsilon = EPSILON);
        }
    }
}
