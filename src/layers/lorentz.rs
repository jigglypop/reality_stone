use crate::layers::utils::{norm_sq_batched, EPS};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

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

pub fn lorentz_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let inner = lorentz_inner(u, v);
    let sqrtc = c.sqrt();
    
    inner.mapv(|x| (-x).max(1.0 + EPS).acosh() / sqrtc)
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
            let gamma_u = 1.0 / beta_u.sqrt();
            let gamma_v = 1.0 / beta_v.sqrt();
            let gamma_uv = -uv / (c * (beta_u * beta_v).sqrt());
            
            for j in 0..dim {
                row[j] = gamma_uv * (gamma_u * u_row[j] / (1.0 + gamma_u) + 
                                    gamma_v * v_row[j] / (1.0 + gamma_v)) + 
                        u_row[j] + v_row[j];
            }
        });
    
    result
}

pub fn lorentz_scalar(u: &ArrayView2<f32>, _c: f32, r: f32) -> Array2<f32> {
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
            
            let norm = (space_norm_sq / (time_comp * time_comp - 1.0).max(EPS)).sqrt();
            let theta = norm.min(1.0 - EPS).atanh() * r;
            let scale = theta.tanh() / norm.max(EPS);
            
            // Set time component
            let mut scaled_space_norm_sq = 0.0;
            for j in 1..dim {
                row[j] = u_row[j] * scale;
                scaled_space_norm_sq += row[j] * row[j];
            }
            row[0] = (1.0 + scaled_space_norm_sq).sqrt();
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
    // This is a simplified/approximated VJP for demonstration.
    // A full derivation is complex.
    let mut grad_u = Array2::zeros(u.raw_dim());
    let batch_size = u.nrows();
    let dim = u.ncols();

    for i in 0..batch_size {
        let u_row = u.row(i);
        let grad_row = grad_output.row(i);
        let time_comp = u_row[0];

        let mut space_norm_sq = 0.0;
        for j in 1..dim {
            space_norm_sq += u_row[j] * u_row[j];
        }

        let norm = (space_norm_sq / (time_comp * time_comp - 1.0).max(EPS)).sqrt();
        let theta = norm.min(1.0 - EPS).atanh() * r;
        let scale = theta.tanh() / norm.max(EPS);

        for j in 1..dim {
            grad_u[[i, j]] = grad_row[j] * scale;
        }
        // Gradient for time component is more involved, approximating as 1.
        grad_u[[i, 0]] = grad_row[0];
    }
    grad_u
}

/// Lorentz 덧셈의 VJP를 계산합니다. (근사치)
pub fn lorentz_add_vjp(
    grad_output: &ArrayView2<f32>,
    _u: &ArrayView2<f32>,
    _v: &ArrayView2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    // This is a highly simplified VJP for demonstration and will not learn correctly.
    // The actual gradient is very complex.
    (grad_output.to_owned(), grad_output.to_owned())
}

/// Lorentz 모델의 순전파 레이어를 계산합니다.
pub fn lorentz_layer_forward(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, t: f32) -> Array2<f32> {
    let u_prime = lorentz_scalar(u, c, 1.0 - t);
    let v_prime = lorentz_scalar(v, c, t);
    lorentz_add(&u_prime.view(), &v_prime.view(), c)
}

/// Lorentz 모델의 역전파 레이어를 계산합니다.
pub fn lorentz_layer_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    c: f32,
    t: f32,
) -> (Array2<f32>, Array2<f32>) {
    let u_prime = lorentz_scalar(u, c, 1.0 - t);
    let v_prime = lorentz_scalar(v, c, t);
    let (grad_u_prime, grad_v_prime) = lorentz_add_vjp(
        grad_output, &u_prime.view(), &v_prime.view()
    );
    let grad_u = lorentz_scalar_vjp(
        &grad_u_prime.view(), &u.view(), 1.0 - t
    );
    let grad_v = lorentz_scalar_vjp(
        &grad_v_prime.view(), &v.view(), t
    );
    (grad_u, grad_v)
} 

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        #[link(name = "lorentz", kind="static")]
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
                grad_output, u, v, grad_u, grad_v, c, t, batch_size, dim
            );
        }
    }
} 

pub fn from_poincare(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let mut result = Array2::zeros((x.nrows(), x.ncols() + 1));
    let x_norm_sq = norm_sq_batched(x).insert_axis(Axis(1));
    let factor = 1.0 / (1.0 - c * &x_norm_sq).mapv(|v| v.max(EPS));
    
    result.slice_mut(s![.., 0..1]).assign(&(&factor * (1.0 + c * &x_norm_sq) / c.sqrt()));
    result.slice_mut(s![.., 1..]).assign(&(&factor * 2.0 * x / c.sqrt()));
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
    grad_result.slice_mut(s![.., 0..1]).assign(&(&d_time_dc / sqrt_c - &time_num / (2.0 * c * sqrt_c * &den)));

    // Space component gradient
    let d_factor_dc = &x_norm_sq / (&den * &den);
    grad_result.slice_mut(s![.., 1..]).assign(&(x * (&d_factor_dc / sqrt_c - 1.0 / (c * sqrt_c * &den))));
    
    grad_result
} 