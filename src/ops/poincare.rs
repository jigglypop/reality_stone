use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use crate::ops::mobius;

const EPS: f32 = 1e-7;

pub fn poincare_distance(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array1<f32> {
    let batch_size = u.nrows();
    let mut result = Array1::zeros(batch_size);
    
    result
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, dist)| {
            let u_row = u.row(i);
            let v_row = v.row(i);
            
            let u2 = u_row.dot(&u_row);
            let v2 = v_row.dot(&v_row);
            let uv = u_row.dot(&v_row);
            
            let sqrtc = c.sqrt();
            let norm_sq_diff = (u2 + v2 - 2.0 * uv).max(EPS);
            let denom = ((1.0 - c * u2) * (1.0 - c * v2)).max(EPS);
            
            *dist = (2.0 / sqrtc) * ((sqrtc * sqrtc * norm_sq_diff) / denom).sqrt().atanh();
        });
    
    result
}

pub fn poincare_to_lorentz(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols();
    let mut result = Array2::zeros((batch_size, dim + 1));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let x_norm_sq = x_row.dot(&x_row);
            let denom = (1.0 - c * x_norm_sq).max(EPS);
            let sqrtc = c.sqrt();
            
            // Time component
            row[0] = (1.0 + c * x_norm_sq) / (denom * sqrtc);
            
            // Space components
            for j in 0..dim {
                row[j + 1] = (2.0 * x_row[j]) / (denom * sqrtc);
            }
        });
    
    result
}

pub fn poincare_to_klein(x: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let batch_size = x.nrows();
    let dim = x.ncols();
    let mut result = Array2::zeros((batch_size, dim));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let x_row = x.row(i);
            let x_norm_sq = x_row.dot(&x_row);
            let denom = (1.0 + c * x_norm_sq).max(EPS);
            
            for j in 0..dim {
                row[j] = 2.0 * x_row[j] / denom;
            }
        });
    
    result
}

pub fn poincare_ball_layer(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, t: f32) -> Array2<f32> {
    let u_prime = mobius::mobius_scalar(u, c, 1.0 - t);
    let v_prime = mobius::mobius_scalar(v, c, t);
    mobius::mobius_add(&u_prime.view(), &v_prime.view(), c)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    #[link(name = "kernel_poincare", kind="static")]
    extern "C" {
        pub fn poincare_distance_cuda_launcher(
            out: *mut f32,
            u: *const f32,
            v: *const f32,
            c: f32,
            batch_size: i64,
            dim: i64,
        );
        pub fn poincare_ball_layer_cuda_launcher(
            out: *mut f32,
            u: *const f32,
            v: *const f32,
            c: f32,
            t: f32,
            batch_size: i64,
            dim: i64,
        );
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
            poincare_distance_cuda_launcher(out, u, v, c, batch_size, dim);
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
            poincare_ball_layer_cuda_launcher(out, u, v, c, t, batch_size, dim);
        }
    }
} 