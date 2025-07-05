use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

const EPS: f32 = 1e-7;
const BOUNDARY_EPS: f32 = 1e-5;
const MIN_DENOMINATOR: f32 = 1e-6;

pub fn mobius_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
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
            
            let u2 = u_row.dot(&u_row);
            let v2 = v_row.dot(&v_row);
            let uv = u_row.dot(&v_row);
            let c2 = c * c;
            
            let coeff_u = (1.0 + 2.0 * c * uv + c * v2) / (1.0 + 2.0 * c * uv + c2 * u2 * v2).max(MIN_DENOMINATOR);
            let coeff_v = (1.0 - c * u2) / (1.0 + 2.0 * c * uv + c2 * u2 * v2).max(MIN_DENOMINATOR);
            
            for j in 0..dim {
                row[j] = coeff_u * u_row[j] + coeff_v * v_row[j];
            }
        });
    
    result
}

pub fn mobius_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let batch_size = u.nrows();
    let dim = u.ncols();
    
    let mut result = Array2::zeros((batch_size, dim));
    
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let u_row = u.row(i);
            let norm = u_row.dot(&u_row).sqrt().max(EPS);
            let sqrtc = c.sqrt();
            let scn = (sqrtc * norm).min(1.0 - BOUNDARY_EPS).max(EPS);
            let alpha = scn.atanh();
            let beta = (r * alpha).tanh();
            let scale = beta / (sqrtc * norm);
            
            for j in 0..dim {
                row[j] = scale * u_row[j];
            }
        });
    
    result
} 

#[cfg(feature = "cuda")]
pub mod cuda {
    use std::os::raw::c_void;

    #[link(name = "kernel_mobius", kind="static")]
    extern "C" {
        pub fn mobius_add_cuda_launcher(
            out: *mut f32,
            u: *const f32,
            v: *const f32,
            c: f32,
            batch_size: i64,
            dim: i64,
        );
        pub fn mobius_scalar_cuda_launcher(
            out: *mut f32,
            u: *const f32,
            c: f32,
            r: f32,
            batch_size: i64,
            dim: i64,
        );
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
            mobius_add_cuda_launcher(out, u, v, c, batch_size, dim);
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
            mobius_scalar_cuda_launcher(out, u, c, r, batch_size, dim);
        }
    }
} 