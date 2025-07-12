use crate::ops::utils::{dot_batched, norm_sq_batched, EPS};
use ndarray::{Array2, ArrayView2, Axis};

const BOUNDARY_EPS: f32 = 1e-5;
const MIN_DENOMINATOR: f32 = 1e-6;

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

pub fn mobius_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    let sqrtc = c.sqrt();
    let scn = (sqrtc * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS).max(EPS));
    let alpha = scn.mapv(f32::atanh);
    let beta = (r * alpha).mapv(f32::tanh);
    let scale = beta / (sqrtc * &norm_clamped);

    scale * u
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        #[link(name = "kernel_mobius", kind="static")]
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