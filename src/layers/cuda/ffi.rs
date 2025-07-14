// src/layers/cuda/ffi.rs
#![allow(dead_code)]

#[link(name = "reality_stone_kernels", kind = "static")]
extern "C" {
    // Bitfield
    pub fn bitfield_forward_cuda(
        // ...
    );
    
    // Spline
    pub fn spline_forward_cuda(
        input: *const f32,
        control_points: *const f32,
        output: *mut f32,
        batch_size: libc::c_int,
        k: libc::c_int,
        in_features: libc::c_int,
        out_features: libc::c_int,
    );

    pub fn spline_backward_cuda(
        grad_output: *const f32,
        input: *const f32,
        grad_control_points: *mut f32,
        batch_size: libc::c_int,
        k: libc::c_int,
        in_features: libc::c_int,
        out_features: libc::c_int,
    );
}
