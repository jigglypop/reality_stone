//! CUDA backend for Riemann-surface attention.
//!
//! cudarc 0.19 driver + NVRTC for runtime PTX compilation. The kernel
//! source is embedded via `include_str!` so no build.rs is required.
//!
//! Two entry points:
//!   * `ce_riemann_fwd_cuda(..host slices..)`  — convenience path for CPU
//!     tensors that still want GPU compute (one alloc + one htod per
//!     input, no per-row Python loop).
//!   * `ce_riemann_fwd_cuda_devptr(..u64 device pointers..)` — zero-copy
//!     entry for PyTorch CUDA tensors. Accepts raw device pointers
//!     (CUdeviceptr cast to u64) and writes results in place. The caller
//!     is responsible for stream synchronization with the producing
//!     PyTorch stream.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const KERNEL_SRC: &str = include_str!("ce_riemann.cu");

const KERNEL_NAME: &str = "ce_riemann_fwd_kernel";
const THREADS_PER_BLOCK: u32 = 128;

static CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();
static MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();

fn ctx() -> Result<Arc<CudaContext>, String> {
    if let Some(c) = CTX.get() {
        return Ok(c.clone());
    }
    let c = CudaContext::new(0).map_err(|e| format!("CudaContext::new failed: {e:?}"))?;
    let _ = CTX.set(c.clone());
    Ok(c)
}

fn module() -> Result<Arc<CudaModule>, String> {
    if let Some(m) = MODULE.get() {
        return Ok(m.clone());
    }
    let ctx = ctx()?;
    let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("NVRTC compile failed: {e:?}"))?;
    let m = ctx
        .load_module(ptx)
        .map_err(|e| format!("load_module failed: {e:?}"))?;
    let _ = MODULE.set(m.clone());
    Ok(m)
}

#[inline]
fn shape_check(bh: usize, n: usize, d_head: usize) -> Result<(), String> {
    if d_head % 2 != 0 {
        return Err(format!("d_head must be even, got {d_head}"));
    }
    if bh == 0 || n == 0 || d_head == 0 {
        return Err("ce_riemann: zero-sized tensor".into());
    }
    // Shared-mem budget: (D + N) * 4 bytes. Default 48 KB / SM is fine
    // for D <= 256 and N <= 8192. Grid-y is u32-bounded.
    if d_head > 1024 {
        return Err(format!("d_head {d_head} exceeds CUDA kernel limit 1024"));
    }
    if n > 16384 {
        return Err(format!("n {n} exceeds CUDA kernel limit 16384"));
    }
    Ok(())
}

#[inline]
fn launch_cfg(bh: usize, n: usize, d_head: usize) -> LaunchConfig {
    let smem_bytes = ((d_head + n) * std::mem::size_of::<f32>()) as u32;
    LaunchConfig {
        grid_dim: (bh as u32, n as u32, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: smem_bytes,
    }
}

/// Host-staging entry: copies inputs to device, runs the kernel,
/// copies the result back. Used when source tensors live on the CPU
/// but compute should run on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn ce_riemann_fwd_cuda(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cos: &[f32],
    sin: &[f32],
    sheet_bias: &[f32],
    bh: usize,
    n: usize,
    d_head: usize,
    causal: bool,
) -> Result<Vec<f32>, String> {
    shape_check(bh, n, d_head)?;
    let half = d_head / 2;
    let total_qkv = bh * n * d_head;
    let total_cs = bh * n * half;
    let total_sb = bh * n * n;
    debug_assert_eq!(q.len(), total_qkv);
    debug_assert_eq!(k.len(), total_qkv);
    debug_assert_eq!(v.len(), total_qkv);
    debug_assert_eq!(cos.len(), total_cs);
    debug_assert_eq!(sin.len(), total_cs);
    debug_assert_eq!(sheet_bias.len(), total_sb);

    let ctx = ctx()?;
    let stream = ctx.default_stream();
    let m = module()?;
    let func = m
        .load_function(KERNEL_NAME)
        .map_err(|e| format!("load_function failed: {e:?}"))?;

    let q_d = stream.clone_htod(q).map_err(|e| format!("htod q: {e:?}"))?;
    let k_d = stream.clone_htod(k).map_err(|e| format!("htod k: {e:?}"))?;
    let v_d = stream.clone_htod(v).map_err(|e| format!("htod v: {e:?}"))?;
    let cos_d = stream
        .clone_htod(cos)
        .map_err(|e| format!("htod cos: {e:?}"))?;
    let sin_d = stream
        .clone_htod(sin)
        .map_err(|e| format!("htod sin: {e:?}"))?;
    let sb_d = stream
        .clone_htod(sheet_bias)
        .map_err(|e| format!("htod sb: {e:?}"))?;
    let mut out_d = stream
        .alloc_zeros::<f32>(total_qkv)
        .map_err(|e| format!("alloc out: {e:?}"))?;

    let cfg = launch_cfg(bh, n, d_head);
    let n_i32 = n as i32;
    let d_i32 = d_head as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&q_d);
    builder.arg(&k_d);
    builder.arg(&v_d);
    builder.arg(&cos_d);
    builder.arg(&sin_d);
    builder.arg(&sb_d);
    builder.arg(&mut out_d);
    builder.arg(&n_i32);
    builder.arg(&d_i32);
    builder.arg(&causal_i32);
    unsafe { builder.launch(cfg) }.map_err(|e| format!("launch: {e:?}"))?;

    stream
        .synchronize()
        .map_err(|e| format!("synchronize: {e:?}"))?;
    let out = stream
        .clone_dtoh(&out_d)
        .map_err(|e| format!("dtoh out: {e:?}"))?;
    Ok(out)
}

/// Zero-copy entry: the caller passes raw CUDA device pointers
/// (e.g. `tensor.data_ptr()` from PyTorch). The kernel writes directly
/// into `out_ptr`. The driver reads pointer values as 64-bit args, so
/// we push them as raw `*const f32` / `*mut f32`.
///
/// Safety: all pointers must be valid CUDA device addresses owned by
/// the calling process, with sufficient backing storage for the shapes
/// implied by `bh`, `n`, `d_head`. Caller must ensure the producing
/// stream has been synchronized before invocation (and synchronize the
/// consuming stream afterwards) when crossing PyTorch ↔ this stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ce_riemann_fwd_cuda_devptr(
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    cos_ptr: u64,
    sin_ptr: u64,
    sb_ptr: u64,
    out_ptr: u64,
    bh: usize,
    n: usize,
    d_head: usize,
    causal: bool,
) -> Result<(), String> {
    shape_check(bh, n, d_head)?;
    let _ = ctx()?;
    let stream = CTX.get().unwrap().default_stream();
    let m = module()?;
    let func = m
        .load_function(KERNEL_NAME)
        .map_err(|e| format!("load_function failed: {e:?}"))?;

    let cfg = launch_cfg(bh, n, d_head);
    let n_i32 = n as i32;
    let d_i32 = d_head as i32;
    let causal_i32 = if causal { 1i32 } else { 0i32 };

    // CUDA driver expects an 8-byte pointer per `float*` arg; u64 has the
    // identical wire representation. cudarc impls `DeviceRepr` for u64
    // but not raw pointer types, so push as u64.
    let mut builder = stream.launch_builder(&func);
    builder.arg(&q_ptr);
    builder.arg(&k_ptr);
    builder.arg(&v_ptr);
    builder.arg(&cos_ptr);
    builder.arg(&sin_ptr);
    builder.arg(&sb_ptr);
    builder.arg(&out_ptr);
    builder.arg(&n_i32);
    builder.arg(&d_i32);
    builder.arg(&causal_i32);
    builder.launch(cfg).map_err(|e| format!("launch: {e:?}"))?;

    stream
        .synchronize()
        .map_err(|e| format!("synchronize: {e:?}"))?;
    Ok(())
}
