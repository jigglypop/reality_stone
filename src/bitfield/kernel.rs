// src/layers/bitfield/kernel.rs

//! # 비트필드 기반 직접 추론 커널
//!
//! 이 모듈은 압축된 비트필드 코드와 기저 테이블로부터 직접 추론을 수행하는
//! 고성능 커널들을 제공합니다. CPU와 GPU 버전을 모두 지원합니다.

use crate::bitfield::{decoder, ops};
use ndarray::{Array1, Array2, ArrayBase, ArrayView, ArrayView2, Axis, Data, Ix2, IxDyn};
use rayon::prelude::*;
// CUDA 타입 정의
#[cfg(feature = "cuda")]
type CudaStreamT = *mut std::ffi::c_void;
// CUDA 커널 바인딩
#[cfg(feature = "cuda")]
extern "C" {
    fn launch_gemm_hyper_bit_kernel(
        x: *const f32,
        codes: *const u32,
        basis_table: *const f32,
        delta: f32,
        output: *mut f32,
        batch_size: i32,
        input_dim: i32,
        output_dim: i32,
        basis_size: i32,
    );

    fn launch_residual_gemm_kernel(
        x: *const f32,
        residual_weights: *const f32,
        output: *mut f32,
        batch_size: i32,
        input_dim: i32,
        output_dim: i32,
    );

    fn launch_gemm_hyper_bit_cached_kernel(
        x: *const f32,
        codes_gpu: *const u32,
        basis_table_gpu: *const f32,
        residual_weights_gpu: *const f32,
        delta: f32,
        output: *mut f32,
        batch_size: i32,
        input_dim: i32,
        output_dim: i32,
        basis_size: i32,
    );

    fn launch_gemm_hyper_bit_optimized_kernel(
        x: *const f32,
        codes_gpu: *const u32,
        basis_table_gpu: *const f32,
        residual_codes_gpu: *const u32,
        delta: f32,
        residual_delta: f32,
        output: *mut f32,
        batch_size: i32,
        input_dim: i32,
        output_dim: i32,
        basis_size: i32,
        stream: CudaStreamT,
    );

    fn launch_gemm_hyper_bit_int8_kernel(
        x: *const f32,
        codes_gpu: *const u32,
        basis_table_int8_gpu: *const i8,
        basis_scales_gpu: *const f32,
        delta: f32,
        output: *mut f32,
        batch_size: i32,
        input_dim: i32,
        output_dim: i32,
        basis_size: i32,
        stream: CudaStreamT,
    );
}

// CUDA 런타임 바인딩
#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
        stream: CudaStreamT,
    ) -> i32;
    fn cudaStreamCreate(pStream: *mut CudaStreamT) -> i32;
    fn cudaStreamDestroy(stream: CudaStreamT) -> i32;
    fn cudaStreamSynchronize(stream: CudaStreamT) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}
// CUDA 메모리 복사 종류 상수
#[cfg(feature = "cuda")]
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
#[cfg(feature = "cuda")]
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

// GPU 메모리 관리를 위한 헬퍼 함수들
#[cfg(feature = "cuda")]
pub unsafe fn cuda_malloc(size: usize) -> *mut f32 {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let result = cudaMalloc(&mut ptr as *mut *mut std::ffi::c_void, size);
    if result != 0 {
        panic!(
            "CUDA malloc failed with error code: {} (size: {} bytes)",
            result, size
        );
    }
    ptr as *mut f32
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_h2d(dst: *mut f32, src: *const f32, size: usize) {
    let result = cudaMemcpy(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_HOST_TO_DEVICE,
    );
    if result != 0 {
        panic!("CUDA memcpy H2D failed with error code: {}", result);
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_d2h(dst: *mut f32, src: *const f32, size: usize) {
    let result = cudaMemcpy(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_DEVICE_TO_HOST,
    );
    if result != 0 {
        panic!("CUDA memcpy D2H failed with error code: {}", result);
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_free<T>(ptr: *mut T) {
    let result = cudaFree(ptr as *mut std::ffi::c_void);
    if result != 0 {
        panic!("CUDA free failed with error code: {}", result);
    }
}

// CUDA 스트림 관리 함수들
#[cfg(feature = "cuda")]
pub unsafe fn cuda_stream_create() -> CudaStreamT {
    let mut stream: CudaStreamT = std::ptr::null_mut();
    let result = cudaStreamCreate(&mut stream);
    if result != 0 {
        panic!("CUDA stream creation failed with error code: {}", result);
    }
    stream
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_stream_destroy(stream: CudaStreamT) {
    let result = cudaStreamDestroy(stream);
    if result != 0 {
        panic!("CUDA stream destruction failed with error code: {}", result);
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_stream_synchronize(stream: CudaStreamT) {
    let result = cudaStreamSynchronize(stream);
    if result != 0 {
        panic!(
            "CUDA stream synchronization failed with error code: {}",
            result
        );
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_async_h2d(
    dst: *mut f32,
    src: *const f32,
    size: usize,
    stream: CudaStreamT,
) {
    let result = cudaMemcpyAsync(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        stream,
    );
    if result != 0 {
        panic!("CUDA async memcpy H2D failed with error code: {}", result);
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_async_d2h(
    dst: *mut f32,
    src: *const f32,
    size: usize,
    stream: CudaStreamT,
) {
    let result = cudaMemcpyAsync(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_DEVICE_TO_HOST,
        stream,
    );
    if result != 0 {
        panic!("CUDA async memcpy D2H failed with error code: {}", result);
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_h2d_async(
    dst: *mut f32,
    src: *const f32,
    size: usize,
    stream: CudaStreamT,
) {
    let result = cudaMemcpyAsync(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_HOST_TO_DEVICE,
        stream as *mut std::ffi::c_void,
    );
    if result != 0 {
        panic!("CUDA async memcpy H2D failed with error code: {}", result);
    }
}

#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_d2h_async(
    dst: *mut f32,
    src: *const f32,
    size: usize,
    stream: CudaStreamT,
) {
    let result = cudaMemcpyAsync(
        dst as *mut std::ffi::c_void,
        src as *const std::ffi::c_void,
        size,
        CUDA_MEMCPY_DEVICE_TO_HOST,
        stream as *mut std::ffi::c_void,
    );
    if result != 0 {
        panic!("CUDA async memcpy D2H failed with error code: {}", result);
    }
}

/// GPU 버전의 비트필드 기반 직접 추론 커널
#[cfg(feature = "cuda")]
pub fn gemm_hyper_bit_gpu<S>(
    x: &ArrayBase<S, Ix2>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    use std::time::Instant;
    let start_total = Instant::now();

    let batch_size = x.shape()[0] as i32;
    let input_dim = x.shape()[1] as i32;
    let output_dim = codes.len() as i32;
    let basis_size = basis_table.shape()[0] as i32;

    unsafe {
        // GPU 메모리 할당
        let start_alloc = Instant::now();
        let x_gpu = cuda_malloc((batch_size * input_dim) as usize * std::mem::size_of::<f32>());
        let codes_gpu = cuda_malloc(output_dim as usize * std::mem::size_of::<u32>()) as *mut u32;
        let basis_gpu = cuda_malloc((basis_size * input_dim) as usize * std::mem::size_of::<f32>());
        let output_gpu =
            cuda_malloc((batch_size * output_dim) as usize * std::mem::size_of::<f32>());
        let alloc_time = start_alloc.elapsed();

        // 데이터를 GPU로 복사
        let start_h2d = Instant::now();
        cuda_memcpy_h2d(
            x_gpu,
            x.as_ptr(),
            (batch_size * input_dim) as usize * std::mem::size_of::<f32>(),
        );
        // codes는 u32 타입이므로 별도 처리
        {
            let codes_ptr = codes.as_ptr() as *const std::ffi::c_void;
            let codes_gpu_ptr = codes_gpu as *mut std::ffi::c_void;
            let result = cudaMemcpy(
                codes_gpu_ptr,
                codes_ptr,
                output_dim as usize * std::mem::size_of::<u32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
            if result != 0 {
                panic!("CUDA memcpy for codes failed with error code: {}", result);
            }
        }
        cuda_memcpy_h2d(
            basis_gpu,
            basis_table.as_ptr(),
            (basis_size * input_dim) as usize * std::mem::size_of::<f32>(),
        );
        let h2d_time = start_h2d.elapsed();

        // CUDA 커널 실행
        let start_kernel = Instant::now();
        launch_gemm_hyper_bit_kernel(
            x_gpu, codes_gpu, basis_gpu, delta, output_gpu, batch_size, input_dim, output_dim,
            basis_size,
        );
        let kernel_time = start_kernel.elapsed();

        // 결과를 CPU로 복사
        let start_d2h = Instant::now();
        let mut output = Array2::<f32>::zeros((batch_size as usize, output_dim as usize));
        cuda_memcpy_d2h(
            output.as_mut_ptr(),
            output_gpu,
            (batch_size * output_dim) as usize * std::mem::size_of::<f32>(),
        );
        let d2h_time = start_d2h.elapsed();

        // GPU 메모리 해제
        let start_free = Instant::now();
        cuda_free(x_gpu);
        cuda_free(codes_gpu as *mut f32);
        cuda_free(basis_gpu);
        cuda_free(output_gpu);
        let free_time = start_free.elapsed();

        let total_time = start_total.elapsed();

        // 첫 번째 호출에서만 타이밍 출력
        static mut FIRST_CALL: bool = true;
        if FIRST_CALL {
            println!(
                "[CUDA Timing] Total: {:.3}ms",
                total_time.as_secs_f64() * 1000.0
            );
            println!("  - Alloc: {:.3}ms", alloc_time.as_secs_f64() * 1000.0);
            println!("  - H2D: {:.3}ms", h2d_time.as_secs_f64() * 1000.0);
            println!("  - Kernel: {:.3}ms", kernel_time.as_secs_f64() * 1000.0);
            println!("  - D2H: {:.3}ms", d2h_time.as_secs_f64() * 1000.0);
            println!("  - Free: {:.3}ms", free_time.as_secs_f64() * 1000.0);
            FIRST_CALL = false;
        }

        output
    }
}

/// GPU 포인터를 직접 사용하는 캐시된 버전의 비트필드 기반 직접 추론 커널
#[cfg(feature = "cuda")]
pub fn gemm_hyper_bit_gpu_cached<S>(
    x: &ArrayBase<S, Ix2>,
    codes_gpu: *const u32,
    basis_table_gpu: *const f32,
    residual_weights_gpu: *const f32,
    delta: f32,
    m: usize,
    n: usize,
    b: usize,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    let batch_size = x.shape()[0] as i32;
    let input_dim = x.shape()[1] as i32;
    let output_dim = m as i32;
    let basis_size = b as i32;

    unsafe {
        // 입력과 출력만 GPU 메모리 할당
        let x_gpu = cuda_malloc((batch_size * input_dim) as usize * std::mem::size_of::<f32>());
        let output_gpu =
            cuda_malloc((batch_size * output_dim) as usize * std::mem::size_of::<f32>());

        // 입력 데이터만 GPU로 복사
        cuda_memcpy_h2d(
            x_gpu,
            x.as_ptr(),
            (batch_size * input_dim) as usize * std::mem::size_of::<f32>(),
        );

        // CUDA 커널 실행 (이미 GPU에 있는 codes, basis_table, residual 사용)
        launch_gemm_hyper_bit_cached_kernel(
            x_gpu,
            codes_gpu,
            basis_table_gpu,
            residual_weights_gpu,
            delta,
            output_gpu,
            batch_size,
            input_dim,
            output_dim,
            basis_size,
        );

        // 결과를 CPU로 복사
        let mut output = Array2::<f32>::zeros((batch_size as usize, output_dim as usize));
        cuda_memcpy_d2h(
            output.as_mut_ptr(),
            output_gpu,
            (batch_size * output_dim) as usize * std::mem::size_of::<f32>(),
        );

        // 입력/출력 GPU 메모리만 해제
        cuda_free(x_gpu);
        cuda_free(output_gpu);

        output
    }
}

/// GPU 버전의 잔차 추론 커널
#[cfg(feature = "cuda")]
pub fn residual_gemm_gpu<S>(
    x: &ArrayBase<S, Ix2>,
    residual_weights: &Array2<f32>,
    output: &mut Array2<f32>,
) where
    S: Data<Elem = f32>,
{
    let batch_size = x.shape()[0] as i32;
    let input_dim = x.shape()[1] as i32;
    let output_dim = residual_weights.shape()[0] as i32;

    unsafe {
        // GPU 메모리 할당
        let x_gpu = cuda_malloc((batch_size * input_dim) as usize * std::mem::size_of::<f32>());
        let residual_gpu =
            cuda_malloc((output_dim * input_dim) as usize * std::mem::size_of::<f32>());
        let output_gpu =
            cuda_malloc((batch_size * output_dim) as usize * std::mem::size_of::<f32>());

        // 데이터를 GPU로 복사
        cuda_memcpy_h2d(x_gpu, x.as_ptr(), (batch_size * input_dim) as usize);
        cuda_memcpy_h2d(
            residual_gpu,
            residual_weights.as_ptr(),
            (output_dim * input_dim) as usize,
        );
        cuda_memcpy_h2d(
            output_gpu,
            output.as_ptr(),
            (batch_size * output_dim) as usize,
        );

        // CUDA 커널 실행
        launch_residual_gemm_kernel(
            x_gpu,
            residual_gpu,
            output_gpu,
            batch_size,
            input_dim,
            output_dim,
        );

        // 결과를 CPU로 복사
        cuda_memcpy_d2h(
            output.as_mut_ptr(),
            output_gpu,
            (batch_size * output_dim) as usize,
        );

        // GPU 메모리 해제
        cuda_free(x_gpu);
        cuda_free(residual_gpu);
        cuda_free(output_gpu);
    }
}

/// 통합된 비트필드 기반 직접 추론 함수 (CPU/GPU 자동 선택)
pub fn gemm_hyper_bit<S>(
    x: &ArrayBase<S, Ix2>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    // 항상 CPU 버전 사용 (CUDA는 별도 함수로 제공)
    gemm_hyper_bit_cpu(x, codes, basis_table, delta)
}

/// 통합된 비트필드 기반 직접 추론 함수 (use_cuda 파라미터로 선택)
pub fn gemm_hyper_bit_with_backend<S>(
    x: &ArrayBase<S, Ix2>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
    use_cuda: bool,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    #[cfg(feature = "cuda")]
    {
        if use_cuda {
            return gemm_hyper_bit_gpu(x, codes, basis_table, delta);
        }
    }

    // CUDA 기능이 비활성화되었거나 use_cuda가 false인 경우 CPU 사용
    gemm_hyper_bit_cpu(x, codes, basis_table, delta)
}

/// 다차원 텐서를 위한 통합된 비트필드 커널
pub fn gemm_hyper_bit_nd(
    x: &ArrayView<f32, IxDyn>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
    use_cuda: bool,
) -> ndarray::Array<f32, IxDyn> {
    let x_shape = x.shape();
    let x_ndim = x_shape.len();

    if x_ndim < 2 {
        // 2D 미만은 CPU로 처리
        return gemm_hyper_bit_cpu_nd(x, codes, basis_table, delta);
    }

    let input_features = x_shape[x_ndim - 1];
    let total_batch_size: usize = x_shape[..x_ndim - 1].iter().product();

    // 2D로 reshape (뷰)
    let x_reshaped = x.to_shape((total_batch_size, input_features)).unwrap();

    // 2D 커널 디스패처 호출
    let output_2d = gemm_hyper_bit_with_backend(&x_reshaped, codes, basis_table, delta, use_cuda);

    // 원래 차원으로 reshape
    let mut output_shape = x_shape.to_vec();
    output_shape[x_ndim - 1] = codes.len();
    output_2d
        .into_shape(output_shape.as_slice())
        .unwrap()
        .into_dyn()
}

#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    // CUDA 장치 확인 로직 (실제 구현에서는 CUDA 런타임 확인)
    // 임시로 true 반환
    true
}

/// W가 압축된 비트필드 코드로 표현될 때 `y = xW^T`를 수행합니다.
///
/// `ndarray`와 병렬 처리를 위한 `rayon`을 사용하는 CPU 기반 참조 구현입니다.
///
/// # 인자
/// * `x` - 입력 벡터/행렬, `[batch_size, n]` 형태.
/// * `codes` - `u32` 비트필드 코드의 1차원 배열, `[m]` 형태.
/// * `basis_table` - 공유 기저 벡터 테이블, `[B, n]` 형태의 `Array2<f32>`.
/// * `delta` - `amp` 필드를 위한 양자화 스텝 사이즈 (`r_max / 255.0`).
///
/// # 반환
/// 곱셈 결과인 `[batch_size, m]` 형태의 `Array2<f32>`.
pub fn gemm_hyper_bit_cpu<S>(
    x: &ArrayBase<S, Ix2>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    // 1. 전처리: 입력과 모든 기저 벡터 간의 내적을 계산합니다.
    // `x` (batch, n) @ `basis_table.T` (n, B) -> `dotb` (batch, B)
    let dotb = x.dot(&basis_table.t());

    // 2. 메인 루프: `m`개의 코드를 순회하며 각 출력 열을 계산합니다.
    // 출력 행렬을 생성하고 rayon을 사용하여 병렬로 열을 채웁니다.
    let m = codes.len();
    let batch_size = x.shape()[0];
    let mut output = Array2::<f32>::zeros((batch_size, m));

    output
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut col)| {
            let code = codes[j];
            let (cat, sub, idx, sign, d, amp, amp_fine, phase) = decoder::decode(code);

            // a. 코드로부터 스케일링 인자 `s_i`를 계산합니다.
            let r_val = (amp as f32 * 4.0 + amp_fine as f32) * delta;
            let signed_r_val = if sign == 0 { r_val } else { -r_val };
            let scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val, phase);

            // b. 미리 계산된 내적 `(x · b_{idx_i})`을 가져옵니다.
            // 이것은 `dotb` 행렬의 한 열에 대한 뷰입니다.
            let dot_col = dotb.column(idx as usize);

            // c. 최종 출력 열 `y_j = s_j * (x · b_{idx_j})`를 계산합니다.
            // `dot_col`은 `[batch_size]` 형태이므로, 브로드캐스팅 곱셈이 수행됩니다.
            let y_col = &dot_col * scale_factor;
            col.assign(&y_col);
        });

    output
}

/// 다차원 텐서를 지원하는 비트필드 커널 함수
///
/// 임의의 차원을 가진 입력 텐서를 처리하되, 마지막 차원을 feature 차원으로 간주합니다.
/// 내부적으로 [..., features] → [total_elements, features] reshape 후 계산하여
/// 메모리 복사 오버헤드를 최소화합니다.
///
/// # 인자
/// * `x` - 입력 텐서, `[..., n]` 형태 (마지막 차원이 feature 차원)
/// * `codes` - `u32` 비트필드 코드의 1차원 배열, `[m]` 형태
/// * `basis_table` - 공유 기저 벡터 테이블, `[B, n]` 형태의 `Array2<f32>`
/// * `delta` - `amp` 필드를 위한 양자화 스텝 사이즈 (`r_max / 255.0`)
///
/// # 반환
/// 곱셈 결과인 `[..., m]` 형태의 다차원 텐서
pub fn gemm_hyper_bit_cpu_nd(
    x: &ArrayView<f32, IxDyn>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> ndarray::Array<f32, IxDyn> {
    let x_shape = x.shape();
    let x_ndim = x_shape.len();

    if x_ndim == 0 {
        panic!("입력 텐서는 최소 1차원 이상이어야 합니다.");
    }

    let input_features = x_shape[x_ndim - 1];
    let output_features = codes.len();

    // 출력 형태 계산: [..., input_features] → [..., output_features]
    let mut output_shape = x_shape.to_vec();
    output_shape[x_ndim - 1] = output_features;

    // 총 배치 크기 계산 (마지막 차원 제외한 모든 차원의 곱)
    let total_batch_size: usize = x_shape[..x_ndim - 1].iter().product();

    // 입력을 2D로 reshape: [total_batch_size, input_features]
    let x_reshaped = x.to_shape((total_batch_size, input_features)).unwrap();

    // 기존 2D 커널 호출
    let output_2d = gemm_hyper_bit_cpu(&x_reshaped, codes, basis_table, delta);

    // 출력을 원래 다차원 형태로 reshape: [total_batch_size, output_features] → [..., output_features]
    let output_nd = output_2d.into_shape(output_shape.as_slice()).unwrap();

    output_nd.into_dyn()
}

/// 다차원 텐서 `x`와 2차원 행렬 `w`의 내적을 계산합니다.
/// `x`의 마지막 차원과 `w`의 첫 번째 차원이 일치해야 합니다.
/// 결과는 `x`의 형태에서 마지막 차원만 `w`의 두 번째 차원으로 바뀐 형태입니다.
pub fn dot_product_nd<S>(
    x: &ArrayView<f32, IxDyn>,
    w: &ArrayBase<S, Ix2>,
) -> ndarray::Array<f32, IxDyn>
where
    S: Data<Elem = f32>,
{
    let x_shape = x.shape();
    let x_ndim = x_shape.len();
    let n = x_shape[x_ndim - 1];
    let m = w.shape()[1];
    assert_eq!(n, w.shape()[0], "내적 차원이 일치하지 않습니다.");

    let total_batch_size: usize = x_shape[..x_ndim - 1].iter().product();
    // 입력을 2D로 재구성 (메모리 복사 없음)
    let x_reshaped = x.to_shape((total_batch_size, n)).unwrap();
    // 2D 내적 수행
    let output_2d = x_reshaped.dot(w);
    // 결과를 원래 다차원 형태로 재구성
    let mut output_shape = x_shape.to_vec();
    output_shape[x_ndim - 1] = m;

    output_2d
        .into_shape(output_shape.as_slice())
        .unwrap()
        .into_dyn()
}

/// 2D 입력을 다차원 함수로 처리하는 편의 함수
///
/// 기존 2D 함수와 호환성을 유지하면서 다차원 인터페이스를 제공합니다.
pub fn gemm_hyper_bit_cpu_2d_as_nd(
    x: &Array2<f32>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> Array2<f32> {
    let x_view = x.view().into_dyn();
    let output_nd = gemm_hyper_bit_cpu_nd(&x_view, codes, basis_table, delta);

    // 결과를 2D로 변환
    let output_shape = output_nd.shape();
    if output_shape.len() != 2 {
        panic!("2D 입력에 대한 출력이 2D가 아닙니다.");
    }

    output_nd.into_dimensionality::<ndarray::Ix2>().unwrap()
}

/// GPU 포인터 직접 사용 forward (복사 없음)
#[cfg(feature = "cuda")]
pub unsafe fn gemm_hyper_bit_gpu_direct(
    input_ptr: *const f32,          // PyTorch 텐서의 GPU 포인터
    output_ptr: *mut f32,           // PyTorch 텐서의 GPU 포인터
    codes_gpu: *const u32,          // 이미 GPU에 있는 codes
    basis_table_gpu: *const f32,    // 이미 GPU에 있는 basis
    residual_codes_gpu: *const u32, // 이미 GPU에 있는 residual
    delta: f32,
    residual_delta: f32,
    batch_size: i32,
    input_dim: i32,
    output_dim: i32,
    basis_size: i32,
    stream: *mut std::ffi::c_void,
) {
    // CUDA 커널 실행 (데이터 복사 없음)
    launch_gemm_hyper_bit_direct_kernel(
        input_ptr,
        codes_gpu,
        basis_table_gpu,
        residual_codes_gpu,
        delta,
        residual_delta,
        output_ptr,
        batch_size,
        input_dim,
        output_dim,
        basis_size,
        stream,
    );
}

/// GPU 포인터 직접 사용 backward (복사 없음)
#[cfg(feature = "cuda")]
pub unsafe fn gemm_hyper_bit_backward_gpu_direct(
    grad_output_ptr: *const f32, // PyTorch grad_output의 GPU 포인터
    grad_input_ptr: *mut f32,    // PyTorch grad_input의 GPU 포인터
    codes_gpu: *const u32,
    basis_table_gpu: *const f32,
    residual_codes_gpu: *const u32,
    delta: f32,
    residual_delta: f32,
    batch_size: i32,
    input_dim: i32,
    output_dim: i32,
    basis_size: i32,
    stream: *mut std::ffi::c_void,
) {
    // Backward 커널 실행
    launch_gemm_hyper_bit_backward_direct_kernel(
        grad_output_ptr,
        codes_gpu,
        basis_table_gpu,
        residual_codes_gpu,
        delta,
        residual_delta,
        grad_input_ptr,
        batch_size,
        input_dim,
        output_dim,
        basis_size,
        stream,
    );
}

// 외부 CUDA 함수 선언
#[cfg(feature = "cuda")]
extern "C" {
    // 직접 포인터 사용 커널
    fn launch_gemm_hyper_bit_direct_kernel(
        x_gpu: *const f32,
        codes_gpu: *const u32,
        basis_table_gpu: *const f32,
        residual_codes_gpu: *const u32,
        delta: f32,
        residual_delta: f32,
        output_gpu: *mut f32,
        batch_size: i32,
        input_dim: i32,
        output_dim: i32,
        basis_size: i32,
        stream: *mut std::ffi::c_void,
    );

    fn launch_gemm_hyper_bit_backward_direct_kernel(
        grad_output_gpu: *const f32,
        codes_gpu: *const u32,
        basis_table_gpu: *const f32,
        residual_codes_gpu: *const u32,
        delta: f32,
        residual_delta: f32,
        grad_input_gpu: *mut f32,
        batch_size: i32,
        input_dim: i32,
        output_dim: i32,
        basis_size: i32,
        stream: *mut std::ffi::c_void,
    );
}

/// 최적화된 GPU 추론 (버퍼 풀과 스트림 사용)
#[cfg(feature = "cuda")]
pub unsafe fn gemm_hyper_bit_gpu_optimized(
    x: &ArrayView2<f32>,
    codes_gpu: *const u32,
    basis_table_gpu: *const f32,
    residual_codes_gpu: *const u32,
    delta: f32,
    residual_delta: f32,
    input_buffer_gpu: *mut f32,
    output_buffer_gpu: *mut f32,
    stream: *mut std::ffi::c_void,
    m: usize,
    n: usize,
    b: usize,
) -> Array2<f32> {
    let batch_size = x.shape()[0];
    let input_dim = x.shape()[1];
    let output_dim = m;

    // 배치 크기 체크
    if batch_size * input_dim * std::mem::size_of::<f32>() > 64 * n * std::mem::size_of::<f32>() {
        panic!("Batch size too large for buffer pool");
    }

    // 비동기 메모리 복사 (스트림 사용)
    cuda_memcpy_async_h2d(
        input_buffer_gpu,
        x.as_ptr(),
        batch_size * input_dim * std::mem::size_of::<f32>(),
        stream,
    );
    // 최적화된 커널 실행
    launch_gemm_hyper_bit_optimized_kernel(
        input_buffer_gpu,
        codes_gpu,
        basis_table_gpu,
        residual_codes_gpu,
        delta,
        residual_delta,
        output_buffer_gpu,
        batch_size as i32,
        input_dim as i32,
        output_dim as i32,
        b as i32,
        stream,
    );

    // 결과를 비동기로 복사
    let mut output = Array2::<f32>::zeros((batch_size, output_dim));
    cuda_memcpy_async_d2h(
        output.as_mut_ptr(),
        output_buffer_gpu,
        batch_size * output_dim * std::mem::size_of::<f32>(),
        stream,
    );
    // 스트림 동기화
    cuda_stream_synchronize(stream);
    output
}

#[cfg(feature = "cuda")]
pub fn gemm_hyper_bit_gpu_int8_optimized<S>(
    x: &ArrayBase<S, Ix2>,
    codes_gpu: *const u32,
    basis_table_int8_gpu: *const i8,
    basis_scales_gpu: *const f32,
    delta: f32,
    input_buffer_gpu: *mut f32,
    output_buffer_gpu: *mut f32,
    stream: CudaStreamT,
    m: usize,
    n: usize,
    basis_size: usize,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    let batch_size = x.shape()[0];

    unsafe {
        // 입력 데이터를 GPU 버퍼로 비동기 복사
        let input_size = batch_size * n * std::mem::size_of::<f32>();
        cuda_memcpy_h2d_async(input_buffer_gpu, x.as_ptr(), input_size, stream);

        // INT8 최적화 커널 실행
        launch_gemm_hyper_bit_int8_kernel(
            input_buffer_gpu,
            codes_gpu,
            basis_table_int8_gpu,
            basis_scales_gpu,
            delta,
            output_buffer_gpu,
            batch_size as i32,
            n as i32,
            m as i32,
            basis_size as i32,
            stream,
        );

        // 결과를 CPU로 복사
        let mut output = Array2::<f32>::zeros((batch_size, m));
        let output_size = batch_size * m * std::mem::size_of::<f32>();
        cuda_memcpy_d2h_async(output.as_mut_ptr(), output_buffer_gpu, output_size, stream);

        // 스트림 동기화
        cuda_stream_synchronize(stream);

        output
    }
}

#[cfg(feature = "cuda")]
pub fn gemm_hyper_bit_gpu_tensorcore_optimized<S>(
    x: &ArrayBase<S, Ix2>,
    codes_gpu: *const u32,
    basis_table_int8_gpu: *const i8,
    basis_scales_gpu: *const f32,
    delta: f32,
    input_buffer_gpu: *mut f32,
    output_buffer_gpu: *mut f32,
    stream: CudaStreamT,
    m: usize,
    n: usize,
    _basis_size: usize,
    _use_cuda: bool,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    let batch_size = x.shape()[0];
    let input_size = batch_size * n * std::mem::size_of::<f32>();
    // 결과를 CPU로 복사
    let mut output = Array2::<f32>::zeros((batch_size, m));
    let output_size = batch_size * m * std::mem::size_of::<f32>();
    output
}
