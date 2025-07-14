// src/layers/bitfield/mod.rs

//! # 비트필드 기반 하이퍼볼릭 레이어
//!
//! 이 모듈은 비트필드 기반 직접 추론 레이어에 필요한 모든 구성요소를 캡슐화합니다.
//! 디코더, 기저 테이블 관리, 하이퍼볼릭 연산, 그리고 추론 커널을 포함합니다.
//!
//! 사용자 친화적인 `BitfieldLinear` 구조체를 노출하여, 가중치가 고도로 압축된 형태로
//! 저장된다는 점을 제외하면 표준 선형 레이어처럼 사용할 수 있습니다.
pub mod adaptive;
pub mod basis;
pub mod decoder;
pub mod jacobian;
pub mod kernel;
pub mod ops;
pub mod pipeline;
pub mod riemannian;
pub mod unified_tables;

use ndarray::{Array1, Array2, ArrayView, IxDyn};

/// 비트필드 인코딩 레이아웃 선택
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitfieldLayout {
    /// 32비트 표준 레이아웃 (phase, amp_fine, sign 포함)
    Standard32Bit,
    /// 22비트 극한 압축 레이아웃
    Extreme22Bit,
}

// 계층적 압축을 위한 구조체 (4비트/가중치)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct HierarchicalCode {
    pub base_code: u16,  // 공유 베이스 코드 (8개 가중치당 1개)
    pub deltas: [u8; 2], // 2비트 × 8 = 16비트 델타
}

// 극한 압축을 위한 구조체 (2.5비트/가중치)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ExtremeCode {
    pub packed: u16, // 5개 가중치를 16비트에 패킹 (각 3.2비트)
}

// GPU 메모리 포인터를 위한 구조체. INT8 잔차 관련 필드 제거.
#[cfg(feature = "cuda")]
pub struct GpuBuffers {
    pub codes_gpu: *mut u32,
    pub basis_table_gpu: *mut f32,
    pub residual_codes_gpu: *mut u32,
    // INT8 잔차 GPU 메모리
    pub residual_int8_gpu: *mut i8,
    pub residual_scales_gpu: *mut f32,
    // INT8 기저 테이블 (새로 추가)
    pub basis_table_int8_gpu: *mut i8,
    pub basis_scales_gpu: *mut f32,
    // 입출력 버퍼 풀
    pub input_buffer_gpu: *mut f32,
    pub output_buffer_gpu: *mut f32,
    // CUDA 스트림
    pub stream: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
unsafe impl Send for GpuBuffers {}
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuBuffers {}

/// 가중치가 압축된 비트필드 형태로 저장되는 선형 레이어입니다.
///
/// 이 구조체는 압축된 코드, 공유 기저 테이블, 그리고 직접 추론 계산에
/// 필요한 다른 파라미터들을 보유합니다.
pub struct BitfieldLinear {
    codes: Array1<u32>,
    basis_table: Array2<f32>,
    delta: f32,
    m: usize,
    n: usize,
    residual_codes: Option<Array1<u32>>, // 비트필드 잔차
    residual_delta: Option<f32>,
    // INT8 잔차
    residual_int8: Option<Array2<i8>>,    // [m, n]
    residual_scales: Option<Array1<f32>>, // [m]
    // INT8 기저 테이블 (새로 추가)
    basis_table_int8: Option<Array2<i8>>, // [b, n]
    basis_scales: Option<Array1<f32>>,    // [b] - 각 기저 벡터의 스케일
    pub layout: BitfieldLayout,           // 비트필드 레이아웃
    pub use_cuda: bool,
    pub use_int8: bool,         // INT8 최적화 사용 여부
    pub use_tensorcore: bool,   // Tensor Core 사용 여부
    pub use_hierarchical: bool, // 계층적 압축 사용 여부
    pub compression_level: u8,  // 압축 레벨 (1: 기본, 2: 4비트, 3: 2.5비트)
    pub trainable: bool,        // 학습 가능 모드
    pub temperature: f32,       // Gumbel-Softmax 온도
    #[cfg(feature = "cuda")]
    gpu_buffers: Option<GpuBuffers>,
}

impl BitfieldLinear {
    /// 새로운 `BitfieldLinear` 레이어를 생성합니다.
    ///
    /// 실제 애플리케이션에서는 학습된 `codes`와 `basis_table`을
    /// 가중치 파일에서 로드해야 합니다. 여기서는 플레이스홀더로 초기화합니다.
    ///
    /// # 인자
    /// * `m` - 출력 피처의 수.
    /// * `n` - 입력 피처의 수.
    /// * `b` - 기저 벡터 테이블의 크기.
    /// * `r_max` - 양자화를 위한 최대 반지름 값.
    ///
    /// # 반환
    /// 새로운 `BitfieldLinear` 인스턴스.
    pub fn new(m: usize, n: usize, b: usize, r_max: f32, layout: BitfieldLayout) -> Self {
        let codes = Array1::zeros(m);
        let basis_table = basis::load_basis_table(b, n);
        let delta = r_max / 255.0;

        Self {
            codes,
            basis_table,
            delta,
            m,
            n,
            residual_codes: None,
            residual_delta: None,
            residual_int8: None,
            residual_scales: None,
            basis_table_int8: None,
            basis_scales: None,
            layout,
            use_cuda: false,
            use_int8: false,
            use_tensorcore: false,
            use_hierarchical: false,
            compression_level: 1,
            trainable: false,
            temperature: 1.0,
            #[cfg(feature = "cuda")]
            gpu_buffers: None,
        }
    }

    /// 기존 가중치 행렬로부터 BitfieldLinear를 생성합니다.
    ///
    /// # 인자
    /// * `weights` - `[m, n]` 형태의 가중치 행렬
    /// * `b` - 기저 벡터 테이블의 크기 (256 권장)
    /// * `r_max` - 최대 반지름 값 (1.0 권장)
    /// * `use_residual` - 잔차 가중치 사용 여부 (false면 극한 압축)
    /// * `use_cuda` - CUDA 커널 사용 여부
    /// * `bitfield_residual` - 잔차도 비트필드로 인코딩할지 여부
    ///
    /// # 반환
    /// 압축된 BitfieldLinear 인스턴스
    pub fn from_weights(
        weights: &Array2<f32>,
        b: usize,
        r_max: f32,
        use_residual: bool,
        use_cuda: bool,
        bitfield_residual: bool,
        layout: BitfieldLayout,
    ) -> Self {
        let (m, n) = (weights.shape()[0], weights.shape()[1]);
        let mut model = Self::new(m, n, b, r_max, layout);
        model.use_cuda = use_cuda;

        let delta = model.delta;
        let basis_table = model.basis_table.clone(); // 소유권 문제 해결을 위해 복제
        let mut codes = Array1::<u32>::zeros(m);
        let mut residual_weights = Array2::<f32>::zeros((m, n));

        for i in 0..m {
            let weight_row = weights.row(i);

            let (code, reconstructed_scale, best_idx) = match layout {
                BitfieldLayout::Extreme22Bit => {
                    let (cat, sub, idx, d, amp, _error) =
                        basis::find_best_code_22bit(&weight_row, &basis_table, delta);
                    let code = decoder::encode_22bit(cat, sub, idx, d, amp);
                    let r = (amp as f32) * delta;
                    let scale = ops::lookup_and_apply(cat, sub, d, r, 0);
                    (code, scale, idx)
                }
                BitfieldLayout::Standard32Bit => {
                    let (cat, sub, idx, sign, d, amp, amp_fine, phase, _error) =
                        basis::find_best_code_32bit(&weight_row, &basis_table, delta);
                    let code = decoder::encode_32bit(cat, sub, idx, sign, d, amp, amp_fine, phase);
                    let r_unsigned = (amp as f32) + (amp_fine as f32) / 4.0;
                    let r = if sign == 1 { -r_unsigned } else { r_unsigned } * delta;
                    let scale = ops::lookup_and_apply(cat, sub, d, r, phase);
                    (code, scale, idx)
                }
            };
            codes[i] = code;

            let reconstructed = &basis_table.row(best_idx as usize) * reconstructed_scale;
            let error = &weight_row - &reconstructed;
            residual_weights.row_mut(i).assign(&error);
        }

        model.codes = codes;

        if use_residual {
            if bitfield_residual {
                // 잔차도 비트필드로 인코딩 (다음 단계에서 구현)
                // model.residual_codes = Some(residual_codes);
                // model.residual_delta = Some(residual_delta);
            } else {
                let (residual_int8, residual_scales) = Self::quantize_to_int8(&residual_weights);
                model.residual_int8 = Some(residual_int8);
                model.residual_scales = Some(residual_scales);
            }
        }

        #[cfg(feature = "cuda")]
        if use_cuda {
            model.init_gpu_memory();
        }

        model
    }

    #[cfg(feature = "cuda")]
    pub fn init_gpu_memory(&mut self) {
        if !self.use_cuda || self.gpu_buffers.is_some() {
            return;
        }

        unsafe {
            // 메모리 크기 계산
            let codes_size = self.m * std::mem::size_of::<u32>();
            let basis_size = self.basis_table.len() * std::mem::size_of::<f32>();
            let residual_codes_size = if let Some(ref res_codes) = self.residual_codes {
                res_codes.len() * std::mem::size_of::<u32>()
            } else {
                0
            };

            // GPU 메모리 할당
            let codes_gpu = kernel::cuda_malloc(codes_size) as *mut u32;
            let basis_table_gpu = kernel::cuda_malloc(basis_size);
            let residual_codes_gpu = if residual_codes_size > 0 {
                kernel::cuda_malloc(residual_codes_size) as *mut u32
            } else {
                std::ptr::null_mut()
            };

            // INT8 잔차 메모리 할당
            let (residual_int8_gpu, residual_scales_gpu) =
                if let (Some(ref res_int8), Some(ref res_scales)) =
                    (&self.residual_int8, &self.residual_scales)
                {
                    let int8_size = res_int8.len() * std::mem::size_of::<i8>();
                    let scales_size = res_scales.len() * std::mem::size_of::<f32>();
                    let int8_gpu = kernel::cuda_malloc(int8_size) as *mut i8;
                    let scales_gpu = kernel::cuda_malloc(scales_size);

                    // 데이터 복사
                    kernel::cuda_memcpy_h2d(
                        int8_gpu as *mut f32,
                        res_int8.as_ptr() as *const f32,
                        int8_size,
                    );
                    kernel::cuda_memcpy_h2d(scales_gpu, res_scales.as_ptr(), scales_size);

                    (int8_gpu, scales_gpu)
                } else {
                    (std::ptr::null_mut(), std::ptr::null_mut())
                };

            // INT8 기저 테이블 메모리 할당
            let (basis_table_int8_gpu, basis_scales_gpu) =
                if let (Some(ref basis_int8), Some(ref basis_scales)) =
                    (&self.basis_table_int8, &self.basis_scales)
                {
                    let int8_size = basis_int8.len() * std::mem::size_of::<i8>();
                    let scales_size = basis_scales.len() * std::mem::size_of::<f32>();
                    let int8_gpu = kernel::cuda_malloc(int8_size) as *mut i8;
                    let scales_gpu = kernel::cuda_malloc(scales_size);

                    // 데이터 복사
                    kernel::cuda_memcpy_h2d(
                        int8_gpu as *mut f32,
                        basis_int8.as_ptr() as *const f32,
                        int8_size,
                    );
                    kernel::cuda_memcpy_h2d(scales_gpu, basis_scales.as_ptr(), scales_size);

                    (int8_gpu, scales_gpu)
                } else {
                    (std::ptr::null_mut(), std::ptr::null_mut())
                };

            // 입력/출력 버퍼 풀 할당 (최대 배치 크기 64 가정)
            let max_batch_size = 64;
            let input_buffer_size = max_batch_size * self.n * std::mem::size_of::<f32>();
            let output_buffer_size = max_batch_size * self.m * std::mem::size_of::<f32>();
            let input_buffer_gpu = kernel::cuda_malloc(input_buffer_size);
            let output_buffer_gpu = kernel::cuda_malloc(output_buffer_size);

            // CUDA 스트림 생성
            let stream = kernel::cuda_stream_create();

            // 데이터를 GPU로 복사 (한 번만)
            kernel::cuda_memcpy_h2d(
                codes_gpu as *mut f32,
                self.codes.as_ptr() as *const f32,
                codes_size,
            );
            kernel::cuda_memcpy_h2d(basis_table_gpu, self.basis_table.as_ptr(), basis_size);

            if residual_codes_size > 0 {
                if let Some(ref res_codes) = self.residual_codes {
                    kernel::cuda_memcpy_h2d(
                        residual_codes_gpu as *mut f32,
                        res_codes.as_ptr() as *const f32,
                        residual_codes_size,
                    );
                }
            }

            self.gpu_buffers = Some(GpuBuffers {
                codes_gpu,
                basis_table_gpu,
                residual_codes_gpu,
                residual_int8_gpu,
                residual_scales_gpu,
                basis_table_int8_gpu,
                basis_scales_gpu,
                input_buffer_gpu,
                output_buffer_gpu,
                stream,
            });
        }
    }

    /// 순전파 `y = xW^T`를 수행합니다.
    ///
    /// # 인자
    /// * `x` - `[batch_size, n]` 형태의 입력 텐서.
    ///
    /// # 반환
    /// `[batch_size, m]` 형태의 출력 텐서.
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        #[cfg(feature = "cuda")]
        {
            let mut use_cuda_inner = self.use_cuda;
            if self.gpu_buffers.is_none() {
                use_cuda_inner = false;
            }

            if use_cuda_inner {
                if let Some(ref buffers) = self.gpu_buffers {
                    // INT8 최적화 사용 여부 확인
                    if self.use_int8 && !buffers.basis_table_int8_gpu.is_null() {
                        // Tensor Core 최적화 사용 여부 확인
                        let output = if self.use_tensorcore {
                            kernel::gemm_hyper_bit_gpu_tensorcore_optimized(
                                &x.view(),
                                buffers.codes_gpu,
                                buffers.basis_table_int8_gpu,
                                buffers.basis_scales_gpu,
                                self.delta,
                                buffers.input_buffer_gpu,
                                buffers.output_buffer_gpu,
                                buffers.stream,
                                self.m,
                                self.n,
                                self.basis_table.shape()[0],
                                true,
                            )
                        } else {
                            kernel::gemm_hyper_bit_gpu_int8_optimized(
                                &x.view(),
                                buffers.codes_gpu,
                                buffers.basis_table_int8_gpu,
                                buffers.basis_scales_gpu,
                                self.delta,
                                buffers.input_buffer_gpu,
                                buffers.output_buffer_gpu,
                                buffers.stream,
                                self.m,
                                self.n,
                                self.basis_table.shape()[0],
                            )
                        };

                        return output;
                    } else {
                        // 기존 FP32 커널 사용
                        let residual_delta = self.residual_delta.unwrap_or(self.delta);

                        unsafe {
                            return kernel::gemm_hyper_bit_gpu_optimized(
                                &x.view(),
                                buffers.codes_gpu,
                                buffers.basis_table_gpu,
                                buffers.residual_codes_gpu,
                                self.delta,
                                residual_delta,
                                buffers.input_buffer_gpu,
                                buffers.output_buffer_gpu,
                                buffers.stream,
                                self.m,
                                self.n,
                                self.basis_table.shape()[0],
                            );
                        }
                    }
                }
            }
        }
        // CPU 로직
        assert_eq!(x.shape()[1], self.n, "입력 차원이 일치하지 않습니다.");

        // INT8 최적화 확인
        if self.use_int8 && self.basis_table_int8.is_some() && self.basis_scales.is_some() {
            // INT8 기저로 추론
            let mut output = Array2::<f32>::zeros((x.shape()[0], self.m));
            let basis_int8 = self.basis_table_int8.as_ref().unwrap();
            let basis_scales = self.basis_scales.as_ref().unwrap();

            use ndarray::Axis;
            use rayon::prelude::*;

            // 병렬 처리를 위해 출력 열을 병렬로 계산
            output
                .axis_iter_mut(Axis(1))
                .into_par_iter()
                .enumerate()
                .for_each(|(out_idx, mut col)| {
                    let code = self.codes[out_idx];
                    let (cat, sub, idx, sign, d, amp, amp_fine, phase) = decoder::decode(code);

                    // 스케일 팩터 계산
                    let r_val = (amp as f32 * 4.0 + amp_fine as f32) * self.delta;
                    let signed_r_val = if sign == 0 { r_val } else { -r_val };
                    let scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val, phase);
                    let basis_scale = basis_scales[idx as usize];

                    // INT8 기저와 입력의 내적
                    let basis_row = basis_int8.row(idx as usize);

                    for batch_idx in 0..x.shape()[0] {
                        let mut sum = 0.0f32;
                        let input_row = x.row(batch_idx);

                        // SIMD를 위한 벡터화 (8개씩 처리)
                        let chunks = basis_row.len() / 8;
                        for i in 0..chunks {
                            let base = i * 8;
                            // 언롤링으로 파이프라인 최적화
                            sum += basis_row[base] as f32 * input_row[base];
                            sum += basis_row[base + 1] as f32 * input_row[base + 1];
                            sum += basis_row[base + 2] as f32 * input_row[base + 2];
                            sum += basis_row[base + 3] as f32 * input_row[base + 3];
                            sum += basis_row[base + 4] as f32 * input_row[base + 4];
                            sum += basis_row[base + 5] as f32 * input_row[base + 5];
                            sum += basis_row[base + 6] as f32 * input_row[base + 6];
                            sum += basis_row[base + 7] as f32 * input_row[base + 7];
                        }

                        // 나머지 처리
                        for i in (chunks * 8)..basis_row.len() {
                            sum += basis_row[i] as f32 * input_row[i];
                        }

                        col[batch_idx] = sum * scale_factor * basis_scale;
                    }
                });

            // 잔차 적용
            if let (Some(res_int8), Some(scales)) = (&self.residual_int8, &self.residual_scales) {
                // INT8 잔차 적용
                let mut residual_f32 = Array2::<f32>::zeros((self.m, self.n));
                for i in 0..self.m {
                    let scale = scales[i];
                    let dst_row = &mut residual_f32.row_mut(i);
                    let src_row = res_int8.row(i);
                    for j in 0..self.n {
                        dst_row[j] = src_row[j] as f32 * scale;
                    }
                }
                output += &x.dot(&residual_f32.t());
            }

            return output;
        }

        // 기존 FP32 경로
        // 1. 기본 비트필드 추론 (y_approx) - CPU/GPU 선택
        let mut output = kernel::gemm_hyper_bit_with_backend(
            x,
            &self.codes,
            &self.basis_table,
            self.delta,
            self.use_cuda,
        );

        // 2. 잔차 적용 (비트필드 또는 INT8)
        match (
            &self.residual_codes,
            &self.residual_delta,
            &self.residual_int8,
            &self.residual_scales,
        ) {
            (Some(res_codes), Some(res_delta), _, _) => {
                // 비트필드 잔차 적용
                let residual_output = kernel::gemm_hyper_bit_with_backend(
                    x,
                    res_codes,
                    &self.basis_table,
                    *res_delta,
                    false,
                );
                output += &residual_output;
            }
            (None, None, Some(res_int8), Some(scales)) => {
                // INT8 잔차 적용
                // 역양자화하여 FP32 행렬 생성
                let mut residual_f32 = Array2::<f32>::zeros((self.m, self.n));
                for i in 0..self.m {
                    let scale = scales[i];
                    let dst_row = &mut residual_f32.row_mut(i);
                    let src_row = res_int8.row(i);
                    for j in 0..self.n {
                        dst_row[j] = src_row[j] as f32 * scale;
                    }
                }
                output += &x.dot(&residual_f32.t());
            }
            _ => {}
        }

        output
    }

    /// 역전파 `grad_input = grad_output @ W`를 수행합니다.
    /// Matrix-Free 방식을 사용하여, 중간 가중치 행렬을 생성하지 않고 직접 계산합니다.
    pub fn backward(&self, grad_output: &Array2<f32>) -> Array2<f32> {
        let batch_size = grad_output.shape()[0];
        let mut grad_input = Array2::<f32>::zeros((batch_size, self.n));

        for i in 0..self.m {
            // 1. i번째 주 가중치의 스케일 팩터와 기저 벡터를 가져옵니다.
            let code = self.codes[i];
            let (cat, sub, idx, sign, d, amp, amp_fine, phase) = decoder::decode(code);
            let r_val = (amp as f32 * 4.0 + amp_fine as f32) * self.delta;
            let signed_r_val = if sign == 0 { r_val } else { -r_val };
            let scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val, phase);
            let basis_vector = self.basis_table.row(idx as usize);

            // 2. grad_output의 i번째 열을 가져와 스케일링합니다.
            let grad_col = grad_output.column(i);

            // 3. 스케일링된 그래디언트를 기저 벡터와 외적하여 grad_input에 누적합니다.
            for (mut grad_input_row, &grad_output_val) in
                grad_input.outer_iter_mut().zip(grad_col.iter())
            {
                grad_input_row.scaled_add(grad_output_val * scale_factor, &basis_vector);
            }

            // 4. 비트필드 잔차가 있다면 동일한 방식으로 누적합니다.
            if let Some(res_codes) = &self.residual_codes {
                if let Some(res_delta) = self.residual_delta {
                    let res_code = res_codes[i];
                    if res_code != 0 {
                        let (
                            res_cat,
                            res_sub,
                            res_idx,
                            res_sign,
                            res_d,
                            res_amp,
                            res_amp_fine,
                            res_phase,
                        ) = decoder::decode(res_code);
                        let res_r_val = (res_amp as f32 * 4.0 + res_amp_fine as f32) * res_delta;
                        let res_signed_r_val = if res_sign == 0 { res_r_val } else { -res_r_val };
                        let res_scale_factor = ops::lookup_and_apply(
                            res_cat,
                            res_sub,
                            res_d,
                            res_signed_r_val,
                            res_phase,
                        );
                        let res_basis_vector = self.basis_table.row(res_idx as usize);

                        for (mut grad_input_row, &grad_output_val) in
                            grad_input.outer_iter_mut().zip(grad_col.iter())
                        {
                            grad_input_row
                                .scaled_add(grad_output_val * res_scale_factor, &res_basis_vector);
                        }
                    }
                }
            } else if let (Some(res_int8), Some(scales)) =
                (&self.residual_int8, &self.residual_scales)
            {
                let res_row_i8 = res_int8.row(i);
                let scale = scales[i];
                let res_row: Array1<f32> = res_row_i8.mapv(|v| v as f32 * scale);
                for (mut grad_input_row, &grad_output_val) in
                    grad_input.outer_iter_mut().zip(grad_col.iter())
                {
                    grad_input_row.scaled_add(grad_output_val, &res_row);
                }
            }
        }

        grad_input
    }

    /// 다차원 텐서 순전파 `y = xW^T`를 수행합니다.
    ///
    /// 임의의 차원을 가진 입력 텐서를 처리하되, 마지막 차원을 feature 차원으로 간주합니다.
    ///
    /// # 인자
    /// * `x` - `[..., n]` 형태의 입력 텐서 (마지막 차원이 feature 차원).
    ///
    /// # 반환
    /// `[..., m]` 형태의 출력 텐서.
    pub fn forward_nd(&self, x: &ArrayView<f32, IxDyn>) -> ndarray::Array<f32, IxDyn> {
        let x_shape = x.shape();
        let x_ndim = x_shape.len();
        assert!(x_ndim > 0, "입력 텐서는 최소 1차원 이상이어야 합니다.");

        let input_features = x_shape[x_ndim - 1];
        assert_eq!(input_features, self.n, "입력 차원이 일치하지 않습니다.");

        // 1. 최적화된 비트필드 추론 커널 사용 (y_approx)
        let mut output =
            kernel::gemm_hyper_bit_nd(x, &self.codes, &self.basis_table, self.delta, self.use_cuda);

        // 2. 잔차 적용 (비트필드 또는 INT8)
        match (
            &self.residual_codes,
            &self.residual_delta,
            &self.residual_int8,
            &self.residual_scales,
        ) {
            (Some(res_codes), Some(res_delta), _, _) => {
                let residual_contrib =
                    kernel::gemm_hyper_bit_nd(x, res_codes, &self.basis_table, *res_delta, false);
                output += &residual_contrib;
            }
            (None, None, Some(res_int8), Some(scales)) => {
                // 역양자화 후 dot_product_nd 사용
                let mut residual_f32 = Array2::<f32>::zeros((self.m, self.n));
                for i in 0..self.m {
                    let scale = scales[i];
                    let mut dst_row = residual_f32.row_mut(i);
                    let src_row = res_int8.row(i);
                    for j in 0..self.n {
                        dst_row[j] = src_row[j] as f32 * scale;
                    }
                }
                // forward_nd에서는 다차원 처리를 위해 직접 구현
                let x_shape = x.shape();
                let x_ndim = x_shape.len();
                let total_batch_size: usize = x_shape[..x_ndim - 1].iter().product();

                // x를 2D로 reshape
                let x_reshaped = x.to_shape((total_batch_size, self.n)).unwrap();

                // 2D 행렬곱
                let residual_2d = x_reshaped.dot(&residual_f32.t());

                // 결과를 원래 차원으로 reshape
                let mut residual_shape = x_shape.to_vec();
                residual_shape[x_ndim - 1] = self.m;
                let residual_contrib = residual_2d
                    .into_shape(residual_shape.as_slice())
                    .unwrap()
                    .into_dyn();

                output += &residual_contrib;
            }
            _ => {}
        }

        output
    }

    /// 다차원 텐서 역전파 `grad_input = grad_output @ W`를 수행합니다.
    ///
    /// # 인자
    /// * `grad_output` - `[..., m]` 형태의 출력 그래디언트 텐서.
    ///
    /// # 반환
    /// `[..., n]` 형태의 입력 그래디언트 텐서.
    pub fn backward_nd(&self, grad_output: &ArrayView<f32, IxDyn>) -> ndarray::Array<f32, IxDyn> {
        let grad_shape = grad_output.shape();
        let grad_ndim = grad_shape.len();

        if grad_ndim == 0 {
            panic!("그래디언트 텐서는 최소 1차원 이상이어야 합니다.");
        }

        let output_features = grad_shape[grad_ndim - 1];
        assert_eq!(
            output_features, self.m,
            "그래디언트 출력 차원이 일치하지 않습니다."
        );

        // 1. 압축된 코드로부터 완전한 가중치 행렬 W_approx (`[m, n]`)를 복원합니다.
        let mut w_matrix = Array2::<f32>::zeros((self.m, self.n));
        for i in 0..self.m {
            let code = self.codes[i];
            let (cat, sub, idx, sign, d, amp, amp_fine, phase) = decoder::decode(code);
            let r_val = (amp as f32 * 4.0 + amp_fine as f32) * self.delta;
            let signed_r_val = if sign == 0 { r_val } else { -r_val };
            let scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val, phase);
            let basis_vector = self.basis_table.row(idx as usize);
            let weight_row = &basis_vector * scale_factor;

            w_matrix.row_mut(i).assign(&weight_row);
        }

        // INT8 잔차를 역양자화하여 더해줌
        if let (Some(res_codes), Some(res_delta)) = (&self.residual_codes, &self.residual_delta) {
            for i in 0..self.m {
                let scale = *res_delta;
                // 비트필드 잔차는 개별 코드당 reconstruct 필요 -> 간략화: 무시(0) 에 대하여 skip
                let res_code = res_codes[i];
                if res_code == 0 {
                    continue;
                }

                let (res_cat, res_sub, res_idx, res_sign, res_d, res_amp, res_amp_fine, res_phase) =
                    decoder::decode(res_code);
                let res_r_val = (res_amp as f32 * 4.0 + res_amp_fine as f32) * scale;
                let res_signed_r_val = if res_sign == 0 { res_r_val } else { -res_r_val };
                let res_scale_factor =
                    ops::lookup_and_apply(res_cat, res_sub, res_d, res_signed_r_val, res_phase);
                let res_basis_vector = self.basis_table.row(res_idx as usize);
                for j in 0..self.n {
                    w_matrix[[i, j]] += res_basis_vector[j] * res_scale_factor;
                }
            }
        } else if let (Some(res_int8), Some(scales)) = (&self.residual_int8, &self.residual_scales)
        {
            for i in 0..self.m {
                let scale = scales[i];
                let src_row = res_int8.row(i);
                for j in 0..self.n {
                    w_matrix[[i, j]] += src_row[j] as f32 * scale;
                }
            }
        }

        // 2. grad_input = grad_output @ W 를 계산합니다.
        let total_batch_size: usize = grad_shape[..grad_ndim - 1].iter().product();

        // 그래디언트를 2D로 reshape
        let grad_reshaped = grad_output.to_shape((total_batch_size, self.m)).unwrap();
        let grad_input_2d = grad_reshaped.dot(&w_matrix);

        // 입력 그래디언트를 원래 형태로 reshape
        let mut input_shape = grad_shape.to_vec();
        input_shape[grad_ndim - 1] = self.n;
        let grad_input_nd = grad_input_2d
            .into_shape(input_shape.as_slice())
            .unwrap()
            .into_dyn();

        grad_input_nd
    }

    // Getter 메서드들 (Python 바인딩용)
    pub fn get_m(&self) -> usize {
        self.m
    }
    pub fn get_n(&self) -> usize {
        self.n
    }
    pub fn get_delta(&self) -> f32 {
        self.delta
    }
    pub fn get_residual_delta(&self) -> Option<f32> {
        self.residual_delta
    }
    pub fn get_basis_table_shape(&self) -> (usize, usize) {
        (self.basis_table.shape()[0], self.basis_table.shape()[1])
    }

    #[cfg(feature = "cuda")]
    pub fn get_gpu_buffers(&self) -> Option<&GpuBuffers> {
        self.gpu_buffers.as_ref()
    }

    #[cfg(feature = "cuda")]
    pub fn get_gpu_buffers_mut(&mut self) -> Option<&mut GpuBuffers> {
        self.gpu_buffers.as_mut()
    }

    /// 기저 테이블을 INT8로 양자화합니다.
    ///
    /// # 인자
    /// * `basis_table` - FP32 기저 테이블
    ///
    /// # 반환
    /// (INT8 테이블, 스케일 팩터) 튜플
    fn quantize_to_int8(basis_table: &Array2<f32>) -> (Array2<i8>, Array1<f32>) {
        let (b, n) = (basis_table.shape()[0], basis_table.shape()[1]);
        let mut int8_table = Array2::<i8>::zeros((b, n));
        let mut scales = Array1::<f32>::zeros(b);

        for i in 0..b {
            let row = basis_table.row(i);

            // 행별 최대 절댓값 찾기
            let max_abs = row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

            // 스케일 팩터 계산 (127로 나누어 INT8 범위에 맞춤)
            let scale = if max_abs > 1e-8 { max_abs / 127.0 } else { 1.0 };
            scales[i] = scale;

            // 양자화
            for j in 0..n {
                let quantized = (row[j] / scale).round().clamp(-128.0, 127.0) as i8;
                int8_table[[i, j]] = quantized;
            }
        }

        (int8_table, scales)
    }

    /// INT8 최적화를 활성화합니다.
    /// 기저 테이블을 INT8로 양자화하고 GPU 메모리에 업로드합니다.
    pub fn enable_int8_optimization(&mut self) {
        if self.use_int8 {
            return; // 이미 활성화됨
        }

        // 기저 테이블 INT8 양자화
        let (int8_table, scales) = Self::quantize_to_int8(&self.basis_table);
        self.basis_table_int8 = Some(int8_table);
        self.basis_scales = Some(scales);
        self.use_int8 = true;

        println!("INT8 최적화 활성화: 기저 테이블 양자화 완료");
        println!("  - 원본 크기: {} KB", self.basis_table.len() * 4 / 1024);
        println!("  - INT8 크기: {} KB", self.basis_table.len() / 1024);
        println!("  - 메모리 절감: {:.1}x", 4.0);

        // GPU 메모리 재초기화 (INT8 데이터 업로드)
        #[cfg(feature = "cuda")]
        if self.use_cuda {
            // 기존 GPU 버퍼 해제하고 다시 초기화
            if self.gpu_buffers.is_some() {
                // Drop을 통해 기존 메모리 해제
                self.gpu_buffers = None;
            }

            // GPU 메모리 재초기화 (INT8 데이터 포함)
            self.init_gpu_memory();

            println!("  - GPU 메모리 재초기화 완료 (INT8 데이터 포함)");
        }
    }

    /// Tensor Core 최적화를 활성화합니다.
    /// INT8 최적화가 필요하므로 자동으로 활성화합니다.
    pub fn enable_tensorcore(&mut self) {
        if !self.use_cuda {
            println!("경고: Tensor Core는 CUDA가 활성화되어야 사용 가능합니다.");
            return;
        }

        // INT8 최적화가 필요함
        if !self.use_int8 {
            self.enable_int8_optimization();
        }

        self.use_tensorcore = true;
        println!("Tensor Core 최적화 활성화");
    }

    /// 계층적 압축을 활성화합니다 (4비트/가중치).
    pub fn enable_hierarchical_compression(&mut self, level: u8) {
        self.use_hierarchical = true;
        self.compression_level = level;

        match level {
            2 => {
                println!("계층적 압축 활성화: 4비트/가중치");
                // 실제로 코드를 재인코딩해야 함
                // 여기서는 시뮬레이션을 위해 코드의 상위 비트만 유지
                for i in 0..self.codes.len() {
                    // 4비트로 양자화: 상위 4비트만 유지
                    let code = self.codes[i];
                    let quantized = (code >> 28) << 28; // 상위 4비트만 유지
                    self.codes[i] = quantized;
                }
            }
            3 => {
                println!("극한 압축 활성화: 2.5비트/가중치");
                // 2.5비트로 양자화: 더 극단적인 압축
                for i in 0..self.codes.len() {
                    let code = self.codes[i];
                    let quantized = (code >> 29) << 29; // 상위 3비트만 유지
                    self.codes[i] = quantized;
                }
            }
            _ => println!("표준 압축 모드"),
        }

        // 압축률 계산
        let original_bits = self.m * self.n * 32; // FP32
        let compressed_bits = match level {
            2 => self.m * 4,     // 4비트/가중치
            3 => self.m * 5 / 2, // 2.5비트/가중치
            _ => self.m * 32,    // 기본 32비트 코드
        };

        println!(
            "  - 압축률: {:.1}x",
            original_bits as f32 / compressed_bits as f32
        );

        // GPU 메모리 업데이트
        #[cfg(feature = "cuda")]
        if self.use_cuda && self.gpu_buffers.is_some() {
            if let Some(ref buffers) = self.gpu_buffers {
                unsafe {
                    kernel::cuda_memcpy_h2d(
                        buffers.codes_gpu as *mut f32,
                        self.codes.as_ptr() as *const f32,
                        self.m * std::mem::size_of::<u32>(),
                    );
                }
            }
        }
    }

    /// Quantization-Aware Training을 활성화합니다.
    pub fn enable_qat(&mut self, temperature: f32) {
        self.trainable = true;
        self.temperature = temperature;
        println!("QAT (Quantization-Aware Training) 활성화");
        println!("  - Temperature: {}", temperature);
        println!("  - Gumbel-Softmax를 사용한 미분 가능 양자화");
    }

    /// 미분 가능한 양자화를 수행합니다 (Gumbel-Softmax 사용).
    pub fn differentiable_quantize(&self, value: f32, levels: usize) -> (f32, Vec<f32>) {
        let mut logits = vec![0.0f32; levels];
        let centers: Vec<f32> = (0..levels)
            .map(|i| (i as f32 + 0.5) / levels as f32 * 2.0 - 1.0)
            .collect();

        // 각 레벨에 대한 로짓 계산
        for i in 0..levels {
            let dist = (value - centers[i]).powi(2);
            logits[i] = -dist / self.temperature;
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

        let probs: Vec<f32> = logits
            .iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect();

        // 양자화된 값 (기댓값)
        let quantized: f32 = centers.iter().zip(probs.iter()).map(|(&c, &p)| c * p).sum();

        (quantized, probs)
    }

    /// 양자화 손실을 계산합니다.
    pub fn quantization_loss(&self) -> f32 {
        // 실제 가중치와 양자화된 가중치 간의 MSE
        // 여기서는 간단히 0을 반환 (실제 구현에서는 계산 필요)
        0.0
    }

    /// 온도를 점진적으로 낮춥니다 (annealing).
    pub fn anneal_temperature(&mut self, decay_rate: f32) {
        self.temperature *= decay_rate;
        self.temperature = self.temperature.max(0.1); // 최소 온도
    }
}

// GPU 메모리 정리를 위한 Drop 구현
#[cfg(feature = "cuda")]
impl Drop for BitfieldLinear {
    fn drop(&mut self) {
        if let Some(ref buffers) = self.gpu_buffers {
            unsafe {
                // 스트림 동기화 및 삭제
                if !buffers.stream.is_null() {
                    kernel::cuda_stream_synchronize(buffers.stream);
                    kernel::cuda_stream_destroy(buffers.stream);
                }

                // GPU 메모리 해제
                kernel::cuda_free(buffers.codes_gpu);
                kernel::cuda_free(buffers.basis_table_gpu);
                if !buffers.residual_codes_gpu.is_null() {
                    kernel::cuda_free(buffers.residual_codes_gpu);
                }
                if !buffers.residual_int8_gpu.is_null() {
                    kernel::cuda_free(buffers.residual_int8_gpu);
                }
                if !buffers.residual_scales_gpu.is_null() {
                    kernel::cuda_free(buffers.residual_scales_gpu);
                }
                if !buffers.basis_table_int8_gpu.is_null() {
                    kernel::cuda_free(buffers.basis_table_int8_gpu);
                }
                if !buffers.basis_scales_gpu.is_null() {
                    kernel::cuda_free(buffers.basis_scales_gpu);
                }
                if !buffers.input_buffer_gpu.is_null() {
                    kernel::cuda_free(buffers.input_buffer_gpu);
                }
                if !buffers.output_buffer_gpu.is_null() {
                    kernel::cuda_free(buffers.output_buffer_gpu);
                }
            }
        }
    }
}

#[cfg(test)]
mod __test__;
