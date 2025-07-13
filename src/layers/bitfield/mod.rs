// src/layers/bitfield/mod.rs

//! # 비트필드 기반 하이퍼볼릭 레이어
//!
//! 이 모듈은 비트필드 기반 직접 추론 레이어에 필요한 모든 구성요소를 캡슐화합니다.
//! 디코더, 기저 테이블 관리, 하이퍼볼릭 연산, 그리고 추론 커널을 포함합니다.
//!
//! 사용자 친화적인 `BitfieldLinear` 구조체를 노출하여, 가중치가 고도로 압축된 형태로
//! 저장된다는 점을 제외하면 표준 선형 레이어처럼 사용할 수 있습니다.

pub mod basis;
pub mod decoder;
pub mod kernel;
pub mod ops;

use ndarray::{Array1, Array2, ArrayView, IxDyn};

// GPU 메모리 포인터를 위한 구조체. INT8 잔차 관련 필드 제거.
#[cfg(feature = "cuda")]
pub struct GpuBuffers {
    pub codes_gpu: *mut u32,
    pub basis_table_gpu: *mut f32,
    pub residual_codes_gpu: *mut u32,
    // INT8 잔차 GPU 메모리
    pub residual_int8_gpu: *mut i8,
    pub residual_scales_gpu: *mut f32,
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
    residual_int8: Option<Array2<i8>>,   // [m, n]
    residual_scales: Option<Array1<f32>>, // [m]
    pub use_cuda: bool,
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
    pub fn new(m: usize, n: usize, b: usize, r_max: f32) -> Self {
        // 플레이스홀더 초기화.
        let codes = Array1::<u32>::zeros(m);
        let basis_table = basis::load_basis_table(b, n);
        
        Self {
            codes,
            basis_table,
            delta: r_max / (255.0 * 1024.0), // 정밀도 향상에 맞춰 delta 조정
            m,
            n,
            residual_codes: None,
            residual_delta: None,
            residual_int8: None,
            residual_scales: None,
            use_cuda: false,
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
    ) -> Self {
        let (m, n) = (weights.shape()[0], weights.shape()[1]);
        
        // 1. 가중치에서 주요 방향 추출하여 기저 벡터 테이블 생성
        let mut basis_table = Array2::<f32>::zeros((b, n));
        
        // 첫 n개는 표준 기저 벡터
        for i in 0..b.min(n) {
            basis_table[[i, i]] = 1.0;
        }
        
        // 가중치 행렬에서 가장 큰 노름을 가진 행들을 기저로 사용
        if m > 0 && b > n {
            let mut norms: Vec<(f32, usize)> = weights.rows()
                .into_iter()
                .enumerate()
                .map(|(i, row)| (row.dot(&row).sqrt(), i))
                .collect();
            norms.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            
            for (idx, &(norm, row_idx)) in norms.iter().enumerate() {
                if idx + n >= b || norm < 1e-6 {
                    break;
                }
                let row = weights.row(row_idx);
                let normalized = &row / norm;
                basis_table.row_mut(n + idx).assign(&normalized);
            }
        }

        // 2. 각 가중치 행을 압축
        let mut codes = Array1::<u32>::zeros(m);
        let fine_amp_max = 1023.0; // 10비트
        
        // 가중치 분포 분석
        let mut max_norm = 0.0f32;
        for i in 0..m {
            let norm = weights.row(i).dot(&weights.row(i)).sqrt();
            max_norm = max_norm.max(norm);
        }
        
        // 적응적 delta: 최대 노름을 18비트 범위에 맞춤
        let delta = max_norm.max(r_max) / (255.0 * 4.0); // 10비트 정밀도 (amp 8비트 + amp_fine 2비트)
        let mut residual_weights = Array2::<f32>::zeros((m, n));
        
        for i in 0..m {
            let w_row = weights.row(i);
            
            // 가중치 행의 노름 계산
            let norm = w_row.dot(&w_row).sqrt();
            if norm < 1e-8 {
                // 영벡터는 코드 0으로
                codes[i] = 0;
                continue;
            }
            
            // 방향 벡터
            let direction = &w_row / norm;
            
            // 가장 가까운 기저 벡터 찾기 (부호도 고려)
            let mut best_idx = 0;
            let mut best_dot = -1.0f32;
            let mut best_sign = 1.0f32;
            
            for j in 0..b {
                let dot = direction.dot(&basis_table.row(j));
                let abs_dot = dot.abs();
                if abs_dot > best_dot {
                    best_dot = abs_dot;
                    best_idx = j;
                    best_sign = if dot > 0.0 { 1.0 } else { -1.0 };
                }
            }

            // 가중치 norm 크기에 따라 적절한 기하학(cat)과 스케일링 함수를 선택
            let (cat, sub, d) = if norm < 0.95 {
                (0, 0, 0)  // norm < 1 이므로 Poincaré 사용
            } else {
                (1, 0, 0)  // norm >= 1 이므로 Lorentz 사용
            };

            // 선택된 기하학에 맞는 역함수를 사용하여 'r' 값을 계산
            let r = match cat {
                0 => 2.0 * norm.min(0.999).atanh(), // s = tanh(r/2) => r = 2 * atanh(s)
                1 => norm.asinh(),                 // s = sinh(r) => r = asinh(s)
                _ => norm, // Unreachable with current logic
            };
            
            // 위상 계산: 가중치 벡터의 주요 2개 성분으로부터 각도 추출
            let phase = if n >= 2 {
                // 가장 큰 두 성분 찾기
                let mut max_idx1 = 0;
                let mut max_idx2 = 1;
                let mut max_val1 = w_row[0].abs();
                let mut max_val2 = w_row[1].abs();
                
                if max_val2 > max_val1 {
                    std::mem::swap(&mut max_idx1, &mut max_idx2);
                    std::mem::swap(&mut max_val1, &mut max_val2);
                }
                
                for j in 2..n {
                    let val = w_row[j].abs();
                    if val > max_val1 {
                        max_idx2 = max_idx1;
                        max_val2 = max_val1;
                        max_idx1 = j;
                        max_val1 = val;
                    } else if val > max_val2 {
                        max_idx2 = j;
                        max_val2 = val;
                    }
                }
                
                // atan2를 사용하여 위상 각도 계산
                let angle = w_row[max_idx2].atan2(w_row[max_idx1]);
                // -π ~ π를 0 ~ 255로 매핑
                ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 255.0) as u8
            } else {
                0
            };
            
            // 부호 비트 결정 (0: 양수, 1: 음수)
            let sign_bit = if best_sign > 0.0 { 0 } else { 1 };

            // 10비트 양자화 (8비트 주 + 2비트 미세)
            let total_quantized = ((r.abs() / delta) as u32).min((256 * 4 - 1) as u32);
            let amp = (total_quantized / 4) as u8;
            let amp_fine = (total_quantized % 4) as u8;
            
            // 새로운 인코딩 함수 호출 (32비트, 위상 포함)
            codes[i] = decoder::encode(cat, sub, best_idx as u8, sign_bit, d, amp, amp_fine, phase);
            
            // 재구성 오차 계산
            let full_res_r = (amp as f32 * 4.0 + amp_fine as f32) * delta;
            let signed_r = if sign_bit == 0 { full_res_r } else { -full_res_r };
            let reconstructed_scale = ops::lookup_and_apply(cat, sub, d, signed_r, phase);
            let reconstructed = &basis_table.row(best_idx) * reconstructed_scale;
            
            // 잔차 저장
            let error = &w_row - &reconstructed;
            residual_weights.row_mut(i).assign(&error);
        }
        
        let (residual_codes, residual_delta, residual_int8, residual_scales) = if use_residual {
            if bitfield_residual {
                // 잔차도 비트필드로 인코딩
                let mut residual_codes = Array1::<u32>::zeros(m);
                
                // 잔차의 최대 노름 계산
                let mut max_residual_norm = 0.0f32;
                for i in 0..m {
                    let norm = residual_weights.row(i).dot(&residual_weights.row(i)).sqrt();
                    max_residual_norm = max_residual_norm.max(norm);
                }
                
                // 잔차용 delta (더 작은 범위)
                let residual_delta = max_residual_norm / (255.0 * 4.0);
                
                // 각 잔차 행을 비트필드로 인코딩
                for i in 0..m {
                    let res_row = residual_weights.row(i);
                    let norm = res_row.dot(&res_row).sqrt();
                    
                    if norm < 1e-8 {
                        residual_codes[i] = 0;
                        continue;
                    }
                    
                    // 방향 벡터
                    let direction = &res_row / norm;
                    
                    // 가장 가까운 기저 벡터 찾기
                    let mut best_idx = 0;
                    let mut best_dot = -1.0f32;
                    let mut best_sign = 1.0f32;
                    
                    for j in 0..b {
                        let dot = direction.dot(&basis_table.row(j));
                        let abs_dot = dot.abs();
                        if abs_dot > best_dot {
                            best_dot = abs_dot;
                            best_idx = j;
                            best_sign = if dot > 0.0 { 1.0 } else { -1.0 };
                        }
                    }
                    
                    // 잔차는 주로 Poincaré 기하학 사용 (작은 값)
                    let (cat, sub, d) = (0, 0, 0);
                    let r = 2.0 * norm.min(0.999).atanh();
                    
                    // 위상 계산
                    let phase = 0; // 잔차는 위상 정보가 덜 중요
                    
                    // 부호 비트
                    let sign_bit = if best_sign > 0.0 { 0 } else { 1 };
                    
                    // 양자화
                    let total_quantized = ((r.abs() / residual_delta) as u32).min(1023);
                    let amp = (total_quantized / 4) as u8;
                    let amp_fine = (total_quantized % 4) as u8;
                    
                    residual_codes[i] = decoder::encode(cat, sub, best_idx as u8, sign_bit, d, amp, amp_fine, phase);
                }
                
                // INT8 잔차 저장
                (Some(residual_codes), Some(residual_delta), Some(Array2::<i8>::zeros((m, n))), Some(Array1::<f32>::zeros(m)))
            } else {
                // 기존 INT8 양자화 방식
                let mut res_int8 = Array2::<i8>::zeros((m, n));
                let mut res_scales = Array1::<f32>::zeros(m);
                
                for i in 0..m {
                    let row = residual_weights.row(i);
                    let max_abs = row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                    
                    if max_abs > 1e-8 {
                        let scale = max_abs / 127.0;
                        res_scales[i] = scale;
                        for j in 0..n {
                            res_int8[[i, j]] = (row[j] / scale).round().max(-128.0).min(127.0) as i8;
                        }
                    } else {
                        res_scales[i] = 1.0;
                    }
                }
                (None, None, None, None)
            }
        } else {
            (None, None, None, None)
        };
        
        Self {
            codes,
            basis_table,
            delta,
            m,
            n,
            residual_codes,
            residual_delta,
            residual_int8,
            residual_scales,
            use_cuda: use_cuda,
            #[cfg(feature = "cuda")]
            gpu_buffers: None,
        }
    }

    /// GPU 메모리를 초기화합니다. 
    /// 한 번만 호출되어 데이터를 GPU에 영구적으로 올립니다.
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
            } else { 0 };

            // GPU 메모리 할당
            let codes_gpu = kernel::cuda_malloc(codes_size) as *mut u32;
            let basis_table_gpu = kernel::cuda_malloc(basis_size);
            let residual_codes_gpu = if residual_codes_size > 0 {
                kernel::cuda_malloc(residual_codes_size) as *mut u32
            } else { std::ptr::null_mut() };
            
            // INT8 잔차 메모리 할당
            let (residual_int8_gpu, residual_scales_gpu) = if let (Some(ref res_int8), Some(ref res_scales)) = 
                (&self.residual_int8, &self.residual_scales) {
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
                kernel::cuda_memcpy_h2d(
                    scales_gpu,
                    res_scales.as_ptr(),
                    scales_size,
                );
                
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
            kernel::cuda_memcpy_h2d(
                basis_table_gpu,
                self.basis_table.as_ptr(),
                basis_size,
            );

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
        assert_eq!(x.shape()[1], self.n, "입력 차원이 일치하지 않습니다.");
        
        // GPU 사용 시 최적화된 경로
        #[cfg(feature = "cuda")]
        if self.use_cuda {
            if self.gpu_buffers.is_none() { self.init_gpu_memory(); }
            if let Some(ref buffers) = self.gpu_buffers {
                let residual_delta = self.residual_delta.unwrap_or(self.delta);
                
                // 최적화된 GPU 추론 사용
                let output = kernel::gemm_hyper_bit_gpu_optimized(
                    x,
                    buffers.codes_gpu,
                    buffers.basis_table_gpu,
                    buffers.residual_codes_gpu, // 영구 버퍼 사용
                    self.delta,
                    residual_delta,
                    buffers.input_buffer_gpu,
                    buffers.output_buffer_gpu,
                    buffers.stream,
                    self.m,
                    self.n,
                    self.basis_table.shape()[0],
                );
                
                return output;
            }
        }
        
        // CPU 경로 또는 GPU 버퍼가 초기화되지 않은 경우
        // 1. 기본 비트필드 추론 (y_approx) - CPU/GPU 선택
        let mut output = kernel::gemm_hyper_bit_with_backend(
            x, &self.codes, &self.basis_table, self.delta, self.use_cuda
        );
        
        // 2. 잔차 적용 (비트필드 또는 INT8)
        match (&self.residual_codes, &self.residual_delta, &self.residual_int8, &self.residual_scales) {
            (Some(res_codes), Some(res_delta), _, _) => {
                // 비트필드 잔차 적용
                let residual_output = kernel::gemm_hyper_bit_with_backend(
                    x, res_codes, &self.basis_table, *res_delta, false
                );
                output += &residual_output;
            },
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
            },
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
            for (mut grad_input_row, &grad_output_val) in grad_input.outer_iter_mut().zip(grad_col.iter()) {
                grad_input_row.scaled_add(grad_output_val * scale_factor, &basis_vector);
            }

            // 4. 비트필드 잔차가 있다면 동일한 방식으로 누적합니다.
            if let Some(res_codes) = &self.residual_codes {
                if let Some(res_delta) = self.residual_delta {
                    let res_code = res_codes[i];
                    if res_code != 0 {
                        let (res_cat, res_sub, res_idx, res_sign, res_d, res_amp, res_amp_fine, res_phase) = decoder::decode(res_code);
                        let res_r_val = (res_amp as f32 * 4.0 + res_amp_fine as f32) * res_delta;
                        let res_signed_r_val = if res_sign == 0 { res_r_val } else { -res_r_val };
                        let res_scale_factor = ops::lookup_and_apply(res_cat, res_sub, res_d, res_signed_r_val, res_phase);
                        let res_basis_vector = self.basis_table.row(res_idx as usize);

                        for (mut grad_input_row, &grad_output_val) in grad_input.outer_iter_mut().zip(grad_col.iter()) {
                            grad_input_row.scaled_add(grad_output_val * res_scale_factor, &res_basis_vector);
                        }
                    }
                }
            } else if let (Some(res_int8), Some(scales)) = (&self.residual_int8, &self.residual_scales) {
                let res_row_i8 = res_int8.row(i);
                let scale = scales[i];
                let res_row: Array1<f32> = res_row_i8.mapv(|v| v as f32 * scale);
                for (mut grad_input_row, &grad_output_val) in grad_input.outer_iter_mut().zip(grad_col.iter()) {
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
        let mut output = kernel::gemm_hyper_bit_nd(
            x, &self.codes, &self.basis_table, self.delta, self.use_cuda
        );
        
        // 2. 잔차 적용 (비트필드 또는 INT8)
        match (&self.residual_codes, &self.residual_delta, &self.residual_int8, &self.residual_scales) {
            (Some(res_codes), Some(res_delta), _, _) => {
                let residual_contrib = kernel::gemm_hyper_bit_nd(
                    x, res_codes, &self.basis_table, *res_delta, false
                );
                output += &residual_contrib;
            },
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
                let residual_contrib = residual_2d.into_shape(residual_shape.as_slice()).unwrap().into_dyn();
                
                output += &residual_contrib;
            },
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
        assert_eq!(output_features, self.m, "그래디언트 출력 차원이 일치하지 않습니다.");
        
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
                if res_code == 0 { continue; }
                let (cat, sub, idx, sign, d, amp, amp_fine, phase) = decoder::decode(res_code);
                let r_val = (amp as f32 * 4.0 + amp_fine as f32) * scale;
                let signed_r_val = if sign == 0 { r_val } else { -r_val };
                let scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val, phase);
                let basis_vector = self.basis_table.row(idx as usize);
                for j in 0..self.n {
                    w_matrix[[i, j]] += basis_vector[j] * scale_factor;
                }
            }
        } else if let (Some(res_int8), Some(scales)) = (&self.residual_int8, &self.residual_scales) {
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
        let grad_input_nd = grad_input_2d.into_shape(input_shape.as_slice()).unwrap().into_dyn();
        
        grad_input_nd
    }
    
    // Getter 메서드들 (Python 바인딩용)
    pub fn get_m(&self) -> usize { self.m }
    pub fn get_n(&self) -> usize { self.n }
    pub fn get_delta(&self) -> f32 { self.delta }
    pub fn get_residual_delta(&self) -> Option<f32> { self.residual_delta }
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