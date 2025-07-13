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
    // 학습 가능한 스케일 파라미터 (정밀도 향상용)
    scale_factors: Option<Array1<f32>>,
    // 잔차 가중치는 정확도 향상보다 복잡성을 증가시켜 제거함
    // residual_weights: Option<Array2<f32>>,
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
            delta: r_max / 255.0,
            m,
            n,
            scale_factors: None,
            // residual_weights: None,
        }
    }

    /// 기존 가중치 행렬로부터 BitfieldLinear를 생성합니다.
    ///
    /// # 인자
    /// * `weights` - `[m, n]` 형태의 가중치 행렬
    /// * `b` - 기저 벡터 테이블의 크기 (256 권장)
    /// * `r_max` - 최대 반지름 값 (1.0 권장)
    ///
    /// # 반환
    /// 압축된 BitfieldLinear 인스턴스
    pub fn from_weights(weights: &Array2<f32>, b: usize, r_max: f32) -> Self {
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
        let delta = r_max / 255.0;
        let mut scale_factors = Array1::<f32>::ones(m);
        // let mut residual_weights = Array2::<f32>::zeros((m, n));
        let mut total_error = 0.0f32;
        
        for i in 0..m {
            let w_row = weights.row(i);
            
            // 가중치 행의 노름 계산
            let norm = w_row.dot(&w_row).sqrt();
            if norm < 1e-8 {
                // 영벡터는 코드 0으로
                codes[i] = 0;
                continue;
            }
            
            // 푸앵카레 볼 제약: ||w|| < 1
            let norm_clamped = norm.min(0.95); // 경계에서 약간 떨어뜨림
            
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
            
            // 푸앵카레 볼에서의 반지름 계산
            // w = tanh(r/2) · u 이므로, r = 2 * atanh(||w||)
            let r = 2.0 * norm_clamped.atanh();
            
            // 부호 비트 결정 (0: 양수, 1: 음수)
            let sign_bit = if best_sign > 0.0 { 0 } else { 1 };
            
            // cat과 sub, d를 결정하는 로직 단순화
            let (cat, sub, d) = if best_dot > 0.95 {
                (0, i as u8 % 4, 0)
            } else if best_dot > 0.9 {
                (1, i as u8 % 4, 0)
            } else if best_dot > 0.8 {
                (2, i as u8 % 4, 0)
            } else {
                (3, i as u8 % 4, 0)
            };

            let r_abs = r.abs().min(r_max);
            let amp = ((r_abs / delta) as u32).min(255);
            
            // 새로운 인코딩 함수 호출
            codes[i] = decoder::encode(cat, sub, best_idx as u8, sign_bit, d, amp as u8);
            
            // 재구성 오차 계산
            let signed_r = if sign_bit == 0 { r_abs } else { -r_abs };
            let reconstructed_scale = ops::lookup_and_apply(cat, sub, d, signed_r);
            let reconstructed = &basis_table.row(best_idx) * reconstructed_scale;
            let error = &w_row - &reconstructed;
            
            // 적응적 스케일 팩터 (재구성 품질에 따라)
            let error_norm = error.dot(&error).sqrt();
            total_error += error_norm;
            scale_factors[i] = 1.0 - error_norm.min(0.5); // 오차가 클수록 스케일 감소 (보정)
        }
        
        Self {
            codes,
            basis_table,
            delta,
            m,
            n,
            scale_factors: Some(scale_factors),
            // residual_weights: None,
        }
    }

    /// 순전파 `y = xW^T`를 수행합니다.
    ///
    /// # 인자
    /// * `x` - `[batch_size, n]` 형태의 입력 텐서.
    ///
    /// # 반환
    /// `[batch_size, m]` 형태의 출력 텐서.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.shape()[1], self.n, "입력 차원이 일치하지 않습니다.");
        
        // 기본 비트필드 추론
        let mut output = kernel::gemm_hyper_bit_cpu(x, &self.codes, &self.basis_table, self.delta);
        
        // 스케일 팩터 적용
        if let Some(ref scale_factors) = self.scale_factors {
            for i in 0..self.m {
                let scale = scale_factors[i];
                output.column_mut(i).mapv_inplace(|v| v * scale);
            }
        }
        
        // 잔차 로직 제거됨
        
        output
    }

    /// 역전파 `grad_input = grad_output @ W`를 수행합니다.
    ///
    /// # 인자
    /// * `grad_output` - `[batch_size, m]` 형태의 출력 그래디언트 텐서.
    ///
    /// # 반환
    /// `[batch_size, n]` 형태의 입력 그래디언트 텐서.
    pub fn backward(&self, grad_output: &Array2<f32>) -> Array2<f32> {
        // 1. 압축된 코드로부터 완전한 가중치 행렬 W (`[m, n]`)를 복원합니다.
        let mut w_matrix = Array2::<f32>::zeros((self.m, self.n));
        for i in 0..self.m {
            let code = self.codes[i];
            let (cat, sub, idx, sign, d, amp) = decoder::decode(code);

            let r_val = (amp as f32) * self.delta;
            let signed_r_val = if sign == 0 { r_val } else { -r_val };
            
            let mut scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val);
            
            // 스케일 팩터 적용
            if let Some(ref scale_factors) = self.scale_factors {
                scale_factor *= scale_factors[i];
            }

            let basis_vector = self.basis_table.row(idx as usize);
            let weight_row = &basis_vector * scale_factor;
            
            w_matrix.row_mut(i).assign(&weight_row);
        }
        
        // 잔차 로직 제거됨

        // 2. grad_input = grad_output @ W 를 계산합니다.
        grad_output.dot(&w_matrix)
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
        
        // 최적화된 비트필드 추론 커널 사용
        let mut output = kernel::gemm_hyper_bit_cpu_nd(x, &self.codes, &self.basis_table, self.delta);
        
        // 스케일 팩터 적용
        if let Some(ref scale_factors) = self.scale_factors {
            let output_shape = output.shape();
            let output_ndim = output_shape.len();
            let total_batch_size: usize = output_shape[..output_ndim - 1].iter().product();
            
            // 출력을 2D로 reshape하여 스케일 적용
            let mut output_2d = output.to_shape((total_batch_size, self.m)).unwrap().to_owned();
            for i in 0..self.m {
                let scale = scale_factors[i];
                output_2d.column_mut(i).mapv_inplace(|v| v * scale);
            }
            
            // 다시 원래 형태로 reshape
            output = output_2d.into_shape(output_shape).unwrap().into_dyn();
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
        
        // 1. 압축된 코드로부터 완전한 가중치 행렬 W (`[m, n]`)를 복원합니다.
        let mut w_matrix = Array2::<f32>::zeros((self.m, self.n));
        for i in 0..self.m {
            let code = self.codes[i];
            let (cat, sub, idx, sign, d, amp) = decoder::decode(code);

            let r_val = (amp as f32) * self.delta;
            let signed_r_val = if sign == 0 { r_val } else { -r_val };
            
            let mut scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val);
            
            // 스케일 팩터 적용
            if let Some(ref scale_factors) = self.scale_factors {
                scale_factor *= scale_factors[i];
            }

            let basis_vector = self.basis_table.row(idx as usize);
            let weight_row = &basis_vector * scale_factor;
            
            w_matrix.row_mut(i).assign(&weight_row);
        }
        
        // 잔차 로직 제거됨

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
} 