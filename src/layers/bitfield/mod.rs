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

use ndarray::{Array1, Array2, Axis};

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
    // 잔차 연결을 위한 작은 가중치 행렬
    residual_weights: Option<Array2<f32>>,
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
            residual_weights: None,
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
        let mut residual_weights = Array2::<f32>::zeros((m, n));
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
            let r_adjusted = r * best_sign; // 부호 정보 포함
            
            // cat과 sub를 활용하여 더 많은 정보 인코딩
            let (cat, sub, d) = if r_adjusted >= 0.0 {
                // 양의 스케일
                if best_dot > 0.98 {
                    // 매우 정확한 매칭 - 기본 함수 사용
                    (0, 0, 0) // Poincaré tanh(r/2)
                } else if best_dot > 0.95 {
                    // 좋은 매칭 - 쌍곡 함수
                    (0, 1, (i % 4) as u8) // sinh, cosh 변형들
                } else if best_dot > 0.9 {
                    // 보통 매칭 - 삼각 함수
                    (0, 2, (i % 4) as u8) // sin, cos 변형들
                } else {
                    // 낮은 매칭 - 특수 함수
                    let cat_choice = (i / 4) % 4;
                    (cat_choice as u8, (i % 4) as u8, (i / 16) as u8 % 4)
                }
            } else {
                // 음의 스케일
                if best_dot > 0.98 {
                    (0, 0, 1) // -tanh(r/2)
                } else if best_dot > 0.95 {
                    (1, 0, (i % 4) as u8) // Lorentz 함수들
                } else if best_dot > 0.9 {
                    (1, 1, (i % 4) as u8) // 수정된 쌍곡 함수
                } else {
                    // Klein 또는 특수 함수
                    let choice = i % 3;
                    match choice {
                        0 => (2, 0, (i % 4) as u8), // Klein 기본
                        1 => (2, 1, (i % 4) as u8), // Klein 투영
                        _ => (3, (i % 4) as u8, (i / 4) as u8 % 4), // 특수 함수
                    }
                }
            };
            
            let r_abs = r_adjusted.abs().min(r_max);
            let amp = ((r_abs / delta) as u32).min(255);
            
            // 더 정교한 인코딩
            codes[i] = decoder::encode(cat, sub, best_idx as u8, d, amp as u8);
            
            // 재구성 오차 계산 및 잔차 저장
            let reconstructed_r = (amp as f32) * delta;
            let reconstructed_scale = if sub == 0 {
                (reconstructed_r * 0.5).tanh()
            } else {
                -(reconstructed_r * 0.5).tanh()
            };
            let reconstructed = &basis_table.row(best_idx) * reconstructed_scale;
            let error = &w_row - &reconstructed;
            residual_weights.row_mut(i).assign(&error);
            
            // 적응적 스케일 팩터 (재구성 품질에 따라)
            let error_norm = error.dot(&error).sqrt();
            total_error += error_norm;
            scale_factors[i] = 1.0 + error_norm.min(0.1); // 오차가 클수록 스케일 증가
        }
        
        // 평균 오차가 너무 크면 잔차 가중치 사용
        let avg_error = total_error / m as f32;
        let use_residual = avg_error > 0.05;
        
        Self {
            codes,
            basis_table,
            delta,
            m,
            n,
            scale_factors: Some(scale_factors),
            residual_weights: if use_residual { Some(residual_weights) } else { None },
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
        
        // 잔차 추가
        if let Some(ref residual) = self.residual_weights {
            let residual_output = x.dot(&residual.t());
            output = output + residual_output * 0.1; // 작은 가중치로 추가
        }
        
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
            let (cat, sub, idx, d, amp) = decoder::decode(code);

            let r_val = (amp as f32) * self.delta;
            let mut scale_factor = ops::lookup_and_apply(cat, sub, d, r_val);
            
            // 스케일 팩터 적용
            if let Some(ref scale_factors) = self.scale_factors {
                scale_factor *= scale_factors[i];
            }

            let basis_vector = self.basis_table.row(idx as usize);
            let weight_row = &basis_vector * scale_factor;
            
            w_matrix.row_mut(i).assign(&weight_row);
        }
        
        // 잔차 가중치 추가
        if let Some(ref residual) = self.residual_weights {
            w_matrix = w_matrix + residual * 0.1;
        }

        // 2. grad_input = grad_output @ W 를 계산합니다.
        grad_output.dot(&w_matrix)
    }
} 