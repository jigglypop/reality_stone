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
        
        // 1. 기저 벡터 테이블 생성
        let basis_table = basis::load_basis_table(b, n);

        // 2. 각 가중치 행을 압축
        let mut codes = Array1::<u32>::zeros(m);
        let delta = r_max / 255.0;
        
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
            let norm_clamped = norm.min(0.999);
            
            // 방향 벡터
            let direction = &w_row / norm;
            
            // 가장 가까운 기저 벡터 찾기
            let mut best_idx = 0;
            let mut best_dot = -1.0f32;
            for j in 0..b {
                let dot = direction.dot(&basis_table.row(j));
                // 절대값이 아닌 원래 내적값 사용 (부호 보존)
                if dot.abs() > best_dot {
                    best_dot = dot.abs();
                    best_idx = j;
                }
            }
            
            // 푸앵카레 볼에서의 반지름 계산
            // w = tanh(r/2) · u 이므로, r = 2 * atanh(||w||)
            let r = 2.0 * norm_clamped.atanh();
            let r_clamped = r.min(r_max);
            let amp = ((r_clamped / delta) as u32).min(255);
            
            // 간단한 인코딩 (cat=0, sub=0, d=0)
            codes[i] = decoder::encode(0, 0, best_idx as u8, 0, amp as u8);
        }
        
        Self {
            codes,
            basis_table,
            delta,
            m,
            n,
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
        kernel::gemm_hyper_bit_cpu(x, &self.codes, &self.basis_table, self.delta)
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
            let scale_factor = ops::lookup_and_apply(cat, sub, d, r_val);

            let basis_vector = self.basis_table.row(idx as usize);
            let weight_row = &basis_vector * scale_factor;
            
            w_matrix.row_mut(i).assign(&weight_row);
        }

        // 2. grad_input = grad_output @ W 를 계산합니다.
        grad_output.dot(&w_matrix)
    }
} 