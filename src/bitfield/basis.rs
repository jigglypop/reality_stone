// src/layers/bitfield/basis.rs

//! # 기저 벡터 테이블 관리
//!
//! 이 모듈은 비트필드 직접 추론 커널에서 사용되는 공유 기저 벡터 테이블 $\{b_j\}$를
//! 생성하고 관리하는 역할을 합니다. 가중치 벡터의 방향은 이 테이블의 인덱스 `idx`로 표현됩니다.

use ndarray::Array2;

/// `B x n` 크기의 기저 벡터 테이블을 생성합니다.
///
/// 실제 시나리오에서는 사전 학습되고 최적화된 테이블을 파일에서 로드해야 합니다.
/// 이 함수는 데모 목적으로, 각 행이 정규화된 n차원 단위 벡터인 테이블을 생성합니다.
///
/// # 인자
/// * `b` - 테이블에 포함된 기저 벡터의 수 (예: 256).
/// * `n` - 각 기저 벡터의 차원.
///
/// # 반환
/// `[b, n]` 형태의 `Array2<f32>`.
pub fn load_basis_table(b: usize, n: usize) -> Array2<f32> {
    let mut basis = Array2::<f32>::zeros((b, n));

    // 첫 n개는 표준 기저 벡터 (단위 벡터)
    for i in 0..b.min(n) {
        basis[[i, i]] = 1.0;
    }

    // 나머지는 다양한 패턴의 정규화된 벡터로 채움
    if b > n {
        use std::f32::consts::PI;

        for i in n..b {
            let pattern_idx = (i - n) % 4;
            match pattern_idx {
                0 => {
                    // 두 차원 조합 (균등)
                    let idx1 = i % n;
                    let idx2 = (i * 7 + 3) % n;
                    if idx1 != idx2 {
                        basis[[i, idx1]] = 0.707107; // 1/sqrt(2)
                        basis[[i, idx2]] = 0.707107;
                    } else {
                        basis[[i, idx1]] = 1.0;
                    }
                }
                1 => {
                    // 삼각 함수 패턴
                    for j in 0..n {
                        let phase = 2.0 * PI * (i as f32) * (j as f32) / (n as f32);
                        basis[[i, j]] = phase.cos() / (n as f32).sqrt();
                    }
                }
                2 => {
                    // 희소 패턴 (3개 차원만 활성)
                    let idx1 = (i * 5) % n;
                    let idx2 = (i * 11) % n;
                    let idx3 = (i * 17) % n;
                    basis[[i, idx1]] = 0.577; // 1/sqrt(3)
                    if idx2 != idx1 {
                        basis[[i, idx2]] = 0.577;
                    }
                    if idx3 != idx1 && idx3 != idx2 {
                        basis[[i, idx3]] = 0.577;
                    }
                }
                _ => {
                    // 랜덤 가우시안
                    let mut sum = 0.0;
                    for j in 0..n {
                        let val = ((i * 13 + j * 7) % 100) as f32 / 50.0 - 1.0;
                        basis[[i, j]] = val;
                        sum += val * val;
                    }
                    // 정규화
                    if sum > 1e-6 {
                        let norm = sum.sqrt();
                        for j in 0..n {
                            basis[[i, j]] /= norm;
                        }
                    }
                }
            }
        }
    }

    basis
}

/// 22비트 레이아웃에 대해 최적의 (cat, sub, d, amp, idx) 조합을 찾습니다.
pub fn find_best_code_22bit(
    weight_row: &ndarray::ArrayView1<f32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> (u8, u8, u8, u8, u8, f32) {
    let mut best_error = f32::INFINITY;
    let mut best_params = (0, 0, 0, 0, 0);

    let dots: Vec<f32> = (0..basis_table.nrows())
        .map(|i| weight_row.dot(&basis_table.row(i)))
        .collect();

    for idx in 0..basis_table.nrows() {
        let dot_product = dots[idx];

        for cat in 0..4 {
            for sub in 0..4 {
                for d in 0..4 {
                    // 최적의 r 값을 찾기 위한 간단한 탐색 (실제로는 더 정교한 최적화 필요)
                    // 여기서는 dot_product를 기반으로 r을 추정
                    let estimated_r = dot_product.abs(); // 매우 단순한 추정
                    let amp = (estimated_r / delta).min(255.0).max(0.0) as u8;
                    let r = amp as f32 * delta * dot_product.signum();

                    let scale = super::ops::lookup_and_apply(cat, sub, d, r, 0);
                    let reconstructed = &basis_table.row(idx) * scale;
                    let error = (&reconstructed - weight_row).mapv(|x| x.powi(2)).sum();

                    if error < best_error {
                        best_error = error;
                        best_params = (cat, sub, idx as u8, d, amp);
                    }
                }
            }
        }
    }
    (
        best_params.0,
        best_params.1,
        best_params.2,
        best_params.3,
        best_params.4,
        best_error,
    )
}

/// 32비트 레이아웃에 대해 최적의 조합을 찾습니다. (phase, amp_fine, sign 포함)
pub fn find_best_code_32bit(
    weight_row: &ndarray::ArrayView1<f32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> (u8, u8, u8, u8, u8, u8, u8, u8, f32) {
    let mut best_error = f32::INFINITY;
    let mut best_params = (0, 0, 0, 0, 0, 0, 0, 0);

    let dots: Vec<f32> = (0..basis_table.nrows())
        .map(|i| weight_row.dot(&basis_table.row(i)))
        .collect();

    for idx in 0..basis_table.nrows() {
        let dot_product = dots[idx];
        let sign = if dot_product < 0.0 { 1 } else { 0 };

        for cat in 0..4 {
            for sub in 0..4 {
                for d_1bit in 0..2 {
                    // d는 1비트
                    // 최적 r 추정
                    let r_unsigned = dot_product.abs();
                    let amp = (r_unsigned / delta).min(255.0) as u8;
                    let amp_fine = (((r_unsigned / delta) - amp as f32) * 4.0).min(3.0) as u8;

                    // TODO: 최적의 phase 찾기 (현재는 0으로 고정)
                    let phase = 0;

                    let r = (amp as f32 + (amp_fine as f32) / 4.0)
                        * delta
                        * if sign == 1 { -1.0 } else { 1.0 };

                    let scale = super::ops::lookup_and_apply(cat, sub, d_1bit, r, phase);
                    let reconstructed = &basis_table.row(idx) * scale;
                    let error = (&reconstructed - weight_row).mapv(|x| x.powi(2)).sum();

                    if error < best_error {
                        best_error = error;
                        best_params = (cat, sub, idx as u8, sign, d_1bit, amp, amp_fine, phase);
                    }
                }
            }
        }
    }
    (
        best_params.0,
        best_params.1,
        best_params.2,
        best_params.3,
        best_params.4,
        best_params.5,
        best_params.6,
        best_params.7,
        best_error,
    )
}
