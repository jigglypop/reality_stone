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
                        basis[[i, idx1]] = 0.707107;  // 1/sqrt(2)
                        basis[[i, idx2]] = 0.707107;
                    } else {
                        basis[[i, idx1]] = 1.0;
                    }
                },
                1 => {
                    // 삼각 함수 패턴
                    for j in 0..n {
                        let phase = 2.0 * PI * (i as f32) * (j as f32) / (n as f32);
                        basis[[i, j]] = phase.cos() / (n as f32).sqrt();
                    }
                },
                2 => {
                    // 희소 패턴 (3개 차원만 활성)
                    let idx1 = (i * 5) % n;
                    let idx2 = (i * 11) % n;
                    let idx3 = (i * 17) % n;
                    basis[[i, idx1]] = 0.577;  // 1/sqrt(3)
                    if idx2 != idx1 {
                        basis[[i, idx2]] = 0.577;
                    }
                    if idx3 != idx1 && idx3 != idx2 {
                        basis[[i, idx3]] = 0.577;
                    }
                },
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