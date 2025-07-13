// src/layers/bitfield/kernel.rs

//! # 비트필드 인코딩 가중치를 위한 직접 추론 커널
//!
//! 이 모듈은 가중치 행렬을 완전히 압축 해제하지 않고,
//! 압축된 비트필드 표현에서 직접 행렬-벡터 곱셈을 수행하는 핵심 로직을 포함합니다.

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

use crate::layers::bitfield::{decoder, ops};

/// W가 압축된 비트필드 코드로 표현될 때 `y = xW^T`를 수행합니다.
///
/// 이것은 `ndarray`와 병렬 처리를 위한 `rayon`을 사용하는 CPU 기반 참조 구현입니다.
///
/// # 인자
/// * `x` - 입력 벡터/행렬, `[batch_size, n]` 형태의 `Array2<f32>`.
/// * `codes` - `u32` 비트필드 코드의 1차원 배열, `[m]` 형태.
/// * `basis_table` - 공유 기저 벡터 테이블, `[B, n]` 형태의 `Array2<f32>`.
/// * `delta` - `amp` 필드를 위한 양자화 스텝 사이즈 (`r_max / 255.0`).
///
/// # 반환
/// 곱셈 결과인 `[batch_size, m]` 형태의 `Array2<f32>`.
pub fn gemm_hyper_bit_cpu(
    x: &Array2<f32>,
    codes: &Array1<u32>,
    basis_table: &Array2<f32>,
    delta: f32,
) -> Array2<f32> {
    // 1. 전처리: 입력과 모든 기저 벡터 간의 내적을 계산합니다.
    // `x` (batch, n) @ `basis_table.T` (n, B) -> `dotb` (batch, B)
    let dotb = x.dot(&basis_table.t());

    // 2. 메인 루프: `m`개의 코드를 순회하며 각 출력 열을 계산합니다.
    // 출력 행렬을 생성하고 rayon을 사용하여 병렬로 열을 채웁니다.
    let m = codes.len();
    let batch_size = x.shape()[0];
    let mut output = Array2::<f32>::zeros((batch_size, m));

    output.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut col)| {
            let code = codes[j];
            let (cat, sub, idx, d, amp) = decoder::decode(code);

            // a. 코드로부터 스케일링 인자 `s_i`를 계산합니다.
            let r_val = (amp as f32) * delta;
            let scale_factor = ops::lookup_and_apply(cat, sub, d, r_val);

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