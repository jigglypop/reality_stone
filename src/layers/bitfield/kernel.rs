// src/layers/bitfield/kernel.rs

//! # 비트필드 인코딩 가중치를 위한 직접 추론 커널
//!
//! 이 모듈은 가중치 행렬을 완전히 압축 해제하지 않고,
//! 압축된 비트필드 표현에서 직접 행렬-벡터 곱셈을 수행하는 핵심 로직을 포함합니다.

use ndarray::{Array1, Array2, Axis, ArrayView, IxDyn, ArrayBase, Data, Ix2};
use rayon::prelude::*;

use crate::layers::bitfield::{decoder, ops};

/// W가 압축된 비트필드 코드로 표현될 때 `y = xW^T`를 수행합니다.
///
/// 이것은 `ndarray`와 병렬 처리를 위한 `rayon`을 사용하는 CPU 기반 참조 구현입니다.
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

    output.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(j, mut col)| {
            let code = codes[j];
            let (cat, sub, idx, sign, d, amp) = decoder::decode(code);

            // a. 코드로부터 스케일링 인자 `s_i`를 계산합니다.
            let r_val = (amp as f32) * delta;
            let signed_r_val = if sign == 0 { r_val } else { -r_val };
            let scale_factor = ops::lookup_and_apply(cat, sub, d, signed_r_val);

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
    
    // 출력을 원래 형태로 reshape: [total_batch_size, output_features] → [..., output_features]
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
    
    output_2d.into_shape(output_shape.as_slice()).unwrap().into_dyn()
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