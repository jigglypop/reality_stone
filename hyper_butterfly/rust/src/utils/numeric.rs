//! 수치 계산 유틸리티 함수들

use crate::config::constants::Constants;
use ndarray::{Array1, Array2, Axis};

/// 수치 안정적인 atanh 함수 구현
pub fn atanh(x: f32) -> f32 {
    // 경계값 클램핑
    let x = x.clamp(
        -1.0 + Constants::BOUNDARY_EPS,
        1.0 - Constants::BOUNDARY_EPS,
    );
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

/// 수치 안정적인 tanh 함수 구현 (오버플로우 방지)
pub fn safe_tanh(x: f32) -> f32 {
    let x = x.clamp(-Constants::MAX_TANH_ARG, Constants::MAX_TANH_ARG);
    x.tanh()
}

/// L2 노름 계산 (차원 1에 대해)
pub fn l2_norm(x: &Array2<f32>) -> Array1<f32> {
    x.map_axis(Axis(1), |row| row.dot(&row).sqrt())
}

/// 배치 내적 계산
pub fn batch_dot(a: &Array2<f32>, b: &Array2<f32>) -> Array1<f32> {
    // 두 2D 텐서의 각 행 간 내적 계산
    assert_eq!(a.shape()[0], b.shape()[0]);
    assert_eq!(a.shape()[1], b.shape()[1]);

    a.outer_iter()
        .zip(b.outer_iter())
        .map(|(a_row, b_row)| a_row.dot(&b_row))
        .collect()
}

/// 다음 2의 거듭제곱 값 찾기
pub fn next_pow2(v: usize) -> usize {
    let mut v = v - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}
