// src/layers/bitfield/ops.rs

//! # 하이퍼볼릭 및 주기적 연산 LUT
//!
//! 이 모듈은 디코딩된 비트필드 코드(`cat`, `sub`, `d`)에 기반하여
//! 올바른 수학적 함수를 적용하기 위한 조회 테이블(LUT) 역할을 합니다.
//! 표준 부동소수점 숫자에 대해 연산을 수행합니다.

/// 제공된 코드에 기반하여 올바른 스케일링 함수를 조회하고 적용합니다.
///
/// # 인자
/// * `cat` - 기하학 카테고리 코드.
/// * `sub` - 기저 함수 족 코드.
/// * `d` - 도함수 차수 코드.
/// * `r` - 반지름을 나타내는 스칼라 `f32` 값.
///
/// # 반환
/// 계산된 스케일링 인자를 담은 스칼라 `f32` 값.
pub fn lookup_and_apply(cat: u8, sub: u8, d: u8, r: f32) -> f32 {
    // 기본적으로 cat=0, sub=0인 경우 단순히 tanh(r/2)를 사용
    match (cat, sub) {
        (0, 0) => {
            // 푸앵카레 볼 기하학의 기본 스케일링
            (r * 0.5).tanh()
        },
        (0, 1) => {
            // sinh/cosh 족
            let base = (r * 0.5).tanh();
            let periodic = if d % 2 == 0 { r.sinh() } else { r.cosh() };
            base * periodic
        },
        (0, 2) => {
            // sin/cos 족
            let base = (r * 0.5).tanh();
            let periodic = match d % 4 {
                0 => r.sin(),
                1 => r.cos(),
                2 => -r.sin(),
                _ => -r.cos(),
            };
            base * periodic
        },
        // TODO: 다른 기하학(로렌츠 등)에 대한 케이스 구현
        _ => (r * 0.5).tanh(), // 기본값으로 푸앵카레 스케일링 사용
    }
} 