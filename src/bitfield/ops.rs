//! # 리만 기하학 함수 적용 인터페이스
//!
//! 이 모듈은 디코딩된 비트필드 코드(`cat`, `sub`, `d`)에 기반하여
//! 올바른 리만 기하학 함수를 적용하기 위한 인터페이스를 제공합니다.

use super::riemannian::get_riemannian_function;

/// 제공된 코드에 기반하여 올바른 스케일링 함수를 조회하고 적용합니다.
///
/// # 인자
/// * `cat` - 기하학 카테고리 코드 (0: Poincaré, 1: Lorentz, 2: Klein, 3: Special)
/// * `sub` - 기저 함수 족 코드 (0: 기본, 1: 쌍곡, 2: 삼각, 3: 지수/로그)
/// * `d` - 도함수 차수 또는 변형 코드
/// * `r` - 반지름을 나타내는 스칼라 `f32` 값.
/// * `phase` - 위상 값 (0-255를 0-2π로 매핑).
///
/// # 반환
/// 계산된 스케일링 인자를 담은 스칼라 `f32` 값.
pub fn lookup_and_apply(cat: u8, sub: u8, d: u8, r: f32, phase: u8) -> f32 {
    // 기본적으로 cat, sub, d에 따른 리만 함수를 사용
    let base_value = get_riemannian_function(cat, sub, d, r);

    // 위상 정보가 있는 경우 추가 변조
    if phase != 0 {
        let phase_rad = (phase as f32 / 255.0) * 2.0 * std::f32::consts::PI;

        // 특정 함수족에서만 위상 변조 적용
        match (cat, sub) {
            // Poincaré 삼각함수족은 위상 변조
            (0, 2) => match d {
                0 | 1 => base_value * phase_rad.cos(),
                2 | 3 => base_value * phase_rad.sin(),
                _ => base_value,
            },
            // Lorentz 혼합 함수는 위상 추가
            (1, 2) => base_value * (phase_rad.sin() + 1.0) * 0.5,
            // 특수 함수 주기적 변조
            (3, 2) | (3, 3) => base_value * (phase_rad * 2.0).cos(),
            // 기타는 위상 무시
            _ => base_value,
        }
    } else {
        base_value
    }
}

/// 디버깅을 위한 함수 이름 반환
pub fn get_function_name_with_phase(cat: u8, sub: u8, d: u8, phase: u8) -> String {
    use super::riemannian::get_function_name;

    let base_name = get_function_name(cat, sub, d);

    if phase != 0 {
        let phase_rad = (phase as f32 / 255.0) * 2.0 * std::f32::consts::PI;
        format!("{} × cos({:.2})", base_name, phase_rad)
    } else {
        base_name.to_string()
    }
}
