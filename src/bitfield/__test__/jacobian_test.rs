//! 비트 야코비안 테스트

use crate::bitfield::jacobian::{BitJacobian, HyperbolicBitJacobian};
use approx::assert_relative_eq;
use ndarray::Array1;

#[test]
fn test_비트_야코비안_생성() {
    let jacobian = BitJacobian::new();
    assert_eq!(jacobian.scale_factor, 1.0 / 127.0);
}

#[test]
fn test_순환_도함수_테이블() {
    let jacobian = BitJacobian::new();

    // sin(x)의 1차 도함수는 cos(x) → 정규화된 값 127
    assert_eq!(jacobian.cyclic_derivatives[0][1], 127);

    // cos(x)의 1차 도함수는 -sin(x) → 정규화된 값 0
    assert_eq!(jacobian.cyclic_derivatives[1][1], 0);

    // -sin(x)의 1차 도함수는 -cos(x) → 정규화된 값 -127
    assert_eq!(jacobian.cyclic_derivatives[2][1], -127);

    // -cos(x)의 1차 도함수는 sin(x) → 정규화된 값 0
    assert_eq!(jacobian.cyclic_derivatives[3][1], 0);
}

#[test]
fn test_비트_마스크_대각_행렬_변환() {
    let mask = 0b1010; // 비트 1과 3이 켜짐
    let diag = BitJacobian::bit_mask_to_diagonal(mask, 4);

    assert_eq!(diag[0], 0.0);
    assert_eq!(diag[1], 1.0);
    assert_eq!(diag[2], 0.0);
    assert_eq!(diag[3], 1.0);
}

#[test]
fn test_대각_원소_계산() {
    let jacobian = BitJacobian::new();

    // 테스트 코드: sin 함수, 위상 0, 진폭 0.5
    let code = 0x00000080; // amp = 128 (0.5)
    let x = 0.0;
    let diff_order = 1; // 1차 도함수

    let result = jacobian.compute_diagonal_element(code, x, diff_order);

    // sin'(0) = cos(0) = 1.0, 진폭 0.5 적용
    assert_relative_eq!(result, 0.5, epsilon = 0.1);
}

#[test]
fn test_위상_변조() {
    let jacobian = BitJacobian::new();

    // 위상이 π/2인 sin 함수
    let phase_90 = ((std::f32::consts::PI / 2.0) / (2.0 * std::f32::consts::PI) * 255.0) as u8;
    let code = (phase_90 as u32) << 24 | 0x000000FF; // 위상 π/2, 진폭 1.0

    let x = 0.0;
    let diff_order = 0;

    let result = jacobian.compute_diagonal_element(code, x, diff_order);

    // sin(0 + π/2) = cos(0) = 1.0
    assert_relative_eq!(result, 1.0, epsilon = 0.1);
}

#[test]
fn test_쌍곡_야코비안() {
    let hyp_jacobian = HyperbolicBitJacobian::new();

    // sinh(0) = 0
    let result = hyp_jacobian.compute_hyperbolic_jacobian(0, 0.0, 0);
    assert_relative_eq!(result, 0.0, epsilon = 0.001);

    // sinh'(0) = cosh(0) = 1.0
    let result = hyp_jacobian.compute_hyperbolic_jacobian(0, 0.0, 1);
    assert_relative_eq!(result, 1.0, epsilon = 0.01);

    // cosh(0) = 1.0
    let result = hyp_jacobian.compute_hyperbolic_jacobian(1, 0.0, 0);
    assert_relative_eq!(result, 1.0, epsilon = 0.01);

    // cosh'(0) = sinh(0) = 0.0
    let result = hyp_jacobian.compute_hyperbolic_jacobian(1, 0.0, 1);
    assert_relative_eq!(result, 0.0, epsilon = 0.001);
}

#[test]
fn test_전체_야코비안_계산() {
    let jacobian = BitJacobian::new();

    // 4개의 테스트 코드
    let codes = Array1::from(vec![
        0x00000080, // sin, amp=0.5
        0x00040080, // cos, amp=0.5
        0x00080080, // -sin, amp=0.5
        0x000C0080, // -cos, amp=0.5
    ]);

    let x = Array1::zeros(4);
    let jac = jacobian.compute_jacobian(&codes, &x.view(), 0);

    assert_eq!(jac.len(), 4);

    // 각 함수의 x=0에서의 값 확인
    assert_relative_eq!(jac[0], 0.0, epsilon = 0.01); // sin(0) = 0
    assert_relative_eq!(jac[1], 0.5, epsilon = 0.01); // cos(0) = 1 * 0.5
    assert_relative_eq!(jac[2], 0.0, epsilon = 0.01); // -sin(0) = 0
    assert_relative_eq!(jac[3], -0.5, epsilon = 0.01); // -cos(0) = -1 * 0.5
}

#[test]
fn test_야코비안_전치_적용() {
    let jacobian = BitJacobian::new();

    let codes = Array1::from(vec![0x00000080, 0x00040080]);
    let x = Array1::zeros(2);
    let grad_output = Array1::ones(2);

    let grad_input = jacobian.apply_jacobian_transpose(&grad_output.view(), &codes, &x.view());

    assert_eq!(grad_input.len(), 2);

    // 대각 야코비안이므로 grad_input = diag(J) * grad_output
    assert_relative_eq!(grad_input[0], 0.0, epsilon = 0.01); // sin(0) * 1.0
    assert_relative_eq!(grad_input[1], 0.5, epsilon = 0.01); // cos(0) * 0.5 * 1.0
}

#[test]
fn test_미분_순환성() {
    let jacobian = BitJacobian::new();

    // sin 함수 코드
    let code = 0x000000FF; // sin, amp=1.0
    let x = std::f32::consts::PI / 4.0; // π/4

    // 0차 도함수: sin(π/4) ≈ 0.707
    let d0 = jacobian.compute_diagonal_element(code, x, 0);
    assert_relative_eq!(d0, 0.707, epsilon = 0.01);

    // 1차 도함수: cos(π/4) ≈ 0.707
    let d1 = jacobian.compute_diagonal_element(code, x, 1);
    assert_relative_eq!(d1, 0.707, epsilon = 0.01);

    // 2차 도함수: -sin(π/4) ≈ -0.707
    let d2 = jacobian.compute_diagonal_element(code, x, 2);
    assert_relative_eq!(d2, -0.707, epsilon = 0.01);

    // 3차 도함수: -cos(π/4) ≈ -0.707
    let d3 = jacobian.compute_diagonal_element(code, x, 3);
    assert_relative_eq!(d3, -0.707, epsilon = 0.01);

    // 4차 도함수 = 0차 도함수 (순환)
    let d4 = jacobian.compute_diagonal_element(code, x, 4);
    assert_relative_eq!(d4, d0, epsilon = 0.001);
}

#[test]
fn test_쌍곡_함수_순환성() {
    let hyp_jacobian = HyperbolicBitJacobian::new();

    let x: f32 = 0.5; // 타입 명시

    // sinh와 그 도함수들
    let sinh0 = hyp_jacobian.compute_hyperbolic_jacobian(0, x, 0); // sinh(x)
    let sinh1 = hyp_jacobian.compute_hyperbolic_jacobian(0, x, 1); // cosh(x)
    let sinh2 = hyp_jacobian.compute_hyperbolic_jacobian(0, x, 2); // sinh(x) (2주기)

    assert_relative_eq!(sinh0, x.sinh(), epsilon = 0.01);
    assert_relative_eq!(sinh1, x.cosh(), epsilon = 0.01);
    assert_relative_eq!(sinh2, sinh0, epsilon = 0.001); // 2주기 순환
}

#[test]
fn test_진폭_미세조정() {
    let jacobian = BitJacobian::new();

    // amp=255 (1.0), amp_fine=3 (0.75) → 총 진폭 ≈ 1.1875
    let code = 0x00C000FF; // amp_fine=3, amp=255
    let x = 0.0;
    let diff_order = 1;

    let result = jacobian.compute_diagonal_element(code, x, diff_order);

    // cos(0) * 1.1875 ≈ 1.1875
    assert_relative_eq!(result, 1.1875, epsilon = 0.1);
}

#[test]
fn test_비트필드_디코딩_디버그() {
    let jacobian = BitJacobian::new();

    // 테스트 코드: SUB=0 (sin), amp=255 (1.0)
    let code = 0x000000FF; // amp = 255 (1.0)

    // 비트필드 분석
    let sub = ((code >> 18) & 0x3) as u8;
    let d = ((code >> 8) & 0x1) as u8;
    let amp = (code & 0xFF) as u8;

    println!("코드: 0x{:08X}", code);
    println!("SUB 필드: {}", sub);
    println!("D 필드: {}", d);
    println!("AMP 필드: {}", amp);

    // 함수값 계산
    let x = 0.0;
    let result = jacobian.compute_diagonal_element(code, x, 0);
    println!("sin(0) 계산 결과: {}", result);

    // 1차 도함수
    let result_d1 = jacobian.compute_diagonal_element(code, x, 1);
    println!("sin'(0) = cos(0) 계산 결과: {}", result_d1);

    // 예상값: sin(0) = 0, cos(0) = 1
    assert_relative_eq!(result, 0.0, epsilon = 0.01);
    assert_relative_eq!(result_d1, 1.0, epsilon = 0.01);
}
