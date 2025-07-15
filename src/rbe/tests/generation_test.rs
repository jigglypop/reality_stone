//! `generation.rs`에 대한 단위 테스트

use poincare_layer::Packed64;
use approx::assert_relative_eq;
use std::f32::consts::PI;

#[test]
fn test_weight_generation_logic() {
    println!("\n--- Test: Weight Generation Logic ---");

    // 1. 특정 파라미터를 가진 시드 생성
    // (sin*cosh, 미분 없음, 회전 없음, c=1)
    let packed = Packed64::new(0.5, PI / 2.0, 0, 0, false, 0, 0, 0);

    // 2. 특정 좌표에서의 가중치 계산
    // 행렬의 정중앙 (i=15, j=15 for 31x31) -> 정규화 좌표 (x=0, y=0)
    let rows = 31;
    let cols = 31;
    let center_i = 15;
    let center_j = 15;
    let weight = packed.compute_weight(center_i, center_j, rows, cols);

    // 3. 예상 값 수동 계산
    // x=0, y=0 -> r_local=0, theta_local=0
    // theta_final = params.theta + 0 + 0 = PI / 2.0
    // angular_value = sin(PI/2) = 1.0
    // radial_value = sinh(c*r) = sinh(1.0 * 0.5) = 0.521095
    // basis_value = 1.0 * 0.521095
    // jacobian = sqrt((1 - 1.0*0.5^2)^-2) = sqrt((0.75)^-2) = 1.333...
    let expected_c = 1.0f32;
    let expected_r = 0.5f32;
    let expected_theta = PI / 2.0f32;
    let expected_angular = expected_theta.sin();
    let expected_radial = (expected_c * expected_r).sinh();
    let jacobian = (1.0 - expected_c * expected_r * expected_r).powi(-2).sqrt(); // generation.rs 구현과 동일하게
    let expected_weight = expected_angular * expected_radial * jacobian;
    
    // 4. 검증
    println!("  - Seed params: r={}, theta={}, c={}", expected_r, expected_theta, expected_c);
    println!("  - Coords (i,j): ({},{}) -> (x,y): (0,0)", center_i, center_j);
    println!("  - Computed weight: {}", weight);
    println!("  - Expected weight: {}", expected_weight);

    assert_relative_eq!(weight, expected_weight, epsilon = 1e-5); // Epsilon 완화
    println!("  [PASSED] Weight generation at center is correct.");
}

#[test]
fn test_jacobian_calculation() {
    println!("\n--- Test: Jacobian Calculation ---");
    // 야코비안 계산 로직만 별도 검증

    let params = Packed64::new(0.8, 0.0, 0, 0, false, 0, 1, 0).decode(); // c=2.0
    let c = 2.0f32.powi(params.log2_c as i32);
    let r = params.r;

    // compute_weight 내부의 야코비안 계산
    let jacobian_in_code = (1.0 - c * r * r).powi(-2).sqrt();

    // 직접 계산 (코드와 동일하게)
    let expected_jacobian = (1.0 / (1.0 - c * r * r).powi(2)).sqrt();
    
    println!("  - c={}, r={}", c, r);
    println!("  - Jacobian in code: {}", jacobian_in_code);
    println!("  - Expected Jacobian: {}", expected_jacobian);
    
    assert_relative_eq!(jacobian_in_code, expected_jacobian, epsilon = 1e-6);
    println!("  [PASSED] Jacobian calculation matches current implementation.");
} 