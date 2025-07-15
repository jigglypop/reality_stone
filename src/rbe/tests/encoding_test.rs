//! `encoding.rs`에 대한 단위 테스트

use poincare_layer::Packed64;
use std::f32::consts::PI;

#[test]
fn test_encoding_bit_packing() {
    println!("\n--- Test: Encoding Bit Packing ---");

    // 1. 테스트할 파라미터 정의
    let r = 0.5;
    let theta = PI;
    let basis_id = 0b1010; // 10
    let d_theta = 0b11;    // 3
    let d_r = true;
    let rot_code = 0b1101; // 13
    let log2_c = -3;       // 2의 보수: 0b101
    let reserved = 0b101010; // 42

    // 2. 인코딩 실행
    let packed = Packed64::new(r, theta, basis_id, d_theta, d_r, rot_code, log2_c, reserved);

    // 3. 예상 비트 값 수동 계산
    let r_bits = (r * ((1u64 << 20) - 1) as f32).round() as u64; // 0.5 -> 0x7FFFF
    let theta_bits = (theta / (2.0 * PI) * ((1u64 << 24) - 1) as f32).round() as u64; // PI -> 0.5 -> 0x7FFFFF
    
    let mut expected_packed = 0u64;
    expected_packed |= (r_bits & 0xFFFFF) << 44;
    expected_packed |= (theta_bits & 0xFFFFFF) << 20;
    expected_packed |= (basis_id as u64 & 0xF) << 16;
    expected_packed |= (d_theta as u64 & 0x3) << 14;
    expected_packed |= (d_r as u64 & 0x1) << 13;
    expected_packed |= (rot_code as u64 & 0xF) << 9;
    expected_packed |= ((log2_c as u64) & 0x7) << 6;
    expected_packed |= (reserved as u64) & 0x3F;

    // 4. 검증
    println!("  -      Packed: 0x{:016X}", packed.0);
    println!("  -    Expected: 0x{:016X}", expected_packed);
    assert_eq!(packed.0, expected_packed, "The bit-packed u64 value is incorrect.");
    println!("  [PASSED] Bit packing is correct.");
}

#[test]
fn test_encoding_clamping_and_normalization() {
    println!("\n--- Test: Encoding Clamping and Normalization ---");

    // 경계값 테스트
    let r_over = 1.5;
    let theta_over = 3.0 * PI;

    let packed = Packed64::new(r_over, theta_over, 0, 0, false, 0, 0, 0);
    let decoded = packed.decode();

    // r은 1.0 미만으로 클램핑되어야 함
    assert!(decoded.r < 1.0, "r value should be clamped to be less than 1.0");
    println!("  [PASSED] r value is clamped correctly.");

    // theta는 [0, 2π) 범위로 정규화되어야 함 (3π -> π)
    assert!(
        (decoded.theta - PI).abs() < 1e-6,
        "theta value should be normalized to the [0, 2π) range"
    );
    println!("  [PASSED] theta value is normalized correctly.");
} 