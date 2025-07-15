//! `decoding.rs`에 대한 단위 테스트

use poincare_layer::Packed64;
use approx::assert_relative_eq;
use std::f32::consts::PI;

#[test]
fn test_decoding_bit_unpacking() {
    println!("\n--- Test: Decoding Bit Unpacking ---");

    // 1. 테스트할 64비트 시드 값 수동 생성
    // (encoding_test.rs의 예상값과 동일한 로직으로 생성)
    let r_bits: u64 = 0x7FFFF;      // r = 0.5
    let theta_bits: u64 = 0x800000;  // theta = PI (정규화 후 0.5 -> 24비트의 절반)
    let basis_id: u64 = 0b1010;
    let d_theta: u64 = 0b11;
    let d_r: u64 = 1;
    let rot_code: u64 = 0b1101;
    let log2_c: u64 = 0b101; // -3 (2의 보수)
    let reserved: u64 = 0b101010;

    let test_packed_val = (r_bits << 44)
        | (theta_bits << 20)
        | (basis_id << 16)
        | (d_theta << 14)
        | (d_r << 13)
        | (rot_code << 9)
        | (log2_c << 6)
        | reserved;
    
    let packed = Packed64(test_packed_val);

    // 2. 디코딩 실행
    let decoded = packed.decode();

    // 3. 검증
    println!("  - Packed Value: 0x{:016X}", test_packed_val);
    println!("  - Decoded Params: {:?}", decoded);

    assert_relative_eq!(decoded.r, 0.5, epsilon = 1e-6);
    assert_relative_eq!(decoded.theta, PI, epsilon = 1e-6);
    assert_eq!(decoded.basis_id, basis_id as u8);
    assert_eq!(decoded.d_theta, d_theta as u8);
    assert_eq!(decoded.d_r, d_r == 1);
    assert_eq!(decoded.rot_code, rot_code as u8);
    assert_eq!(decoded.log2_c, -3);
    assert_eq!(decoded.reserved, reserved as u8);
    println!("  [PASSED] All fields were decoded correctly.");
}

#[test]
fn test_signed_int_decoding() {
    println!("\n--- Test: Signed Integer (log2_c) Decoding ---");
    
    // log2_c: 3비트, 2의 보수, -4 ~ +3
    let test_cases = vec![
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -4),
        (0b101, -3),
        (0b110, -2),
        (0b111, -1),
    ];

    for (bits, expected_val) in test_cases {
        let packed = Packed64(bits << 6); // log2_c 필드에 위치
        let decoded = packed.decode();
        println!("  - Bits: 0b{:03b} -> Decoded: {}", bits, decoded.log2_c);
        assert_eq!(decoded.log2_c, expected_val);
    }
    println!("  [PASSED] 3-bit signed integer decoding is correct.");
} 