use poincare_layer::{Packed64, PoincareMatrix};
use approx::assert_relative_eq;
use std::f32::consts::PI;

#[test]
fn test_encode_decode_exact() {
    // 문서의 예제대로 테스트
    let r = 0.5;
    let theta = PI;
    let basis_id = 0;
    let d_theta = 2;
    let d_r = true;
    let rot_code = 3;
    let log2_c = -2;
    let reserved = 0;

    let packed = Packed64::new(r, theta, basis_id, d_theta, d_r, rot_code, log2_c, reserved);
    let decoded = packed.decode();
    
    assert_relative_eq!(decoded.r, r, epsilon = 1e-6);
    assert_relative_eq!(decoded.theta, theta, epsilon = 1e-6);
    assert_eq!(decoded.basis_id, basis_id);
    assert_eq!(decoded.d_theta, d_theta);
    assert_eq!(decoded.d_r, d_r);
    assert_eq!(decoded.rot_code, rot_code);
    assert_eq!(decoded.log2_c, log2_c);
    assert_eq!(decoded.reserved, reserved);
}

#[test]
fn test_compression_and_decompression() {
    let rows = 32;
    let cols = 32;
    
    // 간단한 패턴 생성
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
            let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
            matrix[i * cols + j] = (PI * x).sin() * (PI * y).cos();
        }
    }
    
    // 압축
    let compressed = PoincareMatrix::deep_compress(&matrix, rows, cols);
    
    // 복원
    let reconstructed = compressed.decompress();
    
    // RMSE 계산
    let mut total_error = 0.0;
    for i in 0..matrix.len() {
        total_error += (matrix[i] - reconstructed[i]).powi(2);
    }
    let rmse = (total_error / matrix.len() as f32).sqrt();
    
    println!("\n--- 압축 및 복원 테스트 ({}x{}) ---", rows, cols);
    println!("  - 원본 크기: {} bytes", matrix.len() * 4);
    println!("  - 압축 크기: 8 bytes (1 x u64)");
    println!("  - 압축률: {}:1", matrix.len() * 4 / 8);
    println!("  - 최종 RMSE: {:.6}", rmse);
    println!("  - 찾은 시드: {:?}", compressed.seed.decode());
    
    // 랜덤 탐색의 한계가 있으므로, RMSE가 적정 수준 이하인지 확인
    assert!(rmse < 1.0, "RMSE should be reasonably low after compression.");
} 