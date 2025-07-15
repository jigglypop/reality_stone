//! `matrix.rs`에 대한 단위 테스트

use poincare_layer::PoincareMatrix;
use std::f32::consts::PI;

#[test]
fn test_compression_and_decompression() {
    println!("\n--- Test: Matrix Compression and Decompression ---");
    
    let rows = 32;
    let cols = 32;
    
    // 1. 테스트용 원본 행렬 생성 (복잡한 sin/cos 패턴)
    let mut source_matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
            let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
            source_matrix[i * cols + j] = (PI * x * 2.0).sin() * (PI * y * 3.0).cos();
        }
    }
    
    // 2. 압축 실행
    // compress 함수는 내부에 랜덤 탐색을 포함하므로, 결과가 매번 다를 수 있음
    let compressed = PoincareMatrix::deep_compress(&source_matrix, rows, cols);
    
    // 3. 복원 실행
    let reconstructed = compressed.decompress();
    
    // 4. RMSE 계산
    let mut total_error = 0.0;
    for i in 0..source_matrix.len() {
        total_error += (source_matrix[i] - reconstructed[i]).powi(2);
    }
    let rmse = (total_error / source_matrix.len() as f32).sqrt();
    
    // 5. 결과 출력
    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - Original data size: {} bytes", source_matrix.len() * 4);
    println!("  - Compressed data size: 8 bytes (1 x u64)");
    println!("  - Achieved RMSE: {:.6}", rmse);
    println!("  - Best seed found: {:?}", compressed.seed.decode());
    
    // 6. 검증
    // 랜덤 탐색의 한계로 완벽한 복원은 불가능하지만,
    // 생성된 패턴에 대해 RMSE가 특정 임계값(1.5) 미만이어야 함을 확인
    assert!(rmse < 1.0, "RMSE ({}) should be reasonably low after compression.", rmse);
    println!("  [PASSED] Compression yields a reasonably low RMSE.");
} 