use poincare_layer::PoincareMatrix;
use std::f32::consts::PI;

#[test]
fn test_deep_compress() {
    let rows = 32;
    let cols = 32;
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
            let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
            matrix[i * cols + j] = (PI * x).sin() * (PI * y).cos();
        }
    }
    let compressed = PoincareMatrix::deep_compress(&matrix, rows, cols);
    let reconstructed = compressed.decompress();
    let mut error = 0.0;
    for i in 0..matrix.len() {
        error += (matrix[i] - reconstructed[i]).powi(2);
    }
    let rmse = (error / matrix.len() as f32).sqrt();
    assert!(rmse < 0.5);
} 