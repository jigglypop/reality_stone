use ndarray::{Array1, ArrayView2, Axis};

pub const EPS: f32 = 1e-7;

/// Computes the squared norm of each row in a batched manner.
/// x: A 2D array of shape (batch_size, dim).
/// Returns a 1D array of shape (batch_size,) where each element is the squared L2 norm of a row.
pub fn norm_sq_batched(x: &ArrayView2<f32>) -> Array1<f32> {
    (x * x).sum_axis(Axis(1))
}

/// Computes the dot product of corresponding rows in two matrices in a batched manner.
/// x: A 2D array of shape (batch_size, dim).
/// y: A 2D array of shape (batch_size, dim).
/// Returns a 1D array of shape (batch_size,) where each element is the dot product of corresponding rows.
pub fn dot_batched(x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
    (x * y).sum_axis(Axis(1))
} 