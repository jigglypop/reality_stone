use ndarray::{Array1, ArrayView2, Axis};

pub const EPS: f32 = 1e-7;

pub fn norm_sq_batched(x: &ArrayView2<f32>) -> Array1<f32> {
    x.map_axis(Axis(1), |row| row.mapv(|a| a.powi(2)).sum())
}

pub fn dot_batched(x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
    (x * y).sum_axis(Axis(1))
} 