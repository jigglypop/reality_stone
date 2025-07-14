use ndarray::{Array1, Array2, ArrayView2, Axis};

pub const EPS: f32 = 1e-7;

pub fn norm_sq_batched(x: &ArrayView2<f32>) -> Array1<f32> {
    x.map_axis(Axis(1), |row| row.mapv(|a| a.powi(2)).sum())
}

pub fn project_to_ball(x: &ArrayView2<f32>, epsilon: f32) -> Array2<f32> {
    let norm = norm_sq_batched(x).mapv(f32::sqrt).insert_axis(Axis(1));
    let max_norm = 1.0 - epsilon;
    let scale = norm.mapv(|n| if n > max_norm { max_norm / n } else { 1.0 });
    x * &scale
}

pub fn dot_batched(x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
    (x * y).sum_axis(Axis(1))
}
