use crate::ops::batch::norm_sq_batched;
use ndarray::{ArrayView2, Axis, Array2};

pub const EPS: f32 = 1e-7;
pub fn project_to_ball(x: &ArrayView2<f32>, epsilon: f32) -> Array2<f32> {
    let norm = norm_sq_batched(x).mapv(f32::sqrt).insert_axis(Axis(1));
    let max_norm = 1.0 - epsilon;
    let scale = norm.mapv(|n| if n > max_norm { max_norm / n } else { 1.0 });
    x * &scale
}
