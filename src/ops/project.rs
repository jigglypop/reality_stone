use crate::ops::batch::norm_sq_batched;
use ndarray::{Array2, ArrayView2, Axis};

pub const EPS: f32 = 1e-7;
pub fn project_to_ball(x: &ArrayView2<f32>, epsilon: f32) -> Array2<f32> {
    let norm = norm_sq_batched(x).mapv(f32::sqrt).insert_axis(Axis(1));
    let max_norm = 1.0 - epsilon;
    let scale = norm.mapv(|n| if n > max_norm { max_norm / n } else { 1.0 });
    x * &scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_project_to_ball_inside() {
        let x = arr2(&[[0.1_f32, 0.2], [0.3, 0.2]]);
        let y = project_to_ball(&x.view(), 1e-5);
        // 내부 포인트는 거의 변화가 없어야 함
        assert!((y[[0, 0]] - 0.1).abs() < 1e-6);
        assert!((y[[1, 1]] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_project_to_ball_outside() {
        let x = arr2(&[[0.9_f32, 0.9]]);
        let y = project_to_ball(&x.view(), 1e-3);
        let norm = (y[[0, 0]].powi(2) + y[[0, 1]].powi(2)).sqrt();
        assert!(norm <= 1.0 - 1e-3 + 1e-5);
    }
}
