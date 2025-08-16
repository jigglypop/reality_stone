use ndarray::{Array1, ArrayView2, Axis};

pub const EPS: f32 = 1e-7;

pub fn norm_sq_batched(x: &ArrayView2<f32>) -> Array1<f32> {
    x.map_axis(Axis(1), |row| row.mapv(|a| a.powi(2)).sum())
}

pub fn dot_batched(x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
    (x * y).sum_axis(Axis(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_norm_sq_batched() {
        let x = arr2(&[[1.0_f32, 2.0, 2.0], [3.0, 4.0, 0.0]]);
        let norms = norm_sq_batched(&x.view());
        assert!((norms[0] - 9.0).abs() < 1e-6);
        assert!((norms[1] - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_batched() {
        let x = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let y = arr2(&[[6.0_f32, 5.0, 4.0], [3.0, 2.0, 1.0]]);
        let dots = dot_batched(&x.view(), &y.view());
        assert!((dots[0] - (1.0 * 6.0 + 2.0 * 5.0 + 3.0 * 4.0)).abs() < 1e-6);
        assert!((dots[1] - (4.0 * 3.0 + 5.0 * 2.0 + 6.0 * 1.0)).abs() < 1e-6);
    }
}
