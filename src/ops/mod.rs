pub mod batch;
pub mod curvature;
pub mod mobius;
pub mod project;

use ndarray::{Array2, Axis};

pub use self::batch::{dot_batched, norm_sq_batched};
pub use self::mobius::{
    mobius_add, mobius_scalar,
    DynamicCurvature, mobius_add_dynamic, mobius_add_dynamic_backward,
    LayerWiseDynamicCurvature, mobius_add_layerwise, mobius_add_layerwise_backward,
    mobius_scalar_grad_c, mobius_add_grad_c
};
pub use self::project::{project_to_ball};

/// MSE loss의 gradient를 계산합니다.
pub fn mse_loss_grad(pred: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    2.0 * (pred - target) / (pred.shape()[0] * pred.shape()[1]) as f32
}
