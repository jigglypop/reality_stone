pub mod batch;
pub mod curvature;
pub mod metrikey;
pub mod mobius;
pub mod project;

use ndarray::Array2;

pub use self::batch::{dot_batched, norm_sq_batched};
pub use self::mobius::{
    mobius_add, mobius_add_dynamic, mobius_add_dynamic_backward, mobius_add_grad_c,
    mobius_add_layerwise, mobius_add_layerwise_backward, mobius_scalar, mobius_scalar_grad_c,
};
pub use self::project::project_to_ball;
pub use curvature::{DynamicCurvature, LayerWiseDynamicCurvature};
pub use metrikey::{
    apply_linear, block_orthogonal_from_key, compose_layers_gravity,
    compose_layers_order_preserving, deterministic_orthogonal_from_key, mahalanobis_distance_sq_g,
    mahalanobis_distance_sq_l, metric_factor_cholesky, rotate_metric_factor_block,
    spd_block_metric_from_key, spd_metric_from_key, spd_metric_from_key_weighted,
};

// f64 high-precision exports (not all functions; only where useful)
pub use metrikey::{
    apply_linear_f64, compose_layers_gravity_compact_f64, compose_layers_gravity_f64,
    deterministic_orthogonal_from_key_f64, spd_metric_from_key_f64,
};

// Implicit transforms
pub use metrikey::{
    givens_chain_apply_from_key, householder_chain_apply_from_key,
    householder_chain_apply_transpose_from_key, lowrank_plus_diag_apply_from_key,
};

/// MSE loss의 gradient를 계산합니다.
pub fn mse_loss_grad(pred: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    2.0 * (pred - target) / (pred.shape()[0] * pred.shape()[1]) as f32
}
