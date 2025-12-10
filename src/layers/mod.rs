// src/layers/mod.rs

//! # Reality Stone 레이어 모듈
//!
//! 리만 기하학에 최적화된 다양한 하이퍼볼릭 레이어를 제공합니다.

// 기하학적 레이어들
pub mod bellman;
pub mod klein;
pub mod lorentz;
pub mod memory;
pub mod poincare;
pub mod riemann;
pub mod spline;
pub mod spline_cache;
pub mod suppression;
pub mod utils;

// 통합 리만 시스템
pub mod bellman_lagrangian;
pub mod decoder;
pub mod diffusion;
pub mod geodesic;
pub mod human_decoder;
pub mod hyper_metric;
pub mod metric;
pub mod rsulf;
pub mod symplectic;
pub mod unified_riemannian;

pub use self::poincare::{
    poincare_ball_layer, poincare_ball_layer_backward, poincare_distance, poincare_exp_at,
    poincare_log_at, poincare_to_klein, poincare_to_lorentz,
};

pub use self::bellman_lagrangian::{
    bellman_potential, kinetic_energy, representation_flow, EnergyComponents, LagrangianParams,
    ValueFunction,
};
pub use self::decoder::RiemannianDecoder;
pub use self::diffusion::RiemannianDiffusion; // Export diffusion
pub use self::geodesic::{exponential_map, geodesic_interpolation, geodesic_path, logarithmic_map};
pub use self::human_decoder::{HumanStyleDecoder, StageWeights};
pub use self::metric::{
    DiagonalMetric, KleinMetric, LorentzMetric, MetricTensor, MetricType, PoincareMetric,
};
pub use self::unified_riemannian::{
    LayerCache, LayerGradients, LayerOutput, UnifiedRiemannianLayer,
};
