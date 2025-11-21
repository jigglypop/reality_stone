// src/layers/mod.rs

//! # Reality Stone 레이어 모듈
//!
//! 리만 기하학에 최적화된 다양한 하이퍼볼릭 레이어를 제공합니다.

// 기하학적 레이어들
pub mod klein;
pub mod lorentz;
pub mod poincare;
pub mod spline;
pub mod riemann;
pub mod suppression;
pub mod bellman;
pub mod utils;

// 통합 리만 시스템
pub mod metric;
pub mod geodesic;
pub mod bellman_lagrangian;
pub mod unified_riemannian;

pub use self::poincare::{
    poincare_ball_layer, poincare_ball_layer_backward, poincare_distance, poincare_exp_at,
    poincare_log_at, poincare_to_klein, poincare_to_lorentz,
};

pub use self::unified_riemannian::{
    UnifiedRiemannianLayer, LayerOutput, LayerCache, LayerGradients,
};
pub use self::metric::{MetricType, MetricTensor, DiagonalMetric, PoincareMetric, LorentzMetric, KleinMetric};
pub use self::geodesic::{exponential_map, logarithmic_map, geodesic_path, geodesic_interpolation};
pub use self::bellman_lagrangian::{
    ValueFunction, LagrangianParams, EnergyComponents,
    bellman_potential, kinetic_energy, representation_flow,
};

