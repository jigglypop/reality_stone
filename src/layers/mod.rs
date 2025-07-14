// src/layers/mod.rs

//! # Reality Stone 레이어 모듈
//!
//! 리만 기하학에 최적화된 다양한 하이퍼볼릭 레이어를 제공합니다.

// 기하학적 레이어들
pub mod klein;
pub mod lorentz;
pub mod poincare;
pub mod spline;
pub mod utils;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use self::poincare::{
    poincare_ball_layer, poincare_ball_layer_backward, poincare_distance, poincare_to_klein,
    poincare_to_lorentz,
};
