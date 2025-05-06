//! CUDA 지원 모듈
//! cudarc 크레이트를 사용한 CUDA 함수 구현

#[cfg(feature = "cuda")]
pub mod butterfly;
#[cfg(feature = "cuda")]
pub mod exp_map;
#[cfg(feature = "cuda")]
pub mod geodesic;
#[cfg(feature = "cuda")]
pub mod log_map;
#[cfg(feature = "cuda")]
pub mod mobius;

#[cfg(feature = "cuda")]
pub use butterfly::butterfly_transform_cuda;
#[cfg(feature = "cuda")]
pub use exp_map::exp_map_cuda;
#[cfg(feature = "cuda")]
pub use geodesic::geodesic_cuda;
#[cfg(feature = "cuda")]
pub use log_map::log_map_cuda;
#[cfg(feature = "cuda")]
pub use mobius::{mobius_add_cuda, mobius_scalar_cuda};

#[cfg(feature = "cuda")]
use cudarc::prelude::*;
use ndarray::Array2;

#[cfg(feature = "cuda")]
pub fn init_cuda() -> CudaDevice {
    CudaDevice::new(0).expect("CUDA 디바이스 초기화 실패")
}
