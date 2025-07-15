//! # 푸앵카레 레이어 압축 라이브러리
//!
//! 이 라이브러리는 `README.md`에 설명된 "단일 64비트 시드"를 사용하여
//! 전체 신경망 레이어를 표현하고 압축/복원하는 기능을 제공합니다.

pub mod types;
pub mod encoder;
pub mod decoder;
pub mod generator;
pub mod math;
pub mod matrix;

// 라이브러리 사용자가 편리하게 접근할 수 있도록 주요 구조체들을 공개합니다.
pub use types::{Packed64, PoincareMatrix, BasisFunction, DecodedParams};
 