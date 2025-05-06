//! 수치 안정성을 위한 상수값 정의
/// 하이퍼볼릭 기하학 연산의 수치 안정성을 위한 상수들
pub struct Constants;

impl Constants {
    pub const EPS: f32 = 1e-6;
    pub const BOUNDARY_EPS: f32 = 1e-6;
    pub const MAX_TANH_ARG: f32 = 15.0;
    pub const MIN_DENOMINATOR: f32 = 1e-8;
    pub const SAFE_LOGEXP_BOUNDARY: f32 = 0.999999;
}
