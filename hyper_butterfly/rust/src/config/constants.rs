//! 수치 안정성을 위한 상수값 정의

/// 하이퍼볼릭 기하학 연산의 수치 안정성을 위한 상수들
pub struct Constants;

impl Constants {
    /// 엡실론 값 (0으로 나누기 방지)
    pub const EPS: f32 = 1e-6;

    /// 경계 엡실론 (경계 근처 수치 문제 방지)
    pub const BOUNDARY_EPS: f32 = 1e-6;

    /// tanh 함수의 최대 입력값 제한
    pub const MAX_TANH_ARG: f32 = 15.0;

    /// 분모의 최소값 (0으로 나누기 방지)
    pub const MIN_DENOMINATOR: f32 = 1e-8;

    /// log/exp 맵 경계 안전 값
    pub const SAFE_LOGEXP_BOUNDARY: f32 = 0.999999;
}
