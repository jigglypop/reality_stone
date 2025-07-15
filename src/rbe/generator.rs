use super::math::*;
use super::types::Packed64;

impl Packed64 {
    /// 가중치 계산 (문서의 방식대로)
    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let params = self.decode();

        // 곡률 계산
        let c = 2.0f32.powi(params.log2_c as i32);

        // 좌표를 [-1, 1] 범위로 정규화
        let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;

        // 로컬 극좌표
        let r_local = (x * x + y * y).sqrt().min(0.999999);
        let theta_local = y.atan2(x);

        // 회전 적용
        let rotation = get_rotation_angle(params.rot_code);
        let theta_final = params.theta + theta_local + rotation;

        // 미분 순환성 적용
        let angular_value = apply_angular_derivative(theta_final, params.d_theta, params.basis_id);
        let radial_value = apply_radial_derivative(c * params.r, params.d_r, params.basis_id);

        // 기저 함수에 따른 계산
        let basis_value = match params.basis_id {
            0..=3 => angular_value * radial_value,
            4 => bessel_j0(r_local * 10.0),
            5 => bessel_i0(r_local * 10.0),
            6 => bessel_k0(r_local * 10.0),
            7 => bessel_y0(r_local * 10.0),
            8 => (c * r_local).tanh() * theta_final.cos().signum(),
            9 => sech(c * r_local) * triangle_wave(theta_final),
            10 => (-c * r_local).exp() * theta_final.sin(),
            11 => morlet_wavelet(r_local, theta_final, 5.0),
            _ => 0.0,
        };

        // 야코비안 계산
        let jacobian = (1.0 - c * params.r * params.r).powi(-2).sqrt();

        basis_value * jacobian
    }
}
