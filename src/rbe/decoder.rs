use super::types::{DecodedParams, Packed64};
use std::f32::consts::PI;

impl Packed64 {
    /// 시드를 디코딩
    pub fn decode(&self) -> DecodedParams {
        let bits = self.0;

        // 비트 언패킹
        let r_bits = (bits >> 44) & 0xFFFFF;
        let theta_bits = (bits >> 20) & 0xFFFFFF;
        let basis_id = ((bits >> 16) & 0xF) as u8;
        let d_theta = ((bits >> 14) & 0x3) as u8;
        let d_r = ((bits >> 13) & 0x1) != 0;
        let rot_code = ((bits >> 9) & 0xF) as u8;
        let log2_c_bits = ((bits >> 6) & 0x7) as u8;
        let reserved = (bits & 0x3F) as u8;

        // 값 복원
        let r = (r_bits as f32) / ((1u64 << 20) - 1) as f32;
        let theta = (theta_bits as f32 / ((1u64 << 24) - 1) as f32) * 2.0 * PI;

        // 3비트 부호있는 정수 복원 (-4 ~ +3)
        let log2_c = if (log2_c_bits & 0x4) != 0 {
            (log2_c_bits as i8) | -8
        } else {
            log2_c_bits as i8
        };

        DecodedParams {
            r,
            theta,
            basis_id,
            d_theta,
            d_r,
            rot_code,
            log2_c,
            reserved,
        }
    }
}
