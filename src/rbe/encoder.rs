use super::types::Packed64;
use std::f32::consts::PI;

impl Packed64 {
    /// 새로운 Packed64 시드 생성 (인코딩)
    pub fn new(
        r: f32,
        theta: f32,
        basis_id: u8,
        d_theta: u8,
        d_r: bool,
        rot_code: u8,
        log2_c: i8,
        reserved: u8,
    ) -> Self {
        // r: [0, 1) 범위로 클램핑
        let r_clamped = r.clamp(0.0, 0.999999);
        // theta: [0, 2π) 범위로 정규화
        let theta_normalized = theta.rem_euclid(2.0 * PI);

        // 비트 패킹
        let r_bits = (r_clamped * ((1u64 << 20) - 1) as f32).round() as u64;
        let theta_bits = (theta_normalized / (2.0 * PI) * ((1u64 << 24) - 1) as f32).round() as u64;

        let mut packed = 0u64;
        packed |= (r_bits & 0xFFFFF) << 44; // 20 bits
        packed |= (theta_bits & 0xFFFFFF) << 20; // 24 bits
        packed |= ((basis_id as u64) & 0xF) << 16; // 4 bits
        packed |= ((d_theta as u64) & 0x3) << 14; // 2 bits
        packed |= ((d_r as u64) & 0x1) << 13; // 1 bit
        packed |= ((rot_code as u64) & 0xF) << 9; // 4 bits
        packed |= ((log2_c as u64) & 0x7) << 6; // 3 bits (2's complement)
        packed |= (reserved as u64) & 0x3F; // 6 bits

        Packed64(packed)
    }
}
