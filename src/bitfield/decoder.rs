// src/layers/bitfield/decoder.rs

//! # 비트필드 기반 가중치 인코딩/디코딩
//!
//! 이 모듈은 가중치 벡터 파라미터를 압축된 정수에서
//! 인코딩 및 디코딩하기 위한 상수와 유틸리티 함수를 정의합니다.
//! 32비트 표준 레이아웃과 22비트 극한 압축 레이아웃을 지원합니다.

// ========== 32-bit 표준 레이아웃 ==========
// | 비트      | 필드      | 설명                               |
// |-----------|-----------|------------------------------------|
// | `31..=24` | `phase`   | 8비트 위상 정보 (0-2π)             |
// | `23..=22` | `amp_fine`| 2비트 미세 진폭 보정               |
// | `21..=20` | `cat`     | 기하학 카테고리                      |
// | `19..=18` | `sub`     | 기저 함수 족                       |
// | `17..=10` | `idx`     | 기저 벡터 인덱스 (최대 256개)      |
// | `9`       | `sign`    | 반지름 부호 (0: 양수, 1: 음수)     |
// | `8`       | `d`       | 함수 내 변형 코드 (1비트)          |
// | `7..=0`   | `amp`     | 8비트 양자화된 반지름              |

pub const PHASE_MASK_32BIT: u32 = 0xFF << 24;
pub const AMP_FINE_MASK_32BIT: u32 = 0b11 << 22;
pub const CAT_MASK_32BIT: u32 = 0b11 << 20;
pub const SUB_MASK_32BIT: u32 = 0b11 << 18;
pub const IDX_MASK_32BIT: u32 = 0xFF << 10;
pub const SIGN_MASK_32BIT: u32 = 0b1 << 9;
pub const D_MASK_32BIT: u32 = 0b1 << 8;
pub const AMP_MASK_32BIT: u32 = 0xFF;

/// 32비트 코드를 8개 필드로 디코딩합니다.
#[inline(always)]
pub fn decode_32bit(code: u32) -> (u8, u8, u8, u8, u8, u8, u8, u8) {
    let phase = ((code & PHASE_MASK_32BIT) >> 24) as u8;
    let amp_fine = ((code & AMP_FINE_MASK_32BIT) >> 22) as u8;
    let cat = ((code & CAT_MASK_32BIT) >> 20) as u8;
    let sub = ((code & SUB_MASK_32BIT) >> 18) as u8;
    let idx = ((code & IDX_MASK_32BIT) >> 10) as u8;
    let sign = ((code & SIGN_MASK_32BIT) >> 9) as u8;
    let d = ((code & D_MASK_32BIT) >> 8) as u8;
    let amp = (code & AMP_MASK_32BIT) as u8;
    (cat, sub, idx, sign, d, amp, amp_fine, phase)
}

/// 8개 필드를 32비트 코드로 인코딩합니다.
#[inline(always)]
pub fn encode_32bit(
    cat: u8,
    sub: u8,
    idx: u8,
    sign: u8,
    d: u8,
    amp: u8,
    amp_fine: u8,
    phase: u8,
) -> u32 {
    ((phase as u32) << 24)
        | ((amp_fine as u32) << 22)
        | ((cat as u32) << 20)
        | ((sub as u32) << 18)
        | ((idx as u32) << 10)
        | ((sign as u32) << 9)
        | ((d as u32) << 8)
        | (amp as u32)
}

// ========== 22-bit 극한 압축 레이아웃 ==========
// | 비트      | 필드      | 설명                       |
// |-----------|-----------|----------------------------|
// | `21..=20` | `cat`     | 기하학 카테고리 (2비트)       |
// | `19..=18` | `sub`     | 기저 함수 족 (2비트)        |
// | `17..=10` | `idx`     | 기저 벡터 인덱스 (8비트)     |
// | `9..=8`   | `d`       | 함수 내 변형 코드 (2비트)    |
// | `7..=0`   | `amp`     | 양자화된 반지름 (8비트)      |

pub const CAT_MASK_22BIT: u32 = 0b11 << 20;
pub const SUB_MASK_22BIT: u32 = 0b11 << 18;
pub const IDX_MASK_22BIT: u32 = 0xFF << 10;
pub const D_MASK_22BIT: u32 = 0b11 << 8;
pub const AMP_MASK_22BIT: u32 = 0xFF;

/// 22비트 코드를 5개 필드로 디코딩합니다.
#[inline(always)]
pub fn decode_22bit(code: u32) -> (u8, u8, u8, u8, u8) {
    let cat = ((code & CAT_MASK_22BIT) >> 20) as u8;
    let sub = ((code & SUB_MASK_22BIT) >> 18) as u8;
    let idx = ((code & IDX_MASK_22BIT) >> 10) as u8;
    let d = ((code & D_MASK_22BIT) >> 8) as u8;
    let amp = (code & AMP_MASK_22BIT) as u8;
    (cat, sub, idx, d, amp)
}

/// 5개 필드를 22비트 코드로 인코딩합니다.
#[inline(always)]
pub fn encode_22bit(cat: u8, sub: u8, idx: u8, d: u8, amp: u8) -> u32 {
    ((cat as u32) << 20)
        | ((sub as u32) << 18)
        | ((idx as u32) << 10)
        | ((d as u32) << 8)
        | (amp as u32)
}

// 기존 함수 이름과의 호환성을 위해 유지
pub use decode_32bit as decode;
pub use encode_32bit as encode;
