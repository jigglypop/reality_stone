// src/layers/bitfield/decoder.rs

//! # 비트필드 기반 가중치 인코딩 디코더
//!
//! 이 모듈은 단일 가중치 벡터의 파라미터를 압축된 32비트 정수에서
//! 디코딩하기 위한 상수와 유틸리티 함수를 정의합니다.
//! 이는 Reality Stone의 핵심적인 초경량 압축 방법론입니다.
//!
//! 32비트 레이아웃은 다음과 같이 확장되었습니다:
//!
//! | 비트      | 필드      | 설명                                                      |
//! |-----------|-----------|-----------------------------------------------------------|
//! | `31..=24` | `phase`   | 8비트 위상 정보 (0-255 -> 0-2π)                           |
//! | `23..=22` | `amp_fine`| 2비트 미세 진폭 보정                                       |
//! | `21..=20` | `cat`     | 기하학 카테고리 (예: 푸앵카레, 로렌츠)                      |
//! | `19..=18` | `sub`     | 기저 함수 족 (예: exp/log, sinh/cosh)                     |
//! | `17..=10` | `idx`     | 공유 기저 벡터 테이블의 인덱스 (최대 256개)               |
//! | `9`       | `sign`    | 반지름(크기)의 부호 (0: 양수, 1: 음수)                    |
//! | `8`       | `d`       | 함수 내 변형 코드 (이제 1비트, 2종류)                     |
//! | `7..=0`   | `amp`     | 8비트로 양자화된 접선 벡터의 크기(반지름)                 |

pub const PHASE_MASK: u32 = 0xFF << 24;
pub const AMP_FINE_MASK: u32 = 0b11 << 22;
pub const CAT_MASK: u32 = 0b11 << 20;
pub const SUB_MASK: u32 = 0b11 << 18;
pub const IDX_MASK: u32 = 0xFF << 10;
pub const SIGN_MASK: u32 = 0b1 << 9;
pub const D_MASK: u32 = 0b1 << 8;
pub const AMP_MASK: u32 = 0xFF;

/// 32비트 정수 코드를 8개의 구성 파라미터 필드로 디코딩합니다.
///
/// 이 함수는 직접 추론 커널과 같이 반복이 많은 루프에서
/// 최고의 성능을 내기 위해 인라인되도록 설계되었습니다.
///
/// # 인자
/// * `code` - 압축된 32비트 코드를 담고 있는 32비트 정수.
///
/// # 반환
/// `(cat, sub, idx, sign, d, amp, amp_fine, phase)` 튜플.
#[inline(always)]
pub fn decode(code: u32) -> (u8, u8, u8, u8, u8, u8, u8, u8) {
    let phase = ((code & PHASE_MASK) >> 24) as u8;
    let amp_fine = ((code & AMP_FINE_MASK) >> 22) as u8;
    let cat = ((code & CAT_MASK) >> 20) as u8;
    let sub = ((code & SUB_MASK) >> 18) as u8;
    let idx = ((code & IDX_MASK) >> 10) as u8;
    let sign = ((code & SIGN_MASK) >> 9) as u8;
    let d = ((code & D_MASK) >> 8) as u8;
    let amp = (code & AMP_MASK) as u8;
    (cat, sub, idx, sign, d, amp, amp_fine, phase)
}

/// 개별 필드를 하나의 u32 코드로 인코딩합니다.
///
/// # 인자
/// * `cat` - 카테고리 (0-3)
/// * `sub` - 서브카테고리 (0-3)
/// * `idx` - 기저 인덱스 (0-255)
/// * `sign` - 부호 비트 (0: 양수, 1: 음수)
/// * `d` - 추가 파라미터 (0-1)
/// * `amp` - 주 진폭 (0-255)
/// * `amp_fine` - 미세 진폭 (0-3)
/// * `phase` - 위상 (0-255)
///
/// # 반환
/// 32비트 인코딩된 u32 값
#[inline]
pub fn encode(cat: u8, sub: u8, idx: u8, sign: u8, d: u8, amp: u8, amp_fine: u8, phase: u8) -> u32 {
    ((phase as u32 & 0xFF) << 24)
        | ((amp_fine as u32 & 0x3) << 22)
        | ((cat as u32 & 0x3) << 20)
        | ((sub as u32 & 0x3) << 18)
        | ((idx as u32 & 0xFF) << 10)
        | ((sign as u32 & 0x1) << 9)
        | ((d as u32 & 0x1) << 8)
        | (amp as u32 & 0xFF)
} 