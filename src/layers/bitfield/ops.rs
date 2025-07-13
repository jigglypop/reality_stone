// src/layers/bitfield/ops.rs

//! # 하이퍼볼릭 및 주기적 연산 LUT
//!
//! 이 모듈은 디코딩된 비트필드 코드(`cat`, `sub`, `d`)에 기반하여
//! 올바른 수학적 함수를 적용하기 위한 조회 테이블(LUT) 역할을 합니다.
//! 리만 기하학에서 자주 사용되는 기초 함수들을 체계적으로 구현합니다.

/// 제공된 코드에 기반하여 올바른 스케일링 함수를 조회하고 적용합니다.
///
/// # 인자
/// * `cat` - 기하학 카테고리 코드 (0: Poincaré, 1: Lorentz, 2: Klein, 3: Special)
/// * `sub` - 기저 함수 족 코드 (0: 기본, 1: 쌍곡, 2: 삼각, 3: 지수/로그)
/// * `d` - 도함수 차수 또는 변형 코드
/// * `r` - 반지름을 나타내는 스칼라 `f32` 값.
///
/// # 반환
/// 계산된 스케일링 인자를 담은 스칼라 `f32` 값.
pub fn lookup_and_apply(cat: u8, sub: u8, d: u8, r: f32) -> f32 {
    const EPS: f32 = 1e-7;
    
    match cat {
        0 => {
            // Poincaré 볼 기하학
            match sub {
                0 => {
                    // 기본 푸앵카레 스케일링
                    match d {
                        0 => (r * 0.5).tanh(),           // tanh(r/2)
                        1 => -(r * 0.5).tanh(),          // -tanh(r/2)
                        2 => 2.0 * (r * 0.25).tanh(),   // 2*tanh(r/4)
                        _ => (r * 0.5).tanh().powi(2),  // tanh²(r/2)
                    }
                },
                1 => {
                    // 쌍곡 함수족 (sinh, cosh)
                    match d {
                        0 => r.sinh() / (1.0 + r.cosh()).max(EPS),  // sinh(r)/(1+cosh(r))
                        1 => (r.cosh() - 1.0) / (1.0 + r.cosh()).max(EPS), // (cosh(r)-1)/(1+cosh(r))
                        2 => r.tanh(),                   // tanh(r)
                        _ => r.sinh() / r.max(EPS),      // sinh(r)/r (sinc 쌍곡)
                    }
                },
                2 => {
                    // 삼각 함수족 (sin, cos)
                    match d {
                        0 => r.sin() / r.max(EPS),      // sin(r)/r (sinc)
                        1 => r.cos(),                    // cos(r)
                        2 => (1.0 - r.cos()) / (r * r).max(EPS), // (1-cos(r))/r²
                        _ => r.sin() * r.cos(),          // sin(r)cos(r)
                    }
                },
                _ => {
                    // 지수/로그 함수족
                    match d {
                        0 => (r.exp() - 1.0) / r.max(EPS),  // (exp(r)-1)/r
                        1 => r.exp() / (1.0 + r.exp()),     // exp(r)/(1+exp(r)) (sigmoid)
                        2 => (r + 1.0).ln().max(0.0),       // ln(r+1)
                        _ => r / (1.0 + r.abs()),           // r/(1+|r|)
                    }
                }
            }
        },
        1 => {
            // Lorentz (쌍곡) 기하학
            match sub {
                0 => {
                    // 기본 로렌츠 스케일링
                    match d {
                        0 => r.sinh(),                   // sinh(r)
                        1 => r.cosh() - 1.0,            // cosh(r) - 1
                        2 => r.tanh(),                  // tanh(r)
                        _ => r.sinh() / r.cosh(),       // tanh(r) 다른 형태
                    }
                },
                1 => {
                    // 수정된 쌍곡 함수
                    match d {
                        0 => (r * 0.5).sinh() * 2.0,   // 2*sinh(r/2)
                        1 => (r * 0.5).cosh(),          // cosh(r/2)
                        2 => r.exp() - 1.0,             // exp(r) - 1
                        _ => 1.0 - (-r).exp(),          // 1 - exp(-r)
                    }
                },
                2 => {
                    // 혼합 함수
                    match d {
                        0 => r.sinh() * r.sin(),        // sinh(r)*sin(r)
                        1 => r.cosh() * r.cos(),        // cosh(r)*cos(r)
                        2 => r.tanh() * r.tan().tanh(), // tanh(r)*tanh(tan(r))
                        _ => (r.sinh() + r.sin()) * 0.5, // (sinh(r)+sin(r))/2
                    }
                },
                _ => {
                    // 특수 함수
                    match d {
                        0 => r.asinh() / r.max(EPS),    // asinh(r)/r
                        1 => r.atanh().min(10.0),        // atanh(r) (제한)
                        2 => (1.0 + r * r).sqrt() - 1.0, // sqrt(1+r²) - 1
                        _ => r / (1.0 + r * r).sqrt(),   // r/sqrt(1+r²)
                    }
                }
            }
        },
        2 => {
            // Klein 기하학
            match sub {
                0 => {
                    // 기본 Klein 스케일링
                    match d {
                        0 => r / (1.0 + r).max(EPS),    // r/(1+r)
                        1 => r / (1.0 + r * r).sqrt(),  // r/sqrt(1+r²)
                        2 => r * r / (1.0 + r * r),     // r²/(1+r²)
                        _ => 1.0 - 1.0 / (1.0 + r),     // 1 - 1/(1+r)
                    }
                },
                1 => {
                    // 투영 함수
                    match d {
                        0 => 2.0 * r / (1.0 + r * r),   // 2r/(1+r²)
                        1 => (1.0 - r * r) / (1.0 + r * r), // (1-r²)/(1+r²)
                        2 => 4.0 * r / (1.0 + r * r).powi(2), // 4r/(1+r²)²
                        _ => r.atan() * 2.0 / std::f32::consts::PI, // 2*atan(r)/π
                    }
                },
                _ => {
                    // 기타 Klein 변환
                    r / (1.0 + r.abs()).max(EPS)
                }
            }
        },
        _ => {
            // 특수/실험적 함수들
            match sub {
                0 => {
                    // Bessel 유사 함수
                    match d {
                        0 => (r * 0.5).sin() / (r * 0.5).max(EPS), // J₀ 근사
                        1 => r.sin() / r.max(EPS) - r.cos(), // J₁ 근사
                        _ => r.exp() * (-r).exp() * r.sin(), // 변조된 Bessel
                    }
                },
                1 => {
                    // Gaussian 유사
                    match d {
                        0 => (-r * r * 0.5).exp(),      // exp(-r²/2)
                        1 => r * (-r * r * 0.5).exp(),  // r*exp(-r²/2)
                        2 => (1.0 - r * r) * (-r * r * 0.5).exp(), // (1-r²)*exp(-r²/2)
                        _ => (-r.abs()).exp(),           // exp(-|r|)
                    }
                },
                2 => {
                    // 주기적 변조
                    match d {
                        0 => (r * std::f32::consts::PI).sin(), // sin(πr)
                        1 => (r * std::f32::consts::PI).cos(), // cos(πr)
                        2 => (r * 2.0 * std::f32::consts::PI).sin() * 0.5 + 0.5, // (sin(2πr)+1)/2
                        _ => ((r * std::f32::consts::PI).sin()).powi(2), // sin²(πr)
                    }
                },
                _ => {
                    // 기본값
                    (r * 0.5).tanh()
                }
            }
        }
    }
} 