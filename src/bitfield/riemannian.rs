//! 리만 기하학 함수 라이브러리
//!
//! 64개의 다양한 리만 기하학 함수들을 제공합니다.
//! 각 함수는 (cat, sub, d) 조합으로 선택됩니다.

use std::f32::consts::{E, PI};

/// # 리만 기하학 함수 선택 및 계산
///
/// ## Arguments
/// * `cat` - 기하학 카테고리 (0-3)
/// * `sub` - 함수 서브카테고리 (0-3)  
/// * `d` - 함수 변형/도함수 차수 (0-3)
/// * `r` - 입력 반지름
///
/// ## Returns
/// 계산된 함수값
#[inline]
pub fn get_riemannian_function(cat: u8, sub: u8, d: u8, r: f32) -> f32 {
    match (cat, sub, d) {
        // ========== Category 0: Poincaré 기하학 ==========

        // 기본 함수족 (SUB=0)
        (0, 0, 0) => r.tanh() / 2.0,           // 표준 Poincaré 매핑
        (0, 0, 1) => -r.tanh() / 2.0,          // 음의 스케일링
        (0, 0, 2) => 2.0 * (r / 4.0).tanh(),   // 완화된 매핑
        (0, 0, 3) => (r / 2.0).tanh().powi(2), // 제곱 매핑

        // 쌍곡 함수족 (SUB=1)
        (0, 1, 0) => r.sinh() / (1.0 + r.cosh()), // 정규화된 sinh
        (0, 1, 1) => (r.cosh() - 1.0) / (1.0 + r.cosh()), // 이동된 cosh
        (0, 1, 2) => r.tanh(),                    // 표준 tanh
        (0, 1, 3) => {
            if r.abs() < 1e-6 {
                1.0
            } else {
                r.sinh() / r
            }
        } // sinc 쌍곡

        // 삼각 함수족 (SUB=2)
        (0, 2, 0) => {
            if r.abs() < 1e-6 {
                1.0
            } else {
                r.sin() / r
            }
        } // sinc 함수
        (0, 2, 1) => r.cos(), // 코사인
        (0, 2, 2) => {
            if r.abs() < 1e-5 {
                0.5
            } else {
                (1.0 - r.cos()) / (r * r)
            }
        } // versine 정규화
        (0, 2, 3) => r.sin() * r.cos(), // 이중 주기

        // 지수/로그 함수족 (SUB=3)
        (0, 3, 0) => {
            if r.abs() < 1e-6 {
                1.0
            } else {
                (r.exp() - 1.0) / r
            }
        } // 정규화된 지수
        (0, 3, 1) => r.exp() / (1.0 + r.exp()), // sigmoid
        (0, 3, 2) => {
            if r > 0.0 {
                if r < 1e-5 {
                    // r이 작을 때 테일러 급수 사용: ln(1+r)/r ≈ 1 - r/2 + r²/3
                    1.0 - r / 2.0 + r * r / 3.0
                } else {
                    (r + 1.0).ln() / r
                }
            } else if r < 0.0 {
                if r > -1e-5 {
                    // r이 작고 음수일 때도 테일러 급수 사용
                    1.0 - r / 2.0 + r * r / 3.0
                } else if r > -1.0 {
                    (r + 1.0).ln() / r
                } else {
                    // r <= -1일 때는 정의되지 않음
                    f32::NAN
                }
            } else {
                // r = 0
                1.0
            }
        } // 로그 변환
        (0, 3, 3) => r / (1.0 + r.abs()),       // 유계 선형

        // ========== Category 1: Lorentz 기하학 ==========

        // 기본 Lorentz (SUB=0)
        (1, 0, 0) => r.sinh(),            // 쌍곡 사인
        (1, 0, 1) => r.cosh() - 1.0,      // 이동된 쌍곡 코사인
        (1, 0, 2) => r.tanh(),            // 쌍곡 탄젠트
        (1, 0, 3) => r.sinh() / r.cosh(), // 비율 함수

        // 수정된 쌍곡 (SUB=1)
        (1, 1, 0) => 2.0 * (r / 2.0).sinh(), // 스케일된 sinh
        (1, 1, 1) => (r / 2.0).cosh(),       // 반각 cosh
        (1, 1, 2) => r.exp() - 1.0,          // 지수 이동
        (1, 1, 3) => 1.0 - (-r).exp(),       // 포화 지수

        // 확장된 쌍곡 (SUB=2)
        (1, 2, 0) => r * r.cosh(), // r-가중 cosh
        (1, 2, 1) => r * r.sinh(), // r-가중 sinh
        (1, 2, 2) => {
            if r.abs() < 1e-6 {
                0.0
            } else {
                (r.sinh() - r) / (r * r)
            }
        } // 고차 보정
        (1, 2, 3) => {
            if r.abs() < 1e-6 {
                0.0
            } else {
                (r.cosh() - 1.0 - r * r / 2.0) / r.powi(3)
            }
        } // 3차 보정

        // 특수 Lorentz (SUB=3)
        (1, 3, 0) => r.cosh().sqrt(),                    // 제곱근 cosh
        (1, 3, 1) => r.signum() * r.sinh().abs().sqrt(), // 제곱근 sinh
        (1, 3, 2) => (r * r).tanh(),                     // 제곱 인수
        (1, 3, 3) => {
            if r.abs() < 1e-6 {
                1.0
            } else {
                r.tanh() / r
            }
        } // 정규화된 tanh

        // ========== Category 2: Klein 기하학 ==========

        // 기본 Klein (SUB=0)
        (2, 0, 0) => r / (1.0 + r),            // 유계 선형
        (2, 0, 1) => r / (1.0 + r * r).sqrt(), // 정규화
        (2, 0, 2) => r * r / (1.0 + r * r),    // 제곱 유계
        (2, 0, 3) => 1.0 - 1.0 / (1.0 + r),    // 역 유계

        // 투영 함수 (SUB=1)
        (2, 1, 0) => 2.0 * r / (1.0 + r * r),       // 원형 투영
        (2, 1, 1) => (1.0 - r * r) / (1.0 + r * r), // 코사인 유사
        (2, 1, 2) => 4.0 * r / (1.0 + r * r).powi(2), // 이중 투영
        (2, 1, 3) => 2.0 * r.atan() / PI,           // 각도 정규화

        // 멱급수 근사 (SUB=2)
        (2, 2, 0) => r / (1.0 + r + r * r / 2.0), // 3차 근사
        (2, 2, 1) => r / (1.0 + r - r * r / 2.0), // 교대 급수
        (2, 2, 2) => r - r.powi(3) / 3.0 + r.powi(5) / 5.0, // 절단 급수
        (2, 2, 3) => r / (1.0 + r * r / 3.0),     // 대각화 근사

        // 변분 함수 (SUB=3)
        (2, 3, 0) => r / (1.0 + r.abs()),         // 절댓값 정규화
        (2, 3, 1) => r.signum() * r.abs().sqrt(), // 제곱근 변형
        (2, 3, 2) => r.powi(3) / (1.0 + r * r),   // 3차 변형
        (2, 3, 3) => r * (-r * r / 2.0).exp(),    // 가우시안 가중

        // ========== Category 3: 특수 함수 ==========

        // Bessel 유사 함수 (SUB=0)
        (3, 0, 0) => bessel_j0_approx(r) * 0.5 + bessel_j1_approx(r) * 0.5, // 베셀 혼합
        (3, 0, 1) => {
            if r.abs() < 1e-6 {
                1.0
            } else {
                (r.sin() / r) * (r / 2.0).cos()
            }
        } // 변조 sinc
        (3, 0, 2) => {
            if r.abs() < 1e-6 {
                0.0
            } else {
                2.0 * bessel_j1_approx(r) / r
            }
        } // 베셀 1차 정규화
        (3, 0, 3) => bessel_j0_approx(r) * r.cos(),                         // 베셀-코사인 곱

        // Gaussian 유사 함수 (SUB=1)
        (3, 1, 0) => (-r * r / 2.0).exp(),     // 가우시안
        (3, 1, 1) => r * (-r * r / 2.0).exp(), // 가우시안 1차
        (3, 1, 2) => (1.0 - r * r) * (-r * r / 2.0).exp(), // 가우시안 2차
        (3, 1, 3) => 1.0 / (1.0 + r * r).sqrt(), // 역 제곱근

        // 주기적 변조 (SUB=2)
        (3, 2, 0) => r.cos() * (-r * r / 4.0).exp(), // 가우시안 변조 코사인
        (3, 2, 1) => r.sin() * (-r * r / 4.0).exp(), // 가우시안 변조 사인
        (3, 2, 2) => (r * r).cos(),                  // 프레넬 코사인
        (3, 2, 3) => (r * r).sin(),                  // 프레넬 사인

        // 실험적 함수 (SUB=3)
        (3, 3, 0) => {
            if r.abs() < 1e-6 {
                1.0
            } else {
                r.tan() / r
            }
        } // 정규화된 탄젠트
        (3, 3, 1) => {
            if r.abs() < 1e-6 {
                0.0
            } else {
                r.sin() * r.cos() / r
            }
        } // 이중 주기 정규화
        (3, 3, 2) => {
            if r.abs() < 1e-6 {
                0.5
            } else {
                (1.0 - r.cos()) / (r * r.sin())
            }
        } // 복합 함수
        (3, 3, 3) => {
            if r.abs() < 1e-5 {
                // r이 작을 때 테일러 급수: ln(1+r²)/r ≈ r - r³/3 + r⁵/5
                r - r.powi(3) / 3.0
            } else {
                (1.0 + r * r).ln() / r
            }
        } // 로그 제곱 정규화

        // 기본값 (정의되지 않은 조합)
        _ => r.tanh() / 2.0,
    }
}

/// Bessel J0 함수의 근사
/// Taylor 급수를 사용한 간단한 근사
fn bessel_j0_approx(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        let x8 = x4 * x4;

        1.0 - x2 / 4.0 + x4 / 64.0 - x6 / 2304.0 + x8 / 147456.0
    }
}

/// Bessel J1 함수의 근사
/// Taylor 급수를 사용한 간단한 근사
fn bessel_j1_approx(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        x / 2.0
    } else {
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        let x7 = x5 * x2;

        x / 2.0 - x3 / 16.0 + x5 / 384.0 - x7 / 18432.0
    }
}

/// 함수 이름 반환 (디버깅용)
pub fn get_function_name(cat: u8, sub: u8, d: u8) -> &'static str {
    match (cat, sub, d) {
        // Poincaré
        (0, 0, 0) => "tanh(r)/2",
        (0, 0, 1) => "-tanh(r)/2",
        (0, 0, 2) => "2*tanh(r/4)",
        (0, 0, 3) => "tanh²(r/2)",

        (0, 1, 0) => "sinh(r)/(1+cosh(r))",
        (0, 1, 1) => "(cosh(r)-1)/(1+cosh(r))",
        (0, 1, 2) => "tanh(r)",
        (0, 1, 3) => "sinh(r)/r",

        (0, 2, 0) => "sin(r)/r",
        (0, 2, 1) => "cos(r)",
        (0, 2, 2) => "(1-cos(r))/r²",
        (0, 2, 3) => "sin(r)cos(r)",

        (0, 3, 0) => "(e^r-1)/r",
        (0, 3, 1) => "e^r/(1+e^r)",
        (0, 3, 2) => "ln(r+1)/r",
        (0, 3, 3) => "r/(1+|r|)",

        // Lorentz
        (1, 0, 0) => "sinh(r)",
        (1, 0, 1) => "cosh(r)-1",
        (1, 0, 2) => "tanh(r)",
        (1, 0, 3) => "sinh(r)/cosh(r)",

        (1, 1, 0) => "2*sinh(r/2)",
        (1, 1, 1) => "cosh(r/2)",
        (1, 1, 2) => "e^r-1",
        (1, 1, 3) => "1-e^(-r)",

        (1, 2, 0) => "r*cosh(r)",
        (1, 2, 1) => "r*sinh(r)",
        (1, 2, 2) => "(sinh(r)-r)/r²",
        (1, 2, 3) => "(cosh(r)-1-r²/2)/r³",

        (1, 3, 0) => "√cosh(r)",
        (1, 3, 1) => "sgn(r)√|sinh(r)|",
        (1, 3, 2) => "tanh(r²)",
        (1, 3, 3) => "tanh(r)/r",

        // Klein
        (2, 0, 0) => "r/(1+r)",
        (2, 0, 1) => "r/√(1+r²)",
        (2, 0, 2) => "r²/(1+r²)",
        (2, 0, 3) => "1-1/(1+r)",

        (2, 1, 0) => "2r/(1+r²)",
        (2, 1, 1) => "(1-r²)/(1+r²)",
        (2, 1, 2) => "4r/(1+r²)²",
        (2, 1, 3) => "2*atan(r)/π",

        (2, 2, 0) => "r/(1+r+r²/2)",
        (2, 2, 1) => "r/(1+r-r²/2)",
        (2, 2, 2) => "r-r³/3+r⁵/5",
        (2, 2, 3) => "r/(1+r²/3)",

        (2, 3, 0) => "r/(1+|r|)",
        (2, 3, 1) => "sgn(r)√|r|",
        (2, 3, 2) => "r³/(1+r²)",
        (2, 3, 3) => "r*exp(-r²/2)",

        // Special
        (3, 0, 0) => "(J₀(r)+J₁(r))/2",
        (3, 0, 1) => "sinc(r)*cos(r/2)",
        (3, 0, 2) => "2J₁(r)/r",
        (3, 0, 3) => "J₀(r)*cos(r)",

        (3, 1, 0) => "exp(-r²/2)",
        (3, 1, 1) => "r*exp(-r²/2)",
        (3, 1, 2) => "(1-r²)*exp(-r²/2)",
        (3, 1, 3) => "1/√(1+r²)",

        (3, 2, 0) => "cos(r)*exp(-r²/4)",
        (3, 2, 1) => "sin(r)*exp(-r²/4)",
        (3, 2, 2) => "cos(r²)",
        (3, 2, 3) => "sin(r²)",

        (3, 3, 0) => "tan(r)/r",
        (3, 3, 1) => "sin(r)cos(r)/r",
        (3, 3, 2) => "(1-cos(r))/(r*sin(r))",
        (3, 3, 3) => "ln(1+r²)/r",

        _ => "default: tanh(r)/2",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_함수_연속성() {
        // r=0 근처에서 연속성 테스트
        for cat in 0..4 {
            for sub in 0..4 {
                for d in 0..4 {
                    let eps = 1e-7;
                    let f_0 = get_riemannian_function(cat, sub, d, 0.0);
                    let f_eps = get_riemannian_function(cat, sub, d, eps);

                    // NaN이 아니어야 함
                    assert!(!f_0.is_nan(), "f({},{},{},0) = NaN", cat, sub, d);
                    assert!(!f_eps.is_nan(), "f({},{},{},{}) = NaN", cat, sub, d, eps);

                    // 0 근처에서 급격한 변화가 없어야 함
                    let diff = (f_0 - f_eps).abs();
                    assert!(
                        diff < 0.1,
                        "불연속: f({},{},{},0)={} vs f({},{},{},{})={}",
                        cat,
                        sub,
                        d,
                        f_0,
                        cat,
                        sub,
                        d,
                        eps,
                        f_eps
                    );
                }
            }
        }
    }

    #[test]
    fn test_함수_범위() {
        // 일반적인 r 값에서 함수값이 합리적인 범위에 있는지 테스트
        let test_r = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0];

        for &r in &test_r {
            for cat in 0..4 {
                for sub in 0..4 {
                    for d in 0..4 {
                        let f = get_riemannian_function(cat, sub, d, r);

                        // NaN이나 무한대가 아니어야 함
                        assert!(!f.is_nan(), "f({},{},{},{}) = NaN", cat, sub, d, r);
                        assert!(!f.is_infinite(), "f({},{},{},{}) = inf", cat, sub, d, r);

                        // 자연스럽게 큰 값을 가질 수 있는 함수들
                        let naturally_unbounded = matches!(
                            (cat, sub, d),
                            (1, 0, 0) | // sinh
                            (1, 0, 1) | // cosh - 1
                            (1, 1, 0) | // 2*sinh(r/2)
                            (1, 1, 2) | // exp(r) - 1
                            (1, 2, 0) | // r*cosh(r) 
                            (1, 2, 1) | // r*sinh(r)
                            (2, 2, 2) | // r - r³/3 + r⁵/5
                            (2, 3, 0) | // r^2
                            (2, 3, 1) | // r^3
                            (3, 2, 0) | // exp(r)
                            (3, 2, 1) // exp(2r)
                        );

                        if !naturally_unbounded {
                            // 합리적인 범위 (-100, 100) 내에 있어야 함
                            assert!(
                                f.abs() < 100.0,
                                "f({},{},{},{}) = {} 너무 큼",
                                cat,
                                sub,
                                d,
                                r,
                                f
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_특정_함수_동작() {
        // 몇 가지 알려진 함수값 테스트
        let eps = 1e-5;

        // tanh(0)/2 = 0
        assert!((get_riemannian_function(0, 0, 0, 0.0) - 0.0).abs() < eps);

        // cos(0) = 1
        assert!((get_riemannian_function(0, 2, 1, 0.0) - 1.0).abs() < eps);

        // sinh(0) = 0
        assert!((get_riemannian_function(1, 0, 0, 0.0) - 0.0).abs() < eps);

        // cosh(0) - 1 = 0
        assert!((get_riemannian_function(1, 0, 1, 0.0) - 0.0).abs() < eps);

        // exp(-0²/2) = 1
        assert!((get_riemannian_function(3, 1, 0, 0.0) - 1.0).abs() < eps);
    }
}
