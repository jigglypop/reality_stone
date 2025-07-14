//! 리만 기하학 함수 테스트

use crate::bitfield::riemannian::{get_function_name, get_riemannian_function};

#[test]
fn test_모든_함수_정의() {
    // 64개 모든 함수가 정의되었는지 확인
    let mut count = 0;
    for cat in 0..4 {
        for sub in 0..4 {
            for d in 0..4 {
                let name = get_function_name(cat, sub, d);
                assert!(
                    !name.starts_with("default"),
                    "함수 ({},{},{})가 정의되지 않음",
                    cat,
                    sub,
                    d
                );
                count += 1;
            }
        }
    }
    assert_eq!(count, 64, "정확히 64개의 함수가 정의되어야 함");
}

#[test]
fn test_poincare_기본_함수() {
    // Poincaré 기본 함수들 테스트
    let eps = 1e-5;

    // tanh(r)/2
    assert!((get_riemannian_function(0, 0, 0, 0.0) - 0.0).abs() < eps);
    assert!((get_riemannian_function(0, 0, 0, 1.0) - 0.5 * 1.0_f32.tanh()).abs() < eps);

    // -tanh(r)/2
    assert!((get_riemannian_function(0, 0, 1, 1.0) - (-0.5 * 1.0_f32.tanh())).abs() < eps);

    // 2*tanh(r/4)
    assert!((get_riemannian_function(0, 0, 2, 0.0) - 0.0).abs() < eps);
    assert!((get_riemannian_function(0, 0, 2, 4.0) - 2.0 * 1.0_f32.tanh()).abs() < eps);

    // tanh²(r/2)
    let tanh_half = (0.5_f32).tanh();
    assert!((get_riemannian_function(0, 0, 3, 1.0) - tanh_half * tanh_half).abs() < eps);
}

#[test]
fn test_삼각함수_특이점() {
    // r=0 근처의 특이점 처리 확인
    let eps = 1e-5;

    // sin(r)/r → 1 as r→0
    assert!((get_riemannian_function(0, 2, 0, 0.0) - 1.0).abs() < eps);
    assert!((get_riemannian_function(0, 2, 0, 1e-7) - 1.0).abs() < 0.01);

    // (1-cos(r))/r² → 1/2 as r→0
    assert!((get_riemannian_function(0, 2, 2, 0.0) - 0.5).abs() < eps);
    assert!((get_riemannian_function(0, 2, 2, 1e-6) - 0.5).abs() < 0.01);
}

#[test]
fn test_lorentz_함수() {
    let eps = 1e-5;

    // sinh(r)
    assert!((get_riemannian_function(1, 0, 0, 0.0) - 0.0).abs() < eps);
    assert!((get_riemannian_function(1, 0, 0, 1.0) - 1.0_f32.sinh()).abs() < eps);

    // cosh(r) - 1
    assert!((get_riemannian_function(1, 0, 1, 0.0) - 0.0).abs() < eps);
    assert!((get_riemannian_function(1, 0, 1, 1.0) - (1.0_f32.cosh() - 1.0)).abs() < eps);

    // 제곱근 cosh(r)
    assert!((get_riemannian_function(1, 3, 0, 0.0) - 1.0).abs() < eps);
    assert!((get_riemannian_function(1, 3, 0, 1.0) - 1.0_f32.cosh().sqrt()).abs() < eps);
}

#[test]
fn test_klein_함수() {
    let eps = 1e-5;

    // r/(1+r)
    assert!((get_riemannian_function(2, 0, 0, 0.0) - 0.0).abs() < eps);
    assert!((get_riemannian_function(2, 0, 0, 1.0) - 0.5).abs() < eps);
    assert!((get_riemannian_function(2, 0, 0, 2.0) - 2.0 / 3.0).abs() < eps);

    // r/√(1+r²)
    assert!((get_riemannian_function(2, 0, 1, 0.0) - 0.0).abs() < eps);
    assert!((get_riemannian_function(2, 0, 1, 1.0) - 1.0 / (2.0_f32.sqrt())).abs() < eps);

    // 2r/(1+r²)
    assert!((get_riemannian_function(2, 1, 0, 0.0) - 0.0).abs() < eps);
    assert!((get_riemannian_function(2, 1, 0, 1.0) - 1.0).abs() < eps);
}

#[test]
fn test_특수_함수() {
    let eps = 1e-5;

    // exp(-r²/2) - 가우시안
    assert!((get_riemannian_function(3, 1, 0, 0.0) - 1.0).abs() < eps);
    assert!((get_riemannian_function(3, 1, 0, 1.0) - (-0.5_f32).exp()).abs() < eps);

    // 1/√(1+r²)
    assert!((get_riemannian_function(3, 1, 3, 0.0) - 1.0).abs() < eps);
    assert!((get_riemannian_function(3, 1, 3, 1.0) - 1.0 / (2.0_f32.sqrt())).abs() < eps);

    // cos(r²) - 프레넬 코사인
    assert!((get_riemannian_function(3, 2, 2, 0.0) - 1.0).abs() < eps);
    assert!((get_riemannian_function(3, 2, 2, 1.0) - 1.0_f32.cos()).abs() < eps);
}

#[test]
fn test_함수_유계성() {
    // 모든 함수가 합리적인 범위 내의 값을 반환하는지 확인
    let test_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    for &r in &test_values {
        for cat in 0..4 {
            for sub in 0..4 {
                for d in 0..4 {
                    let value = get_riemannian_function(cat, sub, d, r);

                    // NaN이나 무한대가 아니어야 함
                    assert!(!value.is_nan(), "f({},{},{},{}) = NaN", cat, sub, d, r);
                    assert!(!value.is_infinite(), "f({},{},{},{}) = inf", cat, sub, d, r);

                    // 일부 발산하는 함수들은 유계성 테스트에서 제외
                    let is_unbounded = (cat == 1) || // Lorentz 함수 대부분 발산
                                     (cat == 0 && sub == 1 && d == 3) || // sinh(r)/r
                                     (cat == 0 && sub == 3 && d == 0) || // (exp(r)-1)/r
                                     (cat == 0 && sub == 3 && d == 2) || // ln(1+r)/r
                                     (cat == 2 && sub == 2 && d == 2) || // 멱급수 근사는 발산
                                     (cat == 3 && sub == 0); // Bessel 함수 근사는 발산

                    if !is_unbounded {
                        assert!(
                            value.abs() < 100.0,
                            "f({},{},{},{}) = {} 너무 큼",
                            cat,
                            sub,
                            d,
                            r,
                            value
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_대칭성() {
    let eps = 1e-5;

    // 일부 함수는 홀함수 (f(-r) = -f(r))
    let odd_functions = [
        (0, 0, 0), // tanh(r)/2
        (0, 0, 1), // -tanh(r)/2
        (1, 0, 0), // sinh(r)
        (0, 1, 2), // tanh(r)
    ];

    for &(cat, sub, d) in &odd_functions {
        for r in [0.5, 1.0, 2.0] {
            let f_pos = get_riemannian_function(cat, sub, d, r);
            let f_neg = get_riemannian_function(cat, sub, d, -r);
            assert!(
                (f_pos + f_neg).abs() < eps,
                "함수 ({},{},{})가 홀함수가 아님: f({})={}, f({})={}",
                cat,
                sub,
                d,
                r,
                f_pos,
                -r,
                f_neg
            );
        }
    }

    // 일부 함수는 짝함수 (f(-r) = f(r))
    let even_functions = [
        (0, 2, 1), // cos(r)
        (1, 0, 1), // cosh(r)-1
        (3, 1, 0), // exp(-r²/2)
    ];

    for &(cat, sub, d) in &even_functions {
        for r in [0.5, 1.0, 2.0] {
            let f_pos = get_riemannian_function(cat, sub, d, r);
            let f_neg = get_riemannian_function(cat, sub, d, -r);
            assert!(
                (f_pos - f_neg).abs() < eps,
                "함수 ({},{},{})가 짝함수가 아님: f({})={}, f({})={}",
                cat,
                sub,
                d,
                r,
                f_pos,
                -r,
                f_neg
            );
        }
    }
}

#[test]
fn test_단조성() {
    // 일부 함수는 단조증가
    let monotonic_functions = [
        (0, 0, 0), // tanh(r)/2
        (0, 1, 2), // tanh(r)
        (2, 0, 0), // r/(1+r)
    ];

    for &(cat, sub, d) in &monotonic_functions {
        let r_values = [0.0, 0.5, 1.0, 1.5, 2.0];
        for i in 1..r_values.len() {
            let f_prev = get_riemannian_function(cat, sub, d, r_values[i - 1]);
            let f_curr = get_riemannian_function(cat, sub, d, r_values[i]);
            assert!(
                f_curr >= f_prev,
                "함수 ({},{},{})가 단조증가하지 않음: f({})={} > f({})={}",
                cat,
                sub,
                d,
                r_values[i - 1],
                f_prev,
                r_values[i],
                f_curr
            );
        }
    }
}
