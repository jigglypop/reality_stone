//! `math.rs`에 대한 단위 테스트

use poincare_layer::math::*;
use std::f32::consts::PI;
use approx::assert_relative_eq;

#[test]
fn test_rotation_angle() {
    println!("\n--- Test: Rotation Angle Calculation ---");
    assert_relative_eq!(get_rotation_angle(0), 0.0);
    assert_relative_eq!(get_rotation_angle(3), PI / 4.0);
    assert_relative_eq!(get_rotation_angle(5), PI / 2.0);
    assert_relative_eq!(get_rotation_angle(15), 0.0); // Out of defined range
    println!("  [PASSED] get_rotation_angle works correctly.");
}

#[test]
fn test_angular_derivatives() {
    println!("\n--- Test: Angular Derivative Cycles ---");
    let theta = PI / 6.0; // 30 degrees

    // sin-based (basis_id is even)
    assert_relative_eq!(apply_angular_derivative(theta, 0, 0), theta.sin());
    assert_relative_eq!(apply_angular_derivative(theta, 1, 0), theta.cos());
    assert_relative_eq!(apply_angular_derivative(theta, 2, 0), -theta.sin());
    assert_relative_eq!(apply_angular_derivative(theta, 3, 0), -theta.cos());
    assert_relative_eq!(apply_angular_derivative(theta, 4, 0), theta.sin()); // 4-cycle

    // cos-based (basis_id is odd)
    assert_relative_eq!(apply_angular_derivative(theta, 0, 1), theta.cos());
    assert_relative_eq!(apply_angular_derivative(theta, 1, 1), -theta.sin());
    assert_relative_eq!(apply_angular_derivative(theta, 2, 1), -theta.cos());
    assert_relative_eq!(apply_angular_derivative(theta, 3, 1), theta.sin());
    println!("  [PASSED] apply_angular_derivative cycles are correct.");
}

#[test]
fn test_radial_derivatives() {
    println!("\n--- Test: Radial Derivative Cycles ---");
    let r = 0.5;

    // sinh-based (basis_id is 0, 1, 4, 5, ...)
    assert_relative_eq!(apply_radial_derivative(r, false, 0), r.sinh());
    assert_relative_eq!(apply_radial_derivative(r, true, 0), r.cosh());

    // cosh-based (basis_id is 2, 3, 6, 7, ...)
    assert_relative_eq!(apply_radial_derivative(r, false, 2), r.cosh());
    assert_relative_eq!(apply_radial_derivative(r, true, 2), r.sinh());
    println!("  [PASSED] apply_radial_derivative works correctly.");
}

#[test]
fn test_wave_functions() {
    println!("\n--- Test: Wave Functions ---");
    let x = 1.5;

    // sech
    assert_relative_eq!(sech(x), 1.0 / x.cosh());

    // triangle_wave (test a few points)
    assert_relative_eq!(triangle_wave(0.0), -1.0);
    assert_relative_eq!(triangle_wave(PI / 2.0), 1.0); // t=0.5
    assert_relative_eq!(triangle_wave(PI), -1.0);     // t=1.0 -> 0.0
    assert_relative_eq!(triangle_wave(3.0 * PI / 2.0), 1.0);

    println!("  [PASSED] sech and triangle_wave work correctly.");
} 