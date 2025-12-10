use _rust::layers::spline_cache::SplineCache;
use ndarray::arr1;

#[test]
fn test_spline_cache_initialization() {
    let cache = SplineCache::new(0.1, 4);
    assert_eq!(cache.control_points.len(), 0);
    assert_eq!(cache.curvature, 0.1);
    assert_eq!(cache.dimension, 4);
}

#[test]
fn test_add_point() {
    let mut cache = SplineCache::new(0.0, 2);
    cache.add_point(0.0, arr1(&[1.0, 2.0]).view(), arr1(&[0.1, 0.2]).view());
    assert_eq!(cache.control_points.len(), 1);
    assert_eq!(cache.control_points[0].time, 0.0);
    assert_eq!(cache.control_points[0].state[0], 1.0);
}

#[test]
fn test_reconstruct_exact_points() {
    let mut cache = SplineCache::new(0.0, 2);
    cache.add_point(0.0, arr1(&[1.0, 1.0]).view(), arr1(&[0.0, 0.0]).view());
    cache.add_point(1.0, arr1(&[2.0, 2.0]).view(), arr1(&[0.0, 0.0]).view());

    let p0 = cache.reconstruct(0.0).unwrap();
    assert!((p0[0] - 1.0).abs() < 1e-5);

    let p1 = cache.reconstruct(1.0).unwrap();
    assert!((p1[0] - 2.0).abs() < 1e-5);
}

#[test]
fn test_reconstruct_interpolation() {
    let mut cache = SplineCache::new(0.0, 1);
    // Linear motion from 0 to 1 over time 0 to 1. Velocity 1.
    cache.add_point(0.0, arr1(&[0.0]).view(), arr1(&[1.0]).view());
    cache.add_point(1.0, arr1(&[1.0]).view(), arr1(&[1.0]).view());

    let mid = cache.reconstruct(0.5).unwrap();
    // Should be 0.5 exactly for cubic hermite with consistent velocity
    assert!((mid[0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_reconstruct_curvature() {
    let mut cache = SplineCache::new(1.0, 1); // Positive curvature
                                              // Linear motion base
    cache.add_point(0.0, arr1(&[0.0]).view(), arr1(&[1.0]).view());
    cache.add_point(1.0, arr1(&[1.0]).view(), arr1(&[1.0]).view());

    let mid = cache.reconstruct(0.5).unwrap();
    // With +1 curvature, correction is u(1-u)*k = 0.25 * 1 = 0.25
    // value = 0.5 * (1 + 0.25) = 0.625
    assert!((mid[0] - 0.625).abs() < 1e-5);
}

#[test]
fn test_batch_reconstruct() {
    let mut cache = SplineCache::new(0.0, 1);
    cache.add_point(0.0, arr1(&[0.0]).view(), arr1(&[1.0]).view());
    cache.add_point(1.0, arr1(&[1.0]).view(), arr1(&[1.0]).view());

    let times = arr1(&[0.0, 0.5, 1.0]);
    let batch = cache.batch_reconstruct(times.view());

    assert_eq!(batch.shape(), &[3, 1]);
    assert!((batch[[0, 0]] - 0.0).abs() < 1e-5);
    assert!((batch[[1, 0]] - 0.5).abs() < 1e-5);
    assert!((batch[[2, 0]] - 1.0).abs() < 1e-5);
}

#[test]
#[should_panic]
fn test_dimension_mismatch() {
    let mut cache = SplineCache::new(0.0, 2);
    cache.add_point(0.0, arr1(&[1.0]).view(), arr1(&[0.1]).view());
}
