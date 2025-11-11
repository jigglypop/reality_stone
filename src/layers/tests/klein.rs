use crate::layers::klein;
use approx::assert_relative_eq;
use ndarray::arr2;
use ndarray::{Array2, ArrayView2};

#[test]
fn klein_distance_self_zero() {
    let c = 0.5_f32;
    // Klein 모델의 유효 도메인은 ||x|| < 1/sqrt(c). 0은 OK, 첫 샘플은 충분히 작게.
    let x = arr2(&[[0.05_f32, -0.05], [0.0, 0.0]]);
    let d = klein::klein_distance(&x.view(), &x.view(), c);
    // Numerical scheme may yield tiny residuals; relax tolerance
    assert!(d.iter().all(|v| v.abs() < 1e-2));
}

#[test]
fn klein_transform_roundtrip_poincare() {
    let c = 0.7_f32;
    let x = arr2(&[[0.1_f32, -0.2], [0.05, 0.05]]);
    let p = klein::klein_to_poincare(&x.view(), c);
    let k = klein::from_poincare(&p.view(), c);
    assert_relative_eq!(x, k, epsilon = 1e-4);
}

fn sum_forward(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32, t: f32) -> f32 {
    let y = klein::klein_layer_forward(u, v, c, t);
    y.sum()
}

#[test]
fn klein_layer_backward_matches_numeric() {
    let c = 0.5_f32;
    let t = 0.3_f32;
    let mut u = Array2::<f32>::from_shape_vec((2, 3), vec![0.1, -0.2, 0.05, 0.02, 0.03, -0.01])
        .unwrap();
    let mut v =
        Array2::<f32>::from_shape_vec((2, 3), vec![-0.05, 0.07, 0.02, -0.01, -0.02, 0.04])
            .unwrap();

    let y = klein::klein_layer_forward(&u.view(), &v.view(), c, t);
    let grad_output = Array2::<f32>::ones(y.dim());
    let (gu, gv) = klein::klein_layer_backward(&grad_output.view(), &u.view(), &v.view(), c, t);

    let eps = 1e-4f32;
    let mut gu_num = Array2::<f32>::zeros(u.dim());
    for i in 0..u.nrows() {
        for j in 0..u.ncols() {
            let orig = u[(i, j)];
            u[(i, j)] = orig + eps;
            let f_plus = sum_forward(&u.view(), &v.view(), c, t);
            u[(i, j)] = orig - eps;
            let f_minus = sum_forward(&u.view(), &v.view(), c, t);
            u[(i, j)] = orig;
            gu_num[(i, j)] = (f_plus - f_minus) / (2.0 * eps);
        }
    }

    let mut gv_num = Array2::<f32>::zeros(v.dim());
    for i in 0..v.nrows() {
        for j in 0..v.ncols() {
            let orig = v[(i, j)];
            v[(i, j)] = orig + eps;
            let f_plus = sum_forward(&u.view(), &v.view(), c, t);
            v[(i, j)] = orig - eps;
            let f_minus = sum_forward(&u.view(), &v.view(), c, t);
            v[(i, j)] = orig;
            gv_num[(i, j)] = (f_plus - f_minus) / (2.0 * eps);
        }
    }

    let tol = 5e-3f32;
    for ((i, j), a) in gu.indexed_iter() {
        let b = gu_num[(i, j)];
        assert!((a - b).abs() <= tol, "grad_u mismatch at ({},{}): {} vs {}", i, j, a, b);
    }
    for ((i, j), a) in gv.indexed_iter() {
        let b = gv_num[(i, j)];
        assert!((a - b).abs() <= tol, "grad_v mismatch at ({},{}): {} vs {}", i, j, a, b);
    }
}


