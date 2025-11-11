use crate::layers::lorentz;
use ndarray::{arr2, Array2};

fn to_lorentz_coords(space: &Array2<f32>, c: f32) -> Array2<f32> {
    // x0 = sqrt(1/c + ||x||^2)
    let mut out = Array2::<f32>::zeros((space.nrows(), space.ncols() + 1));
    for i in 0..space.nrows() {
        let mut norm_sq = 0.0f32;
        for j in 0..space.ncols() {
            let v = space[[i, j]];
            norm_sq += v * v;
            out[[i, j + 1]] = v;
        }
        out[[i, 0]] = (1.0 / c + norm_sq).sqrt();
    }
    out
}

fn sum_forward(u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> f32 {
    let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
    y.sum()
}

#[test]
fn lorentz_distance_self_zero() {
    let c = 1.0_f32;
    // build valid hyperboloid point from space coords
    let sp = arr2(&[[0.1_f32, 0.2]]);
    let x = to_lorentz_coords(&sp, c);
    let d = lorentz::lorentz_distance(&x.view(), &x.view(), c);
    assert!(d[0] >= 0.0 && d[0] <= 1e-3);
}

#[test]
fn lorentz_layer_forward_shapes_and_finiteness() {
    let c = 0.8_f32;
    let t = 0.3_f32;
    let usp = arr2(&[[0.1_f32, 0.2], [-0.1, 0.05]]);
    let vsp = arr2(&[[-0.05_f32, 0.1], [0.02, -0.03]]);
    let u = to_lorentz_coords(&usp, c);
    let v = to_lorentz_coords(&vsp, c);
    let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
    assert_eq!(y.dim(), u.dim());
    assert!(y.iter().all(|z| z.is_finite()));
}

#[test]
fn lorentz_layer_backward_matches_numeric_small() {
    let c = 1.0_f32;
    let t = 0.3_f32;
    let usp = arr2(&[[0.1_f32, 0.2]]);
    let vsp = arr2(&[[-0.05_f32, 0.1]]);
    let mut u = to_lorentz_coords(&usp, c);
    let mut v = to_lorentz_coords(&vsp, c);

    let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
    let grad_output = Array2::<f32>::ones(y.dim());
    let (gu, gv) = lorentz::lorentz_layer_backward(&grad_output.view(), &u.view(), &v.view(), c, t);

    let eps = 1e-4f32;
    let mut gu_num = Array2::<f32>::zeros(u.dim());
    for j in 0..u.ncols() {
        let orig = u[(0, j)];
        u[(0, j)] = orig + eps;
        let f_plus = sum_forward(&u, &v, c, t);
        u[(0, j)] = orig - eps;
        let f_minus = sum_forward(&u, &v, c, t);
        u[(0, j)] = orig;
        gu_num[(0, j)] = (f_plus - f_minus) / (2.0 * eps);
    }
    let mut gv_num = Array2::<f32>::zeros(v.dim());
    for j in 0..v.ncols() {
        let orig = v[(0, j)];
        v[(0, j)] = orig + eps;
        let f_plus = sum_forward(&u, &v, c, t);
        v[(0, j)] = orig - eps;
        let f_minus = sum_forward(&u, &v, c, t);
        v[(0, j)] = orig;
        gv_num[(0, j)] = (f_plus - f_minus) / (2.0 * eps);
    }

    let tol = 1e-1f32; // relax due to nonlinearity and atanh/acosh numerical sensitivity
    for j in 0..u.ncols() {
        assert!((gu[(0, j)] - gu_num[(0, j)]).abs() <= tol);
        assert!((gv[(0, j)] - gv_num[(0, j)]).abs() <= tol);
    }
}


