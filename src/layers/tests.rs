#[cfg(test)]
mod klein_tests {
    use crate::layers::klein;
    use approx::assert_relative_eq;
    use ndarray::arr2;
    use ndarray::{Array2, ArrayView2};

    #[test]
    fn klein_distance_self_zero() {
        let c = 0.5_f32;
        let x = arr2(&[[0.1_f32, -0.2], [0.0, 0.0]]);
        let d = klein::klein_distance(&x.view(), &x.view(), c);
        assert!(d.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn klein_transform_roundtrip_poincare() {
        let c = 0.7_f32;
        let x = arr2(&[[0.1_f32, -0.2], [0.05, 0.05]]);
        let p = klein::klein_to_poincare(&x.view(), c);
        // map back via poincare->klein implemented in klein::from_poincare
        let k = klein::from_poincare(&p.view(), c);
        assert_relative_eq!(x, k, epsilon = 1e-4);
    }
}

#[cfg(test)]
mod klein_grad_tests {
    use crate::layers::klein;
    use ndarray::{Array2, ArrayView2};

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

        // Analytical grads via backward
        let y = klein::klein_layer_forward(&u.view(), &v.view(), c, t);
        let grad_output = Array2::<f32>::ones(y.dim());
        let (gu, gv) = klein::klein_layer_backward(&grad_output.view(), &u.view(), &v.view(), c, t);

        // Numeric grads
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

        // Allow moderate tolerance due to nonlinearity and clamping
        let tol = 5e-3f32;
        for ((i, j), a) in gu.indexed_iter() {
            let b = gu_num[(i, j)];
            assert!(
                (a - b).abs() <= tol,
                "grad_u mismatch at ({},{}) ana={} num={}",
                i,
                j,
                a,
                b
            );
        }
        for ((i, j), a) in gv.indexed_iter() {
            let b = gv_num[(i, j)];
            assert!(
                (a - b).abs() <= tol,
                "grad_v mismatch at ({},{}) ana={} num={}",
                i,
                j,
                a,
                b
            );
        }
    }
}

#[cfg(test)]
mod lorentz_tests {
    use crate::layers::lorentz;
    use approx::assert_relative_eq;
    use ndarray::arr2;
    use ndarray::Array2;

    #[test]
    fn lorentz_distance_self_zero() {
        let c = 1.0_f32;
        // On hyperboloid: x0^2 - ||x||^2 = 1/c
        let x = arr2(&[[1.5_f32, 0.1, 0.2]]);
        let d = lorentz::lorentz_distance(&x.view(), &x.view(), c);
        assert!(d[0] >= 0.0 && d[0] <= 1e-4);
    }

    #[test]
    fn lorentz_layer_forward_shapes_and_finiteness() {
        let c = 0.8_f32;
        let t = 0.3_f32;
        let u = arr2(&[[1.5_f32, 0.1, 0.2], [1.2, -0.1, 0.05]]);
        let v = arr2(&[[1.3_f32, -0.05, 0.1], [1.1, 0.02, -0.03]]);
        let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
        assert_eq!(y.dim(), u.dim());
        assert!(y.iter().all(|z| z.is_finite()));
    }
}

#[cfg(test)]
mod lorentz_grad_tests {
    use crate::layers::lorentz;
    use ndarray::Array2;

    fn sum_forward(u: &Array2<f32>, v: &Array2<f32>, c: f32, t: f32) -> f32 {
        let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
        y.sum()
    }

    #[test]
    fn lorentz_layer_backward_matches_numeric_small() {
        let c = 1.0_f32;
        let t = 0.3_f32;
        let mut u = Array2::<f32>::from_shape_vec((1, 3), vec![1.5, 0.1, 0.2]).unwrap();
        let mut v = Array2::<f32>::from_shape_vec((1, 3), vec![1.3, -0.05, 0.1]).unwrap();

        // Analytical grads
        let y = lorentz::lorentz_layer_forward(&u.view(), &v.view(), c, t);
        let grad_output = Array2::<f32>::ones(y.dim());
        let (gu, gv) =
            lorentz::lorentz_layer_backward(&grad_output.view(), &u.view(), &v.view(), c, t);

        // Numeric grads
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

        let tol = 5e-2f32; // relaxed due to approximations
        for j in 0..u.ncols() {
            assert!((gu[(0, j)] - gu_num[(0, j)]).abs() <= tol);
            assert!((gv[(0, j)] - gv_num[(0, j)]).abs() <= tol);
        }
    }
}
