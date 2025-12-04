use ndarray::{arr1, Array1, Array2};
use _rust::layers::hyper_metric::{HyperMetric, TinyMLP};
use _rust::layers::symplectic::{SymplecticLayer, SymplecticState};

#[test]
fn hypermetric_generate_core_identity() {
    let r = 2_usize;
    let core_flat = arr1(&[1.0_f32, 0.0, 0.0, 3.0]);
    let layer_emb = core_flat.clone();
    let input_dim = core_flat.len();
    let hidden_dim = input_dim;
    let output_dim = input_dim;
    let mut w1 = Array2::<f32>::zeros((input_dim, hidden_dim));
    let mut w2 = Array2::<f32>::zeros((hidden_dim, output_dim));
    for i in 0..input_dim {
        w1[(i, i)] = 1.0;
        w2[(i, i)] = 1.0;
    }
    let b1 = Array1::<f32>::zeros(hidden_dim);
    let b2 = Array1::<f32>::zeros(output_dim);
    let mut u_global = Array2::<f32>::zeros((r, r));
    let mut v_global = Array2::<f32>::zeros((r, r));
    for i in 0..r {
        u_global[(i, i)] = 1.0;
        v_global[(i, i)] = 1.0;
    }
    let mlp = TinyMLP::from_weights(w1, b1, w2, b2);
    let hm = HyperMetric::from_components(u_global, v_global, mlp);
    let core = hm.generate_core(&layer_emb);
    assert_eq!(core.nrows(), r);
    assert_eq!(core.ncols(), r);
    let flat = core.into_raw_vec();
    assert_eq!(flat.len(), core_flat.len());
    for i in 0..flat.len() {
        let diff = (flat[i] - core_flat[i]).abs();
        assert!(diff < 1e-6, "core_flat mismatch at {}: diff={}", i, diff);
    }
}

#[test]
fn hypermetric_project_forward_matches_uv() {
    let r = 2_usize;
    let d = 2_usize;
    let core_vec = vec![1.0_f32, 0.0, 0.0, 2.0];
    let core = Array2::from_shape_vec((r, r), core_vec.clone()).unwrap();
    let core_flat = Array1::from(core_vec);
    let input_dim = core_flat.len();
    let hidden_dim = input_dim;
    let output_dim = input_dim;
    let mut w1 = Array2::<f32>::zeros((input_dim, hidden_dim));
    let mut w2 = Array2::<f32>::zeros((hidden_dim, output_dim));
    for i in 0..input_dim {
        w1[(i, i)] = 1.0;
        w2[(i, i)] = 1.0;
    }
    let b1 = Array1::<f32>::zeros(hidden_dim);
    let b2 = Array1::<f32>::zeros(output_dim);
    let mut u_global = Array2::<f32>::zeros((d, r));
    let mut v_global = Array2::<f32>::zeros((d, r));
    for i in 0..d {
        u_global[(i, i)] = 1.0;
        v_global[(i, i)] = 1.0;
    }
    let mlp = TinyMLP::from_weights(w1, b1, w2, b2);
    let hm = HyperMetric::from_components(u_global, v_global, mlp);
    let x = Array2::from_shape_vec((1, d), vec![1.0_f32, 2.0]).unwrap();
    let out = hm.project_forward(&x, &core_flat);
    let expected = x.dot(&core);
    assert_eq!(out.dim(), expected.dim());
    let diff = (&out - &expected).mapv(|v| v.abs()).sum();
    assert!(diff < 1e-5, "project_forward diff={}", diff);
}

#[test]
fn symplectic_step_dt_zero_no_change() {
    let d = 2_usize;
    let r = 2_usize;
    let core_vec = vec![1.0_f32, 0.0, 0.0, 1.0];
    let core_flat = Array1::from(core_vec.clone());
    let input_dim = core_flat.len();
    let hidden_dim = input_dim;
    let output_dim = input_dim;
    let mut w1 = Array2::<f32>::zeros((input_dim, hidden_dim));
    let mut w2 = Array2::<f32>::zeros((hidden_dim, output_dim));
    for i in 0..input_dim {
        w1[(i, i)] = 1.0;
        w2[(i, i)] = 1.0;
    }
    let b1 = Array1::<f32>::zeros(hidden_dim);
    let b2 = Array1::<f32>::zeros(output_dim);
    let mut u_global = Array2::<f32>::zeros((d, r));
    let mut v_global = Array2::<f32>::zeros((d, r));
    for i in 0..d {
        u_global[(i, i)] = 1.0;
        v_global[(i, i)] = 1.0;
    }
    let mlp = TinyMLP::from_weights(w1, b1, w2, b2);
    let hm = HyperMetric::from_components(u_global, v_global, mlp);
    let dt = 0.0_f32;
    let layer_idx = 0_usize;
    let layer_emb = core_flat.clone();
    let layer = SymplecticLayer::new(layer_idx, layer_emb, hm, dt);
    let q0 = Array2::<f32>::from_shape_vec((1, d), vec![1.0_f32, -2.0]).unwrap();
    let p0 = Array2::<f32>::from_shape_vec((1, d), vec![0.5_f32, 0.25]).unwrap();
    let mut state = SymplecticState { q: q0.clone(), p: p0.clone() };
    let x_input = Array2::<f32>::zeros((1, d));
    let _ = layer.step(&mut state, &x_input);
    let diff_q = (&state.q - &q0).mapv(|v| v.abs()).sum();
    let diff_p = (&state.p - &p0).mapv(|v| v.abs()).sum();
    assert!(diff_q < 1e-6, "q changed with dt=0, diff={}", diff_q);
    assert!(diff_p < 1e-6, "p changed with dt=0, diff={}", diff_p);
}

#[test]
fn symplectic_step_identity_force_matches_manual() {
    let d = 2_usize;
    let r = 2_usize;
    let core_vec = vec![1.0_f32, 0.0, 0.0, 1.0];
    let core_flat = Array1::from(core_vec.clone());
    let input_dim = core_flat.len();
    let hidden_dim = input_dim;
    let output_dim = input_dim;
    let mut w1 = Array2::<f32>::zeros((input_dim, hidden_dim));
    let mut w2 = Array2::<f32>::zeros((hidden_dim, output_dim));
    for i in 0..input_dim {
        w1[(i, i)] = 1.0;
        w2[(i, i)] = 1.0;
    }
    let b1 = Array1::<f32>::zeros(hidden_dim);
    let b2 = Array1::<f32>::zeros(output_dim);
    let mut u_global = Array2::<f32>::zeros((d, r));
    let mut v_global = Array2::<f32>::zeros((d, r));
    for i in 0..d {
        u_global[(i, i)] = 1.0;
        v_global[(i, i)] = 1.0;
    }
    let mlp = TinyMLP::from_weights(w1, b1, w2, b2);
    let hm = HyperMetric::from_components(u_global, v_global, mlp);
    let dt = 0.1_f32;
    let layer_idx = 0_usize;
    let layer_emb = core_flat.clone();
    let layer = SymplecticLayer::new(layer_idx, layer_emb, hm, dt);
    let q0 = Array2::<f32>::from_shape_vec((1, d), vec![1.0_f32, 2.0]).unwrap();
    let p0 = Array2::<f32>::from_shape_vec((1, d), vec![0.5_f32, -0.5]).unwrap();
    let force = hm.project_forward(&q0, &core_flat);
    let expected_p = &p0 + &(&force * dt);
    let expected_q = &q0 + &(&expected_p * dt);
    let mut state = SymplecticState { q: q0.clone(), p: p0.clone() };
    let x_input = Array2::<f32>::zeros((1, d));
    let _ = layer.step(&mut state, &x_input);
    let diff_q = (&state.q - &expected_q).mapv(|v| v.abs()).sum();
    let diff_p = (&state.p - &expected_p).mapv(|v| v.abs()).sum();
    assert!(diff_q < 1e-5, "q mismatch, diff={}", diff_q);
    assert!(diff_p < 1e-5, "p mismatch, diff={}", diff_p);
}

#[test]
fn hypermetric_matches_projected_metric_core() {
    let d = 4_usize;
    let r = 2_usize;
    let wq = Array2::<f32>::from_shape_vec(
        (d, d),
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 4.0,
        ],
    )
    .unwrap();
    let wk = Array2::<f32>::from_shape_vec(
        (d, d),
        vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.5, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();

    let g = wq.t().dot(&wk);
    let g_sym = (&g + &g.t()) * 0.5;

    let mut u_global = Array2::<f32>::zeros((d, r));
    for i in 0..r {
        u_global[(i, i)] = 1.0;
    }

    let g_core = u_global.t().dot(&g_sym).dot(&u_global);

    let core_flat: Array1<f32> = Array1::from(g_core.clone().into_raw_vec());
    let input_dim = core_flat.len();
    let hidden_dim = input_dim;
    let output_dim = input_dim;
    let mut w1 = Array2::<f32>::zeros((input_dim, hidden_dim));
    let mut w2 = Array2::<f32>::zeros((hidden_dim, output_dim));
    for i in 0..input_dim {
        w1[(i, i)] = 1.0;
        w2[(i, i)] = 1.0;
    }
    let b1 = Array1::<f32>::zeros(hidden_dim);
    let b2 = Array1::<f32>::zeros(output_dim);

    let mlp = TinyMLP::from_weights(w1, b1, w2, b2);
    let hm = HyperMetric::from_components(u_global.clone(), u_global.clone(), mlp);

    let x = Array2::<f32>::from_shape_vec(
        (3, d),
        vec![
            1.0, 0.5, -1.0, 2.0,
            -0.5, 1.0, 0.0, -1.5,
            0.3, -0.7, 2.0, 0.0,
        ],
    )
    .unwrap();

    let out_hm = hm.project_forward(&x, &core_flat);

    let x_proj = x.dot(&u_global);
    let x_core = x_proj.dot(&g_core);
    let expected = x_core.dot(&u_global.t());

    assert_eq!(out_hm.dim(), expected.dim());
    let diff = (&out_hm - &expected).mapv(|v| v.abs()).sum();
    assert!(diff < 1e-5, "hypermetric projection diff={}", diff);
}


