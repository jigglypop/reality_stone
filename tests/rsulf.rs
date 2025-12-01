use ndarray::{arr1, Array1, Array2};
use _rust::layers::rsulf::{
    RSULFConfig, RSULFLayer,
    fold_dimension_svd, fold_ffn_svd, create_causal_laplacian,
    compute_curvature, verify_fold_consistency,
    analyze_layer, create_compression_plan, verify_compression_plan,
    LayerType, CompressionStrategy,
};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn compute_ffn_output(x: &Array2<f32>, w1: &Array2<f32>, w2: &Array2<f32>) -> Array2<f32> {
    let pre_act = x.dot(&w1.t());
    let h = pre_act.mapv(silu);
    h.dot(&w2.t())
}

fn compute_phi(f_x: &Array2<f32>) -> f32 {
    0.5 * f_x.iter().map(|v| v * v).sum::<f32>() / f_x.nrows() as f32
}

#[test]
fn rsulf_메트릭_대각_양수() {
    let d = 64;
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig {
        d_model: d,
        r: 16,
        ..Default::default()
    };
    
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.1);
    
    let layer = RSULFLayer::from_transformer(
        wq.view(), wk.view(), w1.view(), w2.view(), config
    );
    
    for i in 0..layer.g_diag.len() {
        assert!(
            layer.g_diag[i] > 0.0,
            "g_diag[{}]={} 양수 아님",
            i, layer.g_diag[i]
        );
        assert!(
            layer.g_inv[i] > 0.0,
            "g_inv[{}]={} 양수 아님",
            i, layer.g_inv[i]
        );
    }
}

#[test]
fn rsulf_메트릭_역행렬_정합성() {
    let d = 32;
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig { d_model: d, r: 8, ..Default::default() };
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    for i in 0..d {
        let product = layer.g_diag[i] * layer.g_inv[i];
        let diff = (product - 1.0).abs();
        assert!(
            diff < 1e-4,
            "g_diag * g_inv != 1 at {}: product={} diff={}",
            i, product, diff
        );
    }
}

#[test]
fn rsulf_포텐셜_비음수() {
    let d = 32;
    let batch = 4;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.1);
    
    let x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 2.0 - 1.0);
    let f_x = compute_ffn_output(&x, &w1, &w2);
    let phi = compute_phi(&f_x);
    
    assert!(
        phi >= 0.0,
        "Phi(x) = 0.5 * ||f(x)||^2 는 반드시 >= 0, got {}",
        phi
    );
}

#[test]
fn rsulf_벨만_메모리_감쇠() {
    let gamma = 0.99_f32;
    let phi_values = vec![1.0_f32, 0.5, 0.3, 0.2, 0.1];
    
    let mut v = 0.0_f32;
    for phi in &phi_values {
        v = gamma * v + (1.0 - gamma) * phi;
    }
    
    assert!(
        v > 0.0 && v < 1.0,
        "V_t 범위 이상: {}",
        v
    );
    
    let mut v_decay = 1.0_f32;
    for _ in 0..100 {
        v_decay = gamma * v_decay;
    }
    assert!(
        v_decay < 0.5,
        "gamma^100 감쇠 실패: {}",
        v_decay
    );
}

#[test]
fn rsulf_라플라시안_인과성() {
    let seq_len = 16;
    let window = 4;
    let l = create_causal_laplacian(seq_len, window);
    
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            assert!(
                l[[i, j]].abs() < 1e-10,
                "L[{},{}]={} 인과성 위반 (미래 참조)",
                i, j, l[[i, j]]
            );
        }
    }
    
    for i in 0..seq_len {
        let row_sum: f32 = l.row(i).sum();
        assert!(
            row_sum.abs() < 1e-5,
            "L 행 합 != 0 at row {}: sum={}",
            i, row_sum
        );
    }
}

#[test]
fn rsulf_곡률_보정_비음수() {
    let s_residual = arr1(&[0.1_f32, 0.2, 0.05]);
    let k = compute_curvature(&s_residual);
    
    assert!(
        k >= 0.0,
        "곡률 K_error 음수: {}",
        k
    );
    
    let expected = (0.1_f32.powi(2) + 0.2_f32.powi(2) + 0.05_f32.powi(2)).sqrt();
    let diff = (k - expected).abs();
    assert!(
        diff < 1e-6,
        "곡률 계산 오류: got={} expected={} diff={}",
        k, expected, diff
    );
}

#[test]
fn rsulf_폴딩_정합성() {
    let d = 64;
    let r = 16;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    
    let folded = fold_dimension_svd(wq.view(), wk.view(), r);
    let result = verify_fold_consistency(wq.view(), wk.view(), &folded);
    
    assert!(
        result.fold_accuracy >= 0.0 && result.fold_accuracy <= 1.0,
        "fold_accuracy 범위 이상: {}",
        result.fold_accuracy
    );
    
    assert!(
        result.reconstruction_error >= 0.0,
        "reconstruction_error 음수: {}",
        result.reconstruction_error
    );
}

#[test]
fn rsulf_forward_출력_유한() {
    let d = 32;
    let batch = 4;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig {
        d_model: d,
        r: 8,
        seq_len: batch,
        ..Default::default()
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    let x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 2.0 - 1.0);
    let (x_next, v_new) = layer.forward(x.view(), None);
    
    assert!(
        x_next.iter().all(|v| v.is_finite()),
        "x_next에 NaN/Inf 포함"
    );
    assert!(
        v_new.iter().all(|v| v.is_finite()),
        "v_new에 NaN/Inf 포함"
    );
}

#[test]
fn rsulf_forward_반복_안정성() {
    let d = 32;
    let batch = 4;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.05);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.05);
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.05);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.05);
    
    let config = RSULFConfig {
        d_model: d,
        r: 8,
        eta: 0.01,
        alpha: 0.01,
        beta: 0.0,
        gamma: 0.99,
        seq_len: batch,
        window: 2,
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    let mut x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 0.5);
    let mut v: Option<Array1<f32>> = None;
    
    for step in 0..10 {
        let (x_new, v_new) = layer.forward(x.view(), v.as_ref().map(|a| a.view()));
        
        let max_val = x_new.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_val < 100.0,
            "step {} 에서 x 폭발: max={}",
            step, max_val
        );
        
        x = x_new;
        v = Some(v_new);
    }
}

#[test]
fn rsulf_그래프_디퓨전_효과() {
    let seq_len = 8;
    let d = 16;
    let window = 4;
    
    let l = create_causal_laplacian(seq_len, window);
    let x = Array2::<f32>::from_shape_fn((seq_len, d), |_| rand::random::<f32>());
    
    let lx = l.dot(&x);
    
    assert!(
        lx.iter().all(|v| v.is_finite()),
        "L*x에 NaN/Inf 포함"
    );
    
    let lx_norm: f32 = lx.iter().map(|v| v * v).sum::<f32>().sqrt();
    let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    
    assert!(
        lx_norm < x_norm * 10.0,
        "L*x 노름 폭발: lx_norm={} x_norm={}",
        lx_norm, x_norm
    );
}

#[test]
fn rsulf_지오데식_스텝_근사() {
    let d = 16;
    let batch = 2;
    
    let x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 0.5);
    let v = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 0.1);
    
    let x_next_flat = &x + &v;
    
    let curvature = 0.01_f32;
    let mut delta = Array2::<f32>::zeros((batch, d));
    for i in 0..batch {
        let v_row = v.row(i);
        let x_row = x.row(i);
        let v_norm_sq: f32 = v_row.iter().map(|val| val * val).sum();
        let scale = -0.5 * curvature * v_norm_sq;
        for j in 0..d {
            delta[[i, j]] = scale * x_row[j];
        }
    }
    
    let x_next_curved = &x + &v + &delta;
    
    let diff: f32 = (&x_next_flat - &x_next_curved).mapv(f32::abs).sum();
    
    assert!(
        diff > 0.0,
        "곡률 보정이 효과 없음"
    );
    assert!(
        diff < 1.0,
        "곡률 보정이 너무 큼: diff={}",
        diff
    );
}

#[test]
fn rsulf_svd_폴딩_랭크() {
    let d = 64;
    let r = 16;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    
    let folded = fold_dimension_svd(wq.view(), wk.view(), r);
    
    assert_eq!(folded.u.ncols(), r.min(d), "U 열 수 불일치");
    assert_eq!(folded.s.len(), r.min(d), "S 길이 불일치");
    assert_eq!(folded.v.ncols(), r.min(d), "V 열 수 불일치");
}

#[test]
fn rsulf_ffn_폴딩_형태() {
    let d = 32;
    let ffn_dim = 128;
    let r = 16;
    
    let w1 = Array2::<f32>::from_shape_fn((ffn_dim, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, ffn_dim), |_| rand::random::<f32>() * 0.1);
    
    let folded = fold_ffn_svd(w1.view(), w2.view(), r);
    
    assert!(folded.u1.nrows() > 0, "u1 비어있음");
    assert!(folded.s1.len() > 0, "s1 비어있음");
    assert!(folded.v1.nrows() > 0, "v1 비어있음");
    assert!(folded.u2.nrows() > 0, "u2 비어있음");
    assert!(folded.s2.len() > 0, "s2 비어있음");
    assert!(folded.v2.nrows() > 0, "v2 비어있음");
}

#[test]
fn rsulf_압축률_계산() {
    let d = 64;
    let r = 16;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig {
        d_model: d,
        r,
        seq_len: 8,
        ..Default::default()
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    let (compressed, original, ratio) = layer.param_count();
    
    assert!(
        compressed < original,
        "압축 후 파라미터가 더 많음: compressed={} original={}",
        compressed, original
    );
    assert!(
        ratio > 1.0,
        "압축률 1 미만: ratio={}",
        ratio
    );
}

#[test]
fn rsulf_폴딩_정확도_임계값_소형() {
    let test_cases = vec![
        (64, 32),
        (64, 16),
        (64, 8),
        (128, 64),
        (128, 32),
        (128, 16),
    ];
    
    for (d, r) in test_cases {
        let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
        let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
        
        let folded = fold_dimension_svd(wq.view(), wk.view(), r);
        let result = verify_fold_consistency(wq.view(), wk.view(), &folded);
        
        assert!(
            result.fold_accuracy >= 0.5,
            "d={} r={}: fold_accuracy={:.4} < 0.5 (최소 임계값)",
            d, r, result.fold_accuracy
        );
    }
}

#[test]
fn rsulf_폴딩_정확도_임계값_대형() {
    use std::thread;
    
    let test_cases = vec![
        (256, 128),
        (256, 64),
        (256, 32),
    ];
    
    for (d, r) in test_cases {
        let handle = thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
                let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
                
                let folded = fold_dimension_svd(wq.view(), wk.view(), r);
                let result = verify_fold_consistency(wq.view(), wk.view(), &folded);
                
                assert!(
                    result.fold_accuracy >= 0.5,
                    "d={} r={}: fold_accuracy={:.4} < 0.5 (최소 임계값)",
                    d, r, result.fold_accuracy
                );
            })
            .unwrap();
        
        handle.join().expect("대형 폴딩 테스트 실패");
    }
}

#[test]
fn rsulf_폴딩_정확도_저랭크() {
    let d = 128;
    let test_ranks = vec![4, 8, 12, 16];
    
    for r in test_ranks {
        let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
        let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
        
        let folded = fold_dimension_svd(wq.view(), wk.view(), r);
        let result = verify_fold_consistency(wq.view(), wk.view(), &folded);
        
        assert!(
            result.fold_accuracy >= 0.3,
            "저랭크 r={}: fold_accuracy={:.4} < 0.3",
            r, result.fold_accuracy
        );
        
        assert!(
            result.reconstruction_error < 1.0,
            "저랭크 r={}: reconstruction_error={:.4} >= 1.0",
            r, result.reconstruction_error
        );
    }
}

#[test]
fn rsulf_폴딩_에너지_보존() {
    let d = 64;
    let r = 32;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    
    let g = wq.t().dot(&wk);
    let frob_g: f32 = g.iter().map(|x| x * x).sum();
    
    let folded = fold_dimension_svd(wq.view(), wk.view(), r);
    let result = verify_fold_consistency(wq.view(), wk.view(), &folded);
    
    assert!(
        result.fold_accuracy >= 0.0 && result.fold_accuracy <= 1.0,
        "fold_accuracy 범위 이상: {:.4}",
        result.fold_accuracy
    );
    
    assert!(
        result.reconstruction_error >= 0.0,
        "reconstruction_error 음수: {:.4}",
        result.reconstruction_error
    );
    
    let energy_captured = result.fold_accuracy;
    assert!(
        energy_captured >= 0.5,
        "에너지 캡처 부족: fold_accuracy={:.4} < 0.5",
        energy_captured
    );
}

#[test]
fn rsulf_ffn_폴딩_정확도() {
    let d = 64;
    let ffn_dim = 256;
    let r = 32;
    
    let w1 = Array2::<f32>::from_shape_fn((ffn_dim, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, ffn_dim), |_| rand::random::<f32>() * 0.1);
    
    let folded = fold_ffn_svd(w1.view(), w2.view(), r);
    
    let frob_w1: f32 = w1.iter().map(|x| x * x).sum();
    let frob_s1: f32 = folded.s1.iter().map(|x| x * x).sum();
    let accuracy_w1 = frob_s1 / frob_w1.max(1e-10);
    
    let frob_w2: f32 = w2.iter().map(|x| x * x).sum();
    let frob_s2: f32 = folded.s2.iter().map(|x| x * x).sum();
    let accuracy_w2 = frob_s2 / frob_w2.max(1e-10);
    
    assert!(
        accuracy_w1 >= 0.3,
        "W1 폴딩 정확도 부족: {:.4}",
        accuracy_w1
    );
    assert!(
        accuracy_w2 >= 0.3,
        "W2 폴딩 정확도 부족: {:.4}",
        accuracy_w2
    );
}

#[test]
fn rsulf_forward_재구성_오차() {
    let d = 32;
    let batch = 4;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig {
        d_model: d,
        r: 16,
        eta: 0.01,
        alpha: 0.0,
        beta: 0.0,
        gamma: 0.0,
        seq_len: batch,
        window: 2,
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    let x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 0.5);
    let (x_next, _) = layer.forward(x.view(), None);
    
    let f_x_original = compute_ffn_output(&x, &w1, &w2);
    
    let diff = &x_next - &x;
    let f_x_norm: f32 = f_x_original.iter().map(|v| v * v).sum::<f32>().sqrt();
    let diff_norm: f32 = diff.iter().map(|v| v * v).sum::<f32>().sqrt();
    
    assert!(
        diff_norm < f_x_norm * 10.0,
        "재구성 오차 과대: diff_norm={:.4} f_x_norm={:.4}",
        diff_norm, f_x_norm
    );
}

#[test]
fn rsulf_고압축_정확도() {
    let d = 128;
    let r = 8;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d * 4, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d * 4), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig {
        d_model: d,
        r,
        seq_len: 8,
        ..Default::default()
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    let (compressed, original, ratio) = layer.param_count();
    
    assert!(
        ratio >= 10.0,
        "고압축 시나리오에서 압축률 부족: ratio={:.1}x (목표: 10x 이상)",
        ratio
    );
    
    let folded = fold_dimension_svd(wq.view(), wk.view(), r);
    let result = verify_fold_consistency(wq.view(), wk.view(), &folded);
    
    assert!(
        result.fold_accuracy >= 0.2,
        "고압축 시나리오에서 정확도 부족: fold_accuracy={:.4} (최소 0.2)",
        result.fold_accuracy
    );
    
    let batch = 4;
    let x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 0.5);
    let (x_next, _) = layer.forward(x.view(), None);
    
    assert!(
        x_next.iter().all(|v| v.is_finite()),
        "고압축 시나리오에서 출력 NaN/Inf"
    );
}

#[test]
fn rsulf_레이어_분석_정확도() {
    let d = 64;
    let d_ff = 128;
    let r = 16;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d_ff, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d_ff), |_| rand::random::<f32>() * 0.1);
    
    let analysis = analyze_layer(wq.view(), wk.view(), w1.view(), w2.view(), 0, r);
    
    assert_eq!(analysis.layer_idx, 0);
    assert!(analysis.spectral_decay >= 0.0 && analysis.spectral_decay <= 1.0);
    assert!(analysis.condition_number >= 1.0);
    assert!(analysis.recommended_rank > 0 && analysis.recommended_rank <= r);
    assert!(analysis.expected_accuracy >= 0.0 && analysis.expected_accuracy <= 1.0);
}

#[test]
fn rsulf_압축_계획_생성() {
    let d = 64;
    let d_ff = 128;
    let r = 16;
    let num_layers = 4;
    
    let mut analyses = Vec::new();
    for i in 0..num_layers {
        let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
        let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
        let w1 = Array2::<f32>::from_shape_fn((d_ff, d), |_| rand::random::<f32>() * 0.1);
        let w2 = Array2::<f32>::from_shape_fn((d, d_ff), |_| rand::random::<f32>() * 0.1);
        
        let analysis = analyze_layer(wq.view(), wk.view(), w1.view(), w2.view(), i, r);
        analyses.push(analysis);
    }
    
    let plan = create_compression_plan(analyses, 10.0);
    
    assert_eq!(plan.layers.len(), num_layers);
    assert!(plan.total_original_params > 0);
    assert!(plan.total_compressed_params > 0);
    assert!(plan.expected_compression_ratio > 0.0);
}

#[test]
fn rsulf_압축_계획_검증() {
    let d = 64;
    let d_ff = 128;
    let r = 32;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d_ff, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d_ff), |_| rand::random::<f32>() * 0.1);
    
    let analysis = analyze_layer(wq.view(), wk.view(), w1.view(), w2.view(), 0, r);
    let plan = create_compression_plan(vec![analysis], 10.0);
    
    let result = verify_compression_plan(&plan, 0.5);
    
    if result.is_err() {
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("accuracy") || err_msg.contains("condition"));
    }
}

#[test]
fn rsulf_어텐션_포함_forward() {
    let d = 32;
    let d_ff = 64;
    let r = 8;
    let batch = 8;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |(i, j)| {
        if i == j { 1.0 } else { rand::random::<f32>() * 0.01 }
    });
    let wk = Array2::<f32>::from_shape_fn((d, d), |(i, j)| {
        if i == j { 1.0 } else { rand::random::<f32>() * 0.01 }
    });
    let w1 = Array2::<f32>::from_shape_fn((d_ff, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d_ff), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig {
        d_model: d,
        r,
        eta: 1.0,
        alpha: 0.0,
        beta: 0.0,
        gamma: 0.99,
        seq_len: batch,
        window: 4,
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    assert_eq!(layer.g_sym.nrows(), d);
    assert_eq!(layer.g_sym.ncols(), d);
    assert_eq!(layer.a_antisym.nrows(), d);
    assert_eq!(layer.a_antisym.ncols(), d);
    
    let x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>());
    let (x_next, v_new) = layer.forward(x.view(), None);
    
    assert_eq!(x_next.nrows(), batch);
    assert_eq!(x_next.ncols(), d);
    assert!(x_next.iter().all(|v| v.is_finite()));
    assert!(v_new.iter().all(|v| v.is_finite()));
    
    let diff: f32 = (&x_next - &x).iter().map(|v| v.abs()).sum::<f32>() / (batch * d) as f32;
    assert!(diff > 1e-6, "Attention이 적용되지 않음: diff={}", diff);
}

#[test]
fn rsulf_지수맵_근사_정확도() {
    let d = 32;
    let d_ff = 64;
    let r = 8;
    let batch = 4;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |(i, j)| {
        if i == j { 1.0 } else { 0.0 }
    });
    let wk = Array2::<f32>::from_shape_fn((d, d), |(i, j)| {
        if i == j { 1.0 } else { 0.0 }
    });
    let w1 = Array2::<f32>::from_shape_fn((d_ff, d), |_| rand::random::<f32>() * 0.01);
    let w2 = Array2::<f32>::from_shape_fn((d, d_ff), |_| rand::random::<f32>() * 0.01);
    
    let config = RSULFConfig {
        d_model: d,
        r,
        eta: 0.1,
        alpha: 0.0,
        beta: 0.0,
        gamma: 0.99,
        seq_len: batch,
        window: 2,
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    let x = Array2::<f32>::from_shape_fn((batch, d), |_| rand::random::<f32>() * 0.5);
    let (x_next, _) = layer.forward(x.view(), None);
    
    let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let x_next_norm: f32 = x_next.iter().map(|v| v * v).sum::<f32>().sqrt();
    
    let norm_ratio = x_next_norm / x_norm.max(1e-10);
    assert!(
        norm_ratio > 0.5 && norm_ratio < 2.0,
        "지수맵이 노름을 과도하게 변경: ratio={}",
        norm_ratio
    );
}

#[test]
fn rsulf_메트릭_대칭_반대칭_분해() {
    let d = 32;
    let d_ff = 64;
    let r = 8;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let wk = Array2::<f32>::from_shape_fn((d, d), |_| rand::random::<f32>() * 0.1);
    let w1 = Array2::<f32>::from_shape_fn((d_ff, d), |_| rand::random::<f32>() * 0.1);
    let w2 = Array2::<f32>::from_shape_fn((d, d_ff), |_| rand::random::<f32>() * 0.1);
    
    let config = RSULFConfig {
        d_model: d,
        r,
        ..Default::default()
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    let g_sym = &layer.g_sym;
    let a_antisym = &layer.a_antisym;
    
    for i in 0..d {
        for j in 0..d {
            let sym_err = (g_sym[[i, j]] - g_sym[[j, i]]).abs();
            assert!(sym_err < 1e-5, "g_sym이 대칭이 아님: [{},{}] err={}", i, j, sym_err);
            
            let antisym_err = (a_antisym[[i, j]] + a_antisym[[j, i]]).abs();
            assert!(antisym_err < 1e-5, "a_antisym이 반대칭이 아님: [{},{}] err={}", i, j, antisym_err);
        }
    }
    
    let b = wq.t().dot(&wk);
    let reconstructed = &layer.g_sym + &layer.a_antisym;
    
    let recon_err: f32 = (&b - &reconstructed).iter().map(|v| v * v).sum::<f32>().sqrt();
    let b_norm: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    
    assert!(
        recon_err / b_norm.max(1e-10) < 1e-4,
        "B = G + A 복원 오류: err/norm={}",
        recon_err / b_norm.max(1e-10)
    );
}

#[test]
fn rsulf_인과적_어텐션_마스킹() {
    let d = 16;
    let d_ff = 32;
    let r = 4;
    let batch = 8;
    
    let wq = Array2::<f32>::from_shape_fn((d, d), |(i, j)| {
        if i == j { 1.0 } else { 0.0 }
    });
    let wk = wq.clone();
    let w1 = Array2::<f32>::zeros((d_ff, d));
    let w2 = Array2::<f32>::zeros((d, d_ff));
    
    let config = RSULFConfig {
        d_model: d,
        r,
        eta: 0.0,
        alpha: 0.0,
        beta: 0.0,
        gamma: 0.99,
        seq_len: batch,
        window: 4,
    };
    
    let layer = RSULFLayer::from_transformer(wq.view(), wk.view(), w1.view(), w2.view(), config);
    
    assert_eq!(layer.g_sym.nrows(), d);
    assert_eq!(layer.g_sym.ncols(), d);
    
    for i in 0..d {
        for j in 0..d {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = layer.g_sym[[i, j]];
            assert!(
                (actual - expected).abs() < 1e-4,
                "g_sym[{},{}] = {} (expected {})",
                i, j, actual, expected
            );
        }
    }
    
    let mut x = Array2::<f32>::zeros((batch, d));
    for i in 0..batch {
        x[[i, i % d]] = 1.0;
    }
    
    let (x_next, _) = layer.forward(x.view(), None);
    
    let row_0 = x_next.row(0);
    let x_0 = x.row(0);
    let diff_0: f32 = (&row_0 - &x_0).iter().map(|v| v.abs()).sum();
    
    assert!(diff_0 < 0.1, "첫 토큰 변화 과대: diff={}", diff_0);
}

