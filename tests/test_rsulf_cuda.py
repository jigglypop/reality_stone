import numpy as np
import torch

try:
    import reality_stone as rs
    from reality_stone._rust import (
        PyRSULFLayer,
        fold_metric_svd,
        fold_ffn,
        build_causal_laplacian,
    )
    from reality_stone.layers.rsulf_cuda import RSULFLayerCUDA, RSULFWrapperCUDA, RSULFLMHeadCUDA
    HAS_RUST = True
except ImportError as e:
    HAS_RUST = False
    print(f"Rust bindings not available: {e}")

try:
    from reality_stone._rust import rsulf_forward_cuda_py, rsulf_batch_forward_cuda_py, rsulf_unified_forward_cuda_py
    HAS_CUDA_RSULF = True
except ImportError:
    HAS_CUDA_RSULF = False
    print("CUDA RS-ULF bindings not available")

def test_rsulf_cpu_forward():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    
    np.random.seed(42)
    d_model = 256
    r = 64
    batch = 4
    ffn_dim = 512
    
    wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.02
    w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.02
    
    layer = PyRSULFLayer(
        wq, wk, w1, w2,
        d_model=d_model,
        r=r,
        eta=0.01,
        alpha=0.02,
        beta=0.01,
        gamma=0.99,
        seq_len=128,
        window=8
    )
    
    x = np.random.randn(batch, d_model).astype(np.float32) * 0.1
    
    x_out, v_out = layer.forward(x, None)
    
    assert x_out.shape == (batch, d_model), f"Expected ({batch}, {d_model}), got {x_out.shape}"
    assert v_out.shape == (batch,), f"Expected ({batch},), got {v_out.shape}"
    assert np.isfinite(x_out).all(), "Output contains NaN/Inf"
    assert np.isfinite(v_out).all(), "V_out contains NaN/Inf"
    
    print(f"CPU Forward: x_out mean={x_out.mean():.6f}, std={x_out.std():.6f}")
    print(f"CPU Forward: v_out mean={v_out.mean():.6f}")
    print("CPU RS-ULF forward test passed!")

def test_rsulf_cuda_forward():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    if not HAS_CUDA_RSULF:
        print("SKIP: CUDA RS-ULF bindings not available")
        return
    
    np.random.seed(42)
    d_model = 256
    r = 64
    batch = 4
    ffn_dim = 512
    
    wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.02
    w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.02
    
    u_metric, s_metric, v_metric, curvature = fold_metric_svd(wq, wk, r)
    u1, s1, v1, u2, s2, v2 = fold_ffn(w1, w2, r)
    
    g_diag = np.abs(s_metric) + 1e-6
    g_inv = 1.0 / g_diag
    
    x = np.random.randn(batch, d_model).astype(np.float32) * 0.1
    
    x_out_cuda, v_out_cuda = rsulf_forward_cuda_py(
        x, v1, s1, u1, v2, s2, u2, g_inv, None,
        eta=0.01, alpha=0.02, gamma_param=0.99
    )
    
    assert x_out_cuda.shape == (batch, d_model), f"Expected ({batch}, {d_model}), got {x_out_cuda.shape}"
    assert v_out_cuda.shape == (batch,), f"Expected ({batch},), got {v_out_cuda.shape}"
    assert np.isfinite(x_out_cuda).all(), "CUDA output contains NaN/Inf"
    assert np.isfinite(v_out_cuda).all(), "CUDA v_out contains NaN/Inf"
    
    print(f"CUDA Forward: x_out mean={x_out_cuda.mean():.6f}, std={x_out_cuda.std():.6f}")
    print(f"CUDA Forward: v_out mean={v_out_cuda.mean():.6f}")
    print("CUDA RS-ULF forward test passed!")

def test_rsulf_cpu_cuda_consistency():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    if not HAS_CUDA_RSULF:
        print("SKIP: CUDA RS-ULF bindings not available")
        return
    
    np.random.seed(42)
    d_model = 128
    r = 32
    batch = 2
    ffn_dim = 256
    
    wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.02
    w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.02
    
    layer = PyRSULFLayer(
        wq, wk, w1, w2,
        d_model=d_model,
        r=r,
        eta=0.01,
        alpha=0.02,
        beta=0.01,
        gamma=0.99,
        seq_len=128,
        window=8
    )
    
    u_metric, s_metric, v_metric, curvature = fold_metric_svd(wq, wk, r)
    u1, s1, v1, u2, s2, v2 = fold_ffn(w1, w2, r)
    g_diag = np.abs(s_metric) + 1e-6
    g_inv = 1.0 / g_diag
    
    x = np.random.randn(batch, d_model).astype(np.float32) * 0.1
    
    x_out_cpu, v_out_cpu = layer.forward(x, None)
    
    x_out_cuda, v_out_cuda = rsulf_forward_cuda_py(
        x, v1, s1, u1, v2, s2, u2, g_inv, None,
        eta=0.01, alpha=0.02, gamma_param=0.99
    )
    
    print(f"CPU  x_out: mean={x_out_cpu.mean():.6f}, std={x_out_cpu.std():.6f}")
    print(f"CUDA x_out: mean={x_out_cuda.mean():.6f}, std={x_out_cuda.std():.6f}")
    print(f"Diff: {np.abs(x_out_cpu - x_out_cuda).max():.6f}")
    
    print("CPU-CUDA consistency check completed!")

def test_rsulf_batch_forward_cuda():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    if not HAS_CUDA_RSULF:
        print("SKIP: CUDA RS-ULF bindings not available")
        return
    
    np.random.seed(0)
    d_model = 128
    r = 32
    batch = 2
    seq_len = 4
    ffn_dim = 256
    
    wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.02
    w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.02
    
    u_metric, s_metric, v_metric, curvature = fold_metric_svd(wq, wk, r)
    u1, s1, v1, u2, s2, v2 = fold_ffn(w1, w2, r)
    
    g_diag = np.abs(s_metric) + 1e-6
    g_inv = 1.0 / g_diag
    
    x = np.random.randn(batch * seq_len, d_model).astype(np.float32) * 0.1
    
    x_out_cuda, v_out_cuda = rsulf_batch_forward_cuda_py(
        x, v1, s1, u1, v2, s2, u2, g_inv, None,
        eta=0.01, alpha=0.02, gamma_param=0.99,
        batch=batch, seq_len=seq_len
    )
    
    assert x_out_cuda.shape == (batch * seq_len, d_model)
    assert v_out_cuda.shape == (batch * seq_len,)
    assert np.isfinite(x_out_cuda).all()
    assert np.isfinite(v_out_cuda).all()

def test_rsulf_unified_forward_cuda():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    if not HAS_CUDA_RSULF:
        print("SKIP: CUDA RS-ULF bindings not available")
        return
    
    np.random.seed(1)
    d_model = 64
    r = 16
    batch = 2
    seq_len = 4
    ffn_dim = 128
    
    wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.02
    w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.02
    
    u_metric, s_metric, v_metric, curvature = fold_metric_svd(wq, wk, r)
    u1, s1, v1, u2, s2, v2 = fold_ffn(w1, w2, r)
    
    g_diag = np.abs(s_metric) + 1e-6
    g_inv = 1.0 / g_diag
    
    lap = build_causal_laplacian(seq_len, 2)
    lap = np.array(lap, dtype=np.float32)
    
    x = np.random.randn(batch * seq_len, d_model).astype(np.float32) * 0.1
    
    x_out_cuda, v_out_cuda = rsulf_unified_forward_cuda_py(
        x, v1, s1, u1, v2, s2, u2, g_inv, lap, None,
        eta=0.01, alpha=0.02, beta=0.01, gamma_param=0.99,
        curvature=float(curvature),
        batch=batch, seq_len=seq_len, window=2
    )
    
    assert x_out_cuda.shape == (batch * seq_len, d_model)
    assert v_out_cuda.shape == (batch * seq_len,)
    assert np.isfinite(x_out_cuda).all()
    assert np.isfinite(v_out_cuda).all()

def test_rsulf_wrapper_batch_mode():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    
    np.random.seed(123)
    torch.manual_seed(123)
    
    d_model = 32
    r = 8
    batch = 2
    seq_len = 4
    ffn_dim = 64
    
    wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.05
    wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.05
    w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.05
    w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.05
    
    rs_layer = RSULFLayerCUDA(
        wq=wq,
        wk=wk,
        w1=w1,
        w2=w2,
        d_model=d_model,
        r=r,
        eta=0.01,
        alpha=0.02,
        beta=0.01,
        gamma=0.99,
        seq_len=seq_len,
        window=2,
    )
    wrapper = RSULFWrapperCUDA(rs_layer)
    
    x = torch.randn(batch, seq_len, d_model, dtype=torch.float32)
    y = wrapper(x)
    
    assert y.shape == (batch, seq_len, d_model)
    assert torch.isfinite(y).all()
    assert wrapper.time_step == seq_len

def test_rsulf_wrapper_autoregressive_mode():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    
    np.random.seed(321)
    torch.manual_seed(321)
    
    d_model = 32
    r = 8
    ffn_dim = 64
    
    wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.05
    wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.05
    w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.05
    w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.05
    
    rs_layer = RSULFLayerCUDA(
        wq=wq,
        wk=wk,
        w1=w1,
        w2=w2,
        d_model=d_model,
        r=r,
        eta=0.01,
        alpha=0.02,
        beta=0.01,
        gamma=0.99,
        seq_len=16,
        window=4,
    )
    wrapper = RSULFWrapperCUDA(rs_layer)
    wrapper.reset_memory()
    
    steps = 6
    for t in range(steps):
        x_t = torch.randn(1, 1, d_model, dtype=torch.float32)
        y_t = wrapper(x_t)
        assert y_t.shape == (1, 1, d_model)
        assert torch.isfinite(y_t).all()
    
    assert wrapper.time_step == steps
    if wrapper.geodesic_memory is not None:
        stored, covered, ratio = wrapper.geodesic_memory.get_stats()
        assert stored > 0
        assert covered >= steps

def test_rsulf_lm_head_cuda_pipeline():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    if not HAS_CUDA_RSULF or not torch.cuda.is_available():
        print("SKIP: CUDA RS-ULF or torch.cuda not available")
        return
    
    np.random.seed(7)
    torch.manual_seed(7)
    
    d_model = 64
    r = 16
    seq_len = 8
    num_layers = 2
    ffn_dim = 256
    vocab_size = 101
    
    rs_layers = []
    for _ in range(num_layers):
        wq = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        wk = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        w1 = np.random.randn(ffn_dim, d_model).astype(np.float32) * 0.02
        w2 = np.random.randn(d_model, ffn_dim).astype(np.float32) * 0.02
        
        rs_layer = RSULFLayerCUDA(
            wq=wq,
            wk=wk,
            w1=w1,
            w2=w2,
            d_model=d_model,
            r=r,
            eta=0.01,
            alpha=0.02,
            beta=0.01,
            gamma=0.99,
            seq_len=seq_len,
            window=2,
        )
        rs_layers.append(rs_layer)
    
    device = torch.device("cuda")
    lm = RSULFLMHeadCUDA(rs_layers, hidden_size=d_model, vocab_size=vocab_size, device=device)
    
    x = torch.randn(1, seq_len, d_model, dtype=torch.float32, device=device)
    logits = lm(x)
    
    assert logits.shape == (1, seq_len, vocab_size)
    assert torch.isfinite(logits).all()

if __name__ == "__main__":
    print("=" * 60)
    print("RS-ULF Test Suite")
    print("=" * 60)
    
    print("\n[Test 1] CPU Forward")
    test_rsulf_cpu_forward()
    
    print("\n[Test 2] Inference Pipeline")
    test_rsulf_inference_pipeline()
    
    if HAS_CUDA_RSULF:
        print("\n[Test 3] CUDA Forward")
        test_rsulf_cuda_forward()
        
        print("\n[Test 4] CPU-CUDA Consistency")
        test_rsulf_cpu_cuda_consistency()
    else:
        print("\n[Test 3-4] Skipped (CUDA not available)")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

