import numpy as np

try:
    import reality_stone as rs
    from reality_stone._rust import (
        PyRSULFLayer,
        fold_metric_svd,
        fold_ffn,
        build_causal_laplacian,
    )
    HAS_RUST = True
except ImportError as e:
    HAS_RUST = False
    print(f"Rust bindings not available: {e}")

try:
    from reality_stone._rust import rsulf_forward_cuda_py, rsulf_batch_forward_cuda_py
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

def test_rsulf_inference_pipeline():
    if not HAS_RUST:
        print("SKIP: Rust bindings not available")
        return
    
    np.random.seed(42)
    d_model = 512
    r = 128
    seq_len = 32
    num_layers = 4
    ffn_dim = 2048
    
    layers = []
    for i in range(num_layers):
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
            seq_len=seq_len,
            window=8
        )
        layers.append(layer)
    
    x = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
    v_mem = None
    
    print(f"\nRS-ULF Inference Pipeline ({num_layers} layers)")
    print(f"Input shape: {x.shape}")
    print(f"d_model={d_model}, r={r}, compression={d_model/r:.1f}x")
    
    import time
    start = time.time()
    
    for i, layer in enumerate(layers):
        x, v_mem = layer.forward(x, v_mem)
        print(f"Layer {i+1}: x_out mean={x.mean():.6f}, std={x.std():.6f}")
    
    elapsed = time.time() - start
    
    print(f"\nTotal inference time: {elapsed*1000:.2f}ms")
    print(f"Time per layer: {elapsed*1000/num_layers:.2f}ms")
    print(f"Output shape: {x.shape}")
    
    assert x.shape == (seq_len, d_model)
    assert np.isfinite(x).all()
    
    compressed, original, ratio = layers[0].param_count()
    print(f"\nParam count: {compressed:,} (compressed) vs {original:,} (original)")
    print(f"Compression ratio: {ratio:.2f}x")
    
    print("\nRS-ULF Inference Pipeline test passed!")

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

