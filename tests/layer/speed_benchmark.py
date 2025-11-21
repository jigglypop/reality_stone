"""Klein vs Lorentz vs Poincaré 속도 비교"""
import torch
import time
import reality_stone as rs

def benchmark_layer(name, layer_fn, u, v, c, t, warmup=10, runs=100):
    """레이어 성능 측정"""
    # Warmup
    for _ in range(warmup):
        _ = layer_fn(u, v, c, t)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(runs):
        result = layer_fn(u, v, c, t)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    avg_ms = (elapsed / runs) * 1000
    return avg_ms, result

def main():
    batch = 256
    dim = 128
    c = 0.1
    t = 0.5
    
    print(f"Benchmark: batch={batch}, dim={dim}, c={c}, t={t}")
    print("=" * 60)
    
    # CPU 테스트
    device = 'cpu'
    u = torch.randn(batch, dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, dim, device=device, dtype=torch.float32)
    
    # Normalize to ball
    u = u / (u.norm(dim=1, keepdim=True) + 1e-5) * 0.5
    v = v / (v.norm(dim=1, keepdim=True) + 1e-5) * 0.5
    
    print(f"\nDevice: {device}")
    print("-" * 60)
    
    # Klein
    try:
        time_klein, _ = benchmark_layer("Klein", rs.klein_layer, u, v, c, t)
        print(f"Klein:     {time_klein:6.3f} ms  (baseline)")
    except Exception as e:
        print(f"Klein:     ERROR - {e}")
        time_klein = None
    
    # Lorentz (dim+1 필요)
    try:
        # Lorentz는 (batch, dim+1) 형태
        u_lor = torch.cat([torch.ones(batch, 1), u], dim=1)
        v_lor = torch.cat([torch.ones(batch, 1), v], dim=1)
        time_lorentz, _ = benchmark_layer("Lorentz", rs.lorentz_layer, u_lor, v_lor, c, t)
        speedup_l = time_klein / time_lorentz if time_klein else 0
        print(f"Lorentz:   {time_lorentz:6.3f} ms  ({speedup_l:.2f}x vs Klein)")
    except Exception as e:
        print(f"Lorentz:   ERROR - {e}")
        time_lorentz = None
    
    # Poincaré  
    try:
        time_poincare, _ = benchmark_layer("Poincaré", rs.poincare_ball_layer, u, v, c, t)
        speedup_p = time_klein / time_poincare if time_klein else 0
        slowdown_p = time_poincare / time_klein if time_klein else 0
        print(f"Poincaré:  {time_poincare:6.3f} ms  ({slowdown_p:.2f}x slower)")
    except Exception as e:
        print(f"Poincaré:  ERROR - {e}")
        time_poincare = None
    
    # 분석
    if all([time_klein, time_lorentz, time_poincare]):
        print("\n" + "=" * 60)
        print("분석:")
        print(f"  Poincaré는 Klein보다 {time_poincare/time_klein:.2f}배 느림")
        print(f"  Poincaré는 Lorentz보다 {time_poincare/time_lorentz:.2f}배 느림")
        
        # 1 epoch (60000 samples) 예상 시간
        batches_per_epoch = 60000 / batch
        klein_epoch = (time_klein * batches_per_epoch) / 1000
        poincare_epoch = (time_poincare * batches_per_epoch) / 1000
        print(f"\nMNIST 1 epoch 예상 (forward only):")
        print(f"  Klein:     {klein_epoch:5.2f}s")
        print(f"  Poincaré:  {poincare_epoch:5.2f}s  (+{poincare_epoch-klein_epoch:.2f}s)")

if __name__ == "__main__":
    print("Reality Stone 속도 벤치마크")
    print()
    main()

