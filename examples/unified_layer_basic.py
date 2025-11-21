#!/usr/bin/env python3
"""
Unified Riemannian Layer - Basic Usage Example

이 예제는 UnifiedRiemannianLayer의 기본 사용법을 보여줍니다.
"""

import numpy as np
import reality_stone as rs

def main():
    print("=" * 60)
    print("Reality Stone - Unified Riemannian Layer Basic Example")
    print("=" * 60)
    
    # 1. 푸앵카레 메트릭으로 레이어 생성
    print("\n1. Creating Poincare layer...")
    poincare_layer = rs.UnifiedRiemannianLayer(
        metric_type="poincare",
        curvature=1.0,
        input_dim=32,
        enable_bellman=False
    )
    print("   -> Poincare layer created (curvature=1.0, dim=32)")
    
    # 2. 입력 데이터 생성
    print("\n2. Generating input data...")
    batch_size = 16
    x = np.random.randn(batch_size, 32).astype(np.float32) * 0.1  # 작은 값으로 초기화
    y = np.random.randn(batch_size, 32).astype(np.float32) * 0.1
    print(f"   -> Generated {batch_size} samples")
    print(f"   Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    
    # 3. 순전파 (목표 없이)
    print("\n3. Forward pass (identity)...")
    output1, energy1 = poincare_layer.forward(x)
    print(f"   Output shape: {output1.shape}")
    print(f"   Energy: {energy1}")
    
    # 4. 순전파 (목표와 함께 - 측지선 보간)
    print("\n4. Forward pass (with target)...")
    output2, energy2 = poincare_layer.forward(x, target=y)
    print(f"   Output shape: {output2.shape}")
    print(f"   Distance moved: {np.linalg.norm(output2 - x, axis=1).mean():.4f}")
    
    # 5. 측지선 경로 생성
    print("\n5. Generating geodesic path...")
    path = poincare_layer.geodesic_path(x[:1], y[:1], num_steps=10)
    print(f"   Path length: {len(path)}")
    print("   Distances between consecutive points:")
    for i in range(len(path) - 1):
        dist = np.linalg.norm(path[i+1] - path[i])
        print(f"      Step {i} -> {i+1}: {dist:.4f}")
    
    # 6. 다른 메트릭 시험
    print("\n6. Testing different metrics...")
    
    # 로렌츠 (차원 +1 필요)
    print("\n   a) Lorentz (Hyperboloid):")
    lorentz_layer = rs.UnifiedRiemannianLayer(
        metric_type="lorentz",
        curvature=1.0,
        input_dim=33,  # +1 for time component
        enable_bellman=False
    )
    x_lorentz = np.random.randn(batch_size, 33).astype(np.float32) * 0.1
    output_lorentz, _ = lorentz_layer.forward(x_lorentz)
    print(f"      Output shape: {output_lorentz.shape}")
    
    # 클라인
    print("\n   b) Klein (Projective):")
    klein_layer = rs.UnifiedRiemannianLayer(
        metric_type="klein",
        curvature=1.0,
        input_dim=32,
        enable_bellman=False
    )
    output_klein, _ = klein_layer.forward(x)
    print(f"      Output shape: {output_klein.shape}")
    
    # 대각 (학습 가능)
    print("\n   c) Diagonal (Learnable):")
    diagonal_layer = rs.UnifiedRiemannianLayer(
        metric_type="diagonal",
        curvature=0.0,  # 곡률 무관
        input_dim=32,
        enable_bellman=False
    )
    output_diagonal, _ = diagonal_layer.forward(x)
    print(f"      Output shape: {output_diagonal.shape}")
    
    # 7. 메트릭별 거리 비교
    print("\n7. Comparing distances across metrics...")
    x_sample = x[:5]
    y_sample = y[:5]
    
    dist_poincare = rs.geodesic_distance(x_sample, y_sample, "poincare", 1.0)
    dist_lorentz = rs.geodesic_distance(
        np.column_stack([np.ones((5, 1), dtype=np.float32), x_sample[:, :32]]),  # Add time component
        np.column_stack([np.ones((5, 1), dtype=np.float32), y_sample[:, :32]]),
        "lorentz",
        1.0
    )
    dist_klein = rs.geodesic_distance(x_sample, y_sample, "klein", 1.0)
    dist_diagonal = rs.geodesic_distance(x_sample, y_sample, "diagonal", 0.0)
    
    print(f"   Poincare:  {dist_poincare.mean():.4f}")
    print(f"   Lorentz:   {dist_lorentz.mean():.4f}")
    print(f"   Klein:     {dist_klein.mean():.4f}")
    print(f"   Diagonal:  {dist_diagonal.mean():.4f}")
    
    # 8. 측지선 보간
    print("\n8. Geodesic interpolation...")
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_single = x[:1]
    y_single = y[:1]
    
    print("   Interpolation at different t:")
    for t in t_values:
        interpolated = rs.geodesic_interpolate(x_single, y_single, "poincare", 1.0, t)
        dist_to_x = np.linalg.norm(interpolated - x_single)
        dist_to_y = np.linalg.norm(interpolated - y_single)
        print(f"      t={t:.2f}: dist_to_x={dist_to_x:.4f}, dist_to_y={dist_to_y:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

