"""
Poincaré Ball Layer 수학적 검증
=================================
이 스크립트는 Poincaré ball layer의 수학적 계산을 단계별로 검증합니다.
"""

import numpy as np
import torch
import reality_stone as rs

# 수학 함수 구현
def mobius_scalar_manual(u, c, r):
    """Möbius 스칼라 곱셈의 수동 구현"""
    norm = np.linalg.norm(u)
    if norm < 1e-7:
        return r * u
    
    sqrtc = np.sqrt(c)
    scn = sqrtc * norm
    scn = min(scn, 1.0 - 1e-5)  # 수치적 안정성
    
    alpha = np.arctanh(scn)
    beta = np.tanh(r * alpha)
    scale = beta / (sqrtc * norm)
    
    return scale * u

def mobius_add_manual(u, v, c):
    """Möbius 덧셈의 수동 구현"""
    u2 = np.dot(u, u)
    v2 = np.dot(v, v)
    uv = np.dot(u, v)
    
    denom = 1.0 + 2.0 * c * uv + c * c * u2 * v2
    denom = max(denom, 1e-6)  # 수치적 안정성
    
    coeff_u = (1.0 + 2.0 * c * uv + c * v2) / denom
    coeff_v = (1.0 - c * u2) / denom
    
    return coeff_u * u + coeff_v * v

def poincare_ball_layer_manual(u, v, c, t):
    """Poincaré ball layer의 수동 구현"""
    u_prime = mobius_scalar_manual(u, c, 1.0 - t)
    v_prime = mobius_scalar_manual(v, c, t)
    result = mobius_add_manual(u_prime, v_prime, c)
    return result

def poincare_distance_manual(u, v, c):
    """Poincaré 거리의 수동 구현"""
    sqrtc = np.sqrt(c)
    diff_norm = np.linalg.norm(u - v)
    u_norm_sq = np.dot(u, u)
    v_norm_sq = np.dot(v, v)
    
    denom = np.sqrt((1 - c * u_norm_sq) * (1 - c * v_norm_sq))
    arg = sqrtc * diff_norm / denom
    arg = min(arg, 1.0 - 1e-7)  # atanh의 정의역 제한
    
    return (2.0 / sqrtc) * np.arctanh(arg)

# 검증 실험
print("=" * 60)
print("Poincaré Ball Layer 수학적 검증")
print("=" * 60)

# 테스트 파라미터
c = 0.001
t_values = [0.5, 0.7, 1.0]

# 테스트 벡터
u = np.array([0.5, 0.3])
v = np.array([0.2, 0.4])

print(f"\n입력 벡터:")
print(f"u = {u}, ||u|| = {np.linalg.norm(u):.4f}")
print(f"v = {v}, ||v|| = {np.linalg.norm(v):.4f}")
print(f"곡률 c = {c}")

# 각 t 값에 대해 계산
for t in t_values:
    print(f"\n{'='*40}")
    print(f"t = {t}에 대한 계산")
    print(f"{'='*40}")
    
    # 수동 계산
    result_manual = poincare_ball_layer_manual(u, v, c, t)
    
    # PyTorch로 검증
    u_torch = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
    v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
    result_torch = rs.poincare_ball_layer(u_torch, v_torch, c, t)
    result_torch_np = result_torch.squeeze(0).numpy()
    
    # 중간 계산 출력
    u_prime = mobius_scalar_manual(u, c, 1.0 - t)
    v_prime = mobius_scalar_manual(v, c, t)
    
    print(f"\n1단계: Möbius 스칼라 곱셈")
    print(f"  (1-t) = {1.0 - t:.3f}")
    print(f"  u_prime = mobius_scalar(u, c, {1.0 - t:.3f}) = {u_prime}")
    print(f"  ||u_prime|| = {np.linalg.norm(u_prime):.4f}")
    
    print(f"\n2단계: Möbius 스칼라 곱셈")
    print(f"  t = {t:.3f}")
    print(f"  v_prime = mobius_scalar(v, c, {t:.3f}) = {v_prime}")
    print(f"  ||v_prime|| = {np.linalg.norm(v_prime):.4f}")
    
    print(f"\n3단계: Möbius 덧셈")
    print(f"  결과 = mobius_add(u_prime, v_prime, c)")
    print(f"  수동 계산: {result_manual}")
    print(f"  PyTorch:   {result_torch_np}")
    print(f"  차이:      {np.abs(result_manual - result_torch_np)}")
    print(f"  ||결과|| = {np.linalg.norm(result_manual):.4f}")

# 측지선 거리 검증
print(f"\n{'='*60}")
print("측지선 거리 검증")
print(f"{'='*60}")

dist_manual = poincare_distance_manual(u, v, c)
print(f"d_c(u, v) = {dist_manual:.6f}")

# Ball 경계 확인
max_norm = 1.0 / np.sqrt(c)
print(f"\nBall 경계: ||x|| < {max_norm:.2f}")

# 리만 계량 계산
def riemannian_metric_factor(x, c):
    """점 x에서의 리만 계량 계수"""
    norm_sq = np.dot(x, x)
    return 4.0 / ((1 - c * norm_sq) ** 2)

metric_u = riemannian_metric_factor(u, c)
metric_v = riemannian_metric_factor(v, c)
print(f"\n리만 계량 계수:")
print(f"  g_u = {metric_u:.4f}")
print(f"  g_v = {metric_v:.4f}")

# 극한 동작 확인
print(f"\n{'='*60}")
print("극한 동작 확인")
print(f"{'='*60}")

# t = 0일 때 (결과가 u여야 함)
result_t0 = poincare_ball_layer_manual(u, v, c, 0.0)
print(f"t = 0: 결과 = {result_t0}")
print(f"       u = {u}")
print(f"       차이 = {np.linalg.norm(result_t0 - u):.6f}")

# t = 1일 때 (결과가 v여야 함)
result_t1 = poincare_ball_layer_manual(u, v, c, 1.0)
print(f"\nt = 1: 결과 = {result_t1}")
print(f"       v = {v}")
print(f"       차이 = {np.linalg.norm(result_t1 - v):.6f}")

# 음수 t에 대한 동작
print(f"\n음수 t에 대한 동작:")
for t in [-0.5, -1.0]:
    result_neg = poincare_ball_layer_manual(u, v, c, t)
    print(f"  t = {t}: 결과 = {result_neg}, ||결과|| = {np.linalg.norm(result_neg):.4f}")

print("\n" + "="*60)
print("검증 완료")
print("="*60) 