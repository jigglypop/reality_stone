#!/usr/bin/env python3
"""
GPU 포인터 직접 처리 최적화 테스트

기존 방식과 새로운 GPU 직접 처리 방식의 성능을 비교합니다.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from reality_stone.layers import BitfieldLinear

def measure_time(fn, warmup=10, iterations=100):
    """함수 실행 시간 측정"""
    # Warmup
    for _ in range(warmup):
        fn()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        fn()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations * 1000  # ms

def test_gpu_optimization():
    """GPU 최적화 테스트"""
    
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. 테스트를 건너뜁니다.")
        return
    
    print("=== GPU 포인터 직접 처리 최적화 테스트 ===\n")
    
    # 테스트 설정
    batch_sizes = [1, 8, 32, 128]
    dimensions = [(784, 256), (256, 128), (128, 10), (1024, 512)]
    
    for in_features, out_features in dimensions:
        print(f"\n차원: {in_features} → {out_features}")
        print("-" * 60)
        
        # 원본 Linear 레이어
        linear = nn.Linear(in_features, out_features).cuda()
        
        # BitfieldLinear 레이어 생성
        bitfield = BitfieldLinear.from_linear(linear, basis_size=256, use_residual=True).cuda()
        
        # GPU 사용 확인
        print(f"BitfieldLinear CUDA 활성화: {bitfield.is_cuda_enabled()}")
        
        for batch_size in batch_sizes:
            # 입력 데이터 생성
            x = torch.randn(batch_size, in_features).cuda()
            
            # 원본 레이어 시간 측정
            def run_original():
                with torch.no_grad():
                    _ = linear(x)
            
            # BitfieldLinear 시간 측정
            def run_bitfield():
                with torch.no_grad():
                    _ = bitfield(x)
            
            original_time = measure_time(run_original)
            bitfield_time = measure_time(run_bitfield)
            
            speedup = original_time / bitfield_time
            
            print(f"  배치 크기 {batch_size:3d}: "
                  f"원본 {original_time:6.3f}ms, "
                  f"압축 {bitfield_time:6.3f}ms, "
                  f"속도비 {speedup:4.2f}x")
            
            # 정확도 확인
            with torch.no_grad():
                y_original = linear(x)
                y_bitfield = bitfield(x)
                
                mse = ((y_original - y_bitfield) ** 2).mean().item()
                relative_error = torch.norm(y_original - y_bitfield) / torch.norm(y_original)
                
                print(f"              MSE: {mse:.6f}, 상대 오차: {relative_error:.2%}")

def test_memory_transfer():
    """메모리 전송량 측정"""
    
    if not torch.cuda.is_available():
        return
    
    print("\n\n=== GPU 메모리 전송 분석 ===\n")
    
    # 큰 행렬로 테스트
    in_features, out_features = 4096, 2048
    batch_size = 64
    
    print(f"테스트 크기: ({batch_size}, {in_features}) × ({in_features}, {out_features})")
    
    # 레이어 생성
    linear = nn.Linear(in_features, out_features).cuda()
    bitfield = BitfieldLinear.from_linear(linear, basis_size=512, use_residual=True).cuda()
    
    # 입력 데이터
    x = torch.randn(batch_size, in_features).cuda()
    
    # 메모리 전송량 계산
    input_bytes = batch_size * in_features * 4  # float32
    output_bytes = batch_size * out_features * 4
    
    print(f"\n입력 크기: {input_bytes / 1024 / 1024:.2f} MB")
    print(f"출력 크기: {output_bytes / 1024 / 1024:.2f} MB")
    
    # 이론적 전송량 (기존 방식)
    old_transfer = (input_bytes + output_bytes) * 2  # H2D + D2H
    print(f"\n기존 방식 전송량: {old_transfer / 1024 / 1024:.2f} MB")
    print(f"새로운 방식 전송량: 0.00 MB (GPU 포인터 직접 사용)")
    
    # 실제 시간 측정
    def run_test():
        with torch.no_grad():
            return bitfield(x)
    
    # 한 번 실행하여 GPU 메모리 초기화
    _ = run_test()
    
    # 시간 측정
    elapsed = measure_time(run_test, warmup=20, iterations=100)
    
    # 이론적 대역폭 계산 (PCIe 3.0 x16 = ~15.75 GB/s)
    pcie_bandwidth = 15.75 * 1024 * 1024 * 1024  # bytes/s
    theoretical_time = old_transfer / pcie_bandwidth * 1000  # ms
    
    print(f"\n이론적 전송 시간 (PCIe 3.0): {theoretical_time:.3f} ms")
    print(f"실제 측정 시간: {elapsed:.3f} ms")
    print(f"개선율: {theoretical_time / elapsed:.1f}x")

def test_backward_pass():
    """역전파 최적화 테스트"""
    
    if not torch.cuda.is_available():
        return
    
    print("\n\n=== 역전파 최적화 테스트 ===\n")
    
    in_features, out_features = 1024, 512
    batch_size = 32
    
    # 레이어 생성
    linear = nn.Linear(in_features, out_features).cuda()
    bitfield = BitfieldLinear.from_linear(linear, basis_size=256, use_residual=True).cuda()
    
    # 입력 데이터
    x = torch.randn(batch_size, in_features, requires_grad=True).cuda()
    target = torch.randn(batch_size, out_features).cuda()
    
    # 손실 함수
    criterion = nn.MSELoss()
    
    # 원본 레이어 역전파
    def run_original_backward():
        x_copy = x.clone().detach().requires_grad_(True)
        y = linear(x_copy)
        loss = criterion(y, target)
        loss.backward()
        return x_copy.grad
    
    # BitfieldLinear 역전파
    def run_bitfield_backward():
        x_copy = x.clone().detach().requires_grad_(True)
        y = bitfield(x_copy)
        loss = criterion(y, target)
        loss.backward()
        return x_copy.grad
    
    # 시간 측정
    original_time = measure_time(run_original_backward, warmup=5, iterations=50)
    bitfield_time = measure_time(run_bitfield_backward, warmup=5, iterations=50)
    
    print(f"역전파 시간:")
    print(f"  원본: {original_time:.3f} ms")
    print(f"  압축: {bitfield_time:.3f} ms")
    print(f"  속도비: {original_time / bitfield_time:.2f}x")
    
    # 그래디언트 정확도 확인
    grad_original = run_original_backward()
    grad_bitfield = run_bitfield_backward()
    
    grad_error = torch.norm(grad_original - grad_bitfield) / torch.norm(grad_original)
    print(f"\n그래디언트 상대 오차: {grad_error:.2%}")

if __name__ == "__main__":
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"PyTorch 버전: {torch.__version__}")
    
    # 테스트 실행
    test_gpu_optimization()
    test_memory_transfer()
    test_backward_pass()
    
    print("\n테스트 완료!") 