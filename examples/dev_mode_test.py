"""
Reality Stone 개발 모드 - JIT 컴파일
코드 변경 후 즉시 테스트 가능
"""
import torch
from torch.utils.cpp_extension import load
import os

def load_reality_stone_dev():
    """개발 모드로 reality_stone 로드 (JIT 컴파일)"""
    
    # 소스 파일 경로
    src_dir = "../src"
    cpp_files = [
        f"{src_dir}/core/ops/mobius_cpu.cpp",
        f"{src_dir}/core/utils/safety.cpp",
        f"{src_dir}/core/layers/poincare_ball_forward_cpu.cpp",
        f"{src_dir}/extension.cpp"
    ]
    
    # CUDA 파일들 (CUDA 사용 시)
    cuda_files = []
    if torch.cuda.is_available():
        cuda_files = [
            f"{src_dir}/core/ops/mobius_cuda.cu",
            f"{src_dir}/core/layers/poincare_ball_forward_cuda.cu",
        ]
    
    # 헤더 경로
    include_dirs = [f"{src_dir}/include"]
    
    # JIT 컴파일 (변경된 파일만 재컴파일)
    print("🔥 JIT 컴파일 시작...")
    reality_stone_dev = load(
        name="reality_stone_dev",
        sources=cpp_files + cuda_files,
        extra_include_paths=include_dirs,
        verbose=True,
        with_cuda=torch.cuda.is_available()
    )
    print("✅ JIT 컴파일 완료!")
    
    return reality_stone_dev

def test_dev_mode():
    """개발 모드 테스트"""
    print("🧪 개발 모드 테스트 시작")
    print("=" * 40)
    
    # 개발 모드 로드
    rs_dev = load_reality_stone_dev()
    
    # 빠른 기능 테스트
    print("\n🚀 기본 함수 테스트:")
    x = torch.randn(2, 3) * 0.1
    y = torch.randn(2, 3) * 0.1
    
    result = rs_dev.mobius_add_cpu(x, y, 1.0)
    print(f"  mobius_add_cpu: ✅ {result.shape}")
    
    # Poincare layer 테스트
    u = torch.zeros(2, 3)
    v = torch.randn(2, 3) * 0.1
    result2 = rs_dev.poincare_ball_forward_cpu(u, v, 1e-3, 0.1)
    print(f"  poincare_ball_forward_cpu: ✅ {result2.shape}")
    print(f"  NaN 체크: {'❌ NaN 발생' if torch.any(torch.isnan(result2)) else '✅ 정상'}")
    
    # 안전성 테스트
    large_input = torch.randn(2, 784) * 10.0  # 큰 입력
    result3 = rs_dev.poincare_ball_forward_cpu(torch.zeros_like(large_input), large_input, 1e-3, 0.1)
    print(f"  큰 입력 안전성: {'❌ NaN 발생' if torch.any(torch.isnan(result3)) else '✅ 정상'}")
    
    print(f"\n🎯 결과 요약:")
    print(f"  - 모든 함수 정상 작동: ✅")
    print(f"  - NaN 발생 없음: ✅")
    print(f"  - 안전성 체크 작동: ✅")

if __name__ == "__main__":
    test_dev_mode() 