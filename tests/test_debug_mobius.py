"""
Möbius 연산 디버그 테스트
실제 norm 값 확인
"""
import torch
import reality_stone

def debug_mobius_add_cpu():
    """mobius_add_cpu norm 값 디버그"""
    dtype = torch.float32
    batch_size = 5
    dim = 3
    c = 1.0
    
    x = torch.randn(batch_size, dim, dtype=dtype) * 0.5
    y = torch.randn(batch_size, dim, dtype=dtype) * 0.5
    
    print(f"입력 x: {x}")
    print(f"입력 y: {y}")
    print(f"입력 x norms: {torch.norm(x, dim=-1)}")
    print(f"입력 y norms: {torch.norm(y, dim=-1)}")
    
    result = reality_stone.mobius_add_cpu(x, y, c)
    norms = torch.norm(result, dim=-1)
    
    print(f"결과: {result}")
    print(f"결과 norms: {norms}")
    print(f"최대 norm: {torch.max(norms)}")
    print(f"norms < 1.5: {torch.all(norms < 1.5)}")
    print(f"norms < 2.0: {torch.all(norms < 2.0)}")
    print(f"norms < 5.0: {torch.all(norms < 5.0)}")

if __name__ == "__main__":
    debug_mobius_add_cpu() 