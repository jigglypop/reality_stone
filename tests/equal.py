import torch
import torch.nn as nn
from reality_stone.layers import EquivalentHyperbolicLinear

def test_equivalence():
    """단순 동등성 테스트"""
    print("Testing EquivalentHyperbolicLinear equivalence...")
    
    # 다양한 크기의 레이어 테스트
    test_configs = [
        (10, 5),
        (768, 2304),  # GPT-2 크기
        (2304, 768),  # 역방향
        (100, 100),   # 정방형
    ]
    
    all_passed = True
    
    for in_features, out_features in test_configs:
        linear = nn.Linear(in_features, out_features)
        equiv = EquivalentHyperbolicLinear.from_linear(linear)
        x = torch.randn(32, in_features)
        with torch.no_grad():
            y_original = linear(x)
            y_equiv = equiv(x)
        diff = (y_original - y_equiv).abs().mean()
        rel_diff = diff / y_original.abs().mean()
        
        print(f"\nLayer ({in_features}, {out_features}):")
        print(f"  Absolute diff: {diff:.6f}")
        print(f"  Relative diff: {rel_diff:.2%}")
        norms = torch.norm(y_equiv, p=2, dim=-1)
        in_ball = (norms < 1.0).all()
        print(f"  In Poincaré ball: {in_ball} (max norm: {norms.max():.4f})")
        if rel_diff > 0.01:  # 1% 이상 차이
            print(f"  ❌ FAILED - difference too large")
            all_passed = False
        else:
            print(f"  ✅ PASSED")
    
    return all_passed

def test_conv1d():
    """Conv1D 호환성 테스트"""
    print("\n\nTesting Conv1D compatibility...")
    # Conv1D 모사 (transformers 스타일)
    class Conv1D(nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.nx = nx
            self.weight = nn.Parameter(torch.empty(nx, nf))
            self.bias = nn.Parameter(torch.zeros(nf))
            nn.init.normal_(self.weight, std=0.02)
        
        def forward(self, x):
            return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    
    # Conv1D 레이어 생성
    conv = Conv1D(2304, 768)
    
    # 변환
    equiv = EquivalentHyperbolicLinear.from_linear(conv)
    
    # 테스트
    x = torch.randn(10, 768)
    
    with torch.no_grad():
        y_conv = conv(x).view(10, -1)
        y_equiv = equiv(x)
    
    diff = (y_conv - y_equiv).abs().mean()
    rel_diff = diff / y_conv.abs().mean()
    
    print(f"Conv1D (768 -> 2304):")
    print(f"  Absolute diff: {diff:.6f}")
    print(f"  Relative diff: {rel_diff:.2%}")
    
    if rel_diff < 0.01:
        print("  ✅ Conv1D PASSED")
        return True
    else:
        print("  ❌ Conv1D FAILED")
        return False

if __name__ == "__main__":
    print("="*60)
    print("EquivalentHyperbolicLinear Equivalence Test")
    print("="*60)
    
    passed1 = test_equivalence()
    passed2 = test_conv1d()
    
    print("\n" + "="*60)
    if passed1 and passed2:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60) 