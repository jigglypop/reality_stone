import torch
import torch.nn as nn
import reality_stone as rs
from typing import Optional

class RiemannianLinear(nn.Module):
    """
    기존 nn.Linear(y = xA^T + b)를 대체.
    """
    def __init__(self, original_layer: nn.Linear, curvature: float = 1.0):
        super().__init__()
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.curvature = curvature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Linear Transform (Euclidean Tangent Space)
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        
        # 2. Riemannian Conformal Correction
        # 입력 x의 에너지(Norm)에 따라 출력을 수축/팽창시킴
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.9)
        conformal_factor = 1.0
        if abs(self.curvature) > 1e-5:
            denom = 1.0 - self.curvature * x_norm_sq
            conformal_factor = 2.0 / torch.clamp(denom, min=1e-5)
            
        return y * conformal_factor

class RiemannianConv1D(nn.Module):
    """
    Hugging Face GPT-2 등의 Conv1D(y = xA + b)를 대체.
    Linear와 달리 가중치가 전치(Transposed)되어 있지 않음.
    """
    def __init__(self, original_layer: nn.Module, curvature: float = 1.0):
        super().__init__()
        # Conv1D stores weights as (in_features, out_features)
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.curvature = curvature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Conv1D Transform (x @ W + b)
        # nn.Linear와 달리 직접 행렬곱 수행
        size_out = x.size()[:-1] + (self.weight.size(1),)
        x_view = x.view(-1, x.size(-1))
        y = torch.addmm(self.bias, x_view, self.weight)
        y = y.view(size_out)
        
        # 2. Riemannian Conformal Correction
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.9)
        conformal_factor = 1.0
        if abs(self.curvature) > 1e-5:
            denom = 1.0 - self.curvature * x_norm_sq
            conformal_factor = 2.0 / torch.clamp(denom, min=1e-5)
            
        return y * conformal_factor

def convert_to_full_riemannian(model: nn.Module, curvature: float = 0.01) -> nn.Module:
    """
    모델의 모든 선형 연산(Linear, Conv1D)을 찾아 리만 기하학 레이어로 완전 교체합니다.
    """
    print("Starting Deep Riemannian Surgery (Full Scope)...")
    
    count = 0
    for name, module in list(model.named_modules()):
        # 1. nn.Linear 교체
        if isinstance(module, nn.Linear):
            # 이미 변환된 레이어는 건너뜀
            if isinstance(module, RiemannianLinear): continue
            
            print(f"Converting Linear: {name}")
            new_layer = RiemannianLinear(module, curvature)
            _replace_module(model, name, new_layer)
            count += 1
            
        # 2. Conv1D 교체 (HF Transformers specific)
        # 클래스 이름으로 체크하여 라이브러리 의존성 제거
        elif "Conv1D" in str(type(module)):
            # 이미 변환된 레이어는 건너뜀
            if isinstance(module, RiemannianConv1D): continue

            print(f"Converting Conv1D: {name}")
            new_layer = RiemannianConv1D(module, curvature)
            _replace_module(model, name, new_layer)
            count += 1
            
    print(f"Surgery Complete. {count} layers converted to Riemannian Manifold.")
    return model

def _replace_module(root_model, path, new_module):
    """모델 트리에서 경로(path)를 따라가 모듈을 교체하는 헬퍼"""
    atoms = path.split('.')
    parent = root_model
    try:
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        setattr(parent, atoms[-1], new_module)
    except AttributeError:
        print(f"Warning: Failed to replace {path}")

def convert_to_hyperbolic(model: nn.Module, c: float = 1.0) -> nn.Module:
    return convert_to_full_riemannian(model, curvature=c)
