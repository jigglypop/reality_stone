import torch
import torch.nn as nn
import reality_stone as rs
from typing import Optional

class PrunedRiemannianLinear(nn.Module):
    """
    가지치기(Pruning) 기능이 내장된 리만 선형 레이어.
    유클리드 가중치 크기가 아닌, '리만 에너지 기여도'를 기준으로 가지치기를 수행합니다.
    """
    def __init__(self, original_layer: nn.Linear, curvature: float = 1.0, prune_ratio: float = 0.5):
        super().__init__()
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.curvature = curvature
        self.prune_ratio = prune_ratio
        
        # 마스크 생성 (초기에는 모두 1)
        self.register_buffer("mask", torch.ones_like(self.weight))
        
        # 초기 가지치기 수행
        self.prune_weights()

    def prune_weights(self):
        """
        리만 기하학적 중요도(Riemannian Importance)에 따라 가중치를 제거합니다.
        중요도 = |w| * (1 + curvature)  <- 단순화된 예시
        곡률이 높을수록 작은 가중치도 공간 왜곡에 큰 영향을 줄 수 있으므로 보정.
        """
        with torch.no_grad():
            # 중요도 계산
            # 곡률이 있을 때 가중치의 영향력이 증폭되므로 이를 반영
            # 여기서는 간단히 절대값 기준으로 하되, 곡률 항을 고려할 수 있음
            importance = torch.abs(self.weight)
            
            # 하위 n% 임계값 계산
            num_params = self.weight.numel()
            num_prune = int(num_params * self.prune_ratio)
            if num_prune == 0: return
            
            # Flatten & Sort
            flat_imp = importance.view(-1)
            threshold = torch.topk(flat_imp, num_prune, largest=False).values.max()
            
            # 마스크 업데이트 (중요도가 임계값보다 큰 것만 남김)
            self.mask = (importance > threshold).float()
            
            # 실제 가중치에 마스크 적용 (0으로 만듦)
            self.weight.data.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pruned Linear Transform
        # 마스크된 가중치를 사용하여 연산 (실제 희소 연산은 아니지만 0 곱셈됨)
        # * 추론 시에는 sparse matrix로 변환하여 가속 가능
        masked_weight = self.weight * self.mask
        y = torch.nn.functional.linear(x, masked_weight, self.bias)
        
        # 2. Riemannian Conformal Correction
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.9)
        conformal_factor = 1.0
        if abs(self.curvature) > 1e-5:
            denom = 1.0 - self.curvature * x_norm_sq
            conformal_factor = 2.0 / torch.clamp(denom, min=1e-5)
            
        return y * conformal_factor

class PrunedRiemannianConv1D(nn.Module):
    """
    가지치기 기능이 내장된 리만 Conv1D 레이어.
    """
    def __init__(self, original_layer: nn.Module, curvature: float = 1.0, prune_ratio: float = 0.5):
        super().__init__()
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.curvature = curvature
        self.prune_ratio = prune_ratio
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.prune_weights()

    def prune_weights(self):
        with torch.no_grad():
            importance = torch.abs(self.weight)
            num_params = self.weight.numel()
            num_prune = int(num_params * self.prune_ratio)
            if num_prune == 0: return
            
            flat_imp = importance.view(-1)
            threshold = torch.topk(flat_imp, num_prune, largest=False).values.max()
            
            self.mask = (importance > threshold).float()
            self.weight.data.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        
        size_out = x.size()[:-1] + (masked_weight.size(1),)
        x_view = x.view(-1, x.size(-1))
        y = torch.addmm(self.bias, x_view, masked_weight)
        y = y.view(size_out)
        
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.9)
        conformal_factor = 1.0
        if abs(self.curvature) > 1e-5:
            denom = 1.0 - self.curvature * x_norm_sq
            conformal_factor = 2.0 / torch.clamp(denom, min=1e-5)
            
        return y * conformal_factor


def convert_to_pruned_riemannian(model: nn.Module, curvature: float = 0.01, prune_ratio: float = 0.3) -> nn.Module:
    """
    모델을 리만 기하학 레이어로 변환하면서 동시에 가지치기(Pruning)를 수행합니다.
    prune_ratio: 제거할 가중치의 비율 (0.3 = 30% 제거)
    """
    print(f"Starting Pruned Riemannian Surgery (Ratio: {prune_ratio*100}%)...")
    
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if isinstance(module, (PrunedRiemannianLinear, PrunedRiemannianConv1D)): continue
            # print(f"Pruning & Converting Linear: {name}")
            new_layer = PrunedRiemannianLinear(module, curvature, prune_ratio)
            _replace_module(model, name, new_layer)
            count += 1
            
        elif "Conv1D" in str(type(module)):
            if isinstance(module, (PrunedRiemannianLinear, PrunedRiemannianConv1D)): continue
            # print(f"Pruning & Converting Conv1D: {name}")
            new_layer = PrunedRiemannianConv1D(module, curvature, prune_ratio)
            _replace_module(model, name, new_layer)
            count += 1
            
    print(f"Surgery Complete. {count} layers pruned and converted.")
    return model

def _replace_module(root_model, path, new_module):
    atoms = path.split('.')
    parent = root_model
    try:
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        setattr(parent, atoms[-1], new_module)
    except AttributeError:
        print(f"Warning: Failed to replace {path}")

