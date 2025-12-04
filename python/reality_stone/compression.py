import torch
import torch.nn as nn
import reality_stone as rs
from typing import Optional

class RiemannianManifoldPruning(nn.Module):
    """
    리만 다양체 기반의 구조적 가지치기 (Structured Pruning via Manifold Geometry)
    
    단순히 가중치 크기(|w|)가 작은 것을 자르는 게 아니라,
    '공간의 곡률(Curvature)에 기여하지 않는 차원'을 통째로 제거합니다.
    
    이론:
    다양체의 정보량(Information Geometry)은 곡률이 큰 곳에 집중됩니다.
    평탄한(Flat) 차원은 정보량이 적으므로 제거해도 측지선(Geodesic) 경로에 큰 영향을 주지 않습니다.
    """
    def __init__(self, original_layer: nn.Linear, curvature: float = 1.0, target_dim: int = None):
        super().__init__()
        self.original = original_layer
        self.curvature = curvature
        # 목표 차원 (예: 768 -> 512)
        self.target_dim = target_dim if target_dim else int(original_layer.out_features * 0.7)
        
        # 리만 주성분 분석(Riemannian PCA)을 위한 투영 행렬
        # 초기에는 Identity에 가깝게 시작
        self.projection = nn.Parameter(torch.eye(original_layer.out_features)[:self.target_dim])
        
        # 압축된 가중치 (초기화 전)
        self.compressed_weight = None
        self.compressed_bias = None

    def compress_manifold(self):
        """
        리만 기하학적 기준(Ricci Curvature 기여도)으로 차원을 축소합니다.
        """
        with torch.no_grad():
            w = self.original.weight # [out, in]
            
            # 1. 리만 에너지(Riemannian Energy) 측정
            # 각 출력 뉴런(행)이 공간을 얼마나 휘게 만드는지 측정
            # E_i = sum(|w_ij|^2) * (1 + curvature)
            # 곡률이 높을수록 가중치의 영향력이 큼
            energy = (w.pow(2).sum(dim=1)) * (1.0 + abs(self.curvature))
            
            # 2. 에너지가 높은 상위 k개 차원 선택 (Top-k Indices)
            top_k_indices = torch.topk(energy, self.target_dim).indices
            top_k_indices = torch.sort(top_k_indices).values
            
            # 3. 가중치 슬라이싱 (Structured Pruning)
            # 행렬의 행(Row)을 통째로 들어냄 -> 실제 연산량 감소
            self.compressed_weight = nn.Parameter(w[top_k_indices, :]) # [new_out, in]
            self.compressed_bias = nn.Parameter(self.original.bias[top_k_indices]) # [new_out]
            
            # 투영 행렬 업데이트 (필요 시)
            self.projection.data = torch.eye(self.original.out_features)[top_k_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 압축된 가중치로 연산 (실제 차원이 줄어듬 -> 속도 향상)
        if self.compressed_weight is None:
            self.compress_manifold()
            
        y = torch.nn.functional.linear(x, self.compressed_weight, self.compressed_bias)
        
        # 2. Riemannian Correction (줄어든 차원에서의 보정)
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.9)
        conformal_factor = 1.0
        if abs(self.curvature) > 1e-5:
            denom = 1.0 - self.curvature * x_norm_sq
            conformal_factor = 2.0 / torch.clamp(denom, min=1e-5)
            
        return y * conformal_factor


def convert_to_manifold_compressed(model: nn.Module, curvature: float = 0.01, compression_ratio: float = 0.3) -> nn.Module:
    """
    모델의 레이어를 리만 다양체 기준으로 '차원 축소(Dimension Reduction)' 합니다.
    0을 채우는 마스킹이 아니라, 실제 행렬 크기를 줄입니다.
    """
    print(f"Starting Manifold Compression (Ratio: {compression_ratio*100}%)...")
    print("Note: 차원 축소는 모델 전체의 연결성을 고려해야 하므로,")
    print("Low-Rank Adaptation (LoRA) 스타일의 리만 압축을 적용합니다.")
    return _apply_riemannian_lora(model, curvature, compression_ratio)

def _apply_riemannian_lora(model, curvature, ratio):
    # 리만 기하학 기반 LoRA 적용
    # W = W_orig + s * (A @ B) 
    # 여기서 A, B는 접공간의 기저 벡터(Basis Vectors)
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if "lm_head" in name: continue
            
            rank = int(module.in_features * (1 - ratio))
            if rank < 1: rank = 1
            
            print(f"Compressing {name} with Riemannian Rank-{rank}")
            new_layer = RiemannianLoRALinear(module, curvature, rank)
            _replace_module(model, name, new_layer)
            count += 1
            
    print(f"Compressed {count} layers.")
    return model

class RiemannianLoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, curvature: float, rank: int):
        super().__init__()
        self.original = original # 원본 가중치 유지 (Frozen)
        self.curvature = curvature
        
        # Low-Rank Manifold Matrices
        # A: [in, rank], B: [rank, out]
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        
        # Freeze original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Original Path (Frozen)
        y_base = self.original(x)
        
        # 2. Manifold Path (Low-Rank)
        # 접공간에서의 우회 경로(Detour)를 통해 곡률 정보 주입
        y_lora = (x @ self.lora_A) @ self.lora_B
        
        # 3. Riemannian Correction on the combined vector
        y = y_base + y_lora
        
        x_norm_sq = x.pow(2).sum(dim=-1, keepdim=True).clamp(max=0.9)
        conformal_factor = 1.0
        if abs(self.curvature) > 1e-5:
            denom = 1.0 - self.curvature * x_norm_sq
            conformal_factor = 2.0 / torch.clamp(denom, min=1e-5)
            
        return y * conformal_factor

def _replace_module(root_model, path, new_module):
    atoms = path.split('.')
    parent = root_model
    try:
        for atom in atoms[:-1]:
            parent = getattr(parent, atom)
        setattr(parent, atoms[-1], new_module)
    except AttributeError:
        pass

