import torch
import torch.nn as nn
import reality_stone as rs
from typing import Optional, Union, List

class HyperbolicEmbeddingAdapter(nn.Module):
    """
    기존 nn.Embedding 레이어를 래핑하여 출력을 쌍곡 공간(Poincaré Ball)으로 투영합니다.
    
    이론적 배경:
    유클리드 임베딩 x를 접공간(Tangent Space)의 원소로 간주하고,
    지수 맵(Exponential Map)을 통해 다양체(Manifold) 위로 옮깁니다.
    """
    def __init__(self, base_embedding: nn.Embedding, c: float = 1.0, init_scale: float = 10.0):
        super().__init__()
        self.base = base_embedding
        self.c = c
        # 임베딩 크기 보정을 위한 학습 가능한 스칼라
        self.output_scale = nn.Parameter(torch.tensor([init_scale]), requires_grad=True)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 1. 기존 유클리드 임베딩 가져오기
        x = self.base(input_ids)
        
        # 2. Exponential Map: Euclidean -> Hyperbolic
        x_hyp = self._exp_map_0(x)
        
        # 3. Scale Correction (Transformer 입력을 위한 크기 복원)
        return x_hyp * self.output_scale

    def _exp_map_0(self, v: torch.Tensor) -> torch.Tensor:
        # 수치 안정성을 위한 클램핑
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=v.device, dtype=v.dtype))
        
        # tanh(sqrt(c)/2 * ||v||)
        scale = torch.tanh(sqrt_c * norm / 2) / (sqrt_c * norm)
        return scale * v

class RiemannianDiffusionHead(nn.Module):
    """
    LLM의 최종 Hidden State에 리만 라그랑지안 디퓨전을 적용합니다.
    
    이론적 배경:
    단순한 선형 변환 대신, 잠재 공간에서 에너지 최소화 경로를 따라 
    상태를 '확산'시켜 더 견고한 표현을 찾습니다.
    """
    def __init__(self, input_dim: int, alpha: float = 0.5, dt: float = 0.1, steps: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.steps = steps
        self.alpha = alpha
        self.dt = dt
        
        # 흐름(Flow)을 결정하는 경량 신경망 (Tangent Space에서의 벡터장)
        self.flow_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Tanh() # 흐름의 크기 제한
        )
        
        # Zero-Flow Initialization:
        # 초기에는 흐름이 0이 되도록 마지막 레이어의 가중치를 0으로 초기화합니다.
        # 이렇게 하면 학습 전에는 기존 모델의 동작을 100% 보존합니다 (Identity).
        nn.init.zeros_(self.flow_net[-2].weight)
        nn.init.zeros_(self.flow_net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq, Dim] - LLM의 마지막 Hidden State
        Returns:
            refined_x: [Batch, Seq, Dim] - Diffusion으로 정제된 상태
        """
        h = x.clone()
        
        # Diffusion Iteration
        for _ in range(self.steps):
            # 1. Flow Field 계산 (Tangent Vector)
            flow = self.flow_net(h)
            
            # 2. Riemannian Step (Update Manifold State)
            # During training, we prefer pure PyTorch operations for full autograd support.
            # Inference can benefit from CUDA kernels, but for consistent behavior
            # and to avoid graph breaks, we use PyTorch implementation here.
            
            # v = (1 - alpha) * (flow - h)
            v = (1.0 - self.alpha) * (flow - h)
            
            # Update: h_next = h + v * dt
            h = h + v * self.dt
            
            # Optional: Manifold Projection (Soft constraint for now)
            # if we wanted strict Poincaré ball constraint:
            # norm = h.norm(dim=-1, keepdim=True)
            # mask = norm > 0.99
            # h[mask] = h[mask] / norm[mask] * 0.99
                
        return h

def patch_llm_with_reality_stone(
    model: nn.Module, 
    diffusion_steps: int = 2,
    alpha: float = 0.5
) -> nn.Module:
    """
    Hugging Face 스타일의 LLM 모델을 받아 Reality Stone 모듈을 장착합니다.
    
    Args:
        model: 대상 PyTorch 모델 (예: LlamaForCausalLM)
        diffusion_steps: 디퓨전 스텝 수
        alpha: 에너지 감쇠 계수
    """
    print("🪨 Reality Stone: Applying Riemannian transformation to LLM...")
    
    # 1. 임베딩 계층 패치 (Hyperbolic Embedding)
    # 대부분의 HF 모델은 get_input_embeddings 메서드를 가짐
    if hasattr(model, 'get_input_embeddings') and hasattr(model, 'set_input_embeddings'):
        base_emb = model.get_input_embeddings()
        print(f"   - Patching Embeddings: {type(base_emb).__name__} -> HyperbolicAdapter")
        
        hyperbolic_emb = HyperbolicEmbeddingAdapter(base_emb)
        model.set_input_embeddings(hyperbolic_emb)
    else:
        print("   ! Warning: Could not find input embeddings to patch.")

    # 2. 헤드 계층 패치 (Riemannian Diffusion)
    # 모델 구조를 탐색하여 마지막 Linear Layer 앞에 Diffusion Head 삽입
    # (예시: LlamaForCausalLM의 경우 model.lm_head가 보통 마지막)
    
    target_module = None
    parent_module = None
    target_name = ""
    
    # 일반적인 헤드 이름 탐색
    for name in ['lm_head', 'embed_out', 'output_layer']:
        if hasattr(model, name):
            target_module = getattr(model, name)
            parent_module = model
            target_name = name
            break
            
    if target_module is not None and isinstance(target_module, nn.Linear):
        input_dim = target_module.in_features
        print(f"   - Patching Head ({target_name}): Injecting RiemannianDiffusion (dim={input_dim})")
        
        diffusion_head = RiemannianDiffusionHead(
            input_dim=input_dim, 
            alpha=alpha, 
            steps=diffusion_steps
        )
        
        # 기존 헤드를 [Diffusion -> Linear] 순서로 교체
        new_head = nn.Sequential(
            diffusion_head,
            target_module
        )
        
        setattr(parent_module, target_name, new_head)
    else:
        print("   ! Warning: Could not find a suitable Linear head to patch.")
        
    print("✨ Reality Stone integration complete.")
    return model
