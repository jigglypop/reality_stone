import torch
import torch.nn as nn
from typing import Optional

try:
    from reality_stone import _rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

class HyperbolicSuppressionField(nn.Module):
    """
    SFE 이론의 가변 억압장(Variable Suppression Field) 구현체.
    
    epsilon(x) = base + alpha * x + beta * tanh(gamma * x)
    
    여기서 x는 '상태의 크기' 또는 '거리'를 의미하며, 
    깊어질수록(x가 클수록) 억압장이 변하여 유효 질량(m_eff)과 창의성(Temperature)을 조절함.
    """
    def __init__(
        self, 
        base: float = 0.37, 
        linear: float = 0.0, 
        hyp: float = 0.0, 
        scale: float = 1.0,
        use_rust: bool = True
    ):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(base))
        self.linear = nn.Parameter(torch.tensor(linear))
        self.hyp = nn.Parameter(torch.tensor(hyp))
        self.scale = nn.Parameter(torch.tensor(scale))
        self.use_rust = use_rust and _HAS_RUST

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 상태 텐서. (보통 Norm 또는 Distance)
            
        Returns:
            epsilon (torch.Tensor): 억압 계수 필드. 값 범위는 이론상 [0, 1)이어야 함.
        """
        if self.use_rust and not x.requires_grad and x.device.type == "cpu":
            # Rust 커널 사용 (Inference 가속)
            return torch.from_numpy(
                _rust.compute_suppression_field(
                    x.numpy(), 
                    self.base.item(), 
                    self.linear.item(), 
                    self.hyp.item(), 
                    self.scale.item()
                )
            ).to(x.device)
        
        # PyTorch 구현 (Autograd 지원)
        return self.base + self.linear * x + self.hyp * torch.tanh(self.scale * x)

    def compute_effective_mass(self, m0: float, x: torch.Tensor) -> torch.Tensor:
        """
        m_eff = m0 * (1 - epsilon(x))
        """
        eps = self.forward(x)
        # epsilon이 1을 넘지 않도록 안전장치 (Soft clamp)
        eps = torch.tanh(eps) 
        return m0 * (1.0 - eps)

    def compute_effective_temperature(self, t0: float, x: torch.Tensor) -> torch.Tensor:
        """
        T_eff = T0 / (1 - epsilon(x))
        질량이 줄어들면(epsilon 증가), 관성이 줄어들어 요동(Temperature)이 커지는 효과.
        """
        eps = self.forward(x)
        eps = 0.9 * torch.tanh(eps) # 1.0 도달 방지
        return t0 / (1.0 - eps + 1e-6)

