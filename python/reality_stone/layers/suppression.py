import torch
import torch.nn as nn
from torch import Tensor


class HyperbolicSuppressionField(nn.Module):
    def __init__(self, base: float = 0.37, linear: float = 0.0, hyp: float = 0.1, scale: float = 1.0) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.tensor(float(base)))
        self.linear = nn.Parameter(torch.tensor(float(linear)))
        self.hyp = nn.Parameter(torch.tensor(float(hyp)))
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def compute_field(self, x: Tensor) -> Tensor:
        x_cast = x.to(dtype=self.base.dtype)
        return self.base + self.linear * x_cast + self.hyp * torch.tanh(self.scale * x_cast)

    def compute_effective_temperature(self, t0, x: Tensor) -> Tensor:
        if torch.is_tensor(t0):
            base_temp = t0.to(device=x.device, dtype=x.dtype)
        else:
            base_temp = torch.as_tensor(t0, device=x.device, dtype=x.dtype)
        field = self.compute_field(x).to(device=x.device, dtype=x.dtype)
        scale = torch.sigmoid(field)
        return base_temp * scale
