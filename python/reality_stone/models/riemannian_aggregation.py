import torch
import torch.nn as nn
from typing import List, Optional, Tuple

try:
    from reality_stone.layers.poincare import (
        poincare_distance,
        project_to_ball,
        exp_map_zero,
        log_map_zero,
    )
    from reality_stone.layers.lorentz import lorentz_distance
    HAS_REALITY_STONE = True
except Exception:
    HAS_REALITY_STONE = False


class RiemannianAggregation(nn.Module):
    def __init__(
        self,
        d_model: int,
        manifold: str = "poincare",
        c: float = 1e-3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.manifold = manifold
        self.c = abs(c)
        self.temperature = temperature
        
    def forward(
        self,
        children_states: torch.Tensor,
        metric_ctx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = children_states.shape
        device = children_states.device
        
        if N == 0:
            return torch.zeros(B, d, device=device)
        
        if metric_ctx is not None:
            d_metric = metric_ctx.size(-1)
            if d_metric != d:
                if d_metric < d:
                    pad_size = d - d_metric
                    metric_ctx_resized = torch.nn.functional.pad(
                        metric_ctx, (0, pad_size, 0, pad_size), value=0.0
                    )
                    for i in range(d_metric, d):
                        metric_ctx_resized[:, i, i] = 1.0
                    metric_ctx = metric_ctx_resized
                else:
                    metric_ctx = metric_ctx[:, :d, :d]
            
            children_states = torch.einsum("bdk,bnk->bnd", metric_ctx, children_states)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
            mu = (children_states * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            mu = children_states.mean(dim=1)  # [B, d]
        
        if not HAS_REALITY_STONE or self.manifold == "euclidean":
            # Fallback: Euclidean mean
            return mu
        
        # Riemannian aggregation
        if self.manifold == "poincare":
            return self._poincare_agg(children_states, mu, mask, temperature_override)
        elif self.manifold == "lorentz":
            return self._lorentz_agg(children_states, mu, mask, temperature_override)
        else:
            return mu
    
    def _poincare_agg(
        self,
        children_states: torch.Tensor,
        mu: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = children_states.shape

        children_states = project_to_ball(children_states.reshape(-1, d)).reshape(B, N, d)
        mu = project_to_ball(mu)

        mu_exp = mu.unsqueeze(1).expand(B, N, d).reshape(B * N, d)
        child_flat = children_states.reshape(B * N, d)
        dist_flat = poincare_distance(mu_exp, child_flat, self.c)
        distances = dist_flat.reshape(B, N)
        temp = temperature_override if temperature_override is not None else torch.as_tensor(self.temperature, device=distances.device, dtype=distances.dtype)
        scores = -distances / temp

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        alpha = torch.softmax(scores, dim=1)

        weighted_mean = (alpha.unsqueeze(-1) * children_states).sum(dim=1)
        result = project_to_ball(weighted_mean)

        return result
    
    def _lorentz_agg(
        self,
        children_states: torch.Tensor,
        mu: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = children_states.shape
        
        distances = []
        for i in range(N):
            dist = lorentz_distance(
                mu,
                children_states[:, i, :],
                self.c
            )
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)
        temp = temperature_override if temperature_override is not None else torch.as_tensor(self.temperature, device=distances.device, dtype=distances.dtype)
        scores = -distances / temp
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        alpha = torch.softmax(scores, dim=1)
        
        weighted_mean = (alpha.unsqueeze(-1) * children_states).sum(dim=1)
        
        return weighted_mean

