import torch
import torch.nn as nn
from typing import List, Dict, Optional

from reality_stone.layers.poincare import poincare_distance, project_to_ball
from reality_stone.layers.lorentz import lorentz_distance


class ProductManifoldDistance(nn.Module):
    def __init__(
        self,
        manifolds: List[str],
        dimensions: List[int],
        lambdas: Optional[List[float]] = None,
        c_poincare: float = 1e-3,
        c_lorentz: float = -1.0,
    ):
        super().__init__()
        self.manifolds = manifolds
        self.dimensions = dimensions
        self.c_poincare = c_poincare
        self.c_lorentz = c_lorentz
        
        if lambdas is None:
            lambdas = [1.0] * len(manifolds)
        
        self.lambdas = nn.Parameter(torch.tensor(lambdas, dtype=torch.float32))
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        total_distance_sq = 0.0
        offset = 0
        
        lambdas_normalized = torch.softmax(self.lambdas, dim=0)
        
        for i, (manifold, dim) in enumerate(zip(self.manifolds, self.dimensions)):
            x_i = x[..., offset:offset+dim]
            y_i = y[..., offset:offset+dim]
            
            if manifold == "poincare":
                x_i_proj = project_to_ball(x_i.reshape(-1, dim)).reshape(x_i.shape)
                y_i_proj = project_to_ball(y_i.reshape(-1, dim)).reshape(y_i.shape)
                
                dist_i = poincare_distance(
                    x_i_proj.reshape(-1, dim),
                    y_i_proj.reshape(-1, dim),
                    self.c_poincare
                ).reshape(x_i.shape[:-1])
                
            elif manifold == "lorentz":
                dist_i = lorentz_distance(
                    x_i.reshape(-1, dim),
                    y_i.reshape(-1, dim),
                    self.c_lorentz
                ).reshape(x_i.shape[:-1])
                
            elif manifold == "euclidean":
                dist_i = torch.norm(x_i - y_i, dim=-1)
                
            else:
                dist_i = torch.norm(x_i - y_i, dim=-1)
            
            total_distance_sq += lambdas_normalized[i] * (dist_i ** 2)
            offset += dim
        
        return torch.sqrt(total_distance_sq + 1e-8)


class ProductManifoldAttention(nn.Module):
    def __init__(
        self,
        manifolds: List[str],
        dimensions: List[int],
        lambdas: Optional[List[float]] = None,
        temperature: float = 0.1,
        c_poincare: float = 1e-3,
        c_lorentz: float = -1.0,
    ):
        super().__init__()
        
        self.distance_fn = ProductManifoldDistance(
            manifolds=manifolds,
            dimensions=dimensions,
            lambdas=lambdas,
            c_poincare=c_poincare,
            c_lorentz=c_lorentz,
        )
        self.temperature = temperature
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = queries.shape
        _, M, _ = keys.shape
        
        q_exp = queries.unsqueeze(2).expand(B, N, M, D)
        k_exp = keys.unsqueeze(1).expand(B, N, M, D)
        
        distances = self.distance_fn(q_exp, k_exp)
        
        scores = -distances / self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        output = torch.einsum('bnm,bmd->bnd', attention_weights, values)
        
        return output


class MultiLevelProductManifold(nn.Module):
    def __init__(
        self,
        level_configs: List[Dict],
        d_model: int = 768,
    ):
        super().__init__()
        self.level_configs = level_configs
        self.d_model = d_model
        
        self.level_projections = nn.ModuleList()
        self.level_distances = nn.ModuleList()
        
        for config in level_configs:
            manifolds = config.get("manifolds", ["euclidean"])
            dimensions = config.get("dimensions", [d_model])
            lambdas = config.get("lambdas", None)
            
            proj = nn.Linear(d_model, sum(dimensions))
            self.level_projections.append(proj)
            
            dist_fn = ProductManifoldDistance(
                manifolds=manifolds,
                dimensions=dimensions,
                lambdas=lambdas,
            )
            self.level_distances.append(dist_fn)
    
    def compute_multilevel_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        level_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if level_weights is None:
            level_weights = torch.ones(len(self.level_configs), device=x.device)
        
        level_weights = torch.softmax(level_weights, dim=0)
        
        total_distance_sq = 0.0
        
        for i, (proj, dist_fn) in enumerate(zip(self.level_projections, self.level_distances)):
            x_proj = proj(x)
            y_proj = proj(y)
            
            dist_i = dist_fn(x_proj, y_proj)
            
            total_distance_sq += level_weights[i] * (dist_i ** 2)
        
        return torch.sqrt(total_distance_sq + 1e-8)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        level_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.compute_multilevel_distance(x, y, level_weights)

