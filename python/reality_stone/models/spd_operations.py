import torch
import torch.nn as nn
from typing import List, Optional


def spd_log(G: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    G_stable = G + epsilon * eye
    
    eigvals, eigvecs = torch.linalg.eigh(G_stable)
    eigvals = torch.clamp(eigvals, min=epsilon)
    log_eigvals = torch.log(eigvals)
    
    log_G = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)
    return log_G


def spd_exp(log_G: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(log_G)
    exp_eigvals = torch.exp(eigvals)
    exp_eigvals = torch.clamp(exp_eigvals, min=epsilon)
    
    G = eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-2, -1)
    return G


def spd_barycenter(
    matrices: List[torch.Tensor],
    weights: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    device = matrices[0].device
    dtype = matrices[0].dtype
    
    log_matrices = [spd_log(G, epsilon) for G in matrices]
    log_matrices_stacked = torch.stack(log_matrices, dim=0)
    
    weights_normalized = weights / (weights.sum() + 1e-10)
    weights_expanded = weights_normalized.view(-1, 1, 1)
    
    log_mean = (log_matrices_stacked * weights_expanded).sum(dim=0)
    
    G_barycenter = spd_exp(log_mean, epsilon)
    
    return G_barycenter


def spd_distance(G1: torch.Tensor, G2: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    eye = torch.eye(G1.shape[-1], device=G1.device, dtype=G1.dtype)
    G1_stable = G1 + epsilon * eye
    
    try:
        L1 = torch.linalg.cholesky(G1_stable)
        L1_inv = torch.linalg.inv(L1)
    except RuntimeError:
        return torch.tensor(0.0, device=G1.device, dtype=G1.dtype)
    
    M = L1_inv @ G2 @ L1_inv.transpose(-2, -1)
    
    eigvals = torch.linalg.eigvalsh(M)
    eigvals = torch.clamp(eigvals, min=epsilon)
    log_eigvals = torch.log(eigvals)
    
    dist = torch.sqrt((log_eigvals ** 2).sum())
    
    return dist


class SPDBarycentricMixer(nn.Module):
    def __init__(self, d_metric: int = 64):
        super().__init__()
        self.d_metric = d_metric
    
    def forward(
        self,
        metric_slots: List[torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return spd_barycenter(metric_slots, weights)


class CrossLevelMetricMixer(nn.Module):
    def __init__(self, d_metric: int = 64):
        super().__init__()
        self.d_metric = d_metric
        
        self.gamma_parent = nn.Parameter(torch.tensor(0.3))
        self.gamma_self = nn.Parameter(torch.tensor(0.5))
        self.gamma_children = nn.Parameter(torch.tensor(0.2))
    
    def forward(
        self,
        G_self: torch.Tensor,
        G_parent: Optional[torch.Tensor] = None,
        G_children: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        matrices = [G_self]
        weights = [self.gamma_self]
        
        if G_parent is not None:
            matrices.append(G_parent)
            weights.append(self.gamma_parent)
        
        if G_children is not None:
            matrices.append(G_children)
            weights.append(self.gamma_children)
        
        weights_tensor = torch.stack(weights)
        weights_normalized = torch.softmax(weights_tensor, dim=0)
        
        G_effective = spd_barycenter(matrices, weights_normalized)
        
        return G_effective

