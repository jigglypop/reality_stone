import torch
import torch.nn as nn
from typing import Dict, List, Optional
import torch.nn.functional as F

from reality_stone.models.spd_operations import spd_barycenter, CrossLevelMetricMixer


class MetricSlot(nn.Module):
    def __init__(self, d_metric: int = 64):
        super().__init__()
        self.d_metric = d_metric
        
        self.diag_params = nn.Parameter(torch.zeros(d_metric))
        self.low_rank_U = nn.Parameter(torch.randn(d_metric, d_metric // 4))
    
    def forward(self) -> torch.Tensor:
        diag = F.softplus(self.diag_params) + 1e-4
        D = torch.diag(diag)
        
        UUT = self.low_rank_U @ self.low_rank_U.t()
        
        G = D + UUT
        
        return G
    
    def get_cholesky(self) -> torch.Tensor:
        G = self.forward()
        
        eye = torch.eye(self.d_metric, device=G.device, dtype=G.dtype)
        G_stable = G + 1e-6 * eye
        
        try:
            L = torch.linalg.cholesky(G_stable)
        except RuntimeError:
            L = eye
        
        return L


class LearnableMetricRouter(nn.Module):
    def __init__(
        self,
        d_metric: int = 64,
        num_slots: int = 16,
        num_topics: int = 8,
    ):
        super().__init__()
        self.d_metric = d_metric
        self.num_slots = num_slots
        self.num_topics = num_topics
        
        self.metric_slots = nn.ModuleList([
            MetricSlot(d_metric) for _ in range(num_slots)
        ])
        
        self.topic_to_slot_weights = nn.Parameter(
            torch.randn(num_topics, num_slots)
        )
        
        self.cross_level_mixer = CrossLevelMetricMixer(d_metric)
    
    def forward(
        self,
        topic_probs: torch.Tensor,
        parent_metric: Optional[torch.Tensor] = None,
        children_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = topic_probs.shape
        device = topic_probs.device
        
        slot_weights = torch.einsum(
            'btc,cs->bts',
            topic_probs,
            self.topic_to_slot_weights,
        )
        slot_weights = F.softmax(slot_weights, dim=-1)
        
        slot_matrices = []
        for slot in self.metric_slots:
            G = slot.forward()
            slot_matrices.append(G)
        
        metric_ctx_list = []
        for b in range(B):
            for t in range(T):
                weights_bt = slot_weights[b, t]
                
                G_bt = spd_barycenter(slot_matrices, weights_bt)
                
                if parent_metric is not None or children_metrics is not None:
                    G_parent = parent_metric[b] if parent_metric is not None else None
                    G_children = children_metrics[b, t] if children_metrics is not None else None
                    
                    G_bt = self.cross_level_mixer(G_bt, G_parent, G_children)
                
                eye = torch.eye(self.d_metric, device=device, dtype=G_bt.dtype)
                G_bt_stable = G_bt + 1e-6 * eye
                
                try:
                    L_bt = torch.linalg.cholesky(G_bt_stable)
                except RuntimeError:
                    L_bt = eye
                
                metric_ctx_list.append(L_bt)
        
        metric_ctx = torch.stack(metric_ctx_list, dim=0).view(B, T, self.d_metric, self.d_metric)
        
        return metric_ctx
    
    def get_metric_parameters(self) -> List[nn.Parameter]:
        params = []
        for slot in self.metric_slots:
            params.extend(list(slot.parameters()))
        params.append(self.topic_to_slot_weights)
        params.extend(list(self.cross_level_mixer.parameters()))
        return params


class MetricRegularization(nn.Module):
    def __init__(self, d_metric: int = 64):
        super().__init__()
        self.d_metric = d_metric
    
    def forward(self, metric_ctx: torch.Tensor) -> torch.Tensor:
        B, T, d, _ = metric_ctx.shape
        device = metric_ctx.device
        
        G = torch.einsum('btij,btkj->btik', metric_ctx, metric_ctx)
        
        eye = torch.eye(d, device=device, dtype=G.dtype)
        
        diff = G - eye.unsqueeze(0).unsqueeze(0)
        loss = (diff ** 2).sum(dim=(-2, -1)).mean()
        
        return loss


class CurvatureRegularization(nn.Module):
    def __init__(
        self,
        c_poincare_target: float = 1e-3,
        c_lorentz_target: float = -1.0,
    ):
        super().__init__()
        self.c_poincare_target = c_poincare_target
        self.c_lorentz_target = c_lorentz_target
    
    def forward(
        self,
        c_poincare: float,
        c_lorentz: float,
    ) -> torch.Tensor:
        loss_p = (c_poincare - self.c_poincare_target) ** 2
        loss_l = (c_lorentz - self.c_lorentz_target) ** 2
        
        loss = torch.tensor(loss_p + loss_l, dtype=torch.float32)
        
        return loss

