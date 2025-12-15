import torch
import torch.nn as nn
from typing import Optional
from reality_stone.layers.poincare import poincare_distance, project_to_ball

class SemanticPreservationLoss(nn.Module):
    def __init__(
        self,
        manifold: str = "poincare",
        c: float = 1e-3,
        reduction: str = "mean",
    ):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.reduction = reduction
    
    def forward(
        self,
        original_embeddings: torch.Tensor,
        edited_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d = original_embeddings.shape
        if self.manifold == "poincare":
            orig_proj = project_to_ball(original_embeddings.reshape(B * T, d))
            edit_proj = project_to_ball(edited_embeddings.reshape(B * T, d))
            distances = poincare_distance(orig_proj, edit_proj, self.c)
            distances = distances.reshape(B, T)
        elif self.manifold == "euclidean":
            distances = torch.norm(
                original_embeddings - edited_embeddings,
                dim=-1,
            )
        else:
            raise ValueError(f"Unsupported manifold: {self.manifold}")
        
        if mask is not None:
            distances = distances * mask
            if self.reduction == "mean":
                loss = distances.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                loss = distances.sum()
            else:
                loss = distances
        else:
            if self.reduction == "mean":
                loss = distances.mean()
            elif self.reduction == "sum":
                loss = distances.sum()
            else:
                loss = distances
        
        return loss


class ContrastiveSemanticLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        manifold: str = "poincare",
        c: float = 1e-3,
    ):
        super().__init__()
        self.temperature = temperature
        self.manifold = manifold
        self.c = c
    
    def forward(
        self,
        original_embeddings: torch.Tensor,
        edited_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d = original_embeddings.shape
        
        orig_flat = original_embeddings.reshape(B * T, d)
        edit_flat = edited_embeddings.reshape(B * T, d)
        
        if self.manifold == "poincare":
            orig_flat = project_to_ball(orig_flat)
            edit_flat = project_to_ball(edit_flat)
            
            pos_distances = poincare_distance(orig_flat, edit_flat, self.c)
            
            orig_exp = orig_flat.unsqueeze(1)
            edit_exp = edit_flat.unsqueeze(0)
            
            neg_distances = torch.cdist(orig_exp, edit_exp, p=2).squeeze(1)
            
        else:
            pos_distances = torch.norm(orig_flat - edit_flat, dim=-1)
            neg_distances = torch.cdist(orig_flat.unsqueeze(0), edit_flat.unsqueeze(0)).squeeze(0)
        
        pos_scores = -pos_distances / self.temperature
        neg_scores = -neg_distances / self.temperature
        
        eye = torch.eye(B * T, device=neg_scores.device).bool()
        neg_scores = neg_scores.masked_fill(eye, float('-inf'))
        
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros(B * T, dtype=torch.long, device=logits.device)
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        return loss

