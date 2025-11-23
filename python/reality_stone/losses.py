import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_same_label(dists_sq: torch.Tensor, labels: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    batch_size = dists_sq.size(0)
    if batch_size < 2:
        return dists_sq.new_tensor(0.0)
    sim = -dists_sq / tau
    adj = F.softmax(sim, dim=-1)
    labels = labels.view(-1, 1)
    device = dists_sq.device
    mask_same = torch.eq(labels, labels.T).to(device)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    mask_same = mask_same & ~self_mask
    if mask_same.sum() == 0:
        return dists_sq.new_tensor(0.0)
    weighted = adj * dists_sq * mask_same.float()
    return weighted.sum() / mask_same.float().sum()


def poincare_kinetic_energy(x_hyp: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    norm_sq = torch.sum(x_hyp ** 2, dim=-1, keepdim=True)
    norm_sq = torch.clamp(norm_sq, max=(1.0 / curvature) - 1e-5)
    lambda_x = 2.0 / (1.0 - curvature * norm_sq)
    kinetic = 0.5 * (lambda_x ** 2) * norm_sq
    return kinetic.mean()


class HyperbolicSupConLoss(nn.Module):
    """
    Hyperbolic Supervised Contrastive Loss
    Uses Poincaré distance for contrastive learning.
    """
    def __init__(self, temperature: float = 0.1, curvature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.curvature = curvature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (batch_size, dim) - Points in Poincaré ball
        labels: (batch_size)
        """
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute pairwise Poincaré distances squared
        # d(u,v) = arccosh(1 + 2 * |u-v|^2 / ((1-|u|^2)(1-|v|^2)))
        # For numerical stability, we use the pre-computed distance or compute it here.
        # Assuming inputs are in the ball.
        
        x_norm_sq = torch.sum(features.pow(2), dim=1, keepdim=True)
        # Clamp for stability
        x_norm_sq = torch.clamp(x_norm_sq, max=(1.0/self.curvature) - 1e-5)
        
        # Pairwise Euclidean distance squared
        dist_euc_sq = torch.cdist(features, features, p=2).pow(2)
        
        alpha = 1.0 - self.curvature * x_norm_sq
        denom = torch.mm(alpha, alpha.T)
        gamma = 1.0 + 2.0 * self.curvature * dist_euc_sq / torch.clamp(denom, min=1e-10)
        dist_hyp = (1.0 / torch.sqrt(torch.tensor(self.curvature))) * torch.acosh(torch.clamp(gamma, min=1.0 + 1e-7))
        
        # Logits: negative distance / temperature
        logits = -dist_hyp / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        
        # Mean log-likelihood for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask.sum(1), min=1.0)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class BellmanConsistencyLoss(nn.Module):
    def __init__(self, lambda_bellman: float = 0.1, gamma: float = 0.99, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.lambda_bellman = lambda_bellman
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, apply_bellman: bool = True) -> dict:
        loss_cls = self.ce_loss(logits, labels)
        if not apply_bellman or self.lambda_bellman == 0:
            zero = logits.new_tensor(0.0)
            return {"total": loss_cls, "classification": loss_cls, "bellman": zero}
        batch_size = logits.size(0)
        idx = torch.arange(batch_size, device=logits.device)
        current_values = logits[idx, labels]
        preds = logits.argmax(dim=1)
        rewards = (preds == labels).float()
        next_values = logits.max(dim=1)[0].detach()
        target = rewards + self.gamma * next_values
        bellman_error = (current_values - target).pow(2).mean()
        total = loss_cls + self.lambda_bellman * bellman_error
        return {"total": total, "classification": loss_cls, "bellman": bellman_error}
