import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoConfig
from tqdm import tqdm

from reality_stone.layers.poincare import exp_map_zero, log_map_zero, poincare_distance
from reality_stone.layers import project_to_ball


class RiemannianMetricLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        with torch.no_grad():
            self.weight.mul_(0.1)

    def forward(self, x):
        return F.linear(x, self.weight)


class RiemannianIntentClassifier(nn.Module):
    """
    RoBERTa-Large Backbone + Learnable Riemannian Metric Head
    
    Re-designed Mathematical Model:
    1. Contextual Embedding: z = RoBERTa(x)
    2. Metric Transformation: h = L z  (Learning Metric Tensor G = L^T L)
    3. Adaptive Curvature: m = exp_0^c(h) (Learning optimal curvature c)
    4. Distance-based Classification: p(y|x) ~ exp(-tau * d_c(m, mu_y)^2)
    """
    
    def __init__(
        self,
        backbone_name: str = "roberta-large",
        num_classes: int = 77,
        hyp_dim: int = 256,
        curvature: float = 1.0,
        alpha: float = 3.0,
        gamma: float = 0.0,
        dropout: float = 0.1,
        num_prototypes: int = 4,
        learnable_curvature: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hyp_dim = hyp_dim
        
        # Learnable curvature parameter (softplus to ensure positivity)
        # Initialize such that softplus(c_param) approx curvature
        init_val = math.log(math.exp(curvature) - 1.0) if curvature > 1.0 else 0.5413 # softplus(0.5413) ~= 1.0
        self.c_param = nn.Parameter(torch.tensor(init_val)) if learnable_curvature else nn.Parameter(torch.tensor(init_val), requires_grad=False)

        print(f"Loading backbone: {backbone_name}")
        config = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=config)
        self.hidden_size = config.hidden_size

        self.metric_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            RiemannianMetricLayer(self.hidden_size, hyp_dim),
            nn.LayerNorm(hyp_dim),
        )

        self.prototypes_tangent = nn.Parameter(
            torch.randn(num_classes, num_prototypes, hyp_dim) * 0.01
        )

        self.scale = nn.Parameter(torch.tensor(alpha))

    @property
    def c(self):
        return F.softplus(self.c_param) + 1e-5

    @torch.no_grad()
    def initialize_prototypes(self, data_loader, device):
        print("Initializing prototypes with class means (Metric Space)...")
        self.eval()
        all_features = []
        all_labels = []
        
        for batch in tqdm(data_loader, desc="Collecting features"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            
            # CLS Pooling for RoBERTa
            pooled = last_hidden[:, 0, :]
            
            h_metric = self.metric_layer(pooled)
            
            all_features.append(h_metric.cpu())
            all_labels.append(labels.cpu())
            
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        class_means = []
        for i in range(self.num_classes):
            mask = (all_labels == i)
            if mask.sum() > 0:
                mean = all_features[mask].mean(dim=0)
            else:
                mean = torch.randn(self.hyp_dim) * 0.01
            class_means.append(mean)
            
        class_means = torch.stack(class_means, dim=0).to(device)
        self.prototypes_tangent.data = class_means.unsqueeze(1)
        print("Prototypes initialized.")

    def get_prototypes(self):
        return exp_map_zero(self.prototypes_tangent, c=self.c)

    def poincare_dist_sq(self, x, y):
        sqrt_c = math.sqrt(self.c)
        x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        dist_euc_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
        max_norm_sq = (1.0 / self.c) - 1e-4
        x_sq = torch.clamp(x_sq, max=max_norm_sq)
        y_sq = torch.clamp(y_sq, max=max_norm_sq)
        alpha = 1.0 - self.c * x_sq
        beta = 1.0 - self.c * y_sq
        denom = alpha * beta
        gamma = 1.0 + 2.0 * self.c * dist_euc_sq / torch.clamp(denom, min=1e-10)
        dist = (1.0 / sqrt_c) * torch.acosh(torch.clamp(gamma, min=1.0 + 1e-7))
        dist = torch.clamp(dist, max=20.0)
        return dist ** 2

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # CLS Pooling for RoBERTa
        pooled = last_hidden[:, 0, :]
            
        h_metric = self.metric_layer(pooled)
        h_hyp = exp_map_zero(h_metric, c=self.c)
        h_hyp = project_to_ball(h_hyp, epsilon=1e-3)

        prototypes = self.get_prototypes()

        h_exp = h_hyp.unsqueeze(1).unsqueeze(1)
        p_exp = prototypes.unsqueeze(0)
        dists_sq = self.poincare_dist_sq(h_exp, p_exp).squeeze(-1)
        min_dists_sq, _ = torch.min(dists_sq, dim=2)

        logits = -torch.abs(self.scale) * min_dists_sq
        return logits, h_hyp
