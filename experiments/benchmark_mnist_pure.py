import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import math
import reality_stone as rs
from reality_stone.optim import PoincareRiemannianAdam

from reality_stone import (
    poincare_distance,
)

# PyTorch native implementations for differentiability
def lorentz_distance_torch(x, y, c, eps=1e-7):
    # x, y: (..., d+1)
    # Minkowski inner: x0y0 - x1y1 - ...
    inner = x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    sqrt_c = torch.sqrt(c)
    arg = (c * inner).clamp(min=1.0 + eps)
    dist = torch.acosh(arg) / sqrt_c
    return dist

def klein_distance_torch(x, y, c, eps=1e-7):
    # x, y: (..., d)
    uv = (x * y).sum(dim=-1)
    u2 = (x * x).sum(dim=-1)
    v2 = (y * y).sum(dim=-1)
    
    num = 1.0 - c * uv
    den_sq = (1.0 - c * u2) * (1.0 - c * v2)
    den = torch.sqrt(den_sq.clamp(min=eps))
    
    arg = (num / den.clamp(min=eps)).clamp(min=1.0 + eps)
    sqrt_c = torch.sqrt(c)
    
    dist = torch.acosh(arg) / sqrt_c
    return dist

def project_to_ball(x: torch.Tensor, c: float = 1.0, epsilon: float = 1e-7) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    max_norm = (1.0 / torch.sqrt(c)) - epsilon
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

def project_klein(x, c, epsilon=1e-7):
    norm = x.norm(dim=-1, keepdim=True)
    max_norm = 1.0 / torch.sqrt(c) - epsilon
    return torch.where(norm > max_norm, x / norm * max_norm, x)

class BioGeometricEncoder(nn.Module):
    """
    학습 가능한 파라미터가 없는 고정된 생물학적/물리적 인코더.
    뇌의 V1 영역처럼 Random/Gabor 필터 역할을 수행하여 입력을 고차원 특징 공간으로 매핑합니다.
    """
    def __init__(self, in_channels=1, out_dim=2048):
        super().__init__()
        # Random projection layer (Fixed weights, not trainable)
        # Expands 784 (28x28) -> 2048 dimensions to create sparse representation
        self.proj = nn.Linear(28*28, out_dim, bias=False)
        # Initialize with orthogonal weights for better isometry
        nn.init.orthogonal_(self.proj.weight)
        self.proj.weight.requires_grad = False # FREEZE! No learning here.

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        # 1. Random Projection (Synaptic connection)
        x = self.proj(x)
        # 2. Non-linearity (Neuron firing)
        x = torch.relu(x)
        return x

class PureManifoldModel(nn.Module):
    def __init__(self, model_type='poincare', in_dim=784, c=1.0):
        super().__init__()
        self.model_type = model_type
        self.c_log = nn.Parameter(torch.tensor(math.log(c)))
        
        # Fixed Biological Encoder (No learning)
        self.feature_dim = 2048
        self.encoder = BioGeometricEncoder(out_dim=self.feature_dim)
        
        # === Only Prototypes are Trainable ===
        init_protos = torch.randn(10, self.feature_dim) * 0.01
        self.prototypes = nn.Parameter(init_protos)

    @property
    def c(self):
        return self.c_log.exp()

    def forward(self, x):
        B = x.size(0)
        
        # Fixed transform: Pixel -> High-dim Feature Space
        with torch.no_grad():
            x = self.encoder(x)
            
        c_val = self.c
        
        if self.model_type == 'poincare':
            # Project high-dim features to Poincare Ball
            h = project_to_ball(x, c=c_val)
            p = project_to_ball(self.prototypes, c=c_val)
            
            h_exp = h.unsqueeze(1).expand(B, 10, self.feature_dim)
            p_exp = p.unsqueeze(0).expand(B, 10, self.feature_dim)
            
            dist = poincare_distance(
                h_exp.reshape(-1, self.feature_dim), 
                p_exp.reshape(-1, self.feature_dim), 
                c=c_val, 
                eps=1e-7
            ).reshape(B, 10)
            
        elif self.model_type == 'klein':
            h = project_klein(x, c=c_val)
            p = project_klein(self.prototypes, c=c_val)
            
            h_exp = h.unsqueeze(1).expand(B, 10, self.feature_dim)
            p_exp = p.unsqueeze(0).expand(B, 10, self.feature_dim)
            
            dist = klein_distance_torch(h_exp, p_exp, c=c_val)
            
        elif self.model_type == 'lorentz':
            sq_h = (x * x).sum(dim=-1, keepdim=True)
            time_h = torch.sqrt(1.0/c_val + sq_h)
            h_l = torch.cat([time_h, x], dim=-1)
            
            sq_p = (self.prototypes * self.prototypes).sum(dim=-1, keepdim=True)
            time_p = torch.sqrt(1.0/c_val + sq_p)
            p_l = torch.cat([time_p, self.prototypes], dim=-1)
            
            h_exp = h_l.unsqueeze(1).expand(B, 10, self.feature_dim + 1)
            p_exp = p_l.unsqueeze(0).expand(B, 10, self.feature_dim + 1)
            
            dist = lorentz_distance_torch(h_exp, p_exp, c=c_val)

        return -dist

def run_pure_experiment(model_type='poincare'):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== [PURE MANIFOLD] Running Experiment: {model_type.upper()} ===")
    print(f"Structure: Input(784) -> Manifold -> Distance(10 Protos)")
    print(f"No Linear Layers. Pure Geometry.")
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = PureManifoldModel(model_type=model_type, in_dim=2048, c=0.1).to(DEVICE)
    crit = nn.CrossEntropyLoss()

    # 1. Optimizer for Curvature
    opt_c = optim.Adam([model.c_log], lr=0.01)

    # 2. Optimizer for Prototypes
    if model_type == 'poincare':
        # Use Riemannian Adam to move prototypes on the manifold naturally
        opt_proto = PoincareRiemannianAdam(
            [{'params': model.prototypes, 'c': model.c.item()}],
            c=model.c.item(),
            lr=0.01, 
            max_norm_eps=1e-7
        )
    else:
        # Standard Adam for others (Euclidean approximation for now)
        opt_proto = optim.Adam([model.prototypes], lr=0.01)

    epochs = 5
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        
        if model_type == 'poincare':
            current_c = model.c.item()
            for group in opt_proto.param_groups:
                group['c'] = current_c

        start = time.time()
        total_loss = 0.0
        total_samples = 0

        for x, y in tqdm(train_loader, desc=f"Pure {model_type} Ep {epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            opt_c.zero_grad()
            opt_proto.zero_grad()
            
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            
            opt_c.step()
            
            if model_type == 'poincare':
                current_c = model.c.item()
                for group in opt_proto.param_groups:
                    group['c'] = current_c
                opt_proto.step()
            else:
                opt_proto.step()
            
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

        avg_loss = total_loss / max(1, total_samples)
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                correct += pred.eq(y).sum().item()
        acc = correct / len(test_dataset)
        best_acc = max(best_acc, acc)
        elapsed = time.time() - start
        
        print(f"  Loss: {avg_loss:.4f} Acc: {acc:.4f} Best: {best_acc:.4f} Time: {elapsed:.2f}s C: {model.c:.4f}")

    print(f"Result Pure {model_type}: Best Acc {best_acc:.4f}")

if __name__ == "__main__":
    for m_type in ['poincare', 'lorentz', 'klein']:
        run_pure_experiment(m_type)

