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

def lorentz_distance_torch(x, y, c, eps=1e-7):
    # x, y: (..., d+1)
    # Minkowski inner product: x0y0 - x1y1 - ...
    inner = x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    sqrt_c = torch.sqrt(c)
    # distance = 1/sqrt(c) * acosh(c * inner)
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


class MnistHyperbolic(nn.Module):
    def __init__(self, model_type='poincare', hidden_dim=128, c=1.0):
        super().__init__()
        self.model_type = model_type
        # Learnable curvature (Log-curvature parameterization)
        self.c_log = nn.Parameter(torch.tensor(math.log(c)))
        self.hidden_dim = hidden_dim
        
        # MLP Layers
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights to be small
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)
        
        # Prototypes (learnable class centers) - initialize spread out
        # We store prototypes in n-dim coordinates for all models.
        # For Lorentz, we lift them to n+1 dim during forward pass.
        init_protos = torch.randn(10, hidden_dim) * 0.3
        # Normalize to stay well inside the ball (for Poincare/Klein validity)
        max_norm = (1.0 / math.sqrt(c)) * 0.5
        norms = init_protos.norm(dim=1, keepdim=True)
        init_protos = init_protos / norms.clamp(min=1e-6) * max_norm * torch.rand(10, 1)
        self.prototypes = nn.Parameter(init_protos)
            
    @property
    def c(self):
        return self.c_log.exp()

    def to_lorentz(self, x):
        # Lift n-dim coords to (n+1)-dim Lorentz coords
        # x0 = sqrt(1/c + ||x||^2)
        c_val = self.c
        sq = (x * x).sum(dim=-1, keepdim=True)
        time_comp = torch.sqrt(1.0/c_val + sq)
        return torch.cat([time_comp, x], dim=-1)
        
    def project_klein(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = 1.0 / np.sqrt(self.c.item()) - 1e-7
        cond = norm > max_norm
        return torch.where(cond, x / norm * max_norm, x)
        
    def project_poincare(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = 1.0 / np.sqrt(self.c.item()) - 1e-7
        cond = norm > max_norm
        return torch.where(cond, x / norm * max_norm, x)

    def forward(self, x):
        B = x.size(0)
        flat = x.view(B, -1)
        c_val = self.c
        
        # Layer 1
        h = self.fc1(flat)
        h = torch.relu(h)
        
        if self.model_type == 'poincare':
            h = self.project_poincare(h)
            h = self.fc2(h)
            h = torch.relu(h)
            h = self.project_poincare(h)
            
            h_exp = h.unsqueeze(1).expand(B, 10, self.hidden_dim)
            p_exp = self.prototypes.unsqueeze(0).expand(B, 10, self.hidden_dim)
            dist = poincare_distance(
                h_exp.reshape(-1, self.hidden_dim), 
                p_exp.reshape(-1, self.hidden_dim), 
                c=c_val,
                eps=1e-7
            ).reshape(B, 10)
            
        elif self.model_type == 'lorentz':
            # Lorentz model: Use Euclidean operations for intermediate layers (simulating tangent space)
            # and only lift to Hyperboloid for distance calculation.
            # Note: Properly we should use exponential map, but standard practice often simplifies 
            # to Euclidean ops + projection/lifting for simple MLPs.
            h = self.fc2(h)
            h = torch.relu(h)
            
            # Lift to Lorentz Hyperboloid
            h_l = self.to_lorentz(h) # (B, hidden+1)
            p_l = self.to_lorentz(self.prototypes) # (10, hidden+1)
            
            h_exp = h_l.unsqueeze(1).expand(B, 10, self.hidden_dim + 1)
            p_exp = p_l.unsqueeze(0).expand(B, 10, self.hidden_dim + 1)
            
            # Note: lorentz_distance in reality_stone might need (B*10, dim) inputs
            dist = lorentz_distance_torch(
                h_exp,
                p_exp,
                c=c_val
            )

        elif self.model_type == 'klein':
            h = self.project_klein(h)
            h = self.fc2(h)
            h = torch.relu(h)
            h = self.project_klein(h)
            
            h_exp = h.unsqueeze(1).expand(B, 10, self.hidden_dim)
            p_exp = self.prototypes.unsqueeze(0).expand(B, 10, self.hidden_dim)
            
            dist = klein_distance_torch(
                h_exp,
                p_exp,
                c=c_val
            )
            
        return -dist # Logits


def run_experiment(model_type='poincare'):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Running Experiment: {model_type.upper()} (Riemannian/Hybrid Adam) ===")
    print(f"Device: {DEVICE}")
    
    if not getattr(rs, "_has_rust_ext", False):
        print("Rust extension not available.")
        return

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = MnistHyperbolic(model_type=model_type, hidden_dim=128, c=0.1).to(DEVICE)
    crit = nn.CrossEntropyLoss()

    # Optimizer Configuration
    # 1. Curvature (c_log) - Always optimized by Adam with high LR for fast adaptation
    opt_c = optim.Adam([model.c_log], lr=0.01, weight_decay=0) # Increased LR for curvature

    # 2. Model Parameters (Weights + Prototypes)
    model_params = [
        {'params': model.prototypes},
        {'params': model.fc1.parameters()},
        {'params': model.fc2.parameters()}
    ]

    if model_type == 'poincare':
        # Use Riemannian Adam for Poincare
        # Note: We pass initial c, but we must update it dynamically
        opt_model = PoincareRiemannianAdam(
            model_params, 
            c=model.c.item(), 
            lr=0.001, 
            max_norm_eps=1e-7
        )
    else:
        # For Lorentz/Klein, use standard Adam (Euclidean approximation)
        # True Riemannian Adam for these manifolds is not yet exposed in Python bindings
        opt_model = optim.Adam(model_params, lr=0.001, weight_decay=1e-5)

    epochs = 1
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        
        # For RADM, we need to update curvature in optimizer groups
        if model_type == 'poincare':
            current_c = model.c.item()
            for group in opt_model.param_groups:
                group['c'] = current_c

        start = time.time()
        total_loss = 0.0
        total_samples = 0

        for x, y in tqdm(train_loader, desc=f"{model_type} Ep {epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            opt_c.zero_grad()
            opt_model.zero_grad()
            
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            
            # Step optimizers
            opt_c.step()
            
            if model_type == 'poincare':
                # Update c for RADM step
                current_c = model.c.item()
                for group in opt_model.param_groups:
                    group['c'] = current_c
                opt_model.step()
            else:
                # Standard Adam step
                opt_model.step()
                
                # For Klein/Lorentz with Adam, we might need to enforce constraints manually if needed
                # But here we use soft constraints via penalty or projection in forward pass.
                # Lorentz: we map to manifold in forward, so weights are Euclidean.
                # Klein: we project in forward.
                pass

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

        avg_loss = total_loss / max(1, total_samples)
        
        # Evaluation
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

    print(f"Result {model_type}: Best Acc {best_acc:.4f}")


if __name__ == "__main__":
    # Run experiments
    # for m_type in ['poincare', 'lorentz', 'klein']:
    #     run_experiment(m_type)
    run_experiment('klein')
