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

from reality_stone import (
    lorentz_layer,
    lorentz_distance,
    klein_layer,
    klein_distance,
    poincare_ball_layer,
    poincare_distance,
    project_to_ball,
    euclidean_to_lorentz,
    lorentz_to_poincare,
)
from reality_stone.layers.poincare import log_map_zero
from reality_stone.layers.klein import project_to_klein
from reality_stone.optim import PoincareRiemannianAdam
from reality_stone.utils.misc import get_device, load_mnist_dataloaders, evaluate_accuracy

class MnistLinear(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    
    def forward(self, x):
        return self.features(x)


class PoincareMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-2, t=0.5):
        super().__init__()
        self.c = c
        self.t = t
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        h = torch.relu(h)
        h = project_to_ball(h, epsilon=1e-5)
        
        u = self.fc2(h)
        u = torch.relu(u)
        u = project_to_ball(u, epsilon=1e-5)
        
        z = poincare_ball_layer(h, u, c=self.c, t=self.t)
        
        if torch.isnan(z).any():
            z = h
        return self.out(z)


class LorentzMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-2, t=0.5):
        super().__init__()
        self.c = c
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.05)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.05)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.05)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.relu(h)
        
        u = h @ self.weights2 + self.bias2
        u = torch.relu(u)

        hl = euclidean_to_lorentz(h, self.c)
        ul = euclidean_to_lorentz(u, self.c)

        z_l = lorentz_layer(hl, ul, self.c, self.t)

        z_p = lorentz_to_poincare(z_l, self.c)
        z = log_map_zero(z_p, self.c)
        
        if torch.isnan(z).any():
            z = h
        output = z @ self.out_weights + self.out_bias
        return output


class KleinMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        h = torch.relu(h)
        h = project_to_klein(h, self.c)
        u = self.fc2(h)
        u = torch.relu(u)
        u = project_to_klein(u, self.c)

        z = klein_layer(h, u, c=self.c, t=self.t)
        if torch.isnan(z).any():
            z = h
        output = self.out(z)
        return output

class MnistHyperbolic(nn.Module):
    def __init__(self, model_type='poincare', hidden_dim=128, c=1.0):
        super().__init__()
        self.model_type = model_type
        self.c = c
        self.hidden_dim = hidden_dim
        
        # Flatten input
        self.input_proj = nn.Linear(28*28, hidden_dim)
        
        # Prototypes (learnable class centers)
        self.prototypes = nn.Parameter(torch.randn(10, hidden_dim) * 0.01)
            
    def forward(self, x):
        B = x.size(0)
        flat = x.view(B, -1)
        h = self.input_proj(flat)
        
        if self.model_type == 'poincare':
            h = torch.tanh(h) # Map to ball approx
            h = project_to_ball(h, epsilon=1e-5)
            
            # Distance
            h_exp = h.unsqueeze(1).expand(B, 10, self.hidden_dim)
            p_exp = project_to_ball(self.prototypes, epsilon=1e-5).unsqueeze(0).expand(B, 10, self.hidden_dim)
            dist = poincare_distance(
                h_exp.reshape(-1, self.hidden_dim), 
                p_exp.reshape(-1, self.hidden_dim), 
                c=self.c
            ).reshape(B, 10)
            
        elif self.model_type == 'lorentz':
            h = torch.tanh(h)
            h_lor = euclidean_to_lorentz(h, self.c)
            p_lor = euclidean_to_lorentz(self.prototypes, self.c)
            
            h_exp = h_lor.unsqueeze(1).expand(B, 10, self.hidden_dim + 1).contiguous()
            p_exp = p_lor.unsqueeze(0).expand(B, 10, self.hidden_dim + 1).contiguous()
            
            dist_sq = lorentz_distance(
                h_exp.reshape(-1, self.hidden_dim + 1),
                p_exp.reshape(-1, self.hidden_dim + 1),
                c=self.c
            ).reshape(B, 10)
            dist = torch.sqrt(dist_sq.clamp(min=1e-8))
            
        elif self.model_type == 'klein':
            h = project_to_klein(h, self.c)
            
            h_exp = h.unsqueeze(1).expand(B, 10, self.hidden_dim).contiguous()
            p_exp = project_to_klein(self.prototypes, self.c).unsqueeze(0).expand(B, 10, self.hidden_dim).contiguous()
            
            dist_sq = klein_distance(
                h_exp.reshape(-1, self.hidden_dim),
                p_exp.reshape(-1, self.hidden_dim),
                c=self.c
            ).reshape(B, 10)
            dist = torch.sqrt(dist_sq.clamp(min=1e-8))
            
        return -dist # Logits


def run_benchmark():
    DEVICE = get_device()
    print(f"Benchmarking on {DEVICE}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    import reality_stone as rs
    print(f"Reality Stone CUDA support: {rs._has_cuda}")
    
    # Data
    train_loader, test_loader = load_mnist_dataloaders(batch_size=256, test_batch_size=1000)
    
    models = {
        'Poincare': PoincareMLP().to(DEVICE),
        'Lorentz': LorentzMLP().to(DEVICE),
        'Klein': KleinMLP().to(DEVICE),
        'Linear': MnistLinear().to(DEVICE),
    }
    
    results = {}
    epochs = 5
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        crit = nn.CrossEntropyLoss()
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            model.train()
            start = time.time()
            total_loss = 0.0
            total_samples = 0
            
            for x, y in tqdm(train_loader, desc=f"{name} Train {epoch}/{epochs}", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                bsz = x.size(0)
                total_loss += loss.item() * bsz
                total_samples += bsz
            
            avg_loss = total_loss / max(1, total_samples)
            scheduler.step()
            
            acc = evaluate_accuracy(model, test_loader, DEVICE)
            best_acc = max(best_acc, acc)
            elapsed = time.time() - start
            print(f"  Ep {epoch} Loss: {avg_loss:.4f} Acc: {acc:.4f} Best: {best_acc:.4f} Time: {elapsed:.2f}s")
            
        results[name] = best_acc
        
    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


def run_hyperbolic_riemannian_adam():
    DEVICE = get_device()
    print(f"Hyperbolic prototypes with Riemannian Adam on {DEVICE}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Reality Stone Rust extension: {rs._has_rust_ext}")  # type: ignore[attr-defined]
    if not getattr(rs, "_has_rust_ext", False):  # type: ignore[attr-defined]
        print("Rust extension not available, skipping Riemannian Adam experiment")
        return

    train_loader, test_loader = load_mnist_dataloaders(batch_size=256, test_batch_size=1000)

    model = MnistHyperbolic(model_type="poincare", hidden_dim=128, c=1.0).to(DEVICE)

    euclid_params = [
        p for n, p in model.named_parameters() if n != "prototypes"
    ]
    opt_euclid = optim.Adam(euclid_params, lr=0.001, weight_decay=1e-5)
    opt_riem = PoincareRiemannianAdam([model.prototypes], c=model.c, lr=0.001)
    crit = nn.CrossEntropyLoss()
    epochs = 5
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_samples = 0

        for x, y in tqdm(
            train_loader,
            desc=f"Hyperbolic-RiemAdam Train {epoch}/{epochs}",
            leave=False,
        ):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt_euclid.zero_grad()
            opt_riem.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt_euclid.step()
            opt_riem.step()
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

        avg_loss = total_loss / max(1, total_samples)

        acc = evaluate_accuracy(model, test_loader, DEVICE)
        best_acc = max(best_acc, acc)
        elapsed = time.time() - start
        print(
            f"  Ep {epoch} Loss: {avg_loss:.4f} Acc: {acc:.4f} Best: {best_acc:.4f} Time: {elapsed:.2f}s"
        )

    print(f"Best accuracy with Riemannian Adam (Poincare prototypes): {best_acc:.4f}")

if __name__ == "__main__":
    run_benchmark()

