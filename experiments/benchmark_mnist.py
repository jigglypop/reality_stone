import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import math

# Import Reality Stone layers
from reality_stone import (
    lorentz_layer,
    lorentz_distance,
    klein_layer,
    klein_distance,
    poincare_ball_layer,
    poincare_distance,
)
from reality_stone.layers.poincare import log_map_zero


def project_to_ball(x: torch.Tensor, c: float = 1.0, epsilon: float = 1e-5) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    max_norm = (1.0 / math.sqrt(c)) - epsilon  # ← c 반영
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

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
        h = project_to_ball(h, c=self.c)
        u = self.fc2(h)
        u = torch.relu(u)
        u = project_to_ball(u, c=self.c)

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
        # Lorentz model can handle unbounded spatial coordinates
        # h = project_to_ball(h) 
        u = h @ self.weights2 + self.bias2
        u = torch.relu(u)
        # u = project_to_ball(u)

        def to_lorentz_coords(sp: torch.Tensor, c: float) -> torch.Tensor:
            x2 = (sp * sp).sum(dim=1, keepdim=True)
            x0 = torch.sqrt(torch.clamp(1.0 / c + x2, min=1e-6))
            return torch.cat([x0, sp], dim=1)

        hl = to_lorentz_coords(h, self.c)
        ul = to_lorentz_coords(u, self.c)

        z_l = lorentz_layer(hl, ul, self.c, self.t)

        def lorentz_log0_space(x_l: torch.Tensor, c: float) -> torch.Tensor:
            x0 = x_l[:, :1]
            xs = x_l[:, 1:]
            sqrtc = math.sqrt(c)
            s = torch.acosh(torch.clamp(sqrtc * x0, min=1.0 + 1e-6))
            denom = torch.clamp(torch.sinh(s), min=1e-6)
            scale = s / (denom * sqrtc)
            return xs * scale

        z = lorentz_log0_space(z_l, self.c)
        if torch.isnan(z).any():
            z = h
        output = z @ self.out_weights + self.out_bias
        return output


def _project_to_klein_with_c(x: torch.Tensor, c: float, epsilon: float = 1e-5) -> torch.Tensor:
    radius = (1.0 / math.sqrt(c)) if c > 0 else 1.0
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    max_norm = radius - epsilon
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


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
        h = _project_to_klein_with_c(h, self.c)
        u = self.fc2(h)
        u = torch.relu(u)
        u = _project_to_klein_with_c(u, self.c)

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
            
    def to_lorentz(self, x):
        # x0 = sqrt(1/c + ||x||^2)
        sq = (x * x).sum(dim=-1, keepdim=True)
        time_comp = torch.sqrt(1.0/self.c + sq)
        return torch.cat([time_comp, x], dim=-1)
        
    def project_klein(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = 1.0 / np.sqrt(self.c) - 1e-5
        cond = norm > max_norm
        return torch.where(cond, x / norm * max_norm, x)
        
    def project_poincare(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = 1.0 / np.sqrt(self.c) - 1e-5
        cond = norm > max_norm
        return torch.where(cond, x / norm * max_norm, x)

    def forward(self, x):
        B = x.size(0)
        flat = x.view(B, -1)
        h = self.input_proj(flat)
        
        if self.model_type == 'poincare':
            h = torch.tanh(h) # Map to ball approx
            h = self.project_poincare(h)
            # Poincare layer mixing (just identity mixing for now to test distance)
            # h = poincare_ball_layer(h, h, self.c, 0.5) 
            
            # Distance
            h_exp = h.unsqueeze(1).expand(B, 10, self.hidden_dim)
            p_exp = self.project_poincare(self.prototypes).unsqueeze(0).expand(B, 10, self.hidden_dim)
            dist = poincare_distance(
                h_exp.reshape(-1, self.hidden_dim), 
                p_exp.reshape(-1, self.hidden_dim), 
                c=self.c
            ).reshape(B, 10)
            
        elif self.model_type == 'lorentz':
            h = torch.tanh(h)
            h_lor = self.to_lorentz(h)
            p_lor = self.to_lorentz(self.prototypes)
            
            h_exp = h_lor.unsqueeze(1).expand(B, 10, self.hidden_dim + 1).contiguous()
            p_exp = p_lor.unsqueeze(0).expand(B, 10, self.hidden_dim + 1).contiguous()
            
            dist_sq = lorentz_distance(
                h_exp.reshape(-1, self.hidden_dim + 1),
                p_exp.reshape(-1, self.hidden_dim + 1),
                c=self.c
            ).reshape(B, 10)
            dist = torch.sqrt(dist_sq.clamp(min=1e-8))
            
        elif self.model_type == 'klein':
            h = self.project_klein(h)
            
            h_exp = h.unsqueeze(1).expand(B, 10, self.hidden_dim).contiguous()
            p_exp = self.project_klein(self.prototypes).unsqueeze(0).expand(B, 10, self.hidden_dim).contiguous()
            
            dist_sq = klein_distance(
                h_exp.reshape(-1, self.hidden_dim),
                p_exp.reshape(-1, self.hidden_dim),
                c=self.c
            ).reshape(B, 10)
            dist = torch.sqrt(dist_sq.clamp(min=1e-8))
            
        return -dist # Logits

def run_benchmark():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {DEVICE}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    import reality_stone as rs
    print(f"Reality Stone CUDA support: {rs._has_cuda}")
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
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
            print(f"  Ep {epoch} Loss: {avg_loss:.4f} Acc: {acc:.4f} Best: {best_acc:.4f} Time: {elapsed:.2f}s")
            
        results[name] = best_acc
        
    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    run_benchmark()

