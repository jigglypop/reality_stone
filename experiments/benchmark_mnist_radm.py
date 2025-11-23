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
        # Learnable curvature (Log-curvature parameterization)
        self.c_log = nn.Parameter(torch.tensor(math.log(c)))
        self.hidden_dim = hidden_dim
        
        # MLP Layers (similar to PoincareMLP in benchmark_mnist.py)
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights to be small (to stay near origin initially)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)
        
        # Prototypes (learnable class centers) - initialize spread out
        init_protos = torch.randn(10, hidden_dim) * 0.3
        # Normalize to stay well inside the ball
        max_norm = (1.0 / math.sqrt(c)) * 0.5
        norms = init_protos.norm(dim=1, keepdim=True)
        init_protos = init_protos / norms.clamp(min=1e-6) * max_norm * torch.rand(10, 1)
        self.prototypes = nn.Parameter(init_protos)
            
    @property
    def c(self):
        return self.c_log.exp()

    def to_lorentz(self, x):
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
            # Project layer 1 output to ball
            h = self.project_poincare(h)
            
            # Layer 2
            h = self.fc2(h)
            h = torch.relu(h)
            # Project layer 2 output to ball
            h = self.project_poincare(h)
            
            # Distance to learnable prototypes living directly on the Poincaré ball
            h_exp = h.unsqueeze(1).expand(B, 10, self.hidden_dim)
            p_exp = self.prototypes.unsqueeze(0).expand(B, 10, self.hidden_dim)
            dist = poincare_distance(
                h_exp.reshape(-1, self.hidden_dim), 
                p_exp.reshape(-1, self.hidden_dim), 
                c=c_val,
                eps=1e-7
            ).reshape(B, 10)
            
        elif self.model_type == 'lorentz':
            # ... (Lorentz implementation omitted for brevity as we focus on Poincare RADM)
            # Fallback to simple projection for now if needed
            pass 
            
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


def run_hyperbolic_riemannian_adam():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hyperbolic prototypes with Riemannian Adam on {DEVICE}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Reality Stone Rust extension: {rs._has_rust_ext}")  # type: ignore[attr-defined]
    if not getattr(rs, "_has_rust_ext", False):  # type: ignore[attr-defined]
        print("Rust extension not available, skipping Riemannian Adam experiment")
        return

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = MnistHyperbolic(model_type="poincare", hidden_dim=128, c=0.1).to(DEVICE)

    # Update: Only c_log is Euclidean
    euclid_params = [model.c_log]
    opt_euclid = optim.Adam(euclid_params, lr=0.001, weight_decay=1e-5)
    
    # All other parameters (MLP weights + Prototypes) go to RADM
    # We treat MLP weights as if they are on the manifold (or simply use RADM's update rule which handles c=0 case or constrained case)
    # Since we project activations to Poincare ball, the weights act on tangent space approximately.
    riem_params = [
        {'params': model.prototypes, 'c': model.c.item()}, # Prototypes are definitely on manifold
        {'params': model.fc1.parameters(), 'c': model.c.item()}, # Treat MLP weights as manifold parameters too
        {'params': model.fc2.parameters(), 'c': model.c.item()}
    ]
    
    # RADM gets aggressive clamping relaxation
    opt_riem = PoincareRiemannianAdam(riem_params, c=model.c.item(), lr=0.001, max_norm_eps=1e-7)
    
    crit = nn.CrossEntropyLoss()
    epochs = 5
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        
        # Dynamic curvature update in optimizer for ALL groups
        current_c = model.c.item()
        for group in opt_riem.param_groups:
            group['c'] = current_c

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
            
            # Update curvature again before step if needed (though per-batch change is small)
            current_c = model.c.item()
            for group in opt_riem.param_groups:
                group['c'] = current_c
            opt_riem.step()
            
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
        print(
            f"  Ep {epoch} Loss: {avg_loss:.4f} Acc: {acc:.4f} Best: {best_acc:.4f} Time: {elapsed:.2f}s C: {model.c:.4f}"
        )

    print(f"Best accuracy with Riemannian Adam (Poincare prototypes): {best_acc:.4f}")
    

if __name__ == "__main__":
    run_hyperbolic_riemannian_adam()
