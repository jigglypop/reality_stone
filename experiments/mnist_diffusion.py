import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
import reality_stone as rs
from reality_stone.layers.diffusion import RiemannianDiffusionStep
from reality_stone.utils.misc import get_device, load_mnist_dataloaders, evaluate_accuracy

DEVICE = get_device()
print(f"Device: {DEVICE}")


class BioGeometricEncoder(nn.Module):
    """Fixed Biological Encoder"""
    def __init__(self, in_channels=1, out_dim=2048):
        super().__init__()
        self.proj = nn.Linear(28*28, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.proj.weight.requires_grad = False # FREEZE

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.proj(x)
        x = torch.relu(x)
        return x

class ManifoldDiffusionModel(nn.Module):
    def __init__(self, hidden_dim=2048, num_classes=10, steps=5, alpha=0.9, dt=0.1):
        super().__init__()
        self.steps = steps
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.alpha_val = alpha
        self.dt = dt
        
        # Encoder (Fixed)
        self.encoder = BioGeometricEncoder(out_dim=hidden_dim)
        
        # === Learnable Structure ===
        # We interpret these weights as defining the "Potential Energy Landscape"
        self.W_hidden = nn.Parameter(torch.eye(hidden_dim) * 0.9 + torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, num_classes) * 0.01)
        
        # === Reality Stone Engine ===
        # Rust-based Riemannian Diffusion Engine
        print(f"Initializing Reality Stone Rust Engine (Dim={hidden_dim}, Alpha={alpha}, dt={dt})...")
        self.rs_engine = rs.PyRiemannianDiffusion(hidden_dim, alpha, dt)

    def forward(self, x):
        # 1. Input Injection (Initial State)
        with torch.no_grad():
            h = self.encoder(x) # (B, 2048)
            
        # 2. Riemannian Diffusion Process
        for t in range(self.steps):
            # Calculate Flow (PyTorch does this fast on GPU)
            flow = torch.tanh(h @ self.W_hidden)
            
            # Apply Riemannian Step (Rust+CUDA does this fast on GPU)
            # h(t+1) = Exp_h( -grad_E * dt )
            h = RiemannianDiffusionStep.apply(
                h,
                flow,
                self.rs_engine,
                self.alpha_val,
                self.dt,
            )
            
        # 3. Readout
        out = h @ self.W_out
        return out

def run_diffusion_experiment():
    print(f"\n=== Running Experiment: Riemannian Lagrangian Diffusion (Rust+CUDA Backend) ===")
    print(f"Backend: {DEVICE.upper()}")
    
    train_loader, test_loader = load_mnist_dataloaders(batch_size=256, test_batch_size=1000)

    # Initialize model
    model = ManifoldDiffusionModel(steps=5, alpha=0.9).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_samples = 0

        for x, y in tqdm(train_loader, desc=f"Riemann Ep {epoch}", leave=False):
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
        elapsed = time.time() - start
        
        # Evaluation
        acc = evaluate_accuracy(model, test_loader, DEVICE)
        best_acc = max(best_acc, acc)
        
        print(f"  Loss: {avg_loss:.4f} Acc: {acc:.4f} Best: {best_acc:.4f} Time: {elapsed:.2f}s")

    print(f"Result Riemannian Diffusion: Best Acc {best_acc:.4f}")

if __name__ == "__main__":
    run_diffusion_experiment()
