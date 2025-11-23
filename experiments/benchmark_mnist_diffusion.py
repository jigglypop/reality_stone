import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import math

class BioGeometricEncoder(nn.Module):
    """Fixed Biological Encoder (Same as before)"""
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
    def __init__(self, hidden_dim=2048, num_classes=10, steps=5):
        super().__init__()
        self.steps = steps # Time steps for diffusion
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Encoder (Fixed)
        self.encoder = BioGeometricEncoder(out_dim=hidden_dim)
        
        # === Structure Learning ===
        # Learnable Adjacency / Conductivity Matrix
        # Instead of full matrix (2048x2048 is too big), we learn connections 
        # from hidden nodes to class nodes directly, plus internal recurrent connections.
        # This simulates "Synaptic Weights" governing energy flow.
        
        # 1. Internal recurrent connections (Self-diffusion within hidden layer)
        # Diagonal-heavy initialization to preserve signal
        self.W_hidden = nn.Parameter(torch.eye(hidden_dim) * 0.9 + torch.randn(hidden_dim, hidden_dim) * 0.01)
        
        # 2. Hidden to Class connections (Readout)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, num_classes) * 0.01)
        
        # Energy dampening factor (Leakage)
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        # 1. Input Injection
        with torch.no_grad():
            h = self.encoder(x) # (B, 2048) - Initial Energy State
            
        # 2. Diffusion Process (Time Evolution)
        # Energy flows through the network for T steps
        # h(t+1) = alpha * h(t) + (1-alpha) * tanh(W_h * h(t))
        # This is a simple dynamical system.
        
        for t in range(self.steps):
            # Normalize adjacency to ensure stability (Graph Laplacian style)
            # W_h = self.W_hidden / self.W_hidden.norm(dim=1, keepdim=True).clamp(min=1e-6)
            
            # Flow
            flow = h @ self.W_hidden
            
            # Update state (Leaky Integrator)
            h = self.alpha * h + (1 - self.alpha) * torch.tanh(flow)
            
        # 3. Energy Readout
        # Which class node accumulated the most energy/signal?
        out = h @ self.W_out
        return out

def run_diffusion_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Running Experiment: Manifold Diffusion (Energy Flow) ===")
    print(f"Structure: Input -> Fixed Encoder -> Dynamical System (T=5) -> Readout")
    print(f"Learning: Synaptic Conductivity (Weights) of the graph")
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model
    # T=5 steps of diffusion
    model = ManifoldDiffusionModel(steps=5).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    
    # We use standard Adam here to learn the graph structure (weights)
    opt = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_samples = 0

        for x, y in tqdm(train_loader, desc=f"Diffusion Ep {epoch}", leave=False):
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
        
        print(f"  Loss: {avg_loss:.4f} Acc: {acc:.4f} Best: {best_acc:.4f} Time: {elapsed:.2f}s")

    print(f"Result Diffusion: Best Acc {best_acc:.4f}")

if __name__ == "__main__":
    run_diffusion_experiment()

