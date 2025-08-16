import time
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()
import reality_stone as rs
import math

def project_to_ball(x, epsilon=1e-5):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    max_norm = 1.0 - epsilon
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

class LorentzMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7, use_dynamic=False, c_min=1e-4, c_max=0.05):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        self.use_dynamic = use_dynamic
        self.c_min = c_min
        self.c_max = c_max
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.002)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.002)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.002)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        if use_dynamic:
            self.kappas = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        # project optional (keep bounded)
        h = project_to_ball(h)
        u = h @ self.weights2 + self.bias2
        u = torch.tanh(u)
        u = project_to_ball(u)

        # Build Lorentz Minkowski coordinates (time + space)
        def to_lorentz_coords(sp: torch.Tensor, c: float) -> torch.Tensor:
            # x0 = sqrt(1/c + ||x||^2)
            x2 = (sp * sp).sum(dim=1, keepdim=True)
            x0 = torch.sqrt(torch.clamp(1.0 / c + x2, min=1e-6))
            return torch.cat([x0, sp], dim=1)

        hl = to_lorentz_coords(h, self.c)
        ul = to_lorentz_coords(u, self.c)

        # Lorentz layer → Minkowski output
        z_l = rs.lorentz_layer(hl, ul, c=self.c, t=self.t)

        # Log-map to tangent at origin for Euclidean head
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

        # Constraint penalty (hyperboloid): (x0^2 - ||x||^2 - 1/c)^2
        h0 = hl[:, :1]
        hs = hl[:, 1:]
        u0 = ul[:, :1]
        us = ul[:, 1:]
        res_h = (h0 * h0 - (hs * hs).sum(dim=1, keepdim=True) - 1.0 / self.c)
        res_u = (u0 * u0 - (us * us).sum(dim=1, keepdim=True) - 1.0 / self.c)
        self._constraint_penalty = ((res_h ** 2).mean() + (res_u ** 2).mean())
        return output

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = nn.functional.cross_entropy(logits, labels)
        # add constraint penalty if computed
        if hasattr(model, '_constraint_penalty') and isinstance(model._constraint_penalty, torch.Tensor):
            loss = loss + 1e-4 * model._constraint_penalty.to(loss.dtype).to(loss.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    
    return total_loss / len(loader.dataset), time.time() - t0

def test_epoch(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(loader.dataset)

def train_model(model_name, model, loader_train, loader_test, epochs=10, lr=1e-3, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    display_name = model_name
    if hasattr(model, 't'):
        display_name = f"{model_name} (t={model.t})"
    print(f"\n--- {display_name} Training ---")
    
    test_accs = []
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, loader_train, optimizer, device)
        acc = test_epoch(model, loader_test, device)
        test_accs.append(acc)
        
        print(f"[{display_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
    
    best_acc = max(test_accs) * 100
    print(f"[{display_name}] Best accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="MNIST Lorentz MLP test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--t", type=float, default=0.7)
    parser.add_argument("--c", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true", help="use small subset for quick run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=os.path.join("tests", "data"))
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    os.makedirs(args.data_dir, exist_ok=True)
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.quick:
        train_ds = torch.utils.data.Subset(train_ds, list(range(0, min(10000, len(train_ds)))))
        test_ds = torch.utils.data.Subset(test_ds, list(range(0, min(2000, len(test_ds)))))
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    model = LorentzMLP(use_dynamic=False, t=args.t, c=args.c).to(device)
    _ = train_model("LorentzMLP (Static Curvature)", model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
