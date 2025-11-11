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
from reality_stone.layers.klein import project_to_klein as _project_to_klein
from reality_stone.layers.poincare import log_map_zero as _log_map_zero

def project_to_ball_with_c(x, c: float, epsilon=1e-5):
    return _project_to_klein(x, c, epsilon)

class KleinMLP(nn.Module):
    def __init__(self, in_dim=784, hid=256, out_dim=10, c=1e-3, L=2, t=0.7, use_dynamic=False, c_min=1e-4, c_max=0.05):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        self.use_dynamic = use_dynamic
        self.c_min = c_min
        self.c_max = c_max
        # Use standard Linear layers for better initialization and fused kernels
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, out_dim)
        
        if use_dynamic:
            self.kappas = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        h = torch.relu(h)
        h = project_to_ball_with_c(h, self.c)
        u = self.fc2(h)
        u = torch.relu(u)
        u = project_to_ball_with_c(u, self.c)
        
        # Klein 정식 레이어 사용 (동적 곡률 미지원)
        z = rs.klein_layer(h, u, c=self.c, t=self.t)
            
        if torch.isnan(z).any():
            z = h
        output = self.out(z)
        return output

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = nn.functional.cross_entropy(logits, labels)
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
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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

    parser = argparse.ArgumentParser(description="MNIST Klein MLP test")
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
    # Encourage parallelism in Rust ndarray (rayon) and PyTorch CPU
    os.environ.setdefault("RAYON_NUM_THREADS", str(max(2, (os.cpu_count() or 4) // 2)))
    try:
        torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
    except Exception:
        pass

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # If CUDA kernels for Klein are unavailable, avoid GPU<->CPU transfers by using CPU
    if device.type == "cuda" and not rs._has_cuda:
        print("Note: Klein CUDA kernels unavailable. Switching to CPU to avoid transfer overhead.")
        device = torch.device("cpu")

    # Optional kernel-level speedups
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    os.makedirs(args.data_dir, exist_ok=True)
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.quick:
        train_ds = torch.utils.data.Subset(train_ds, list(range(0, min(10000, len(train_ds)))))
        test_ds = torch.utils.data.Subset(test_ds, list(range(0, min(2000, len(test_ds)))))
    use_pin = (device.type == "cuda") and rs._has_cuda
    workers = max(2, os.cpu_count() // 2)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=workers, pin_memory=use_pin, persistent_workers=True, prefetch_factor=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, pin_memory=use_pin, persistent_workers=True, prefetch_factor=2
    )

    model = KleinMLP(use_dynamic=False, t=args.t, c=args.c).to(device)
    _ = train_model("KleinMLP (Static Curvature)", model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
