import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()
import reality_stone as rs

def project_to_ball(x, epsilon=1e-5):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    max_norm = 1.0 - epsilon
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

class PoincareMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7, use_dynamic=False, c_min=1e-4, c_max=0.05):
        super().__init__()
        self.c = c
        self.L = L
        self.t = t
        self.use_dynamic = use_dynamic
        self.c_min = c_min
        self.c_max = c_max
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        if use_dynamic:
            self.kappas = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        h = project_to_ball(h)
        u = h @ self.weights2 + self.bias2
        u = torch.tanh(u)
        u = project_to_ball(u)
        
        if self.use_dynamic:
            z = rs.poincare_ball_layer(h, u, c=None, t=self.t, kappas=self.kappas, layer_idx=0, c_min=self.c_min, c_max=self.c_max)
        else:
            z = rs.poincare_ball_layer(h, u, c=self.c, t=self.t)
            
        if torch.isnan(z).any():
            z = h
        output = z @ self.out_weights + self.out_bias
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
        loss.backward()
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
    
    if hasattr(model, 'use_dynamic') and model.use_dynamic:
        print(f"Initial kappa: {model.kappas.item():.4f}")
        initial_kappa = model.kappas.item()
    
    test_accs = []
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, loader_train, optimizer, device)
        acc = test_epoch(model, loader_test, device)
        test_accs.append(acc)
        
        if hasattr(model, 'use_dynamic') and model.use_dynamic:
            kappa = model.kappas.item()
            c = model.c_min + (model.c_max - model.c_min) * torch.sigmoid(model.kappas).item()
            print(f"[{display_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}% | kappa={kappa:.4f}, c={c:.4f}")
        else:
            print(f"[{display_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
    
    if hasattr(model, 'use_dynamic') and model.use_dynamic:
        final_kappa = model.kappas.item()
        print(f"Kappa change: {initial_kappa:.4f} -> {final_kappa:.4f} (Δ={final_kappa-initial_kappa:.4f})")
        
    best_acc = max(test_accs) * 100
    print(f"[{display_name}] Best accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, lr, epochs = 256, 1e-3, 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    print("=== Dynamic Curvature Test ===")
    dynamic_model = PoincareMLP(use_dynamic=True, t=0.7).to(device)
    
    params = [
        {'params': [p for n, p in dynamic_model.named_parameters() if 'kappas' not in n], 'lr': lr},
        {'params': dynamic_model.kappas, 'lr': lr}
    ]
    
    optimizer = optim.Adam(params)
    display_name = "PoincareMLP (Dynamic Curvature)"
    if hasattr(dynamic_model, 't'):
        display_name = f"{display_name} (t={dynamic_model.t})"
    print(f"\n--- {display_name} Training ---")
    
    if hasattr(dynamic_model, 'use_dynamic') and dynamic_model.use_dynamic:
        initial_kappa = dynamic_model.kappas.item()
    
    test_accs = []
    for ep in range(1, epochs+1):
        loss, t = train_epoch(dynamic_model, train_loader, optimizer, device)
        acc = test_epoch(dynamic_model, test_loader, device)
        test_accs.append(acc)
        
        if hasattr(dynamic_model, 'use_dynamic') and dynamic_model.use_dynamic:
            kappa = dynamic_model.kappas.item()
            c = dynamic_model.c_min + (dynamic_model.c_max - dynamic_model.c_min) * torch.sigmoid(dynamic_model.kappas).item()
            print(f"[{display_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}% | kappa={kappa:.4f}, c={c:.4f}")
        else:
            print(f"[{display_name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
    
    if hasattr(dynamic_model, 'use_dynamic') and dynamic_model.use_dynamic:
        final_kappa = dynamic_model.kappas.item()
        print(f"Kappa change: {initial_kappa:.4f} -> {final_kappa:.4f} (Δ={final_kappa-initial_kappa:.4f})")
        
    best_acc = max(test_accs) * 100
    print(f"\n동적 곡률 최종 정확도: {best_acc:.2f}%")
