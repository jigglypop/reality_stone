import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import faulthandler; faulthandler.enable()

# hyper_butterfly 패키지에서 레이어와 함수 import
import hyper_butterfly as hb
from hyper_butterfly import GeodesicButterflyLayer

print("hyper_butterfly 모듈 임포트 성공!")
#
# 1) Geodesic-Butterfly MLP
#
class GeodesicMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=2, t=0.7):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        # GeodesicButterflyLayer(dim, curvature, n_layers, t_param)
        self.geo_layer = GeodesicButterflyLayer(hid, c, L, t)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        z = self.geo_layer(h)
        return self.fc2(z)


#
# 2) Hyper-Butterfly MLP
#
class HyperMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=1):
        super().__init__()
        self.c = c
        self.L = L
        self.fc1 = nn.Linear(in_dim, hid)
        # butterfly params 개수 계산
        log2_h = int(torch.log2(torch.tensor(float(hid))).item())
        total_params = sum((hid // (2 * (1 << (l % log2_h)))) * 2 for l in range(L))
        print(f"[Hyper] hid={hid}, total_params={total_params}")
        self.params = nn.Parameter(torch.randn(total_params) * 1e-3)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        u = hb.hyper_butterfly(h, self.params, self.c, self.L)
        if torch.isnan(u).any():
            u = torch.relu(h)
        return self.fc2(u)


#
# 3) Euclid MLP
#
class EuclidMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


# 학습/평가 함수
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, lr, epochs = 256, 1e-3, 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # GeodesicMLP
    ge = GeodesicMLP(c=1e-3, L=2, t=0.7).to(device)
    opt_ge = optim.Adam(ge.parameters(), lr=lr)
    print("\n--- Geodesic-Butterfly MLP Training ---")
    for ep in range(1, epochs+1):
        loss, t = train_epoch(ge, train_loader, opt_ge, device)
        acc = test_epoch(ge, test_loader, device)
        print(f"[Geodesic] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")

    # HyperMLP
    hy = HyperMLP(c=1e-3, L=1).to(device)
    opt_hy = optim.Adam(hy.parameters(), lr=lr)
    print("\n--- Hyper-Butterfly MLP Training ---")
    for ep in range(1, epochs+1):
        loss, t = train_epoch(hy, train_loader, opt_hy, device)
        acc = test_epoch(hy, test_loader, device)
        print(f"[Hyper] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")

    # EuclidMLP
    eu = EuclidMLP().to(device)
    opt_eu = optim.Adam(eu.parameters(), lr=lr)
    print("\n--- Euclid MLP Training ---")
    for ep in range(1, epochs+1):
        loss, t = train_epoch(eu, train_loader, opt_eu, device)
        acc = test_epoch(eu, test_loader, device)
        print(f"[Euclid] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
