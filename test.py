import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import riemutils

# 확인: hyper_butterfly 바인딩
print("hyper_butterfly binding:", riemutils.hyper_butterfly)

# 1) Euclid MLP
class EuclidMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2) Hyper-Butterfly MLP
class HyperMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=1):
        super().__init__()
        self.c = c
        self.L = L

        # 입력→hidden 변환
        self.fc1 = nn.Linear(in_dim, hid)

        # hid 기반 파라미터 수 계산 및 작은 스케일 초기화
        log2_hid = int(torch.log2(torch.tensor(hid)).item())
        total_params = 0
        for l in range(L):
            bs = 1 << (l % log2_hid)
            nb = hid // (2 * bs)
            total_params += nb * 2
        print(f"[Hyper] hid={hid}, total_params={total_params}")
        self.params = nn.Parameter(torch.randn(total_params) * 1e-3)

        # hidden→output 변환
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))

        # Hyper-Butterfly 적용
        u = riemutils.hyper_butterfly(h, self.params, self.c, self.L)

        # NaN 감지 및 대체
        if torch.isnan(u).any():
            mn = torch.nanmin(u).item()
            mx = torch.nanmax(u).item()
            print(f"[WARN] NaN in hyper output: min={mn:.3e}, max={mx:.3e} -> using ReLU fallback")
            u = torch.relu(h)

        return self.fc2(u)

# 학습/평가 함수

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    start = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset), time.time() - start


def test_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
    return correct / total

# 메인 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    lr = 1e-3
    epochs = 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Hyper-Butterfly MLP
    hy = HyperMLP(c=1e-3, L=1).to(device)
    opt_hy = optim.Adam(hy.parameters(), lr=lr)
    print("\n--- Hyper-Butterfly MLP Training ---")
    for ep in range(1, epochs+1):
        loss, t = train_epoch(hy, train_loader, opt_hy, device)
        acc = test_epoch(hy, test_loader, device)
        print(f"[Hyper] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")

    # Euclid MLP
    eu = EuclidMLP().to(device)
    opt_eu = optim.Adam(eu.parameters(), lr=lr)
    print("\n--- Euclid MLP Training ---")
    for ep in range(1, epochs+1):
        loss, t = train_epoch(eu, train_loader, opt_eu, device)
        acc = test_epoch(eu, test_loader, device)
        print(f"[Euclid] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
