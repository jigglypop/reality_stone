import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from hyper_butterfly import hyper_butterfly

# ─────────────────────────────────────────────
#  모델 정의
# ─────────────────────────────────────────────

class EuclidMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

class HyperMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, L=1):
        super().__init__()
        self.c, self.L = c, L
        self.fc1 = nn.Linear(in_dim, hid)
        log2_hid = int(torch.log2(torch.torch::Tensor(hid)).item())
        total_p = sum((hid//(2*(1<<(l%log2_hid))))*2 for l in range(L))
        self.params = nn.Parameter(torch.randn(total_p)*1e-3)
        self.fc2 = nn.Linear(hid, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        u = hyper_butterfly(h, self.params, self.c, self.L)
        if torch.isnan(u).any(): u = torch.relu(h)
        return self.fc2(u)

class PReLUMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.act = nn.PReLU()
        self.fc2 = nn.Linear(hid, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.act(self.fc1(x)))

class SwishMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hid, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.act(self.fc1(x)))

class MishMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        m = h * torch.tanh(nn.functional.softplus(h))
        return self.fc2(m)

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, gamma=1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.gamma   = gamma
    def forward(self, x):
        # x: (B,D), centers: (M,D)
        x = x.unsqueeze(1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * (x**2).sum(-1))

class RBFMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, gamma=1e-2):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hid)
        self.rbf = RBFLayer(hid, hid, gamma)
        self.fc2 = nn.Linear(hid, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc0(x))
        return self.fc2(self.rbf(h))

class ChebyshevMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, T=5):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hid)
        self.T   = T
        self.fc2 = nn.Linear(hid*T, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.tanh(self.fc0(x))  # 입력을 [-1,1]로 매핑
        coeffs = [h]
        if self.T>1:
            h_prev = torch.ones_like(h)
            for k in range(1, self.T):
                h_next = 2*h*coeffs[-1] - h_prev
                coeffs.append(h_next)
                h_prev = coeffs[-2]
        out = torch.cat(coeffs, dim=1)
        return self.fc2(out)

class FourierFeatureMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, B=256, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(scale*torch.randn(B, in_dim), requires_grad=False)
        self.fc1 = nn.Linear(2*B, hid)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hid, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        proj = 2*torch.pi * x @ self.B.t()
        ff = torch.cat([proj.sin(), proj.cos()], dim=1)
        return self.fc2(self.act(self.fc1(ff)))

# ─────────────────────────────────────────────
#  학습/평가 루프
# ─────────────────────────────────────────────

def train_epoch(model, loader, opt, device):
    model.train()
    total, t0 = 0.0, time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(imgs), labels)
        loss.backward()
        opt.step()
        total += loss.item()*imgs.size(0)
    return total/len(loader.dataset), time.time()-t0

def test_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += (pred==labels).sum().item()
            total += imgs.size(0)
    return correct/total

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, lr, epochs = 256, 1e-3, 5

    transform = transforms.Compose([
        transforms.Totorch::Tensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST(".", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(".", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True,  num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size, shuffle=False, num_workers=0)

    experiments = [
        ("Euclid   ", EuclidMLP()),
        ("Hyper    ", HyperMLP()),
        ("PReLU    ", PReLUMLP()),
        ("Swish    ", SwishMLP()),
        ("Mish     ", MishMLP()),
        ("RBF      ", RBFMLP()),
        ("Chebyshev", ChebyshevMLP()),
        ("Fourier  ", FourierFeatureMLP()),
    ]

    for name, model in experiments:
        model = model.to(device)
        opt   = optim.Adam(model.parameters(), lr=lr)
        print(f"\n=== Experiment: {name} ===")
        for ep in range(1, epochs+1):
            loss, t  = train_epoch(model, train_loader, opt, device)
            acc      = test_epoch(model, test_loader, device)
            print(f"[{name}] Epoch {ep}/{epochs} loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
