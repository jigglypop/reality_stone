import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class ChebyshevMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10,
                 T=5, weight_decay=0.0, dropout_p=0.0, use_batchnorm=False):
        super().__init__()
        self.T = T
        self.hid = hid
        self.fc0 = nn.Linear(in_dim, hid)
        self.bn0 = nn.BatchNorm1d(hid) if use_batchnorm else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p>0 else None
        self.fc2 = nn.Linear(hid*(T+1), out_dim)  # Chebyshev 합치면 차원 (T+1)*hid

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
        h = torch.tanh(self.fc0(x))
        if self.bn0: h = self.bn0(h)
        # Chebyshev basis
        cheb = [h]
        if self.T>=1:
            h_prev = torch.zeros_like(h)
            h_curr = h
            for k in range(1, self.T+1):
                h_next = 2*h*h_curr - h_prev
                cheb.append(h_next)
                h_prev, h_curr = h_curr, h_next
        out = torch.cat(cheb, dim=1)
        if self.dropout: out = self.dropout(out)
        return self.fc2(out)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total, t0 = 0.0, time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(imgs), labels)
        loss.backward()
        optimizer.step()
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

def run_experiment(name, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    tr_ds = datasets.MNIST(".", train=True,  download=True, transform=transform)
    te_ds = datasets.MNIST(".", train=False, download=True, transform=transform)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True,  num_workers=0)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=0)

    model = ChebyshevMLP(**kwargs).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=kwargs.get("weight_decay",0.0))

    print(f"\n=== Experiment: {name:<10} ===")
    for ep in range(1,6):
        loss, t = train_epoch(model, tr_loader, opt, device)
        acc = test_epoch(model, te_loader, device)
        print(f"[{name:<10}] Epoch {ep}/5 loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")

if __name__=="__main__":
    # 1) 기본 Chebyshev (T=5)
    run_experiment("Chebyshev", T=5, hid=128, dropout_p=0, use_batchnorm=False, weight_decay=0)
    # 2) 차수 축소 T=3
    run_experiment("ChebT=3",    T=3, hid=128, dropout_p=0, use_batchnorm=False, weight_decay=0)
    # 3) 가중치 감소
    run_experiment("ChebWD",     T=5, hid=128, dropout_p=0, use_batchnorm=False, weight_decay=1e-4)
    # 4) 드롭아웃
    run_experiment("ChebDrop",   T=5, hid=128, dropout_p=0.3, use_batchnorm=False, weight_decay=0)
    # 5) 배치정규화
    run_experiment("ChebBN",     T=5, hid=128, dropout_p=0, use_batchnorm=True,  weight_decay=0)
