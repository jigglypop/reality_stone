import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import reality_stone as hb
import torch.fft as fft

# ———————— 비선형 사인·코사인 사영 ————————
class FourierNonlinearMap(nn.Module):
    def __init__(self, dim, N=16, omega=1.0):
        super().__init__()
        self.N     = N
        self.omega = omega
        self.a0 = nn.Parameter(torch.ones(1))
        self.a  = nn.Parameter(torch.zeros(N))
        self.b  = nn.Parameter(torch.zeros(N))

    def forward(self, x):
        r = x.norm(dim=1, keepdim=True).clamp_min(1e-6)  
        n = torch.arange(1, self.N+1, device=x.device, dtype=r.dtype).view(1, -1)
        angle = self.omega * (r @ n)
        f_r = self.a0 \
            + (torch.cos(angle) * self.a).sum(dim=1, keepdim=True) \
            + (torch.sin(angle) * self.b).sum(dim=1, keepdim=True)
        return f_r * x

# ———————— Fourier + Poincaré + Nonlinear 융합 블록 ————————
class FourierSpectralHB(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=100, L=10, c=1e-3, N=16, omega=1.0):
        super().__init__()
        self.L = L
        self.c = c

        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        self.fnmap = FourierNonlinearMap(hidden_dim, N=N, omega=omega)

        self.phi   = nn.Parameter(torch.zeros(hidden_dim))
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.zeros(1))

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = x.view(x.size(0), -1)
        h1 = self.relu(self.bn1(self.fc1(h)))

        u0 = hb._C.log_map_origin_cuda(h1, self.c)
        u1 = self.fnmap(u0)

        U  = fft.fft(u1, dim=1)
        H  = torch.exp(1j * (self.L * self.phi))
        V  = U * H.unsqueeze(0)
        v0 = fft.ifft(V, dim=1).real.contiguous()

        v1 = self.fnmap(v0)
        y  = hb._C.exp_map_origin_cuda(v1, self.c)

        h2 = self.alpha * y + self.beta * h1
        h3 = self.relu(self.bn2(h2))
        return self.fc2(h3)


def train_and_test(
    in_dim=32*32*3,
    hidden_dim=512,
    out_dim=100,
    L=10,
    epochs=10,
    batch_size=128,
    lr=5e-4
):
    device = torch.device("cuda")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])

    train_ds = datasets.CIFAR100('.', train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR100('.', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    model = FourierSpectralHB(in_dim, hidden_dim, out_dim, L=L).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    print(f"\n=== FourierSpectralHB on CIFAR-100 (L={L}) ===")
    for ep in range(1, epochs+1):
        model.train()
        corr, tot, t0 = 0, 0, time.time()
        for idx, (imgs, labs) in enumerate(train_loader):
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = lossf(out, labs)
            loss.backward()
            opt.step()
            pred = out.argmax(dim=1)
            corr += (pred==labs).sum().item()
            tot  += labs.size(0)
            if idx % 50 == 0:
                print(f"[Ep{ep}] batch {idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  acc={100.*corr/tot:.2f}%")
        print(f"[Ep{ep}] train time={(time.time()-t0):.1f}s acc={100.*corr/tot:.2f}%")

        model.eval()
        tc, tt, t1 = 0, 0, time.time()
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                tc += (model(imgs).argmax(dim=1)==labs).sum().item()
                tt += labs.size(0)
        print(f"[Ep{ep}] test acc={100.*tc/tt:.2f}% time={(time.time()-t1):.2f}s\n")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_and_test(L=1000, epochs=10)
