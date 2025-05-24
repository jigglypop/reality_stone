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
        self.dim   = dim
        self.N     = N
        self.omega = omega
        self.a0 = nn.Parameter(torch.ones(1))
        self.a  = nn.Parameter(torch.zeros(N))
        self.b  = nn.Parameter(torch.zeros(N))

    def forward(self, x):
        # x: (B, dim)
        r = x.norm(dim=1, keepdim=True).clamp_min(1e-6)           # (B,1)
        n = torch.arange(1, self.N+1, device=x.device, dtype=r.dtype).view(1, -1)  # (1,N)
        angle = self.omega * (r @ n)                             # (B,N)
        cosn  = torch.cos(angle)                                 # (B,N)
        sinn  = torch.sin(angle)                                 # (B,N)
        f_r = self.a0 \
            + (cosn * self.a).sum(dim=1, keepdim=True) \
            + (sinn * self.b).sum(dim=1, keepdim=True)           # (B,1)
        return f_r * x                                           # (B,dim)


# ———————— Fourier + Poincaré + Nonlinear 융합 스펙트럴 블록 ————————
class FourierSpectralHB(nn.Module):
    def __init__(self,
                 in_dim:    int   = 784,
                 hidden_dim:int   = 256,
                 out_dim:   int   = 10,
                 L:         int   = 10,   # “가짜” 레이어 수
                 c:         float = 1e-3,
                 N:         int   = 16,
                 omega:     float = 1.0):
        super().__init__()
        self.L = L
        self.c = c

        # 1) 인풋→히든
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # 2) 푸리에 비선형 맵
        self.fnmap = FourierNonlinearMap(hidden_dim, N=N, omega=omega)

        # 3) 순수 위상 필터 파라미터
        self.phi   = nn.Parameter(torch.zeros(hidden_dim))  # phase ∈ ℝ
        # residual scale
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.zeros(1))

        # 4) 아웃풋
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # (B,1,28,28) or (B,784) → (B,784)
        h0 = x.view(x.size(0), -1)
        h1 = self.relu(self.bn1(self.fc1(h0)))                # (B,hidden_dim)

        # — Poincaré 로그 맵 —
        u0 = hb._C.log_map_origin_cuda(h1, self.c)            # (B,hidden_dim)
        # — 1차 비선형 푸리에 맵 —
        u1 = self.fnmap(u0)                                   # (B,hidden_dim)

        # — FFT → 위상 필터 → IFFT —
        U   = fft.fft(u1, dim=1)                              # complex (B,hidden_dim)
        H   = torch.exp(1j * (self.L * self.phi))             # |H|=1, (hidden_dim,)
        V   = U * H.unsqueeze(0)                              # complex (B,hidden_dim)
        v0  = fft.ifft(V, dim=1).real.contiguous()            # (B,hidden_dim)

        # — 2차 비선형 푸리에 맵 —
        v1 = self.fnmap(v0)                                   # (B,hidden_dim)
        # — Poincaré 지수 맵 —
        y  = hb._C.exp_map_origin_cuda(v1, self.c)            # (B,hidden_dim)

        # — Residual —
        h2 = self.alpha * y + self.beta * h1                  # (B,hidden_dim)
        h3 = self.relu(self.bn2(h2))
        return self.fc2(h3)


# ———————— 학습 · 평가 루프 ————————
def train_and_test(L=10, epochs=5, batch_size=128, lr=5e-4):
    device = torch.device("cuda")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tr_ds = datasets.MNIST('.', train=True,  download=True, transform=tf)
    te_ds = datasets.MNIST('.', train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = FourierSpectralHB(L=L, N=16, omega=1.0).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    print(f"\n=== FourierSpectralHB (L={L}) 학습 시작 ===")
    for ep in range(1, epochs+1):
        model.train()
        corr, tot = 0, 0
        t0 = time.time()
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
                print(f"[Epoch{ep}] batch {idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f} acc={100.*corr/tot:.2f}%")
        print(f"[Epoch{ep}] train time={(time.time()-t0):.1f}s acc={100.*corr/tot:.2f}%")

        model.eval()
        tc, tt = 0, 0
        t1 = time.time()
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                tc += (model(imgs).argmax(dim=1)==labs).sum().item()
                tt += labs.size(0)
        print(f"[Epoch{ep}] test acc={100.*tc/tt:.2f}% time={(time.time()-t1):.2f}s\n")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_and_test()
