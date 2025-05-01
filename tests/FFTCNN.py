import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import hyper_butterfly as hb
import torch.fft as fft

# ————— 비선형 사인·코사인 맵 —————
class FourierNonlinearMap(nn.Module):
    def __init__(self, dim, N=16, omega=1.0):
        super().__init__()
        self.N     = N
        self.omega = omega
        self.a0 = nn.Parameter(torch.ones(1))
        self.a  = nn.Parameter(torch.zeros(N))
        self.b  = nn.Parameter(torch.zeros(N))

    def forward(self, x):
        # x: (M, C)
        r = x.norm(dim=1, keepdim=True).clamp_min(1e-6)   # (M,1)
        n = torch.arange(1, self.N+1, device=x.device, dtype=r.dtype).view(1, -1)  # (1,N)
        angle = self.omega * (r @ n)                     # (M,N)
        f_r = self.a0 \
            + (torch.cos(angle) * self.a).sum(dim=1, keepdim=True) \
            + (torch.sin(angle) * self.b).sum(dim=1, keepdim=True)
        return f_r * x                                   # (M,C)

# ———— 채널 스펙트럴 + 비선형 융합 블록 ————
class ChannelNonlinearSpectralBlock(nn.Module):
    def __init__(self, channels, c=1e-3, L=10, N=16, omega=1.0):
        super().__init__()
        self.c = c
        self.L = L
        # 1차 + 2차 비선형 맵
        self.fnmap1 = FourierNonlinearMap(channels, N, omega)
        self.fnmap2 = FourierNonlinearMap(channels, N, omega)
        # 순수 위상 φ ∈ ℝ^C
        self.phi   = nn.Parameter(torch.zeros(channels))
        # residual 스케일
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # (M, C), M = B*H*W
        h0 = x.permute(0,2,3,1).reshape(-1, C)
        # Poincaré 로그 사상
        u = hb._C.log_map_origin_cuda(h0, self.c)        # (M,C)
        u1 = self.fnmap1(u)                              # (M,C)
        # FFT ↑ 위상필터 ↓ IFFT
        U   = fft.rfft(u1, dim=1)                        # (M, C//2+1)
        Hf  = torch.exp(1j * (self.L * self.phi))        # (C,)
        Hf_r= Hf[:U.shape[-1]].unsqueeze(0)              # (1, C//2+1)
        V   = U * Hf_r                                   # (M, C//2+1)
        v0  = fft.irfft(V, n=C, dim=1).contiguous()      # (M,C)
        v1  = self.fnmap2(v0)                            # (M,C)
        y   = hb._C.exp_map_origin_cuda(v1, self.c)      # (M,C)
        # residual & 복원
        out = self.alpha * y + self.beta * h0            # (M,C)
        out = out.view(B, H, W, C).permute(0,3,1,2)      # (B,C,H,W)
        return out

# ————— CNN + 스펙트럴 블록 —————
class FourierCNN(nn.Module):
    def __init__(self, num_classes=100, c=1e-3, L=10, N=16, omega=1.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.spec1 = ChannelNonlinearSpectralBlock(64, c, L, N, omega)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.spec2 = ChannelNonlinearSpectralBlock(128, c, L, N, omega)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x); x = self.spec1(x)
        x = self.layer2(x); x = self.spec2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ————— 학습 · 평가 루프 —————
def train_and_eval(batch_size=128, epochs=20, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
    ])
    train_ds = torchvision.datasets.CIFAR100('.', train=True,  download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR100('.', train=False, download=True, transform=transform_test)
    tr = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    te = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    model = FourierCNN().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        correct, total, t0 = 0,0, time.time()
        for imgs, labs in tr:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs)
            lossf(out, labs).backward()
            opt.step()
            pred = out.argmax(1)
            correct += (pred==labs).sum().item()
            total   += labs.size(0)
        print(f"[Ep{ep}] train acc={100*correct/total:.2f}% time={(time.time()-t0):.1f}s")

        model.eval()
        correct, total, t1 = 0,0, time.time()
        with torch.no_grad():
            for imgs, labs in te:
                imgs, labs = imgs.to(device), labs.to(device)
                pred = model(imgs).argmax(1)
                correct += (pred==labs).sum().item()
                total   += labs.size(0)
        print(f"[Ep{ep}]  test acc={100*correct/total:.2f}% time={(time.time()-t1):.1f}s\n")

if __name__=="__main__":
    train_and_eval()
