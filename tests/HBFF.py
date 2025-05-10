import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import reality_stone as hb

class FFTStackedHB_SegRes(nn.Module):
    def __init__(self,
                 in_dim:     int   = 784,
                 hidden_dim: int   = 256,
                 out_dim:    int   = 10,
                 L:          int   = 1000,
                 segments:   int   = 20,
                 c:          float = 1e-3):
        super().__init__()
        assert L % segments == 0, "L은 segments의 배수여야 합니다"
        self.seg_len  = L // segments
        self.segments = segments
        self.c        = c

        # ── 인풋 MLP ──
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # ── 세그먼트별 복소 필터 파라미터 & residual 스케일 ──
        self.H_reals = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim)) for _ in range(segments)
        ])
        self.H_imags = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(segments)
        ])
        self.alphas  = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(segments)
        ])
        self.betas   = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(segments)
        ])

        # ── 출력 MLP ──
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # 1) 인풋 처리
        h = x.view(x.size(0), -1)
        h = torch.relu(self.bn1(self.fc1(h)))    # 첫 레이어만 ReLU

        # 2) 세그먼트별 FFT + residual
        for i in range(self.segments):
            # Poincaré 로그 맵
            u = hb._C.log_map_origin_cuda(h, self.c)
            # FFT
            U = torch.fft.fft(u, dim=1)
            # 필터 거듭제곱
            H_i = torch.complex(self.H_reals[i], self.H_imags[i])
            H_eff = H_i.pow(self.seg_len)
            # 필터링 → IFFT → real
            v = torch.fft.ifft(U * H_eff.unsqueeze(0), dim=1).real.contiguous()
            # Poincaré 지수 맵
            y = hb._C.exp_map_origin_cuda(v, self.c)
            # 세그먼트 residual
            h = self.alphas[i] * y + self.betas[i] * h

        # 3) 출력 MLP
        h = torch.relu(self.bn2(h))
        return self.fc2(h)

def train_and_test(L=1000, segments=20, epochs=5, bs=128, lr=5e-4):
    device = torch.device("cuda")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tr_ds = datasets.MNIST('.', train=True, download=True, transform=tf)
    te_ds = datasets.MNIST('.', train=False, download=True, transform=tf)
    tr = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=True)
    te = torch.utils.data.DataLoader(te_ds, batch_size=bs, shuffle=False)

    model = FFTStackedHB_SegRes(L=L, segments=segments).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n=== FFTStackedHB_SegRes (L={L}, S={segments}) 학습 시작 ===")
    for ep in range(1, epochs+1):
        model.train()
        corr, tot = 0, 0
        t0 = time.time()
        for idx, (imgs, labs) in enumerate(tr):
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = lossf(out, labs)
            loss.backward()
            opt.step()
            pred = out.argmax(1)
            corr += (pred == labs).sum().item()
            tot  += labs.size(0)
            if idx % 50 == 0:
                print(f"[Epoch{ep}] batch {idx}/{len(tr)} "
                      f"loss={loss.item():.4f} acc={100.*corr/tot:.2f}%")
        print(f"[Epoch{ep}] train time={(time.time()-t0):.1f}s acc={100.*corr/tot:.2f}%")

        model.eval()
        tc, tt = 0, 0
        t1 = time.time()
        with torch.no_grad():
            for imgs, labs in te:
                imgs, labs = imgs.to(device), labs.to(device)
                pred = model(imgs).argmax(1)
                tc += (pred == labs).sum().item()
                tt += labs.size(0)
        print(f"[Epoch{ep}] test acc={100.*tc/tt:.2f}% time={(time.time()-t1):.2f}s\n")

if __name__=="__main__":
    torch.backends.cudnn.benchmark = True
    train_and_test()
