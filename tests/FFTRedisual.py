import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import reality_stone as hb

class FFTStackedHB(nn.Module):
    def __init__(self,
            in_dim:    int   = 784,
            hidden_dim:int   = 256,
            out_dim:   int   = 10,
            L:         int   = 100000000,
            c:         float = 1e-3):
        super().__init__()
        self.L  = 100000000
        self.c  = c
        
        # 1) 인풋→히든
        self.fc1   = nn.Linear(in_dim, hidden_dim).cuda()
        self.bn1   = nn.BatchNorm1d(hidden_dim).cuda()
        self.relu  = nn.ReLU()
        
        # 2) FFT용 스펙트럴 필터 파라미터
        self.g_real = nn.Parameter(torch.ones(hidden_dim).cuda())
        self.g_imag = nn.Parameter(torch.zeros(hidden_dim).cuda())
        # residual scale
        self.alpha  = nn.Parameter(torch.ones(1).cuda())
        self.beta   = nn.Parameter(torch.zeros(1).cuda())
        
        # 3) 아웃풋
        self.bn2 = nn.BatchNorm1d(hidden_dim).cuda()
        self.fc2 = nn.Linear(hidden_dim, out_dim).cuda()

    def forward(self, x):
        # (B,1,28,28) 혹은 (B,784) → (B,784)
        x0 = x.view(x.size(0), -1).cuda()
        # 인풋 레이어
        h = self.relu(self.bn1(self.fc1(x0)))              # → (B, hidden_dim)

        # 1) Poincaré 로그 맵
        u = hb._C.log_map_origin_cuda(h, self.c)           # → (B, hidden_dim)
        # 2) FFT
        U = torch.fft.fft(u, dim=1)                        # → complex (B, hidden_dim)
        # 3) 필터 거듭제곱
        H = torch.complex(self.g_real, self.g_imag)        # (hidden_dim,)
        H_eff = H.pow(self.L)                              # (hidden_dim,)
        # 4) 필터링
        V = U * H_eff.unsqueeze(0)                         # (B, hidden_dim)
        # 5) IFFT → real
        v = torch.fft.ifft(V, dim=1).real.contiguous()     # → (B, hidden_dim)
        # 6) Poincaré 지수 맵
        y = hb._C.exp_map_origin_cuda(v, self.c)           # → (B, hidden_dim)
        # 7) residual 연결
        h2 = self.alpha * y + self.beta * h                # → (B, hidden_dim)
        # 8) 마무리
        h3 = self.relu(self.bn2(h2))
        return self.fc2(h3)

def train_and_test(L=100000000, epochs=5, batch_size=128, lr=5e-4):
    device = torch.device("cuda")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tr_ds = datasets.MNIST('.', train=True,  download=True, transform=tf)
    te_ds = datasets.MNIST('.', train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    
    model = FFTStackedHB(L=100000000).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    
    print(f"\n=== FFTStackedHB (L={L}) 학습 시작 ===")
    for ep in range(1, epochs+1):
        model.train()
        corr, tot = 0,0
        t0 = time.time()
        for idx,(imgs,labs) in enumerate(train_loader):
            imgs,labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = lossf(out, labs)
            loss.backward()
            opt.step()
            pred = out.argmax(dim=1)
            corr += (pred==labs).sum().item()
            tot  += labs.size(0)
            if idx%50==0:
                print(f"[Epoch{ep}] batch {idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f} acc={100.*corr/tot:.2f}%")
        print(f"[Epoch{ep}] train time={(time.time()-t0):.1f}s acc={100.*corr/tot:.2f}%")
        
        model.eval()
        tc,tt=0,0
        t1=time.time()
        with torch.no_grad():
            for imgs,labs in test_loader:
                imgs,labs = imgs.to(device), labs.to(device)
                pred = model(imgs).argmax(dim=1)
                tc += (pred==labs).sum().item()
                tt += labs.size(0)
        print(f"[Epoch{ep}] test acc={100.*tc/tt:.2f}% time={(time.time()-t1):.2f}s\n")

if __name__=="__main__":
    torch.backends.cudnn.benchmark=True
    train_and_test()
