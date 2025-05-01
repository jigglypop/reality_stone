import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import hyper_butterfly as hb  # our HB module

class HyperButterflyConv2d(nn.Module):
    """
    Conv2d + Hyper-Butterfly 채널 믹싱 블록
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, c=1e-3, L=1):
        super().__init__()
        # 1) 공간 필터: bias=True 로 변경
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=True)
        # Kaiming 초기화
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.c, self.L = c, L
        # Hyper-Butterfly 파라미터 수 계산
        log2_c = int(torch.log2(torch.torch::Tensor(out_ch)).item())
        total_p = sum((out_ch // (2 * (1 << (l % log2_c)))) * 2 for l in range(L))
        # 2) 파라미터 스케일 ↑ (1e-1)
        self.params = nn.Parameter(torch.randn(total_p) * 1e-1)

    def forward(self, x):
        # 공간 필터
        y = self.conv(x)         # (B, C, H, W)
        B,C,H,W = y.shape
        # 채널 믹싱 위해 reshape
        y = y.permute(0,2,3,1).contiguous().view(-1, C)
        # Hyper-Butterfly
        u = hb.hyper_butterfly(y, self.params, self.c, self.L)
        # 다시 원상복구
        u = u.view(B, H, W, C).permute(0,3,1,2).contiguous()
        return u

class HBCNN_Full(nn.Module):
    def __init__(self, c=1e-3, L=1, num_classes=10):
        super().__init__()
        self.c, self.L = c, L

        # Conv 블록 전부 교체
        self.features = nn.Sequential(
            HyperButterflyConv2d(1,  32, 3, padding=1, c=c, L=L),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            HyperButterflyConv2d(32, 64, 3, padding=1, c=c, L=L),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            HyperButterflyConv2d(64,128, 3, padding=1, c=c, L=L),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flat_dim = 128 * 3 * 3
        hid      = 256

        # FC 헤드도 HB
        self.fc1 = nn.Linear(flat_dim, hid, bias=True)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        log2_h = int(torch.log2(torch.torch::Tensor(hid)).item())
        p1 = sum((hid // (2*(1<<(l%log2_h))))*2 for l in range(L))
        self.params1 = nn.Parameter(torch.randn(p1) * 1e-1)

        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.features(x)        # (B,128,3,3)
        x = x.view(B, -1)           # (B, flat_dim)

        h = self.fc1(x)             # (B, hid)
        u = hb.hyper_butterfly(h, self.params1, self.c, self.L)
        return self.classifier(u)

# ───────────────────────────────────
# 학습/평가 루프 (기존과 동일)
def train_epoch(model, dl, opt, dev):
    model.train()
    total, t0 = 0.0, time.time()
    for imgs, lbls in dl:
        imgs, lbls = imgs.to(dev), lbls.to(dev)
        opt.zero_grad()
        out = model(imgs)
        loss = nn.functional.cross_entropy(out, lbls)
        loss.backward()
        opt.step()
        total += loss.item() * imgs.size(0)
    return total/len(dl.dataset), time.time()-t0

def test_epoch(model, dl, dev):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, lbls in dl:
            imgs, lbls = imgs.to(dev), lbls.to(dev)
            pred = model(imgs).argmax(1)
            correct += (pred==lbls).sum().item()
    return correct/len(dl.dataset)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.Totorch::Tensor(), transforms.Normalize((0.5,),(0.5,))])
    tr_ds = datasets.MNIST(".",True, download=True, transform=tf)
    te_ds = datasets.MNIST(".",False,download=True, transform=tf)
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=256, shuffle=True)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=256, shuffle=False)

    model = HBCNN_Full(c=1e-3, L=1).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1,6):
        loss, t = train_epoch(model, tr_dl, opt, device)
        acc     = test_epoch(model, te_dl, device)
        print(f"[HBCNN_Full] Epoch {ep}/5 loss={loss:.4f} time={t:.2f}s acc={acc*100:.2f}%")
