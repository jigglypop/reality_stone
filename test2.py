# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import hyper_butterfly as hb
# 
# class PureHBMLP(nn.Module):
#     """
#     (1) 입력 784 차원을 Linear 로 128로 축소
#     (2) 첫 HyperButterfly
#     (3) 두 번째 HyperButterfly
#     (4) final linear classifier
#     """
#     def __init__(self, in_dim=784, hidden_dim=128, num_classes=10, c=1e-3, L=1):
#         super().__init__()
#         self.c, self.L = c, L
#         self.in_dim = in_dim
#         self.hid    = hidden_dim
# 
#         # 1) 입력→hidden 매핑
#         self.fc0 = nn.Linear(in_dim, hidden_dim)
# 
#         # 2) HB 파라미터 계산
#         log2_h = int(torch.log2(torch.tensor(hidden_dim)).item())
#         total_p = sum((hidden_dim // (2 * (1 << (l % log2_h)))) * 2 for l in range(L))
#         print(f"[PureHB] hidden={hidden_dim}, params_per_layer={total_p}")
# 
#         self.params1 = nn.Parameter(torch.randn(total_p) * 1e-3)
#         self.params2 = nn.Parameter(torch.randn(total_p) * 1e-3)
# 
#         # 3) 분류용 가중치·편향
#         self.class_w = nn.Parameter(torch.randn(hidden_dim, num_classes) * 0.01)
#         self.class_b = nn.Parameter(torch.zeros(num_classes))
# 
#     def forward(self, x):
#         b = x.size(0)
#         x = x.view(b, self.in_dim)       # (b, 784)
#         h = torch.relu(self.fc0(x))      # (b, 128)
# 
#         # 첫 번째 HB
#         u1 = hb.hyper_butterfly(h.contiguous(),
#                                 self.params1, self.c, self.L)
#         # 두 번째 HB
#         u2 = hb.hyper_butterfly(u1.contiguous(),
#                                 self.params2, self.c, self.L)
# 
#         # final classifier
#         # u2: (b, 128), class_w: (128,10) → logits: (b,10)
#         logits = u2 @ self.class_w + self.class_b
#         return logits
# 
# def train_epoch(model, loader, opt, device):
#     model.train()
#     total, t0 = 0.0, time.time()
#     for imgs, labels in loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#         opt.zero_grad()
#         loss = nn.functional.cross_entropy(model(imgs), labels)
#         loss.backward()
#         opt.step()
#         total += loss.item() * imgs.size(0)
#     return total / len(loader.dataset), time.time() - t0
# 
# def test_epoch(model, loader, device):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             pred = model(imgs).argmax(dim=1)
#             correct += (pred == labels).sum().item()
#             total   += imgs.size(0)
#     return correct / total
# 
# if __name__=="__main__":
#     device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     bs, lr, ep = 256, 1e-3, 10
# 
#     tf = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     tr = datasets.MNIST(".", train=True,  download=True, transform=tf)
#     te = datasets.MNIST(".", train=False, download=True, transform=tf)
#     trl = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True,  num_workers=0)
#     tel = torch.utils.data.DataLoader(te, batch_size=bs, shuffle=False, num_workers=0)
# 
#     model = PureHBMLP(in_dim=784, hidden_dim=128, num_classes=10, c=1e-3, L=1).to(device)
#     opt   = optim.Adam(model.parameters(), lr=lr)
# 
#     print("\n--- PureHBMLP Training (with fc0) ---")
#     for epoch in range(1, ep+1):
#         loss, t   = train_epoch(model, trl, opt, device)
#         acc       = test_epoch(model, tel, device)
#         print(f"Epoch {epoch}/{ep}  loss={loss:.4f}  time={t:.2f}s  acc={acc*100:.2f}%")
# 
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import hyper_butterfly as hb

class PureHBNoActMLP(nn.Module):
    """
    활성화 함수 없이
    (1) 입력→Linear(784→128)
    (2) HyperButterfly1
    (3) HyperButterfly2
    (4) Linear 분류기
    """
    def __init__(self, in_dim=784, hidden_dim=128, num_classes=10, c=1e-3, L=1):
        super().__init__()
        self.c, self.L = c, L
        self.in_dim = in_dim
        self.hid    = hidden_dim

        # 1) 입력→hidden (activation 제거)
        self.fc0 = nn.Linear(in_dim, hidden_dim)

        # HB 파라미터 수 계산
        log2_h = int(torch.log2(torch.tensor(hidden_dim)).item())
        total_p = sum((hidden_dim // (2 * (1 << (l % log2_h)))) * 2 for l in range(L))
        print(f"[PureHBNoAct] hidden={hidden_dim}, params={total_p}")
        self.params1 = nn.Parameter(torch.randn(total_p) * 1e-3)
        self.params2 = nn.Parameter(torch.randn(total_p) * 1e-3)

        # 4) 분류용 weight/bias
        self.class_w = nn.Parameter(torch.randn(hidden_dim, num_classes) * 0.01)
        self.class_b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, self.in_dim)        # (b,784)
        h = self.fc0(x)                   # (b,128)  ← 여기서 ReLU 제거!

        # 첫 번째 HB
        u1 = hb.hyper_butterfly(h.contiguous(),
                                self.params1, self.c, self.L)
        # 두 번째 HB
        u2 = hb.hyper_butterfly(u1.contiguous(),
                                self.params2, self.c, self.L)

        # 최종 분류
        logits = u2 @ self.class_w + self.class_b  # (b,10)
        return logits

def train_epoch(model, loader, opt, device):
    model.train()
    total, t0 = 0.0, time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(imgs), labels)
        loss.backward()
        opt.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset), time.time() - t0

def test_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
    return correct / total

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, lr, epochs = 256, 1e-3, 10

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST(".", train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(".", train=False, download=True, transform=tf)
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    te_loader = torch.utils.data.DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

    model = PureHBNoActMLP(784, 128, 10, c=1e-3, L=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("\n--- PureHBNoActMLP Training (No Activation Anywhere) ---")
    for ep in range(1, epochs+1):
        loss, t = train_epoch(model, tr_loader, optimizer, device)
        acc     = test_epoch(model, te_loader, device)
        print(f"Epoch {ep}/{epochs}  loss={loss:.4f}  time={t:.2f}s  acc={acc*100:.2f}%")
