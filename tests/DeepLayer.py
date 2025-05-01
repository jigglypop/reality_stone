import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

# 1) 순수 FFT 스펙트럴 필터 정의
class PureSpectralFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # real+imag 파라미터를 하나의 복소수 텐서로
        self.complex_weight = nn.Parameter(
            torch.randn(dim, dtype=torch.cfloat) * 0.01
        )

    def forward(self, x):
        # x: (B, dim) -> FFT -> 필터링 -> IFFT -> real
        X = torch.fft.fft(x, dim=1)
        Y = X * self.complex_weight.unsqueeze(0)
        return torch.fft.ifft(Y, dim=1).real.contiguous()

# 2) DeepPureSpectral: L번 순수 FFT 필터만 적용 (Residual 없음)
class DeepPureSpectral(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, out_dim=10, L=100):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, device='cuda')
        self.bn1 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.relu = nn.ReLU()
        self.spectral = PureSpectralFilter(hidden_dim).cuda()
        self.L = L
        self.bn2 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.fc2 = nn.Linear(hidden_dim, out_dim, device='cuda')

    def forward(self, x):
        x = x.cuda().view(x.size(0), -1)
        h = self.relu(self.bn1(self.fc1(x)))
        # 이 루프가 ‘L번 겹친’ 부분입니다
        for layer_idx in range(self.L):
            h = self.spectral(h)
        h = self.relu(self.bn2(h))
        return self.fc2(h)

# 3) 학습·평가 루프: 배치마다 로그 출력
def train_and_test(L=1000):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=128, shuffle=False)

    model = DeepPureSpectral(L=L).cuda()
    name = f"PureFFT (L={L})"
    print(f"\n=== {name} 학습 시작 ===")

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 4):  # 3 에포크만 실행
        model.train()
        correct, total = 0, 0
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total   += target.size(0)

            # 배치마다 (혹은 50단위) 로그
            if batch_idx % 50 == 0 or batch_idx == len(train_loader)-1:
                acc = 100. * correct / total
                print(f"[{name}] 에포크{epoch} 배치{batch_idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f} acc={acc:.2f}%")

        epoch_time = time.time() - start
        train_acc  = 100. * correct / total
        print(f"[{name}] 에포크{epoch} 완료 time={epoch_time:.1f}s train_acc={train_acc:.2f}%")

        # 테스트
        model.eval()
        test_correct, test_total = 0, 0
        t0 = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                out = model(data)
                test_correct += (out.argmax(dim=1) == target).sum().item()
                test_total   += target.size(0)
        test_time = time.time() - t0
        test_acc  = 100. * test_correct / test_total
        print(f"[{name}] 테스트_acc={test_acc:.2f}% inference_time={test_time:.2f}s\n")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train_and_test(L=1000)
