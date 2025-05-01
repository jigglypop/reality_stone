import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import hyper_butterfly as hb

class EnhancedSpectralFilter(nn.Module):
    """개선된 FFT 기반 스펙트럴 필터"""
    def __init__(self, dim, curvature=1e-3):
        super().__init__()
        self.dim = dim
        self.c = curvature
        
        # 1. 필터 계수 (주파수 영역)
        self.g_real = nn.Parameter(torch.ones(dim, device='cuda'))
        self.g_imag = nn.Parameter(torch.zeros(dim, device='cuda'))
        
        # 2. 추가 파라미터 - 혼합 계수 (잔차 연결용)
        self.alpha = nn.Parameter(torch.ones(1, device='cuda'))
        self.beta = nn.Parameter(torch.zeros(1, device='cuda'))
        
        # 3. 주파수 변조 레이어 (비선형성 추가)
        self.freq_modulation = nn.Sequential(
            nn.Linear(dim, dim, device='cuda'),
            nn.Tanh()
        )
        
        # 4. 필터 초기화 - 항등 매핑에 가깝게
        with torch.no_grad():
            # 모든 주파수 통과하게 초기화
            self.g_real.fill_(1.0)
            self.g_imag.fill_(0.0)
    
    def forward(self, x):
        # 원본 신호 저장 (잔차 연결용)
        x_orig = x
        
        # 1) 로그 맵 적용
        u = hb._C.log_map_origin_cuda(x, self.c)
        
        # 2) 주파수 변조 (비선형성 추가)
        u_mod = self.freq_modulation(u)
        
        # 3) FFT 적용
        U = torch.fft.fft(u_mod, dim=1)
        
        # 4) 스펙트럴 필터링
        filter_complex = torch.complex(self.g_real, self.g_imag)
        V = U * filter_complex.unsqueeze(0)
        
        # 5) 역변환
        v = torch.fft.ifft(V, dim=1).real.contiguous()
        
        # 6) 지수 맵 적용
        y = hb._C.exp_map_origin_cuda(v, self.c)
        
        # 7) 잔차 연결 - 원본 신호와 필터링된 신호 혼합
        # α*y + β*x_orig 형태로 결합
        output = self.alpha * y + self.beta * x_orig
        
        return output

class SpectralHyperNet(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, out_dim=10, c=1e-3):
        super().__init__()
        # 1) 입력 처리
        self.fc1 = nn.Linear(in_dim, hidden_dim, device='cuda')
        self.bn1 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.relu = nn.ReLU()
        
        # 2) 스펙트럴 필터
        self.spectral = EnhancedSpectralFilter(hidden_dim, c)
        
        # 3) 추가 비선형성
        self.bn2 = nn.BatchNorm1d(hidden_dim, device='cuda')
        
        # 4) 출력 분류
        self.fc2 = nn.Linear(hidden_dim, out_dim, device='cuda')
    
    def forward(self, x):
        x = x.cuda()
        x = x.view(x.size(0), -1)
        
        # 1) 입력 처리
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        
        # 2) 스펙트럴 필터 (수천 레이어 대체)
        h_trans = self.spectral(h)
        
        # 3) 추가 비선형성
        h_trans = self.bn2(h_trans)
        h_trans = self.relu(h_trans)
        
        # 4) 출력 분류
        return self.fc2(h_trans)

# 기존 하이퍼-버터플라이 모델 (비교용)
class HBClassicNet(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=256, out_dim=10, L=3, c=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, device='cuda')
        self.bn1 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.relu1 = nn.ReLU()
        
        self.c, self.L = c, L
        # 하이퍼-버터플라이 파라미터
        log2_h = int(torch.log2(torch.tensor(hidden_dim)).item())
        total_params = sum((hidden_dim // (2 * (1 << (l % log2_h)))) * 2 for l in range(L))
        self.params = nn.Parameter(torch.randn(total_params, device='cuda') * 0.01)
        
        self.bn2 = nn.BatchNorm1d(hidden_dim, device='cuda')
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, device='cuda')
    
    def forward(self, x):
        x = x.cuda()
        x = x.view(x.size(0), -1)
        
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        
        h_trans = hb.hyper_butterfly(h, self.params, self.c, self.L)
        
        h_trans = self.bn2(h_trans)
        h_trans = self.relu2(h_trans)
        
        return self.fc2(h_trans)

def run_experiment():
    # 데이터 로드
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('.', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    
    # 모델 초기화
    classic_model = HBClassicNet(L=3, c=1e-3).cuda()
    spectral_model = SpectralHyperNet(c=1e-3).cuda()
    
    # 학습/테스트 함수
    def train_epoch(model, loader, optimizer, name="모델"):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 정확도 계산
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # 진행 상황 출력
            if batch_idx % 50 == 0:
                acc = 100. * correct / total
                print(f"[{name}] 배치 {batch_idx}/{len(loader)}  loss={loss.item():.4f}  acc={acc:.2f}%")
        
        # 에포크 요약
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        final_acc = 100. * correct / total
        
        print(f"[{name}] 에포크 완료 ▶ time={epoch_time:.2f}s  loss={avg_loss:.4f}  acc={final_acc:.2f}%")
        return avg_loss, final_acc, epoch_time
    
    def test(model, loader, name="모델"):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        acc = 100. * correct / total
        print(f"[{name}] 테스트 ▶ acc={acc:.2f}%")
        return acc
    
    # 실험 1: 기존 하이퍼-버터플라이
    print("=== Experiment 1: Hyper-Butterfly MLP ===")
    classic_opt = optim.Adam(classic_model.parameters(), lr=1e-3)
    classic_accs = []
    classic_times = []
    
    for epoch in range(1, 6):
        print(f"--- Epoch {epoch}/5 ---")
        _, _, epoch_time = train_epoch(classic_model, train_loader, classic_opt, "HB-MLP")
        acc = test(classic_model, test_loader, "HB-MLP")
        classic_accs.append(acc)
        classic_times.append(epoch_time)
    
    # 실험 2: FFT 스펙트럴 모델
    print("\n=== Experiment 2: FFT Spectral HB-MLP ===")
    # 스펙트럴 모델에는 더 높은 학습률 사용
    spectral_opt = optim.Adam(spectral_model.parameters(), lr=5e-3)
    spectral_accs = []
    spectral_times = []
    
    for epoch in range(1, 6):
        print(f"--- Epoch {epoch}/5 ---")
        _, _, epoch_time = train_epoch(spectral_model, train_loader, spectral_opt, "Spectral-HB")
        acc = test(spectral_model, test_loader, "Spectral-HB")
        spectral_accs.append(acc)
        spectral_times.append(epoch_time)
    
    # 결과 요약
    print("\n=== 결과 요약 ===")
    print(f"기존 HB-MLP: 최종 정확도 {classic_accs[-1]:.2f}%, 평균 시간 {sum(classic_times)/5:.2f}초")
    print(f"FFT Spectral: 최종 정확도 {spectral_accs[-1]:.2f}%, 평균 시간 {sum(spectral_times)/5:.2f}초")
    print(f"속도 비율: {sum(classic_times)/sum(spectral_times):.2f}x")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run_experiment()