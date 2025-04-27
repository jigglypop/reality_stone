"""
버터플라이 구조의 속도 향상 테스트 스크립트
이 스크립트는 riemannian_manifold 패키지를 사용하여
다양한 차원에서 버터플라이 변환과 기존 선형 변환의 속도를 비교합니다.
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# GPU 설정 및 정보 출력
print("===== GPU/CUDA 정보 확인 =====")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CPU에서 실행 중")

# 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ------------ 모델 정의 ---------------

# 표준 MLP 모델
class StandardMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(StandardMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# 버터플라이 변환 (O(n log n) 복잡도)
class ButterflyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ButterflyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 차원을 2의 거듭제곱으로 조정
        self.log_dim = int(np.ceil(np.log2(max(in_features, out_features))))
        self.butterfly_dim = 2 ** self.log_dim
        
        # 버터플라이 파라미터
        self.butterfly_params = nn.ParameterList()
        
        # 각 층의 파라미터 초기화
        for layer in range(self.log_dim):
            block_size = 2 ** layer
            num_blocks = self.butterfly_dim // (2 * block_size)
            
            # a, b 파라미터 (회전 행렬)
            theta = torch.rand(num_blocks) * 0.01
            a = nn.Parameter(torch.cos(theta))
            b = nn.Parameter(torch.sin(theta))
            self.butterfly_params.append(nn.ParameterList([a, b]))
        
        # 입출력 차원 조정을 위한 선형 변환
        self.input_proj = nn.Linear(in_features, self.butterfly_dim, bias=False)
        self.output_proj = nn.Linear(self.butterfly_dim, out_features, bias=True)
        
    def forward(self, x):
        # 입력 차원 조정
        x = self.input_proj(x)
        
        # 각 버터플라이 레이어 적용
        batch_size = x.size(0)
        for layer_idx, (a, b) in enumerate(self.butterfly_params):
            x = self._apply_butterfly_layer(x, a, b, layer_idx)
        
        # 출력 차원 조정
        x = self.output_proj(x)
        return x
    
    def _apply_butterfly_layer(self, x, a, b, layer_idx):
        # 현재 블록 크기
        block_size = 2 ** layer_idx
        batch_size, dim = x.size(0), x.size(1)
        
        # 배치 연산을 위한 텐서 재구성
        # [batch_size, dim] -> [batch_size, num_blocks, 2, block_size/2]
        num_blocks = dim // (2 * block_size)
        x_view = x.view(batch_size, num_blocks, 2, block_size)
        
        # 회전 파라미터 확장
        a_expanded = a.view(num_blocks, 1, 1).expand(num_blocks, 1, block_size)
        b_expanded = b.view(num_blocks, 1, 1).expand(num_blocks, 1, block_size)
        
        # 회전 연산 (Givens 회전)
        x_rotated = torch.zeros_like(x_view)
        x_rotated[:, :, 0, :] = a_expanded * x_view[:, :, 0, :] + b_expanded * x_view[:, :, 1, :]
        x_rotated[:, :, 1, :] = -b_expanded * x_view[:, :, 0, :] + a_expanded * x_view[:, :, 1, :]
        
        # 원래 형태로 복원
        return x_rotated.view(batch_size, dim)


# 버터플라이 MLP
class ButterflyMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(ButterflyMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.butterfly = ButterflyLayer(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.butterfly(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# ------------ 속도 테스트 함수 ---------------

def test_layer_speed(dims, batch_size=1024, repeats=10):
    """다양한 차원에서 버터플라이 층과 선형 층의 속도 비교"""
    butterfly_times = []
    linear_times = []
    
    for dim in dims:
        print(f"\n테스트 차원: {dim}")
        
        # 테스트 데이터
        x = torch.randn(batch_size, dim, device=device)
        
        # 버터플라이 층 테스트
        butterfly = ButterflyLayer(dim, dim).to(device)
        
        # 선형 층 테스트
        linear = nn.Linear(dim, dim).to(device)
        
        # 파라미터 수 비교
        butterfly_params = sum(p.numel() for p in butterfly.parameters())
        linear_params = sum(p.numel() for p in linear.parameters())
        print(f"버터플라이 층 파라미터: {butterfly_params:,}")
        print(f"선형 층 파라미터: {linear_params:,}")
        
        # 메모리 초기화
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 워밍업
        with torch.no_grad():
            for _ in range(5):
                _ = butterfly(x)
                _ = linear(x)
        
        # 버터플라이 층 시간 측정
        butterfly_time = 0
        with torch.no_grad():
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = butterfly(x)
                end.record()
                
                torch.cuda.synchronize()
                butterfly_time += start.elapsed_time(end)
            
            butterfly_time /= repeats
            butterfly_times.append(butterfly_time)
        
        # 선형 층 시간 측정
        linear_time = 0
        with torch.no_grad():
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = linear(x)
                end.record()
                
                torch.cuda.synchronize()
                linear_time += start.elapsed_time(end)
            
            linear_time /= repeats
            linear_times.append(linear_time)
        
        # 속도 비교
        speedup = linear_time / butterfly_time if butterfly_time > 0 else 0
        theoretical = (dim*dim) / (dim*np.log2(dim))
        print(f"버터플라이 층 시간: {butterfly_time:.3f} ms")
        print(f"선형 층 시간: {linear_time:.3f} ms")
        print(f"속도 향상: {speedup:.2f}x (이론값: {theoretical:.2f}x)")
    
    return dims, butterfly_times, linear_times


def test_model_speed():
    """MNIST에서 모델 속도 비교"""
    # MNIST 데이터셋 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # 표준 MLP 모델
    standard_mlp = StandardMLP().to(device)
    
    # 버터플라이 MLP 모델
    butterfly_mlp = ButterflyMLP().to(device)
    
    # 파라미터 수 비교
    standard_params = sum(p.numel() for p in standard_mlp.parameters())
    butterfly_params = sum(p.numel() for p in butterfly_mlp.parameters())
    print(f"\n표준 MLP 파라미터: {standard_params:,}")
    print(f"버터플라이 MLP 파라미터: {butterfly_params:,}")
    
    # 추론 시간 측정
    standard_times = []
    butterfly_times = []
    
    # 메모리 초기화
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 워밍업
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = standard_mlp(data)
            _ = butterfly_mlp(data)
            break
    
    # 추론 시간 측정
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 표준 MLP 시간 측정
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = standard_mlp(data)
            end.record()
            
            torch.cuda.synchronize()
            standard_times.append(start.elapsed_time(end))
            
            # 버터플라이 MLP 시간 측정
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = butterfly_mlp(data)
            end.record()
            
            torch.cuda.synchronize()
            butterfly_times.append(start.elapsed_time(end))
    
    # 평균 시간 계산
    avg_standard = np.mean(standard_times)
    avg_butterfly = np.mean(butterfly_times)
    speedup = avg_standard / avg_butterfly if avg_butterfly > 0 else 0
    
    print(f"\n표준 MLP 추론 시간: {avg_standard:.3f} ms")
    print(f"버터플라이 MLP 추론 시간: {avg_butterfly:.3f} ms")
    print(f"속도 향상: {speedup:.2f}x")
    
    hidden_dim = 256
    theoretical = (hidden_dim*hidden_dim) / (hidden_dim*np.log2(hidden_dim))
    print(f"히든 레이어의 이론적 속도 향상: {theoretical:.2f}x")
    
    return avg_standard, avg_butterfly


def plot_results(dimensions, butterfly_times, linear_times):
    """결과 시각화"""
    plt.figure(figsize=(12, 6))
    
    # 시간 비교 그래프
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, butterfly_times, 'go-', linewidth=2, label='버터플라이 층 (O(n log n))')
    plt.plot(dimensions, linear_times, 'ro-', linewidth=2, label='선형 층 (O(n²))')
    plt.xlabel('차원 (n)')
    plt.ylabel('실행 시간 (ms)')
    plt.title('버터플라이 층 vs. 선형 층 속도')
    plt.legend()
    plt.grid(True)
    
    # 속도 향상 그래프
    plt.subplot(1, 2, 2)
    speedups = [linear / butterfly if butterfly > 0 else 0 
                for linear, butterfly in zip(linear_times, butterfly_times)]
    theoretical = [(dim*dim) / (dim*np.log2(dim)) for dim in dimensions]
    
    plt.plot(dimensions, speedups, 'bo-', linewidth=2, label='실제 속도 향상')
    plt.plot(dimensions, theoretical, 'k--', linewidth=2, label='이론적 속도 향상')
    plt.xlabel('차원 (n)')
    plt.ylabel('속도 향상 (x배)')
    plt.title('버터플라이 구조의 속도 이점')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('butterfly_speedup.png')
    print("결과가 'butterfly_speedup.png'에 저장되었습니다.")


if __name__ == "__main__":
    print("\n======= 버터플라이 구조 속도 테스트 =======")
    
    # 다양한 차원에서 버터플라이 층과 선형 층 비교
    if device.type == 'cuda':
        dimensions = [64, 128, 256, 512, 1024, 2048]
    else:
        # CPU의 경우 작은 차원만 테스트
        dimensions = [64, 128, 256, 512]
    
    dimensions, butterfly_times, linear_times = test_layer_speed(dimensions)
    
    # 결과 시각화
    plot_results(dimensions, butterfly_times, linear_times)
    
    # MNIST에서 모델 속도 비교
    print("\n======= MNIST 모델 속도 테스트 =======")
    avg_standard, avg_butterfly = test_model_speed()
    
    print("\n테스트 완료!") 