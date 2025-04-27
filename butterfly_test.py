import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from riemannian_manifold import (
    HyperbolicOperations,
    ButterflyTransform,
    HyperButterflyLayer,
    HyperButterflyMLP
)

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

def benchmark_butterfly_transform():
    """
    버터플라이 변환의 연산 속도를 일반 선형 변환과 비교
    """
    print("\n===== 버터플라이 변환 벤치마킹 =====")
    
    # 테스트할 차원 목록
    dimensions = [64, 128, 256, 512, 1024, 2048]
    batch_size = 128
    
    butterfly_times = []
    linear_times = []
    
    for dim in dimensions:
        print(f"\n--- 입력 차원: {dim} ---")
        
        # 테스트 데이터 생성
        x = torch.randn(batch_size, dim, device=device)
        
        # 버터플라이 변환 초기화
        butterfly = ButterflyTransform(dim, device=device).to(device)
        
        # 동일한 파라미터 수의 선형 레이어 초기화
        butterfly_params = sum(p.numel() for p in butterfly.parameters())
        linear = torch.nn.Linear(dim, dim, bias=False).to(device)
        print(f"버터플라이 파라미터 수: {butterfly_params:,}")
        print(f"선형 레이어 파라미터 수: {dim*dim:,}")
        
        # 버터플라이 변환 시간 측정
        print("버터플라이 변환 실행 중...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(100):  # 여러 번 실행하여 정확한 측정
            _ = butterfly(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        butterfly_time = (time.time() - start_time) / 100
        butterfly_times.append(butterfly_time)
        
        # 선형 레이어 시간 측정
        print("선형 레이어 실행 중...")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(100):
            _ = linear(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        linear_time = (time.time() - start_time) / 100
        linear_times.append(linear_time)
        
        print(f"버터플라이 변환 시간: {butterfly_time*1000:.2f}ms")
        print(f"선형 변환 시간: {linear_time*1000:.2f}ms")
        print(f"속도 향상: {linear_time/butterfly_time:.2f}x")
        print(f"이론적 속도 향상: {(dim*dim)/(dim*np.log2(dim)):.2f}x")
    
    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(dimensions, [t*1000 for t in butterfly_times], 'go-', linewidth=2, label='버터플라이 변환 (O(n log n))')
    plt.plot(dimensions, [t*1000 for t in linear_times], 'ro-', linewidth=2, label='선형 레이어 (O(n²))')
    plt.xlabel('차원 (n)')
    plt.ylabel('실행 시간 (ms)')
    plt.title('버터플라이 변환 vs. 선형 레이어 속도 비교')
    plt.legend()
    plt.grid(True)
    plt.xscale('log2')
    plt.yscale('log2')
    plt.savefig('butterfly_benchmark.png')
    
    print("\n결과가 'butterfly_benchmark.png'에 저장되었습니다.")
    return dimensions, butterfly_times, linear_times

def test_hyperbolic_operations():
    """
    하이퍼볼릭 연산 테스트
    """
    print("\n===== 하이퍼볼릭 연산 테스트 =====")
    
    # 하이퍼볼릭 연산 클래스 초기화
    hyper_ops = HyperbolicOperations()
    
    # 테스트 데이터 생성
    batch_size = 16
    dim = 64
    curvature = 0.1
    
    # 유클리드 공간의 점 생성
    x_euc = torch.randn(batch_size, dim, device=device) * 0.1
    
    # 포인카레 볼로 변환
    x_poincare = hyper_ops.euclidean_to_poincare(x_euc, curvature)
    
    # 포인카레 볼 경계 체크
    max_norm = torch.norm(x_poincare, dim=1).max().item()
    max_allowed = 1.0 / np.sqrt(curvature)
    print(f"최대 노름: {max_norm:.6f}, 허용 최대 노름: {max_allowed:.6f}")
    
    # 접공간 매핑 테스트
    origin = torch.zeros_like(x_poincare)
    v = hyper_ops.batch_poincare_log_map(origin, x_poincare, curvature)
    
    # 다시 하이퍼볼릭 공간으로 변환
    x_back = hyper_ops.batch_poincare_exp_map(origin, v, curvature)
    
    # 원래 점과 비교
    error = torch.norm(x_poincare - x_back, dim=1).mean().item()
    print(f"로그-지수 사상 평균 오차: {error:.6f}")
    
    # 유클리드 변환 테스트
    x_euclidean = hyper_ops.poincare_to_euclidean(x_poincare, curvature)
    x_poincare_back = hyper_ops.euclidean_to_poincare(x_euclidean, curvature)
    error = torch.norm(x_poincare - x_poincare_back, dim=1).mean().item()
    print(f"유클리드 변환 후 복원 평균 오차: {error:.6f}")
    
    return error < 1e-5

def test_hyper_butterfly_layer():
    """
    하이퍼볼릭 버터플라이 레이어 테스트
    """
    print("\n===== 하이퍼볼릭 버터플라이 레이어 테스트 =====")
    
    # 모델 파라미터
    input_dim = 128
    num_layers = 3
    curvature = 0.1
    batch_size = 32
    
    # 하이퍼볼릭 버터플라이 레이어 초기화
    model = HyperButterflyLayer(dim=input_dim, num_layers=num_layers, curvature=curvature, device=device)
    model = model.to(device)
    print(f"모델을 {device}로 이동했습니다.")
    
    # 입력 데이터 생성 (포인카레 볼 내부의 점들)
    x = torch.randn(batch_size, input_dim, device=device) * 0.1
    hyper_ops = HyperbolicOperations()
    x_poincare = hyper_ops.euclidean_to_poincare(x, curvature)
    print(f"입력 텐서 크기: {x_poincare.shape}, 장치: {x_poincare.device}")
    
    # 성능 측정
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    y = model(x_poincare)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time = time.time() - start_time
    
    print(f"출력 텐서 크기: {y.shape}, 장치: {y.device}")
    print(f"추론 시간: {inference_time*1000:.2f} ms")
    
    # 결과가 하이퍼볼릭 공간 내에 있는지 체크
    max_norm = torch.norm(y, dim=1).max().item()
    max_allowed = 1.0 / np.sqrt(curvature)
    print(f"출력 최대 노름: {max_norm:.6f}, 허용 최대 노름: {max_allowed:.6f}")
    
    # 그래디언트 체크
    if device.type == 'cuda':
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    return max_norm < max_allowed

def mnist_speed_test():
    """
    MNIST 데이터셋에서 버터플라이 구현과 일반 구현의 속도 비교
    """
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
    except ImportError:
        print("MNIST 테스트를 위해 torchvision이 필요합니다.")
        return
    
    print("\n===== MNIST 속도 테스트 =====")
    
    # 데이터 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # 일반 MLP (O(n²) 복잡도)
        class StandardMLP(torch.nn.Module):
            def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
                super(StandardMLP, self).__init__()
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = x.view(-1, 784)
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.relu(self.fc2(x))
                x = self.fc3(x)
                return torch.nn.functional.log_softmax(x, dim=1)
        
        # 모델 초기화
        standard_mlp = StandardMLP().to(device)
        hyper_butterfly = HyperButterflyMLP(input_dim=784, hidden_dim=128, output_dim=10, device=device).to(device)
        
        # 파라미터 수 비교
        standard_params = sum(p.numel() for p in standard_mlp.parameters())
        butterfly_params = sum(p.numel() for p in hyper_butterfly.parameters())
        
        print(f"일반 MLP 파라미터 수: {standard_params:,}")
        print(f"하이퍼볼릭 버터플라이 MLP 파라미터 수: {butterfly_params:,}")
        
        # 추론 시간 측정
        standard_times = []
        butterfly_times = []
        
        # GPU 캐시 초기화
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 워밍업
        print("워밍업 실행 중...")
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                _ = standard_mlp(data)
                _ = hyper_butterfly(data)
                break
        
        # 추론 시간 측정
        print("추론 시간 측정 중...")
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                
                # 일반 MLP 시간 측정
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                _ = standard_mlp(data)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                standard_times.append(time.time() - start_time)
                
                # 하이퍼볼릭 버터플라이 MLP 시간 측정
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                _ = hyper_butterfly(data)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                butterfly_times.append(time.time() - start_time)
        
        # 결과 출력
        avg_standard_time = np.mean(standard_times) * 1000  # ms로 변환
        avg_butterfly_time = np.mean(butterfly_times) * 1000
        
        print(f"일반 MLP 추론 시간: {avg_standard_time:.2f}ms")
        print(f"하이퍼볼릭 버터플라이 MLP 추론 시간: {avg_butterfly_time:.2f}ms")
        print(f"속도 비율: {avg_standard_time/avg_butterfly_time:.2f}x")
        
        # 히든 레이어 차원이 128이라면 이론적 속도 향상은 logn 근처
        theoretical_speedup = (128**2) / (128 * np.log2(128))
        print(f"히든 레이어 (n={128}) 이론적 속도 향상: {theoretical_speedup:.2f}x")
        
        return avg_standard_time, avg_butterfly_time
        
    except Exception as e:
        print(f"MNIST 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("\n======= 하이퍼볼릭 버터플라이 네트워크 테스트 =======")
    
    # 버터플라이 변환 벤치마크
    dimensions, butterfly_times, linear_times = benchmark_butterfly_transform()
    
    # 하이퍼볼릭 연산 테스트
    test_hyperbolic_operations()
    
    # 하이퍼볼릭 버터플라이 레이어 테스트
    test_hyper_butterfly_layer()
    
    # MNIST 속도 테스트
    mnist_speed_test()
    
    # 속도 향상 비교 테이블
    print("\n===== 속도 향상 요약 =====")
    print(f"{'차원':<10} {'버터플라이 시간(ms)':<20} {'선형 시간(ms)':<20} {'실제 향상':<15} {'이론적 향상':<15}")
    print("-" * 80)
    for i, dim in enumerate(dimensions):
        theoretical = (dim*dim)/(dim*np.log2(dim))
        actual = linear_times[i]/butterfly_times[i]
        print(f"{dim:<10} {butterfly_times[i]*1000:<20.2f} {linear_times[i]*1000:<20.2f} {actual:<15.2f}x {theoretical:<15.2f}x")
    
    print("\n테스트 완료!") 