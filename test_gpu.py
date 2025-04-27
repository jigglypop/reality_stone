import torch
import time
from riemannian_manifold import HyperButterflyLayer

print("===== 하이퍼볼릭 버터플라이 GPU 테스트 =====")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CPU에서 실행 중")

# 모델 파라미터
batch_size = 32
input_dim = 128
hidden_dim = 128
num_layers = 3
curvature = 0.1

# 하이퍼볼릭 버터플라이 레이어 초기화
model = HyperButterflyLayer(dim=hidden_dim, num_layers=num_layers, curvature=curvature)
model = model.to(device)
print(f"모델을 {device}로 이동했습니다.")

# 입력 데이터 생성
x = torch.randn(batch_size, input_dim, device=device)
print(f"입력 텐서 크기: {x.shape}, 장치: {x.device}")

# 추론 시간 측정
start_time = time.time()
y = model(x)
end_time = time.time()

print(f"출력 텐서 크기: {y.shape}, 장치: {y.device}")
print(f"추론 시간: {(end_time - start_time) * 1000:.2f} ms")

# 역전파 테스트
start_time = time.time()
loss = y.sum()
loss.backward()
end_time = time.time()

print(f"역전파 시간: {(end_time - start_time) * 1000:.2f} ms")
print("테스트 완료!")

# 메모리 사용량 확인
if device.type == "cuda":
    print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved() / 1024**2:.2f} MB") 