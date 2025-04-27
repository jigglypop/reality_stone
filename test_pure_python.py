import torch
import time
import numpy as np

# 디버깅 모드
DEBUG = False  # 출력을 줄이기 위해 False로 설정

def debug_print(msg, tensor=None):
    if DEBUG:
        print(msg)
        if tensor is not None:
            print(f"  Shape: {tensor.shape}")
            print(f"  Min/Max: {tensor.min().item():.6f}/{tensor.max().item():.6f}")
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

# 단순화된 하이퍼볼릭 함수들
def poincare_exp_map(x, v, c=1.0):
    # 수치적 안정성을 위한 엡실론
    eps = 1e-8
    
    # 기준점의 노름 제곱
    x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
    x_norm_squared = torch.clamp(x_norm_squared, max=1.0/c - eps)
    
    # 공형적 인자
    lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
    
    # 벡터 노름
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=eps)
    
    # c를 텐서로 변환
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    sqrt_c = torch.sqrt(c_tensor)
    
    # 스케일 계수 계산
    argument = sqrt_c * lambda_x * v_norm / 2.0
    argument = torch.clamp(argument, max=15.0)  # tanh의 안정적인 영역으로 제한
    scale = torch.tanh(argument) / (sqrt_c * v_norm)
    
    # 스케일된 벡터 계산
    scaled_v = scale * v
    
    # 분자 계산
    numerator = (1.0 - c * x_norm_squared) * scaled_v
    
    # 분모 계산
    x_scaled_v_inner = torch.sum(x * scaled_v, dim=-1, keepdim=True)
    scaled_v_norm_squared = torch.sum(scaled_v * scaled_v, dim=-1, keepdim=True)
    denominator = 1.0 - 2.0 * c * x_scaled_v_inner + c * c * x_norm_squared * scaled_v_norm_squared + eps
    
    # 모빌리우스 덧셈 계산
    result = x + numerator / denominator
    
    # 결과의 노름이 1/sqrt(c)보다 작게 보장
    result_norm = torch.norm(result, p=2, dim=-1, keepdim=True)
    max_norm = (1.0 - eps) / sqrt_c
    scale_factor = torch.ones_like(result_norm)
    mask = result_norm > max_norm
    scale_factor[mask] = max_norm[mask] / result_norm[mask]
    result = result * scale_factor
    
    # NaN 검사
    if torch.isnan(result).any():
        if DEBUG:
            print("NaN in poincare_exp_map result")
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
    
    return result

def butterfly_transform(x, params, num_layers):
    """단순화된 버터플라이 변환 함수
    x: 입력 텐서 [batch_size, dim]
    params: 파라미터 텐서
    num_layers: 레이어 수
    """
    batch_size, dim = x.shape
    output = x.clone()
    
    # 각 레이어의 버터플라이 변환 적용
    for l in range(num_layers):
        block_size = 1 << l  # 2^l
        num_blocks = dim // block_size
        
        # 현재 레이어의 파라미터 인덱스 계산
        param_idx = 0
        for b in range(num_blocks // 2):  # 각 블록 쌍마다
            for i in range(0, block_size, 2):  # 각 블록 내의 짝수 인덱스
                idx1 = b * block_size + i
                idx2 = idx1 + 1
                
                if idx2 >= dim:  # 범위 체크
                    continue
                
                # 현재 파라미터 가져오기
                if param_idx + 1 < len(params):
                    a = params[param_idx].item()
                    b_val = params[param_idx + 1].item()
                    param_idx += 2
                    
                    # 배치 전체에 대해 한번에 연산
                    temp1 = a * output[:, idx1] + b_val * output[:, idx2]
                    temp2 = -b_val * output[:, idx1] + a * output[:, idx2]
                    
                    output[:, idx1] = temp1
                    output[:, idx2] = temp2
    
    return output

# 모델 파라미터
batch_size = 16
input_dim = 64
hidden_dim = 64
num_layers = 2
curvature = 0.05

class SimpleButterfly(torch.nn.Module):
    """매우 단순화된 버터플라이 레이어 구현"""
    def __init__(self, dim, num_layers, curvature=0.1):
        super(SimpleButterfly, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.curvature = curvature
        
        # 단순 선형 변환으로 근사
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        
        # 모든 레이어에 대해 Xavier 초기화
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        # 입력 스케일링 (하이퍼볼릭 공간 경계 내로 유지)
        input_norm = torch.norm(x, dim=1, keepdim=True)
        max_norm = 1.0 / torch.sqrt(torch.tensor(self.curvature, device=x.device)) * 0.9
        scaling = torch.ones_like(input_norm)
        mask = input_norm > max_norm
        scaling[mask] = max_norm / input_norm[mask]
        x_scaled = x * scaling
        
        # 단순 선형 변환 적용
        h = torch.tanh(self.linear1(x_scaled))
        out = self.linear2(h)
        
        # 출력 스케일링
        output_norm = torch.norm(out, dim=1, keepdim=True)
        scaling = torch.ones_like(output_norm)
        mask = output_norm > max_norm
        scaling[mask] = max_norm / output_norm[mask]
        out_scaled = out * scaling
        
        return out_scaled

print("===== 단순화된 버터플라이 레이어 테스트 =====")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CPU에서 실행 중")

# 난수 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 모델 초기화
model = SimpleButterfly(dim=input_dim, num_layers=num_layers, curvature=curvature)
model = model.to(device)
print(f"모델을 {device}로 이동했습니다.")

# 작은 크기의 랜덤 입력 데이터 생성
x = torch.randn(batch_size, input_dim, device=device) * 0.1
print(f"입력 텐서 크기: {x.shape}, 장치: {x.device}")

# CPU 테스트
print("\n[1] CPU에서 테스트")
x_cpu = x.cpu()
model_cpu = model.cpu()
start_time = time.time()
y_cpu = model_cpu(x_cpu)
cpu_time = time.time() - start_time
print(f"CPU 추론 시간: {cpu_time * 1000:.2f} ms")
print(f"출력 형태: {y_cpu.shape}")
print(f"출력 범위: [{y_cpu.min().item():.6f}, {y_cpu.max().item():.6f}]")
print(f"NaN 값 개수: {torch.isnan(y_cpu).sum().item()}")

if torch.cuda.is_available():
    # GPU 테스트
    print("\n[2] GPU에서 테스트")
    model_gpu = model.cuda()
    start_time = time.time()
    y_gpu = model_gpu(x.cuda())
    gpu_time = time.time() - start_time
    print(f"GPU 추론 시간: {gpu_time * 1000:.2f} ms")
    print(f"GPU vs CPU 속도 비율: {cpu_time/gpu_time:.2f}x")
    
    # 결과 비교
    print("\n[3] 결과 정확성 확인")
    y_gpu_cpu = y_gpu.cpu()
    max_diff = torch.max(torch.abs(y_gpu_cpu - y_cpu))
    print(f"출력 형태: {y_gpu.shape}")
    print(f"출력 범위: [{y_gpu_cpu.min().item():.6f}, {y_gpu_cpu.max().item():.6f}]")
    print(f"NaN 값 개수: {torch.isnan(y_gpu_cpu).sum().item()}")
    print(f"최대 차이: {max_diff:.6f}")
    
    # 메모리 사용량
    print(f"\nGPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

print("\n테스트 완료!") 