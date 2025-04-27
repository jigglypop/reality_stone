import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

# GPU 관련 정보 확인 및 설정
print("\n===== GPU/CUDA 정보 확인 =====")
print(f"- PyTorch 버전: {torch.__version__}")
print(f"- CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"- CUDA 버전: {torch.version.cuda}")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"- GPU 개수: {device_count}")
    for i in range(device_count):
        print(f"- GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - 현재 메모리 사용량: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"  - 최대 메모리 용량: {torch.cuda.get_device_properties(i).total_memory/1024**2:.2f} MB")
else:
    print("- CUDA 사용 불가: GPU를 찾을 수 없거나 CUDA가 올바르게 설치되지 않았습니다.")
    print("- 설치된 PyTorch가 CUDA를 지원하는지 확인해주세요.")
print("==============================\n")

# GPU 설정 최적화
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 사용할 GPU 지정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# GPU 성능 최적화 설정
if torch.cuda.is_available():
    # cuDNN 최적화 활성화
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 입력 크기가 일정하면 성능 향상
    torch.backends.cudnn.deterministic = False  # 결정론적 알고리즘 비활성화 (성능 향상)
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32 사용 허용 (Ampere 이상)
    torch.backends.cudnn.allow_tf32 = True  # cuDNN에서 TF32 사용 허용
    
    # GPU 메모리 관리
    torch.cuda.empty_cache()  # 시작 전 캐시 정리
    print(f"GPU 메모리 초기 상태: {torch.cuda.memory_allocated()/1024**2:.2f} MB 사용 중")

# GPU 최적화된 하이퍼볼릭 공간 연산 함수
class HyperbolicOperations:
    @staticmethod
    def batch_poincare_exp_map(x, v, c=1.0):
        """
        배치 처리된 포인카레 볼 지수 사상(exponential map)
        x: 배치 기준점 [batch_size, dim]
        v: 배치 접공간 벡터 [batch_size, dim]
        c: 곡률
        """
        eps = 1e-8
        
        # 기준점의 노름 제곱 [batch_size, 1]
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        
        # Poincare 공간의 공형적 인자 [batch_size, 1]
        lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
        
        # 벡터의 노름 계산 [batch_size, 1]
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=eps)
        
        # c를 텐서로 변환
        c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c_tensor)
        
        # 스케일 계수 계산 [batch_size, 1]
        scale = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm)
        
        # 스케일된 벡터 [batch_size, dim]
        scaled_v = scale * v
        
        # 분자 계산 [batch_size, dim]
        numerator = (1.0 - c * x_norm_squared) * scaled_v
        
        # 분모 계산 [batch_size, 1]
        x_scaled_v_inner = torch.sum(x * scaled_v, dim=-1, keepdim=True)
        scaled_v_norm_squared = torch.sum(scaled_v * scaled_v, dim=-1, keepdim=True)
        denominator = 1.0 - 2.0 * c * x_scaled_v_inner + c * c * x_norm_squared * scaled_v_norm_squared
        
        # 결과 계산 (모빌리우스 덧셈) [batch_size, dim]
        result = x + numerator / (denominator + eps)
        
        # 수치적 안정성 검사
        mask = torch.isfinite(result).all(dim=-1, keepdim=True)
        result = torch.where(mask, result, torch.zeros_like(result))
        
        return result

    @staticmethod
    def batch_poincare_log_map(x, y, c=1.0):
        """
        배치 처리된 포인카레 볼 로그 사상(logarithmic map)
        x: 배치 기준점 [batch_size, dim]
        y: 배치 목표점 [batch_size, dim]
        c: 곡률
        """
        eps = 1e-8
        
        # 기준점의 노름 제곱 [batch_size, 1]
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        
        # Poincare 공간의 공형적 인자 [batch_size, 1]
        lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
        
        # 모빌리우스 뺄셈을 위한 계산 [batch_size, 1]
        y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
        xy_inner_prod = torch.sum(x * y, dim=-1, keepdim=True)
        
        # 분자 계산 [batch_size, dim]
        numerator = (1.0 - 2.0 * c * xy_inner_prod + c * y_norm_squared) * x
        numerator = numerator - (1.0 - c * x_norm_squared) * y
        
        # 분모 계산 [batch_size, 1]
        denominator = 1.0 - 2.0 * c * xy_inner_prod + c * c * x_norm_squared * y_norm_squared
        
        # 차이 벡터 계산 [batch_size, dim]
        diff = numerator / (denominator + eps)
        
        # 차이 벡터의 노름 [batch_size, 1]
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=eps)
        
        # c를 텐서로 변환
        c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c_tensor)
        
        # 최종 결과 계산 [batch_size, dim]
        return 2.0 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * diff_norm) * diff / diff_norm

    @staticmethod
    def batch_poincare_distance(x, y, c=1.0):
        """
        배치 처리된 포인카레 볼에서의 측지선 거리(geodesic distance)
        x: 배치 점 1 [batch_size, dim]
        y: 배치 점 2 [batch_size, dim]
        c: 곡률
        """
        eps = 1e-8
        
        # 각 점의 노름 제곱 [batch_size, 1]
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
        
        # 내적 [batch_size, 1]
        xy_inner_prod = torch.sum(x * y, dim=-1, keepdim=True)
        
        # 분자 계산 [batch_size, 1]
        numerator = 2.0 * torch.norm(x - y, p=2, dim=-1, keepdim=True) ** 2
        
        # 분모 계산 [batch_size, 1]
        denominator = (1.0 - c * x_norm_squared) * (1.0 - c * y_norm_squared) + eps
        
        # 아코시 함수를 위한 인자 계산 [batch_size, 1]
        argument = 1.0 + c * numerator / denominator
        argument = torch.clamp(argument, min=1.0 + eps)  # 수치적 안정성
        
        # c를 텐서로 변환
        c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c_tensor)
        
        # 결과 계산 [batch_size, 1]
        return torch.acosh(argument) / sqrt_c
    
    @staticmethod
    def euclidean_to_poincare(x, c=1.0, max_norm=0.9):
        """
        유클리드 벡터를 포인카레 볼로 안전하게 변환
        x: 배치 유클리드 벡터 [batch_size, dim]
        c: 곡률
        max_norm: 최대 노름 (1보다 작은 값으로 경계에 가까워지는 것 방지)
        """
        # 노름 계산 [batch_size, 1]
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        
        # 노름이 0인 경우 처리
        zeros_mask = (norm == 0)
        safe_norm = torch.where(zeros_mask, torch.ones_like(norm), norm)
        
        # 곡률을 고려한 스케일 계산 [batch_size, 1]
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        scale = max_norm * torch.tanh(sqrt_c * norm) / (sqrt_c * safe_norm)
        
        # 스케일된 벡터 반환, 노름이 0인 경우 0벡터 반환
        return torch.where(zeros_mask, torch.zeros_like(x), scale * x)
    
    @staticmethod
    def poincare_to_euclidean(x, c=1.0):
        """
        포인카레 볼 벡터를 유클리드 공간으로 변환 (원점 기준 로그 맵 근사)
        x: 배치 포인카레 벡터 [batch_size, dim]
        c: 곡률
        """
        # 노름 계산 및 클램핑 [batch_size, 1]
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8, max=1.0-1e-8)
        
        # c를 텐서로 변환
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        
        # 유클리드 공간으로 변환 [batch_size, dim]
        return x * torch.atanh(sqrt_c * norms) / (sqrt_c * norms)


# GPU 최적화된 Butterfly 변환 구현
class ButterflyTransform(nn.Module):
    def __init__(self, dim, device=None):
        super(ButterflyTransform, self).__init__()
        self.dim = dim
        self.device = device
        
        # 차원이 2의 거듭제곱인지 확인하고 조정
        self.log_dim = int(np.ceil(np.log2(dim)))
        self.adjusted_dim = 2 ** self.log_dim
        
        # 각 레이어 및 단계별 파라미터 초기화
        self.layers = nn.ModuleList()
        for layer in range(self.log_dim):
            # 각 레이어의 블록 수 계산
            block_size = 2 ** layer
            num_blocks = self.adjusted_dim // (2 * block_size)
            
            # 레이어의 회전 파라미터 (a, b) 초기화
            # a, b: 회전 행렬의 파라미터, a^2 + b^2 = 1을 만족해야 함
            theta = torch.randn(num_blocks, device=device) * 0.01
            a = nn.Parameter(torch.cos(theta))
            b = nn.Parameter(torch.sin(theta))
            
            # 현재 레이어에 파라미터 추가
            self.layers.append(nn.ParameterList([a, b]))
    
    def forward(self, x):
        """
        배치 입력에 Butterfly 변환 적용
        x: [batch_size, dim] 또는 [dim] 형태의 입력 텐서
        """
        # 입력이 1D인 경우 배치 차원 추가
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = x.size(0)
        
        # 입력 차원이 2의 거듭제곱보다 작은 경우 패딩
        if x.size(1) < self.adjusted_dim:
            padding = torch.zeros(batch_size, self.adjusted_dim - x.size(1), device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :self.adjusted_dim]
        
        # 각 레이어 적용
        for layer_idx, (a, b) in enumerate(self.layers):
            block_size = 2 ** layer_idx
            
            # 현재 레이어에 대한 변환 행렬 구성
            x_padded = self._apply_butterfly_layer(x_padded, a, b, block_size)
        
        # 원래 차원으로 복원
        if single_input:
            return x_padded[0, :x.size(1)]
        else:
            return x_padded[:, :x.size(1)]
    
    def _apply_butterfly_layer(self, x, a, b, block_size):
        """
        단일 Butterfly 레이어 적용 (GPU 최적화 버전)
        x: [batch_size, dim] 형태의 입력 텐서
        a, b: 회전 파라미터
        block_size: 현재 블록 크기
        """
        batch_size = x.size(0)
        dim = x.size(1)
        
        # 현재 레이어의 블록 재구성
        # 블록화된 연산을 위해 텐서 형태 변경
        num_blocks = dim // (2 * block_size)
        x_view = x.view(batch_size, num_blocks, 2, block_size)
        
        # 회전 행렬 파라미터 준비
        a_expanded = a.view(num_blocks, 1, 1).expand(num_blocks, 1, block_size)
        b_expanded = b.view(num_blocks, 1, 1).expand(num_blocks, 1, block_size)
        
        # 배치 연산으로 회전 적용
        # [batch_size, num_blocks, 2, block_size] 형태에서 연산
        x_rotated = torch.zeros_like(x_view)
        
        # 첫 번째 행에 대한 회전
        x_rotated[:, :, 0, :] = a_expanded * x_view[:, :, 0, :] + b_expanded * x_view[:, :, 1, :]
        
        # 두 번째 행에 대한 회전
        x_rotated[:, :, 1, :] = -b_expanded * x_view[:, :, 0, :] + a_expanded * x_view[:, :, 1, :]
        
        # 원래 형태로 복원
        return x_rotated.view(batch_size, dim)


# MNIST 데이터셋 로드 (GPU 최적화)
def load_mnist(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # DataLoader에 num_workers와 pin_memory 추가 (GPU 성능 향상)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader


# 일반 MLP 모델 (유클리드 공간)
class EuclideanMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(EuclideanMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 배치 정규화 추가
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # 배치 정규화 추가
        
        # GPU 최적화를 위한 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# GPU 최적화된 Hyper-Butterfly MLP 모델
class HyperButterflyMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, curvature=0.1, device=None):
        super(HyperButterflyMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature
        self.device = device
        
        # 차원이 2의 거듭제곱이 되도록 조정
        power = int(np.ceil(np.log2(hidden_dim)))
        self.butterfly_dim = 2 ** power
        
        # 입력층: 유클리드 -> 중간 표현
        self.fc_in = nn.Linear(input_dim, self.butterfly_dim)
        self.bn_in = nn.BatchNorm1d(self.butterfly_dim)  # 배치 정규화 추가
        
        # Butterfly 변환 모듈
        self.butterfly = ButterflyTransform(self.butterfly_dim, device=device)
        
        # 출력층: 중간 표현 -> 출력
        self.fc_out = nn.Linear(self.butterfly_dim, output_dim)
        
        # 하이퍼볼릭 연산을 위한 인스턴스
        self.hyper_ops = HyperbolicOperations()
        
        # GPU 최적화를 위한 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    @torch.cuda.amp.autocast(enabled=True)  # 혼합 정밀도 연산 사용
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. 유클리드 입력 처리 [batch_size, input_dim] -> [batch_size, butterfly_dim]
        x = x.view(batch_size, -1)
        h = F.relu(self.bn_in(self.fc_in(x)))
        
        # 2. 유클리드 -> 하이퍼볼릭 변환 (배치 처리)
        h_hyperbolic = self.hyper_ops.euclidean_to_poincare(h, self.curvature)
        
        # 3. 원점 기준 로그 맵 (하이퍼볼릭 -> 접공간)
        # 배치 처리를 위해 원점 텐서 생성
        origin = torch.zeros_like(h_hyperbolic)
        v = self.hyper_ops.batch_poincare_log_map(origin, h_hyperbolic, self.curvature)
        
        # 4. Butterfly 변환 (접공간에서 연산)
        v_transformed = self.butterfly(v)
        
        # 5. 접공간 -> 하이퍼볼릭 (원점 기준 지수 맵)
        h_transformed = self.hyper_ops.batch_poincare_exp_map(origin, v_transformed, self.curvature)
        
        # 6. 하이퍼볼릭 -> 유클리드 (출력층 처리를 위한 변환)
        h_euclidean = self.hyper_ops.poincare_to_euclidean(h_transformed, self.curvature)
        
        # 7. 출력층
        out = self.fc_out(h_euclidean)
        return F.log_softmax(out, dim=1)


# 혼합 정밀도 학습(Mixed Precision Training) 구현
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        # 스케일러가 제공되지 않으면 새로 생성
        self.scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler()
    
    def train_step(self, data, target, device):
        self.optimizer.zero_grad()
        
        # 혼합 정밀도 컨텍스트에서 순전파
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = F.nll_loss(output, target)
        
        # 스케일링된 그래디언트 계산 및 역전파
        self.scaler.scale(loss).backward()
        
        # 그래디언트 클리핑
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 스케일링된 그래디언트로 옵티마이저 스텝
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return output, loss


# 모델 학습 함수 (GPU 최적화)
def train(model, train_loader, optimizer, epoch, device, log_interval=100, mixed_precision=True):
    model.train()
    start_time = time.time()
    
    # 학습 통계
    total_loss = 0
    correct = 0
    total = 0
    
    # 혼합 정밀도 학습 설정
    if mixed_precision and torch.cuda.is_available():
        trainer = MixedPrecisionTrainer(model, optimizer)
        print("혼합 정밀도 학습 활성화")
    else:
        trainer = None
        print("일반 정밀도 학습 사용")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        try:
            if trainer:
                # 혼합 정밀도 학습 사용
                output, loss = trainer.train_step(data, target, device)
            else:
                # 일반 학습 방식
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                # GPU 메모리 사용량 출력
                if device.type == 'cuda':
                    print(f"GPU 메모리: {torch.cuda.memory_allocated()/1024**2:.1f}MB / "
                          f"{torch.cuda.memory_reserved()/1024**2:.1f}MB (할당/예약)")
                
                # 학습 과정에서 NaN 체크
                if torch.isnan(loss):
                    print("경고: NaN 손실 발생!")
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"NaN 그래디언트 발견: {name}")
        
        except Exception as e:
            print(f"학습 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    train_time = time.time() - start_time
    train_acc = 100. * correct / total if total > 0 else 0
    
    print(f'Train Epoch: {epoch} 완료 | 평균 손실: {total_loss / (batch_idx + 1):.4f} | 정확도: {train_acc:.2f}%')
    print(f'학습 시간: {train_time:.2f}초 (배치당 {train_time/(batch_idx+1)*1000:.2f}ms)')
    
    return train_acc, train_time


# 모델 테스트 함수 (GPU 최적화)
def test(model, test_loader, device, mixed_precision=True):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            try:
                # 혼합 정밀도 추론 사용 (선택적)
                if mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = model(data)
                else:
                    output = model(data)
                
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            except Exception as e:
                print(f"테스트 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    test_time = time.time() - start_time
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'테스트 시간: {test_time:.2f}초\n')
    
    return accuracy, test_time


def run_experiment(model_name, epochs=5, batch_size=128, mixed_precision=True):
    # 데이터 로드 (GPU 최적화)
    num_workers = min(4, os.cpu_count() or 1)  # CPU 코어 수에 따라 최적화
    train_loader, test_loader = load_mnist(batch_size=batch_size, num_workers=num_workers)
    
    # 장치 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU 사용 중: {torch.cuda.get_device_name(0)}")
        # CUDA 메모리 시작 상태
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA 시작 메모리: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    else:
        device = torch.device("cpu")
        print("주의: CPU에서 실행 중입니다.")
        mixed_precision = False  # CPU에서는 혼합 정밀도 비활성화
    
    # 모델 초기화 및 GPU로 이동
    if model_name == 'euclidean':
        model = EuclideanMLP().to(device)
        print("유클리드 MLP 모델 생성 완료")
    elif model_name == 'hyperbolic':
        model = HyperButterflyMLP(curvature=0.1, device=device).to(device)
        print("하이퍼볼릭 Butterfly MLP 모델 생성 완료")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 모델 디바이스 확인
    print(f"모델 디바이스: {next(model.parameters()).device}")
    
    # 모델 요약 정보 출력
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델: {model_name}")
    print(f"파라미터 수: {num_params:,}")
    
    # 옵티마이저 설정 (인자 최적화)
    if model_name == 'hyperbolic':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.0005, 
            weight_decay=1e-5,
            betas=(0.9, 0.99),  # 모멘텀 조정
            eps=1e-8,
            amsgrad=True  # AMS 알고리즘 사용
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=0.001,
            weight_decay=1e-4,
            betas=(0.9, 0.99),
            amsgrad=True
        )
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5, 
        patience=1, 
        verbose=True,
        min_lr=1e-6
    )
    
    # 학습 및 평가
    train_accs = []
    train_times = []
    test_accs = []
    test_times = []
    
    for epoch in range(1, epochs + 1):
        # 학습 전 GPU 메모리 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"\n===== 에폭 {epoch}/{epochs} =====")
        print(f"현재 학습률: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 학습
        train_acc, train_time = train(model, train_loader, optimizer, epoch, device, 
                                     mixed_precision=mixed_precision)
        
        # 평가
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        test_acc, test_time = test(model, test_loader, device, mixed_precision=mixed_precision)
        
        # 학습률 조정
        scheduler.step(test_acc)
        
        # 결과 저장
        train_accs.append(train_acc)
        train_times.append(train_time)
        test_accs.append(test_acc)
        test_times.append(test_time)
        
        # GPU 메모리 사용량 확인
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"에폭 {epoch} 최대 GPU 메모리 사용량: {peak_memory:.2f} MB")
            torch.cuda.reset_peak_memory_stats()
    
    # 결과 출력
    print(f"\n--- {model_name} 최종 결과 ---")
    print(f"평균 학습 시간 (에폭당): {np.mean(train_times):.2f} 초")
    print(f"평균 추론 시간 (테스트셋): {np.mean(test_times):.2f} 초")
    print(f"최종 정확도: {test_accs[-1]:.2f}%")
    
    return {
        'model': model_name,
        'train_accs': train_accs,
        'train_times': train_times,
        'test_accs': test_accs,
        'test_times': test_times,
        'num_params': num_params,
        'device': str(device)
    }


# 메인 함수
if __name__ == "__main__":
    print("\n======= CUDA/CUDNN 상세 설정 =======")
    print(f"cudnn 사용 가능: {torch.backends.cudnn.is_available()}")
    print(f"cudnn 활성화: {torch.backends.cudnn.enabled}")
    
    # 혼합 정밀도 학습 지원 확인
    amp_supported = (
        torch.cuda.is_available() and 
        hasattr(torch.cuda, 'amp') and 
        hasattr(torch.cuda.amp, 'autocast')
    )
    print(f"혼합 정밀도 학습 지원: {amp_supported}")
    
    # 텐서코어 지원 확인
    tensor_cores_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    print(f"텐서코어 지원: {tensor_cores_supported}")
    if tensor_cores_supported:
        print("텐서코어 최적화 활성화 (TF32/FP16)")
    
    # 실험 설정
    epochs = 5
    batch_size = 128  # 메모리에 따라 조정
    mixed_precision = amp_supported  # 지원되는 경우 혼합 정밀도 사용
    
    print(f"에폭 수: {epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"혼합 정밀도 학습: {mixed_precision}")
    print("===============================\n")
    
    # 모델 비교 실험
    print("\n[1/2] 유클리드 MLP 실험 시작...")
    euclidean_results = run_experiment('euclidean', epochs=epochs, batch_size=batch_size, 
                                      mixed_precision=mixed_precision)
    
    print("\n[2/2] 하이퍼볼릭 Butterfly MLP 실험 시작...")
    hyperbolic_results = run_experiment('hyperbolic', epochs=epochs, batch_size=batch_size, 
                                       mixed_precision=mixed_precision)
    
    # 결과 시각화
    epoch_range = range(1, epochs + 1)
    
    # 정확도 비교 그래프
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epoch_range, euclidean_results['test_accs'], 'bo-', label='유클리드 MLP')
    plt.plot(epoch_range, hyperbolic_results['test_accs'], 'ro-', label='하이퍼볼릭 Butterfly MLP')
    plt.title('테스트 정확도 비교')
    plt.xlabel('에폭')
    plt.ylabel('정확도 (%)')
    plt.legend()
    plt.grid(True)
    
    # 학습 시간 비교 (막대 그래프)
    plt.subplot(2, 2, 2)
    models = ['유클리드 MLP', '하이퍼볼릭\nButterfly MLP']
    train_times = [
        np.mean(euclidean_results['train_times']), 
        np.mean(hyperbolic_results['train_times'])
    ]
    plt.bar(models, train_times, color=['blue', 'red'])
    plt.title('평균 학습 시간 (에폭당)')
    plt.ylabel('시간 (초)')
    plt.grid(True)
    
    # 추론 시간 비교 (막대 그래프)
    plt.subplot(2, 2, 3)
    test_times = [
        np.mean(euclidean_results['test_times']), 
        np.mean(hyperbolic_results['test_times'])
    ]
    plt.bar(models, test_times, color=['blue', 'red'])
    plt.title('평균 추론 시간 (테스트셋)')
    plt.ylabel('시간 (초)')
    plt.grid(True)
    
    # 파라미터 수 비교 (막대 그래프)
    plt.subplot(2, 2, 4)
    params = [
        euclidean_results['num_params'],
        hyperbolic_results['num_params']
    ]
    plt.bar(models, params, color=['blue', 'red'])
    plt.title('파라미터 수')
    plt.ylabel('개수')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison_gpu.png')
    
    # 학습 곡선 그래프
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, euclidean_results['train_accs'], 'b--', label='유클리드 (학습)')
    plt.plot(epoch_range, euclidean_results['test_accs'], 'b-', label='유클리드 (테스트)')
    plt.plot(epoch_range, hyperbolic_results['train_accs'], 'r--', label='하이퍼볼릭 (학습)')
    plt.plot(epoch_range, hyperbolic_results['test_accs'], 'r-', label='하이퍼볼릭 (테스트)')
    plt.title('학습/테스트 정확도')
    plt.xlabel('에폭')
    plt.ylabel('정확도 (%)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('learning_curves_gpu.png')
    
    # 표 형태로 결과 요약
    print("\n=== GPU 성능 비교 ===")
    print(f"{'모델':<25} {'파라미터':<12} {'학습 시간':<15} {'추론 시간':<15} {'정확도':<10} {'디바이스':<10}")
    print("-" * 90)
    print(f"{'유클리드 MLP':<25} {euclidean_results['num_params']:,}<{'':<10} "
          f"{np.mean(euclidean_results['train_times']):.2f}s{'':<10} "
          f"{np.mean(euclidean_results['test_times']):.2f}s{'':<10} "
          f"{euclidean_results['test_accs'][-1]:.2f}%{'':<5} {euclidean_results['device']}")
    print(f"{'하이퍼볼릭 Butterfly MLP':<25} {hyperbolic_results['num_params']:,}<{'':<10} "
          f"{np.mean(hyperbolic_results['train_times']):.2f}s{'':<10} "
          f"{np.mean(hyperbolic_results['test_times']):.2f}s{'':<10} "
          f"{hyperbolic_results['test_accs'][-1]:.2f}%{'':<5} {hyperbolic_results['device']}")
    
    print("\n실험 완료. 결과가 model_comparison_gpu.png 및 learning_curves_gpu.png 파일로 저장되었습니다.") 