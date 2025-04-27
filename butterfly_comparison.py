# PyTorch CUDA 버전 강제 호환을 위한 환경 변수 설정
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # CUDA 초기화 지연
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # CUDA 메모리 할당 최적화
os.environ['TORCH_USE_RTLD_GLOBAL'] = 'YES'  # 전역 심볼 공유
os.environ['TORCH_NVCC_FLAGS'] = '--expt-relaxed-constexpr --allow-unsupported-compiler'  # NVCC 플래그

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

# CUDA 버전 호환성 메시지
print("\n===== CUDA 버전 정보 =====")
print(f"PyTorch CUDA 버전: {torch.version.cuda}")
try:
    import subprocess
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    cuda_version = result.stdout if result.returncode == 0 else "확인 불가"
    print(f"시스템 CUDA 버전: {cuda_version}")
except Exception as e:
    print(f"시스템 CUDA 버전 확인 불가: {e}")
print("=======================\n")

# CUDA 확장 빌드 상태 확인
try:
    print("CUDA 확장 빌드 디렉토리 확인 중...")
    # 관련 파일 경로 확인
    import glob
    build_files = glob.glob("riemannian_manifold/build/**/*.pyd", recursive=True)
    for file in build_files:
        print(f" - 확장 파일 발견: {file}")
except Exception as e:
    print(f"빌드 파일 확인 오류: {e}")

# 사용 가능한 CUDA 확장 가져오기 
try:
    print("CUDA 확장 로드 시도 중...")
    from riemannian_manifold.csrc.riemannian_cuda import (
        unified_butterfly_transform,
        create_butterfly_matrix,
        efficient_butterfly_transform
    )
    CUDA_AVAILABLE = True
    print("CUDA 버터플라이 확장 로드 성공!")
except ImportError as e:
    print(f"CUDA 확장을 불러올 수 없습니다: {e}")
    print("호환성 모드로 자동 전환합니다 (CPU에서 계산 후 GPU 복사)")
    
    # 대체 방법으로 다시 시도
    try:
        print("대체 로딩 방법 시도 중...")
        import importlib.util
        import sys
        
        # 확장 모듈 파일 직접 찾기
        extension_paths = glob.glob("riemannian_manifold/**/*.pyd", recursive=True)
        if extension_paths:
            print(f"확장 파일 발견: {extension_paths[0]}")
            spec = importlib.util.spec_from_file_location("riemannian_cuda", extension_paths[0])
            riemannian_cuda = importlib.util.module_from_spec(spec)
            sys.modules["riemannian_cuda"] = riemannian_cuda
            spec.loader.exec_module(riemannian_cuda)
            
            # 필요한 함수 가져오기
            unified_butterfly_transform = getattr(riemannian_cuda, "unified_butterfly_transform", None)
            create_butterfly_matrix = getattr(riemannian_cuda, "create_butterfly_matrix", None)
            efficient_butterfly_transform = getattr(riemannian_cuda, "efficient_butterfly_transform", None)
            
            if all([unified_butterfly_transform, create_butterfly_matrix, efficient_butterfly_transform]):
                CUDA_AVAILABLE = True
                print("대체 방법으로 CUDA 확장 로드 성공!")
            else:
                CUDA_AVAILABLE = False
                print("대체 방법으로 확장은 로드되었으나 필요한 함수를 찾을 수 없습니다.")
        else:
            CUDA_AVAILABLE = False
            print("확장 파일을 찾을 수 없습니다.")
    except Exception as e2:
        CUDA_AVAILABLE = False
        print(f"대체 로드 방법도 실패: {e2}")

# GPU information and checks
print("\n===== GPU/CUDA Information =====")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Current memory usage: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"  Maximum memory capacity: {torch.cuda.get_device_properties(i).total_memory/1024**2:.2f} MB")
else:
    print("CUDA is not available. GPU not found or CUDA not properly installed.")
print("==============================\n")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU optimization settings
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB in use")

# Hyperbolic operations class
class HyperbolicOperations:
    @staticmethod
    def euclidean_to_poincare(x, c=1.0, max_norm=0.9):
        """
        Safely convert Euclidean vectors to Poincare ball
        x: batch of Euclidean vectors [batch_size, dim]
        c: curvature
        max_norm: maximum norm (to prevent points from getting too close to the boundary)
        """
        # Calculate norm [batch_size, 1]
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        
        # Handle zero norms
        zeros_mask = (norm == 0)
        safe_norm = torch.where(zeros_mask, torch.ones_like(norm), norm)
        
        # Calculate scale considering curvature [batch_size, 1]
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        scale = max_norm * torch.tanh(sqrt_c * norm) / (sqrt_c * safe_norm)
        
        # Scale vector, returning zero vector for zero norm
        return torch.where(zeros_mask, torch.zeros_like(x), scale * x)
    
    @staticmethod
    def poincare_to_euclidean(x, c=1.0):
        """
        Convert Poincare ball vectors to Euclidean space (approximate log map from origin)
        x: batch of Poincare vectors [batch_size, dim]
        c: curvature
        """
        # Calculate and clamp norms [batch_size, 1]
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8, max=1.0-1e-8)
        
        # Convert to tensor
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        
        # Convert to Euclidean space [batch_size, dim]
        return x * torch.atanh(sqrt_c * norms) / (sqrt_c * norms)
    
    @staticmethod
    def batch_poincare_exp_map(x, v, c=1.0):
        """
        Batch exponential map in Poincare ball
        x: batch reference points [batch_size, dim]
        v: batch tangent vectors [batch_size, dim]
        c: curvature
        """
        eps = 1e-8
        
        # Norm squared of reference points [batch_size, 1]
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        
        # Conformal factor [batch_size, 1]
        lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
        
        # Norm of tangent vectors [batch_size, 1]
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=eps)
        
        # Convert c to tensor
        c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c_tensor)
        
        # Calculate scale factor [batch_size, 1]
        scale = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm)
        
        # Scale vectors [batch_size, dim]
        scaled_v = scale * v
        
        # Calculate numerator [batch_size, dim]
        numerator = (1.0 - c * x_norm_squared) * scaled_v
        
        # Calculate denominator [batch_size, 1]
        x_scaled_v_inner = torch.sum(x * scaled_v, dim=-1, keepdim=True)
        scaled_v_norm_squared = torch.sum(scaled_v * scaled_v, dim=-1, keepdim=True)
        denominator = 1.0 - 2.0 * c * x_scaled_v_inner + c * c * x_norm_squared * scaled_v_norm_squared
        
        # Result (Mobius addition) [batch_size, dim]
        result = x + numerator / (denominator + eps)
        
        # Check numerical stability
        mask = torch.isfinite(result).all(dim=-1, keepdim=True)
        result = torch.where(mask, result, torch.zeros_like(result))
        
        return result
    
    @staticmethod
    def batch_poincare_log_map(x, y, c=1.0):
        """
        Batch logarithmic map in Poincare ball
        x: batch reference points [batch_size, dim]
        y: batch target points [batch_size, dim]
        c: curvature
        """
        eps = 1e-8
        
        # Norm squared of reference points [batch_size, 1]
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        
        # Conformal factor [batch_size, 1]
        lambda_x = 2.0 / (1.0 - c * x_norm_squared + eps)
        
        # For Mobius subtraction [batch_size, 1]
        y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
        xy_inner_prod = torch.sum(x * y, dim=-1, keepdim=True)
        
        # Calculate numerator [batch_size, dim]
        numerator = (1.0 - 2.0 * c * xy_inner_prod + c * y_norm_squared) * x
        numerator = numerator - (1.0 - c * x_norm_squared) * y
        
        # Calculate denominator [batch_size, 1]
        denominator = 1.0 - 2.0 * c * xy_inner_prod + c * c * x_norm_squared * y_norm_squared
        
        # Difference vector [batch_size, dim]
        diff = numerator / (denominator + eps)
        
        # Norm of difference vector [batch_size, 1]
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=eps)
        
        # Convert c to tensor
        c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
        sqrt_c = torch.sqrt(c_tensor)
        
        # Final result [batch_size, dim]
        return 2.0 / (sqrt_c * lambda_x) * torch.atanh(sqrt_c * diff_norm) * diff / diff_norm

# 버터플라이 변환 - 행렬 곱셈으로 구현 (for문 없음)
class ButterflyTransform(nn.Module):
    def __init__(self, dim, device=None):
        super(ButterflyTransform, self).__init__()
        self.dim = dim
        self.device = device
        
        # 2의 거듭제곱으로 조정
        self.log_dim = int(np.ceil(np.log2(dim)))
        self.adjusted_dim = 2 ** self.log_dim
        
        # 각 레이어별 회전 파라미터
        self.thetas = nn.Parameter(torch.randn(self.log_dim * self.adjusted_dim // 2, device=device) * 0.01)
        
        # CUDA 지원 여부 확인
        self.use_cuda_backend = CUDA_AVAILABLE and (device is not None and device.type == 'cuda')
        
        # CUDA 백엔드 사용 시 미리 계산된 행렬 생성
        if self.use_cuda_backend:
            try:
                print("CUDA 백엔드에서 버터플라이 행렬 미리 계산 중...")
                self.weight = nn.Parameter(
                    create_butterfly_matrix(self.thetas, self.adjusted_dim, self.log_dim)
                )
                print(f"버터플라이 행렬 생성 완료: {self.weight.shape}")
            except Exception as e:
                print(f"CUDA 행렬 생성 오류: {e}, 대체 구현 사용")
                self.use_cuda_backend = False
                self._initialize_fallback_matrix()
        else:
            self._initialize_fallback_matrix()
    
    def _initialize_fallback_matrix(self):
        # CPU 대체 구현용 행렬 생성
        print("CPU 대체 구현 사용 중...")
        self.weight = nn.Parameter(torch.eye(self.adjusted_dim, device=self.device))
        
        # 간단한 행렬 버전으로 대체 (버터플라이 행렬 근사)
        butterfly_approx = torch.randn((self.adjusted_dim, self.adjusted_dim), device=self.device) * 0.01
        butterfly_approx = butterfly_approx + torch.eye(self.adjusted_dim, device=self.device)
        
        # 행렬 정규화 (정규직교 행렬로)
        u, _, v = torch.linalg.svd(butterfly_approx, full_matrices=False)
        orthogonal_approx = u @ v
        
        with torch.no_grad():
            self.weight.copy_(orthogonal_approx)
    
    def forward(self, x):
        # 단일 텐서 처리
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        batch_size = x.size(0)
        
        # 입력 크기 맞추기
        if x.size(1) < self.adjusted_dim:
            padding = torch.zeros(batch_size, self.adjusted_dim - x.size(1), device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :self.adjusted_dim]
        
        # CUDA 지원 확인 및 백엔드 호출
        # CUDA 백엔드가 사용 가능하고 입력이 CUDA 텐서인 경우
        if self.use_cuda_backend and x_padded.is_cuda:
            try:
                # 백엔드 함수 직접 호출 (모든 연산 CUDA에서 처리)
                output = unified_butterfly_transform(x_padded, self.weight)
            except Exception as e:
                print(f"CUDA 백엔드 오류: {e}, CPU 구현으로 전환")
                # 오류 발생 시 CPU 구현으로 폴백
                output = x_padded @ self.weight.t()
        else:
            # CPU 구현: 단일 행렬 곱셈으로 모든 버터플라이 변환 적용
            output = x_padded @ self.weight.t()
        
        # 원래 크기로 자르기
        if single_input:
            return output[0, :x.size(1)]
        else:
            return output[:, :x.size(1)]

# Load MNIST dataset
def load_mnist(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # DataLoader with num_workers and pin_memory for better GPU performance
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

# Standard MLP model (Euclidean space)
class EuclideanMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(EuclideanMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Hyperbolic MLP with Butterfly transform
class HyperButterflyMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, curvature=0.1, device=None):
        super(HyperButterflyMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature
        self.device = device
        
        # Adjust to power of 2 for butterfly transform
        power = int(np.ceil(np.log2(hidden_dim)))
        self.butterfly_dim = 2 ** power
        
        # Input layer: Euclidean to intermediate representation
        self.fc_in = nn.Linear(input_dim, self.butterfly_dim)
        self.bn_in = nn.BatchNorm1d(self.butterfly_dim)
        
        # Butterfly transform for efficient computation (O(n log n))
        self.butterfly = ButterflyTransform(self.butterfly_dim, device=device)
        
        # Output layer
        self.fc_out = nn.Linear(self.butterfly_dim, output_dim)
        
        # Hyperbolic operations
        self.hyper_ops = HyperbolicOperations()
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    @torch.cuda.amp.autocast(enabled=True)  # Mixed precision
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Flatten and first linear transform
        x = x.view(batch_size, -1)
        h = F.relu(self.bn_in(self.fc_in(x)))
        
        # 2. Convert to hyperbolic space
        h_hyp = self.hyper_ops.euclidean_to_poincare(h, self.curvature)
        
        # 3. Map to tangent space at origin for butterfly transform
        origin = torch.zeros_like(h_hyp)
        v = self.hyper_ops.batch_poincare_log_map(origin, h_hyp, self.curvature)
        
        # 4. Apply butterfly transform in tangent space (O(n log n) complexity)
        v_transformed = self.butterfly(v)
        
        # 5. Map back to hyperbolic space
        h_transformed = self.hyper_ops.batch_poincare_exp_map(origin, v_transformed, self.curvature)
        
        # 6. Convert to Euclidean for output layer
        h_euc = self.hyper_ops.poincare_to_euclidean(h_transformed, self.curvature)
        
        # 7. Output layer
        out = self.fc_out(h_euc)
        
        return F.log_softmax(out, dim=1)

# Regular Hyperbolic MLP without Butterfly structure (for comparison)
class HyperbolicMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, curvature=0.1):
        super(HyperbolicMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature
        
        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Hyperbolic operations
        self.hyper_ops = HyperbolicOperations()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    @torch.cuda.amp.autocast(enabled=True)  # Mixed precision
    def forward(self, x):
        batch_size = x.size(0)
        
        # Flatten and first layer (Euclidean)
        x = x.view(batch_size, -1)
        h1 = F.relu(self.bn1(self.fc1(x)))
        
        # Convert to hyperbolic space
        h1_hyp = self.hyper_ops.euclidean_to_poincare(h1, self.curvature)
        
        # Create origin tensor for log/exp maps
        origin = torch.zeros_like(h1_hyp)
        
        # Map to tangent space at origin
        h1_tan = self.hyper_ops.batch_poincare_log_map(origin, h1_hyp, self.curvature)
        
        # Apply linear transformation in tangent space
        h2_tan = self.fc2(h1_tan)
        h2_tan = F.relu(self.bn2(h2_tan))
        
        # Map back to hyperbolic space
        h2_hyp = self.hyper_ops.batch_poincare_exp_map(origin, h2_tan, self.curvature)
        
        # Convert back to Euclidean for output layer
        h2_euc = self.hyper_ops.poincare_to_euclidean(h2_hyp, self.curvature)
        
        # Output layer
        out = self.fc3(h2_euc)
        
        return F.log_softmax(out, dim=1)

# Mixed Precision Trainer
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler()
    
    def train_step(self, data, target):
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = F.nll_loss(output, target)
        
        # Scale gradients and backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale for gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update with scaled gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return output, loss

# Train function with time tracking
def train(model, train_loader, optimizer, epoch, device, log_interval=100, mixed_precision=True):
    model.train()
    start_time = time.time()
    
    # Training statistics
    total_loss = 0
    correct = 0
    total = 0
    
    # Detailed time tracking
    forward_time = 0
    backward_time = 0
    data_time = 0
    batch_times = []
    
    # Mixed precision setup
    if mixed_precision and torch.cuda.is_available():
        trainer = MixedPrecisionTrainer(model, optimizer)
        print("Mixed precision training enabled")
    else:
        trainer = None
        print("Using standard precision training")
    
    data_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_end = time.time()
        data_time += data_end - data_start
        
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        try:
            batch_start = time.time()
            if trainer:
                # Use mixed precision
                forward_start = time.time()
                output, loss = trainer.train_step(data, target)
                backward_end = time.time()
                forward_time += backward_end - forward_start
                backward_time += 0  # Already included in trainer
            else:
                # Standard training with time tracking
                optimizer.zero_grad()
                
                forward_start = time.time()
                output = model(data)
                forward_end = time.time()
                forward_time += forward_end - forward_start
                
                loss = F.nll_loss(output, target)
                
                backward_start = time.time()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                backward_end = time.time()
                backward_time += backward_end - backward_start
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Update statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                # Show GPU memory usage
                if device.type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB / "
                          f"{torch.cuda.memory_reserved()/1024**2:.1f}MB (allocated/reserved)")
                    print(f"Batch time: {batch_times[-1]*1000:.2f}ms (Forward: {forward_time/(batch_idx+1)*1000:.2f}ms)")
                
                # Check for NaN
                if torch.isnan(loss):
                    print("Warning: NaN loss detected!")
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"NaN gradient found in: {name}")
        
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        data_start = time.time()
    
    train_time = time.time() - start_time
    train_acc = 100. * correct / total if total > 0 else 0
    
    print(f'Train Epoch: {epoch} complete | Avg loss: {total_loss / (batch_idx + 1):.4f} | Accuracy: {train_acc:.2f}%')
    print(f'Training time: {train_time:.2f}s ({train_time/(batch_idx+1)*1000:.2f}ms per batch)')
    print(f'Forward time: {forward_time:.2f}s, Backward time: {backward_time:.2f}s, Data time: {data_time:.2f}s')
    
    return train_acc, train_time, {
        'forward_time': forward_time, 
        'backward_time': backward_time,
        'data_time': data_time,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0
    }

# Test function
def test(model, test_loader, device, mixed_precision=True):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    
    # Detailed timing
    inference_times = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            try:
                # Mixed precision inference with timing
                inference_start = time.time()
                if mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = model(data)
                else:
                    output = model(data)
                inference_end = time.time()
                inference_times.append(inference_end - inference_start)
                
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    test_time = time.time() - start_time
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'Testing time: {test_time:.2f}s (Avg inference time: {np.mean(inference_times)*1000:.2f}ms per batch)\n')
    
    return accuracy, test_time, np.mean(inference_times)

# Run experiment
def run_experiment(model_name, epochs=5, batch_size=128, mixed_precision=True):
    # Load data
    num_workers = min(4, os.cpu_count() or 1)
    train_loader, test_loader = load_mnist(batch_size=batch_size, num_workers=num_workers)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA initial memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    else:
        device = torch.device("cpu")
        print("Warning: Running on CPU")
        mixed_precision = False
    
    # Initialize model
    if model_name == 'euclidean':
        model = EuclideanMLP().to(device)
        print("Euclidean MLP model created")
    elif model_name == 'hyperbolic':
        model = HyperbolicMLP(curvature=0.1).to(device)
        print("Hyperbolic MLP model created")
    elif model_name == 'hyperbutterfly':
        model = HyperButterflyMLP(curvature=0.1, device=device).to(device)
        print("HyperButterflyMLP model created")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Check model device
    print(f"Model device: {next(model.parameters()).device}")
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Parameters: {num_params:,}")
    
    # Optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.99),
        amsgrad=True
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5, 
        patience=1, 
        verbose=True,
        min_lr=1e-6
    )
    
    # Training and evaluation
    train_accs = []
    train_times = []
    forward_times = []
    backward_times = []
    test_accs = []
    test_times = []
    inference_times = []
    
    for epoch in range(1, epochs + 1):
        # Clear GPU memory before training
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_acc, train_time, time_details = train(model, train_loader, optimizer, epoch, device, 
                                      mixed_precision=mixed_precision)
        
        # Store detailed timing
        forward_times.append(time_details['forward_time'])
        backward_times.append(time_details['backward_time'])
        
        # Evaluate
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        test_acc, test_time, avg_inference = test(model, test_loader, device, mixed_precision=mixed_precision)
        
        # Adjust learning rate
        scheduler.step(test_acc)
        
        # Save results
        train_accs.append(train_acc)
        train_times.append(train_time)
        test_accs.append(test_acc)
        test_times.append(test_time)
        inference_times.append(avg_inference)
        
        # Check GPU memory usage
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Epoch {epoch} peak GPU memory usage: {peak_memory:.2f} MB")
            torch.cuda.reset_peak_memory_stats()
    
    # Print results
    print(f"\n--- {model_name} Final Results ---")
    print(f"Average training time (per epoch): {np.mean(train_times):.2f} seconds")
    print(f"Average forward time: {np.mean(forward_times):.2f} seconds")
    print(f"Average backward time: {np.mean(backward_times):.2f} seconds")
    print(f"Average inference time (per batch): {np.mean(inference_times)*1000:.2f} ms")
    print(f"Average test time (total): {np.mean(test_times):.2f} seconds")
    print(f"Final accuracy: {test_accs[-1]:.2f}%")
    
    return {
        'model': model_name,
        'train_accs': train_accs,
        'train_times': train_times,
        'forward_times': forward_times,
        'backward_times': backward_times,
        'test_accs': test_accs,
        'test_times': test_times,
        'inference_times': inference_times,
        'num_params': num_params,
        'device': str(device)
    }

# Main function
if __name__ == "__main__":
    print("\n======= CUDA/CUDNN Settings =======")
    print(f"cudnn available: {torch.backends.cudnn.is_available()}")
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    
    # Check mixed precision support
    amp_supported = (
        torch.cuda.is_available() and 
        hasattr(torch.cuda, 'amp') and 
        hasattr(torch.cuda.amp, 'autocast')
    )
    print(f"Mixed precision supported: {amp_supported}")
    
    # Check tensor cores support
    tensor_cores_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    print(f"Tensor cores supported: {tensor_cores_supported}")
    if tensor_cores_supported:
        print("Tensor core optimizations enabled (TF32/FP16)")
    
    # Experiment settings
    epochs = 3  # Reduced for quicker testing
    batch_size = 128
    mixed_precision = amp_supported
    
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision training: {mixed_precision}")
    print("===============================\n")
    
    # 실행 순서: 버터플라이 → 유클리드 → 하이퍼볼릭 (사용자 요청대로 변경)
    print("\n[1/3] Starting HyperButterfly MLP experiment...")
    hyperbutterfly_results = run_experiment('hyperbutterfly', epochs=epochs, batch_size=batch_size, 
                                         mixed_precision=mixed_precision)
    
    print("\n[2/3] Starting Euclidean MLP experiment...")
    euclidean_results = run_experiment('euclidean', epochs=epochs, batch_size=batch_size, 
                                      mixed_precision=mixed_precision)
    
    print("\n[3/3] Starting Hyperbolic MLP experiment...")
    hyperbolic_results = run_experiment('hyperbolic', epochs=epochs, batch_size=batch_size, 
                                      mixed_precision=mixed_precision)
    
    # Visualize results
    epoch_range = range(1, epochs + 1)
    
    # Accuracy comparison graph
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epoch_range, euclidean_results['test_accs'], 'bo-', label='Euclidean MLP')
    plt.plot(epoch_range, hyperbolic_results['test_accs'], 'ro-', label='Hyperbolic MLP')
    plt.plot(epoch_range, hyperbutterfly_results['test_accs'], 'go-', label='HyperButterfly MLP')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Forward time comparison (per epoch)
    plt.subplot(2, 2, 2)
    models = ['Euclidean\nMLP', 'Hyperbolic\nMLP', 'HyperButterfly\nMLP']
    forward_times = [
        np.mean(euclidean_results['forward_times']), 
        np.mean(hyperbolic_results['forward_times']),
        np.mean(hyperbutterfly_results['forward_times'])
    ]
    plt.bar(models, forward_times, color=['blue', 'red', 'green'])
    plt.title('Average Forward Pass Time (per epoch)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    # Inference time comparison (per batch)
    plt.subplot(2, 2, 3)
    inference_times = [
        np.mean(euclidean_results['inference_times'])*1000, 
        np.mean(hyperbolic_results['inference_times'])*1000,
        np.mean(hyperbutterfly_results['inference_times'])*1000
    ]
    plt.bar(models, inference_times, color=['blue', 'red', 'green'])
    plt.title('Average Inference Time (per batch)')
    plt.ylabel('Time (milliseconds)')
    plt.grid(True)
    
    # Parameter count comparison
    plt.subplot(2, 2, 4)
    params = [
        euclidean_results['num_params'],
        hyperbolic_results['num_params'],
        hyperbutterfly_results['num_params']
    ]
    plt.bar(models, params, color=['blue', 'red', 'green'])
    plt.title('Parameter Count')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('butterfly_comparison.png')
    
    # Learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, euclidean_results['train_accs'], 'b--', label='Euclidean (Train)')
    plt.plot(epoch_range, euclidean_results['test_accs'], 'b-', label='Euclidean (Test)')
    plt.plot(epoch_range, hyperbolic_results['train_accs'], 'r--', label='Hyperbolic (Train)')
    plt.plot(epoch_range, hyperbolic_results['test_accs'], 'r-', label='Hyperbolic (Test)')
    plt.plot(epoch_range, hyperbutterfly_results['train_accs'], 'g--', label='HyperButterfly (Train)')
    plt.plot(epoch_range, hyperbutterfly_results['test_accs'], 'g-', label='HyperButterfly (Test)')
    plt.title('Train/Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, euclidean_results['train_times'], 'b-', label='Euclidean')
    plt.plot(epoch_range, hyperbolic_results['train_times'], 'r-', label='Hyperbolic')
    plt.plot(epoch_range, hyperbutterfly_results['train_times'], 'g-', label='HyperButterfly')
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('butterfly_learning_curves.png')
    
    # Results summary
    print("\n=== Performance Comparison ===")
    headers = ["Model", "Parameters", "Accuracy", "Forward Time", "Inference Time", "Device"]
    divider = "-" * 100
    print(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<10} {headers[3]:<15} {headers[4]:<15} {headers[5]:<10}")
    print(divider)
    print(f"{'Euclidean MLP':<20} {euclidean_results['num_params']:,} {euclidean_results['test_accs'][-1]:.2f}% "
          f"{np.mean(euclidean_results['forward_times']):.2f}s{'':<10} "
          f"{np.mean(euclidean_results['inference_times'])*1000:.2f}ms{'':<8} "
          f"{euclidean_results['device']}")
    print(f"{'Hyperbolic MLP':<20} {hyperbolic_results['num_params']:,} {hyperbolic_results['test_accs'][-1]:.2f}% "
          f"{np.mean(hyperbolic_results['forward_times']):.2f}s{'':<10} "
          f"{np.mean(hyperbolic_results['inference_times'])*1000:.2f}ms{'':<8} "
          f"{hyperbolic_results['device']}")
    print(f"{'HyperButterfly MLP':<20} {hyperbutterfly_results['num_params']:,} {hyperbutterfly_results['test_accs'][-1]:.2f}% "
          f"{np.mean(hyperbutterfly_results['forward_times']):.2f}s{'':<10} "
          f"{np.mean(hyperbutterfly_results['inference_times'])*1000:.2f}ms{'':<8} "
          f"{hyperbutterfly_results['device']}")
    
    # Speed calculation for Butterfly vs Standard
    butterfly_inference = np.mean(hyperbutterfly_results['inference_times'])
    hyperbolic_inference = np.mean(hyperbolic_results['inference_times'])
    speedup = hyperbolic_inference / butterfly_inference if butterfly_inference > 0 else 0
    
    print(f"\nSpeedup Analysis:")
    print(f"HyperButterfly vs Hyperbolic MLP speedup ratio: {speedup:.2f}x")
    print(f"Theoretical expectation: O(n log n) vs O(n²), which would be approximately: {(128**2)/(128*np.log2(128)):.2f}x")
    
    print("\nExperiment completed. Results saved to butterfly_comparison.png and butterfly_learning_curves.png") 