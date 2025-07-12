# 테스트 가이드

Reality Stone 프로젝트의 테스트 작성 및 실행 방법을 설명합니다. 포괄적인 테스트는 코드 품질과 안정성을 보장하는 핵심 요소입니다.

## 테스트 전략

### 테스트 피라미드

```
    E2E 테스트 (적음)
   ┌─────────────────┐
   │   통합 테스트    │  (보통)
   ├─────────────────┤
   │   단위 테스트    │  (많음)
   └─────────────────┘
```

1. **단위 테스트 (Unit Tests)**: 개별 함수나 클래스 테스트
2. **통합 테스트 (Integration Tests)**: 모듈 간 상호작용 테스트
3. **E2E 테스트 (End-to-End Tests)**: 전체 워크플로우 테스트

### 테스트 원칙

- **F.I.R.S.T 원칙**:
  - **Fast**: 빠른 실행
  - **Independent**: 독립적 실행
  - **Repeatable**: 반복 가능
  - **Self-Validating**: 자체 검증
  - **Timely**: 적시 작성

## Python 테스트

### pytest 설정

```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=reality_stone
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: 느린 테스트 (GPU 필요)
    integration: 통합 테스트
    unit: 단위 테스트
    cuda: CUDA 필요 테스트
```

### 테스트 디렉토리 구조

```
tests/
├── conftest.py                 # pytest 설정 및 fixture
├── unit/                       # 단위 테스트
│   ├── test_poincare_ops.py
│   ├── test_lorentz_ops.py
│   ├── test_klein_ops.py
│   └── test_utilities.py
├── integration/                # 통합 테스트
│   ├── test_layer_integration.py
│   ├── test_model_training.py
│   └── test_coordinate_conversion.py
├── performance/                # 성능 테스트
│   ├── test_benchmarks.py
│   └── test_memory_usage.py
└── e2e/                       # E2E 테스트
    ├── test_mnist_example.py
    └── test_full_pipeline.py
```

### 기본 테스트 작성

```python
# tests/unit/test_poincare_ops.py
import pytest
import torch
import numpy as np
import reality_stone as rs
from reality_stone.errors import DimensionMismatchError, InvalidCurvatureError

class TestPoincareBallOps:
    """Poincaré Ball 연산 단위 테스트."""
    
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터 생성."""
        torch.manual_seed(42)
        return {
            'x': torch.randn(16, 32) * 0.1,
            'y': torch.randn(16, 32) * 0.1,
            'curvature': 1.0
        }
    
    def test_distance_계산_정확성(self, sample_data):
        """거리 계산의 정확성을 테스트합니다."""
        x, y, c = sample_data['x'], sample_data['y'], sample_data['curvature']
        
        # 실제 계산
        distances = rs.poincare_distance(x, y, c=c)
        
        # 기본 검증
        assert distances.shape == (16,)
        assert torch.all(distances >= 0), "거리는 음수가 될 수 없습니다"
        assert torch.isfinite(distances).all(), "거리에 NaN/Inf가 있습니다"
        
        # 동일 점 거리는 0
        same_distances = rs.poincare_distance(x, x, c=c)
        torch.testing.assert_close(same_distances, torch.zeros_like(same_distances), atol=1e-6)
    
    def test_distance_대칭성(self, sample_data):
        """거리의 대칭성을 테스트합니다: d(x,y) = d(y,x)"""
        x, y, c = sample_data['x'], sample_data['y'], sample_data['curvature']
        
        dist_xy = rs.poincare_distance(x, y, c=c)
        dist_yx = rs.poincare_distance(y, x, c=c)
        
        torch.testing.assert_close(dist_xy, dist_yx, rtol=1e-5)
    
    def test_distance_삼각부등식(self, sample_data):
        """삼각부등식을 테스트합니다: d(x,z) ≤ d(x,y) + d(y,z)"""
        x, y, c = sample_data['x'], sample_data['y'], sample_data['curvature']
        z = torch.randn_like(x) * 0.1
        
        dist_xz = rs.poincare_distance(x, z, c=c)
        dist_xy = rs.poincare_distance(x, y, c=c)
        dist_yz = rs.poincare_distance(y, z, c=c)
        
        # 삼각부등식: d(x,z) ≤ d(x,y) + d(y,z)
        triangle_inequality = dist_xz <= dist_xy + dist_yz + 1e-6  # 수치적 오차 고려
        assert triangle_inequality.all(), "삼각부등식이 위반되었습니다"
    
    @pytest.mark.parametrize("curvature", [1e-3, 1e-2, 1e-1, 1.0, 10.0])
    def test_distance_다양한_곡률(self, sample_data, curvature):
        """다양한 곡률 값에서 거리 계산을 테스트합니다."""
        x, y = sample_data['x'], sample_data['y']
        
        distances = rs.poincare_distance(x, y, c=curvature)
        
        assert torch.isfinite(distances).all()
        assert torch.all(distances >= 0)
    
    def test_distance_예외_처리(self, sample_data):
        """거리 계산의 예외 처리를 테스트합니다."""
        x, y = sample_data['x'], sample_data['y']
        
        # 잘못된 곡률
        with pytest.raises(InvalidCurvatureError):
            rs.poincare_distance(x, y, c=-1.0)
        
        with pytest.raises(InvalidCurvatureError):
            rs.poincare_distance(x, y, c=0.0)
        
        # 차원 불일치
        y_wrong = torch.randn(16, 64)  # 다른 차원
        with pytest.raises(DimensionMismatchError):
            rs.poincare_distance(x, y_wrong, c=1.0)
    
    def test_mobius_add_항등원(self, sample_data):
        """Mobius 덧셈의 항등원을 테스트합니다: x ⊕ 0 = x"""
        x, c = sample_data['x'], sample_data['curvature']
        zero = torch.zeros_like(x)
        
        result = rs.mobius_add(x, zero, c=c)
        torch.testing.assert_close(result, x, rtol=1e-5)
    
    def test_mobius_add_역원(self, sample_data):
        """Mobius 덧셈의 역원을 테스트합니다: x ⊕ (-x) = 0"""
        x, c = sample_data['x'], sample_data['curvature']
        neg_x = rs.mobius_neg(x, c=c)
        
        result = rs.mobius_add(x, neg_x, c=c)
        zero = torch.zeros_like(x)
        
        torch.testing.assert_close(result, zero, atol=1e-5)
    
    @pytest.mark.slow
    def test_gradient_수치적_검증(self, sample_data):
        """그래디언트의 수치적 검증을 수행합니다."""
        x, y, c = sample_data['x'], sample_data['y'], sample_data['curvature']
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # 자동 미분으로 그래디언트 계산
        distances = rs.poincare_distance(x, y, c=c)
        loss = distances.sum()
        loss.backward()
        
        auto_grad_x = x.grad.clone()
        auto_grad_y = y.grad.clone()
        
        # 수치 미분으로 그래디언트 계산
        eps = 1e-5
        numerical_grad_x = torch.zeros_like(x)
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_plus = x.clone().detach()
                x_minus = x.clone().detach()
                x_plus[i, j] += eps
                x_minus[i, j] -= eps
                
                dist_plus = rs.poincare_distance(x_plus, y.detach(), c=c).sum()
                dist_minus = rs.poincare_distance(x_minus, y.detach(), c=c).sum()
                
                numerical_grad_x[i, j] = (dist_plus - dist_minus) / (2 * eps)
        
        # 그래디언트 비교 (일부 요소만)
        torch.testing.assert_close(
            auto_grad_x[:4, :4], 
            numerical_grad_x[:4, :4], 
            rtol=1e-3, 
            atol=1e-4
        )
```

### Fixture 활용

```python
# tests/conftest.py
import pytest
import torch
import numpy as np

@pytest.fixture(scope="session")
def device():
    """테스트에 사용할 디바이스를 반환합니다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def random_seed():
    """재현 가능한 테스트를 위한 랜덤 시드 설정."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

@pytest.fixture
def small_tensors():
    """작은 크기의 테스트 텐서들."""
    return {
        'x': torch.randn(8, 16) * 0.1,
        'y': torch.randn(8, 16) * 0.1,
        'batch_size': 8,
        'dim': 16
    }

@pytest.fixture
def large_tensors():
    """큰 크기의 테스트 텐서들."""
    return {
        'x': torch.randn(256, 512) * 0.1,
        'y': torch.randn(256, 512) * 0.1,
        'batch_size': 256,
        'dim': 512
    }

@pytest.fixture
def hyperbolic_constraints():
    """하이퍼볼릭 제약 조건을 만족하는 텐서들."""
    x = torch.randn(32, 64) * 0.1
    y = torch.randn(32, 64) * 0.1
    
    # 단위 원 내부로 클리핑
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    y_norm = torch.norm(y, dim=-1, keepdim=True)
    
    x = x / torch.clamp(x_norm, min=1.01) * 0.9
    y = y / torch.clamp(y_norm, min=1.01) * 0.9
    
    return {'x': x, 'y': y}

@pytest.fixture
def model_config():
    """테스트용 모델 설정."""
    return {
        'input_dim': 64,
        'hidden_dim': 32,
        'output_dim': 10,
        'curvature': 1e-2,
        'learning_rate': 1e-3,
        'batch_size': 16
    }
```

### 성능 테스트

```python
# tests/performance/test_benchmarks.py
import pytest
import torch
import time
import psutil
import reality_stone as rs

class TestPerformance:
    """성능 테스트 클래스."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size,dim", [
        (32, 64), (128, 128), (512, 256), (1024, 512)
    ])
    def test_poincare_layer_속도(self, batch_size, dim):
        """Poincaré 레이어의 실행 속도를 측정합니다."""
        u = torch.randn(batch_size, dim) * 0.1
        v = torch.randn(batch_size, dim) * 0.1
        
        # 워밍업
        for _ in range(10):
            _ = rs.poincare_ball_layer(u, v, c=1.0, t=0.5)
        
        # 실제 측정
        start_time = time.time()
        for _ in range(100):
            result = rs.poincare_ball_layer(u, v, c=1.0, t=0.5)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size / avg_time
        
        print(f"Batch size: {batch_size}, Dim: {dim}")
        print(f"Average time: {avg_time:.4f}s")
        print(f"Throughput: {throughput:.2f} samples/sec")
        
        # 성능 기준 (조정 가능)
        assert avg_time < 0.1, f"Too slow: {avg_time:.4f}s"
    
    @pytest.mark.slow
    @pytest.mark.cuda
    def test_cuda_vs_cpu_성능비교(self):
        """CUDA와 CPU 성능을 비교합니다."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        batch_size, dim = 512, 256
        u = torch.randn(batch_size, dim) * 0.1
        v = torch.randn(batch_size, dim) * 0.1
        
        # CPU 측정
        u_cpu, v_cpu = u.cpu(), v.cpu()
        start_time = time.time()
        for _ in range(50):
            result_cpu = rs.poincare_ball_layer(u_cpu, v_cpu, c=1.0, t=0.5)
        cpu_time = time.time() - start_time
        
        # GPU 측정
        u_gpu, v_gpu = u.cuda(), v.cuda()
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            result_gpu = rs.poincare_ball_layer(u_gpu, v_gpu, c=1.0, t=0.5)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # 결과 일치성 확인
        torch.testing.assert_close(
            result_cpu, result_gpu.cpu(), rtol=1e-4, atol=1e-5
        )
    
    def test_메모리_사용량(self):
        """메모리 사용량을 모니터링합니다."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 큰 텐서 생성 및 연산
        u = torch.randn(1000, 1000) * 0.1
        v = torch.randn(1000, 1000) * 0.1
        
        result = rs.poincare_ball_layer(u, v, c=1.0, t=0.5)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # 메모리 사용량 기준 (조정 가능)
        assert memory_increase < 1000, f"Too much memory used: {memory_increase:.2f} MB"
        
        # 메모리 해제 확인
        del u, v, result
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

### 통합 테스트

```python
# tests/integration/test_layer_integration.py
import pytest
import torch
import torch.nn as nn
import reality_stone as rs

class TestLayerIntegration:
    """레이어 통합 테스트."""
    
    def test_다중_레이어_연결(self):
        """여러 하이퍼볼릭 레이어를 연결한 모델 테스트."""
        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder1 = nn.Linear(64, 32)
                self.encoder2 = nn.Linear(64, 32)
                self.encoder3 = nn.Linear(32, 16)
                self.encoder4 = nn.Linear(32, 16)
                self.classifier = nn.Linear(16, 10)
                
            def forward(self, x):
                # 첫 번째 하이퍼볼릭 레이어
                u1 = torch.tanh(self.encoder1(x))
                v1 = torch.tanh(self.encoder2(x))
                h1 = rs.poincare_ball_layer(u1, v1, c=1e-2, t=0.5)
                
                # 두 번째 하이퍼볼릭 레이어
                u2 = torch.tanh(self.encoder3(h1))
                v2 = torch.tanh(self.encoder4(h1))
                h2 = rs.poincare_ball_layer(u2, v2, c=1e-2, t=0.5)
                
                return self.classifier(h2)
        
        model = MultiLayerModel()
        x = torch.randn(32, 64)
        
        # Forward pass
        output = model(x)
        assert output.shape == (32, 10)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # 모든 파라미터에 그래디언트가 있는지 확인
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"
    
    def test_좌표_변환_일관성(self):
        """좌표 변환의 일관성을 테스트합니다."""
        x = torch.randn(16, 32) * 0.1
        c = 1.0
        
        # Poincaré → Lorentz → Poincaré
        x_lorentz = rs.poincare_to_lorentz(x, c=c)
        x_back = rs.lorentz_to_poincare(x_lorentz, c=c)
        
        torch.testing.assert_close(x, x_back, rtol=1e-5, atol=1e-6)
        
        # Poincaré → Klein → Poincaré
        x_klein = rs.poincare_to_klein(x, c=c)
        x_back2 = rs.klein_to_poincare(x_klein, c=c)
        
        torch.testing.assert_close(x, x_back2, rtol=1e-5, atol=1e-6)
    
    def test_배치_처리_일관성(self):
        """배치 처리와 개별 처리 결과의 일관성을 테스트합니다."""
        batch_x = torch.randn(8, 32) * 0.1
        batch_y = torch.randn(8, 32) * 0.1
        c = 1.0
        
        # 배치로 처리
        batch_result = rs.poincare_distance(batch_x, batch_y, c=c)
        
        # 개별로 처리
        individual_results = []
        for i in range(8):
            x_i = batch_x[i:i+1]
            y_i = batch_y[i:i+1]
            result_i = rs.poincare_distance(x_i, y_i, c=c)
            individual_results.append(result_i)
        
        individual_result = torch.cat(individual_results)
        
        torch.testing.assert_close(batch_result, individual_result, rtol=1e-6)
```

### E2E 테스트

```python
# tests/e2e/test_mnist_example.py
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import reality_stone as rs

@pytest.mark.slow
class TestMNISTExample:
    """MNIST 예제 E2E 테스트."""
    
    def test_mnist_훈련_파이프라인(self):
        """전체 MNIST 훈련 파이프라인을 테스트합니다."""
        class HyperbolicMNIST(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder1 = nn.Linear(784, 64)
                self.encoder2 = nn.Linear(784, 64)
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = x.view(x.size(0), -1)
                u = torch.tanh(self.encoder1(x))
                v = torch.tanh(self.encoder2(x))
                h = rs.poincare_ball_layer(u, v, c=1e-2, t=0.5)
                return self.classifier(h)
        
        # 모델 및 옵티마이저 설정
        model = HyperbolicMNIST()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # 더미 데이터 생성
        batch_size = 32
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        
        # 훈련 루프
        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
        
        final_loss = loss.item()
        
        # 기본 검증
        assert output.shape == (batch_size, 10)
        assert torch.isfinite(output).all()
        assert final_loss < initial_loss, "Loss should decrease during training"
        
        # 예측 정확도 (랜덤 데이터이므로 낮음)
        predictions = output.argmax(dim=1)
        accuracy = (predictions == y).float().mean()
        assert 0.0 <= accuracy <= 1.0
```

## Rust 테스트

### 단위 테스트

```rust
// src/layers/poincare.rs
#[cfg(test)]
mod tests {
    use super::*;
    use torch::{Tensor, Kind, Device};
    
    fn create_test_tensors() -> (Tensor, Tensor) {
        let x = Tensor::randn(&[16, 32], (Kind::Float, Device::Cpu)) * 0.1;
        let y = Tensor::randn(&[16, 32], (Kind::Float, Device::Cpu)) * 0.1;
        (x, y)
    }
    
    #[test]
    fn test_distance_non_negative() {
        let (x, y) = create_test_tensors();
        let distances = poincare_distance(&x, &y, 1.0).unwrap();
        
        // 모든 거리가 음이 아닌지 확인
        let min_distance = f64::from(distances.min());
        assert!(min_distance >= 0.0, "Distance should be non-negative");
    }
    
    #[test]
    fn test_distance_symmetry() {
        let (x, y) = create_test_tensors();
        
        let dist_xy = poincare_distance(&x, &y, 1.0).unwrap();
        let dist_yx = poincare_distance(&y, &x, 1.0).unwrap();
        
        let diff = (&dist_xy - &dist_yx).abs().max();
        assert!(f64::from(diff) < 1e-6, "Distance should be symmetric");
    }
    
    #[test]
    fn test_invalid_curvature() {
        let (x, y) = create_test_tensors();
        
        // 음수 곡률
        assert!(poincare_distance(&x, &y, -1.0).is_err());
        
        // 0 곡률
        assert!(poincare_distance(&x, &y, 0.0).is_err());
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let x = Tensor::randn(&[16, 32], (Kind::Float, Device::Cpu));
        let y = Tensor::randn(&[16, 64], (Kind::Float, Device::Cpu)); // 다른 차원
        
        assert!(poincare_distance(&x, &y, 1.0).is_err());
    }
    
    #[test]
    fn test_mobius_add_identity() {
        let x = Tensor::randn(&[8, 16], (Kind::Float, Device::Cpu)) * 0.1;
        let zero = Tensor::zeros(&[8, 16], (Kind::Float, Device::Cpu));
        
        let result = mobius_add(&x, &zero, 1.0).unwrap();
        let diff = (&result - &x).abs().max();
        
        assert!(f64::from(diff) < 1e-6, "x ⊕ 0 should equal x");
    }
    
    #[test]
    fn test_numerical_stability() {
        // 경계 근처 값들로 테스트
        let x = Tensor::ones(&[4, 8], (Kind::Float, Device::Cpu)) * 0.99;
        let y = Tensor::ones(&[4, 8], (Kind::Float, Device::Cpu)) * 0.98;
        
        let result = poincare_distance(&x, &y, 1e-3).unwrap();
        
        // NaN이나 Inf가 없어야 함
        assert!(result.isfinite().all().bool_value(), "Result should be finite");
    }
}
```

### 통합 테스트

```rust
// tests/integration_test.rs
use reality_stone::*;
use torch::{Tensor, Kind, Device};

#[test]
fn test_layer_integration() {
    let batch_size = 32;
    let dim = 64;
    
    let u = Tensor::randn(&[batch_size, dim], (Kind::Float, Device::Cpu)) * 0.1;
    let v = Tensor::randn(&[batch_size, dim], (Kind::Float, Device::Cpu)) * 0.1;
    
    // Poincaré 레이어
    let poincare_result = poincare_ball_layer(&u, &v, 1e-2, 0.5).unwrap();
    
    // Lorentz로 변환
    let u_lorentz = poincare_to_lorentz(&u, 1e-2).unwrap();
    let v_lorentz = poincare_to_lorentz(&v, 1e-2).unwrap();
    let lorentz_result = lorentz_layer(&u_lorentz, &v_lorentz, 1e-2, 0.5).unwrap();
    
    // 다시 Poincaré로 변환
    let converted_result = lorentz_to_poincare(&lorentz_result, 1e-2).unwrap();
    
    // 결과 비교 (약간의 오차 허용)
    let diff = (&poincare_result - &converted_result).abs().max();
    assert!(f64::from(diff) < 1e-4, "Coordinate conversion should be consistent");
}

#[test]
fn test_gradient_computation() {
    let u = Tensor::randn(&[16, 32], (Kind::Float, Device::Cpu)) * 0.1;
    let v = Tensor::randn(&[16, 32], (Kind::Float, Device::Cpu)) * 0.1;
    u.set_requires_grad(true);
    v.set_requires_grad(true);
    
    let result = poincare_ball_layer(&u, &v, 1.0, 0.5).unwrap();
    let loss = result.sum(Kind::Float);
    
    loss.backward();
    
    // 그래디언트가 계산되었는지 확인
    assert!(u.grad().defined(), "u should have gradients");
    assert!(v.grad().defined(), "v should have gradients");
    
    // 그래디언트가 유한한지 확인
    assert!(u.grad().isfinite().all().bool_value(), "u gradients should be finite");
    assert!(v.grad().isfinite().all().bool_value(), "v gradients should be finite");
}
```

## 테스트 실행

### 기본 실행

```bash
# 모든 테스트 실행
pytest

# 특정 디렉토리 테스트
pytest tests/unit/

# 특정 파일 테스트
pytest tests/unit/test_poincare_ops.py

# 특정 테스트 함수
pytest tests/unit/test_poincare_ops.py::TestPoincareBallOps::test_distance_계산_정확성

# 마커 기반 실행
pytest -m "not slow"  # 느린 테스트 제외
pytest -m "cuda"      # CUDA 테스트만
pytest -m "unit"      # 단위 테스트만
```

### 커버리지 측정

```bash
# 커버리지와 함께 실행
pytest --cov=reality_stone --cov-report=html

# 누락된 라인 표시
pytest --cov=reality_stone --cov-report=term-missing

# 최소 커버리지 요구
pytest --cov=reality_stone --cov-fail-under=80
```

### 병렬 실행

```bash
# pytest-xdist 사용
pip install pytest-xdist

# 4개 프로세스로 병렬 실행
pytest -n 4

# 자동 CPU 수 감지
pytest -n auto
```

### Rust 테스트

```bash
# 단위 테스트
cargo test

# 통합 테스트
cargo test --test integration_test

# 릴리스 모드로 테스트
cargo test --release

# 특정 테스트만
cargo test test_distance_non_negative

# CUDA 기능 포함
cargo test --features cuda
```

## CI/CD 통합

### GitHub Actions 설정

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test-python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest --cov=reality_stone --cov-report=xml -n auto
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  test-rust:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    
    - name: Run Rust tests
      run: cargo test --verbose

  test-cuda:
    runs-on: ubuntu-latest
    if: github.event_name == 'push'  # PR에서는 CUDA 테스트 생략
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up CUDA
      uses: Jimver/cuda-toolkit@v0.2.11
      with:
        cuda: '11.8'
    
    - name: Run CUDA tests
      run: |
        cargo test --features cuda
        pytest -m cuda
```

## 테스트 모범 사례

### 테스트 작성 가이드라인

1. **명확한 테스트명**: 무엇을 테스트하는지 명확히 표현
2. **AAA 패턴**: Arrange, Act, Assert 구조 사용
3. **독립성**: 각 테스트는 독립적으로 실행 가능해야 함
4. **재현성**: 동일한 조건에서 동일한 결과 보장
5. **경계값 테스트**: 극한 상황과 경계값 테스트

### 수치적 테스트 주의사항

```python
# 부동소수점 비교 시 허용 오차 사용
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

# NaN/Inf 검증
assert torch.isfinite(result).all()

# 수치적 안정성 테스트
assert not torch.isnan(result).any()
assert not torch.isinf(result).any()
```

### 성능 회귀 방지

```python
@pytest.mark.benchmark
def test_performance_regression():
    """성능 회귀를 방지하는 벤치마크 테스트."""
    # 기준 성능 측정
    baseline_time = 0.1  # 초
    
    start_time = time.time()
    # 테스트할 연산
    end_time = time.time()
    
    actual_time = end_time - start_time
    assert actual_time < baseline_time * 1.2  # 20% 이내 허용
```

이러한 포괄적인 테스트 전략을 통해 Reality Stone의 품질과 안정성을 보장할 수 있습니다. 