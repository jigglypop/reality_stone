#!/usr/bin/env python3
"""
MNIST Integration Test with Dynamic Curvature
동적 곡률을 사용한 MNIST 분류 테스트
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import reality_stone
import pytest
import time
import numpy as np
import sys


class DynamicCurvatureNet(nn.Module):
    """동적 곡률을 사용하는 하이퍼볼릭 MNIST 분류기"""
    
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 일반 선형 레이어들
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
        # 정규화용 파라미터
        self.base_curvature = 1.0
        self.reg_lambda = 0.01
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        
        # Layer 1: 기본 선형
        h1 = torch.relu(self.fc1(x))
        
        # Layer 2: 하이퍼볼릭 정규화 적용
        h2 = torch.relu(self.fc2(h1))
        
        # 하이퍼볼릭 정규화 (경계 페널티) 적용
        if hasattr(reality_stone._C, 'boundary_penalty'):
            # 포인카레 디스크 내부로 정규화
            h2_norm = torch.norm(h2, dim=-1, keepdim=True)
            max_norm = 1.0 / np.sqrt(self.base_curvature) - 0.01
            h2_clamped = torch.where(
                h2_norm > max_norm,
                h2 * (max_norm / (h2_norm + 1e-7)),
                h2
            )
            h2 = h2_clamped
        
        # 출력 레이어
        logits = self.fc3(h2)
        
        return logits
    
    def get_regularization_loss(self):
        """정규화 손실 계산"""
        total_reg = 0.0
        
        # 모든 가중치에 대해 정규화 적용
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                if hasattr(reality_stone._C, 'boundary_penalty'):
                    reg_loss = reality_stone._C.boundary_penalty(
                        param.view(param.size(0), -1), 
                        self.base_curvature, 
                        0.01
                    )
                    total_reg += torch.mean(reg_loss)
        
        return self.reg_lambda * total_reg


class AdvancedDynamicNet(nn.Module):
    """진짜 고급 기능들을 사용하는 네트워크"""
    
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 동적 곡률 예측용 파라미터
        self.curvature_weight = nn.Parameter(torch.randn(1, 1) * 0.1)
        self.curvature_bias = nn.Parameter(torch.zeros(1))
        
        # 메인 네트워크
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
        # 측지선 활성화용 앵커들
        if hasattr(reality_stone._C, 'geodesic_activation'):
            self.num_anchors = 4
            self.anchors = nn.Parameter(torch.randn(self.num_anchors, hidden_dim) * 0.1)
            self.anchor_weights = nn.Parameter(torch.ones(self.num_anchors) / self.num_anchors)
            self.t_values = nn.Parameter(torch.randn(self.num_anchors) * 0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 동적 곡률 예측
        if hasattr(reality_stone._C, 'dynamic_curvature_pred'):
            # 입력의 L2 norm을 특징으로 사용
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            curvatures = reality_stone._C.dynamic_curvature_pred(
                x_norm, self.curvature_weight, self.curvature_bias, 1.0
            )
        else:
            curvatures = torch.ones(batch_size, device=x.device)
        
        # Layer 1
        h1 = torch.relu(self.fc1(x))
        
        # Layer 2 with 융합 연산
        if hasattr(reality_stone._C, 'fused_linear'):
            h2_raw, reg_loss = reality_stone._C.fused_transform_reg(h1, 1.0, 0.01)
            h2 = torch.relu(h2_raw)
        else:
            h2 = torch.relu(self.fc2(h1))
            reg_loss = torch.tensor(0.0, device=x.device)
        
        # 측지선 활성화 적용
        if hasattr(reality_stone._C, 'geodesic_activation'):
            h2_geo = reality_stone._C.geodesic_activation(
                h2, self.anchors, self.t_values, 
                torch.softmax(self.anchor_weights, dim=0), 1.0
            )
            h2 = h2_geo
        
        # 출력
        logits = self.fc3(h2)
        
        return logits, curvatures, reg_loss


def create_mnist_dataloaders(batch_size=128, num_workers=0):
    """MNIST 데이터로더 생성"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 데이터셋 다운로드 (이미 있으면 skip)
    train_dataset = torchvision.datasets.MNIST(
        root='./MNIST', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./MNIST', train=False, download=True, transform=transform
    )
    
    # 빠른 테스트를 위해 데이터 개수 제한
    train_subset = torch.utils.data.Subset(train_dataset, range(0, 1000))
    test_subset = torch.utils.data.Subset(test_dataset, range(0, 200))
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def test_basic_mnist():
    """기본 MNIST 분류 테스트"""
    print("\n" + "="*60)
    print("🧪 Basic MNIST Classification Test")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # 데이터 로더
    train_loader, test_loader = create_mnist_dataloaders(batch_size=64)
    
    # 모델 생성
    model = DynamicCurvatureNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련 (짧게)
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 5:  # 빠른 테스트를 위해 5배치만
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # 분류 손실
        loss = criterion(output, target)
        
        # 정규화 손실 추가
        reg_loss = model.get_regularization_loss()
        total_loss_val = loss + reg_loss
        
        total_loss_val.backward()
        optimizer.step()
        
        total_loss += total_loss_val.item()
        num_batches += 1
        
        if batch_idx % 2 == 0:
            print(f"   Batch {batch_idx}: Loss {total_loss_val.item():.4f} "
                  f"(CE: {loss.item():.4f}, Reg: {reg_loss.item():.4f})")
    
    avg_loss = total_loss / num_batches
    print(f"✅ Training completed. Average loss: {avg_loss:.4f}")
    
    # 테스트
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 3:  # 빠른 테스트
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    print(f"✅ Test accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    assert accuracy > 10.0, f"너무 낮은 정확도: {accuracy:.2f}%"
    print("🎉 Basic MNIST test passed!")


def test_advanced_mnist():
    """고급 기능을 사용하는 MNIST 테스트"""
    print("\n" + "="*60)
    print("🧪 Advanced MNIST with Dynamic Curvature")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # 고급 기능 사용 가능 확인
    advanced_funcs = ['dynamic_curvature_pred', 'fused_transform_reg', 'geodesic_activation']
    available_funcs = []
    
    for func in advanced_funcs:
        if hasattr(reality_stone._C, func):
            available_funcs.append(func)
            print(f"   ✅ {func} available")
        else:
            print(f"   ❌ {func} not available")
    
    if not available_funcs:
        print("⚠️  No advanced functions available. Skipping advanced test.")
        pytest.skip("Advanced functions not implemented")
        return
    
    # 데이터 로더
    train_loader, test_loader = create_mnist_dataloaders(batch_size=32)
    
    # 고급 모델 생성
    model = AdvancedDynamicNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"📊 Advanced model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련
    model.train()
    total_loss = 0.0
    total_reg = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 3:  # 빠른 테스트
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 전방 패스
        output, curvatures, reg_loss = model(data)
        
        # 손실 계산
        ce_loss = criterion(output, target)
        total_loss_val = ce_loss + reg_loss
        
        total_loss_val.backward()
        optimizer.step()
        
        total_loss += total_loss_val.item()
        total_reg += reg_loss.item()
        num_batches += 1
        
        print(f"   Batch {batch_idx}: Loss {total_loss_val.item():.4f} "
              f"(CE: {ce_loss.item():.4f}, Reg: {reg_loss.item():.4f})")
        print(f"      Curvature range: [{curvatures.min():.3f}, {curvatures.max():.3f}]")
    
    avg_loss = total_loss / num_batches
    avg_reg = total_reg / num_batches
    print(f"✅ Advanced training completed. Loss: {avg_loss:.4f}, Reg: {avg_reg:.4f}")
    
    # 테스트
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 2:
                break
                
            data, target = data.to(device), target.to(device)
            output, curvatures, _ = model(data)
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    print(f"✅ Advanced test accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    assert accuracy > 5.0, f"고급 모델 정확도가 너무 낮음: {accuracy:.2f}%"
    print(f"🎉 Advanced MNIST test passed with {len(available_funcs)} advanced features!")


def test_nan_detection():
    """NaN 문제 감지 테스트"""
    print("\n" + "="*60)
    print("🧪 NaN Detection Test")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 극단적인 값들로 테스트
    extreme_input = torch.tensor([
        [10.0] * 784,    # 매우 큰 값
        [-10.0] * 784,   # 매우 작은 값
        [0.0] * 784,     # 영벡터
    ], device=device)
    
    model = DynamicCurvatureNet().to(device)
    
    print("🔍 Testing with extreme inputs...")
    
    with torch.no_grad():
        output = model(extreme_input)
        
        # NaN 체크
        if torch.isnan(output).any():
            print("❌ NaN detected in output!")
            print(f"   Output: {output}")
            assert False, "NaN values detected with extreme inputs"
        else:
            print("✅ No NaN detected with extreme inputs")
        
        # 무한대 체크
        if torch.isinf(output).any():
            print("❌ Infinite values detected!")
            assert False, "Infinite values detected"
        else:
            print("✅ No infinite values detected")
    
    print("🎉 NaN detection test passed!")


def test_performance_comparison():
    """성능 비교 테스트"""
    print("\n" + "="*60)
    print("🧪 Performance Comparison Test")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    
    # 테스트 데이터
    test_input = torch.randn(batch_size, 784, device=device)
    
    # 기본 모델
    basic_model = DynamicCurvatureNet().to(device)
    
    # 고급 모델 (가능한 기능들만 사용)
    advanced_model = AdvancedDynamicNet().to(device)
    
    # 성능 측정
    def measure_time(model, input_data, num_runs=50):
        model.eval()
        torch.cuda.synchronize() if device == 'cuda' else None
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                if hasattr(model, 'forward'):
                    if isinstance(model, AdvancedDynamicNet):
                        output, _, _ = model(input_data)
                    else:
                        output = model(input_data)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        return (end_time - start_time) / num_runs * 1000  # ms per run
    
    # 측정
    basic_time = measure_time(basic_model, test_input)
    advanced_time = measure_time(advanced_model, test_input)
    
    print(f"📊 Performance Results:")
    print(f"   Basic model: {basic_time:.3f} ms/batch")
    print(f"   Advanced model: {advanced_time:.3f} ms/batch")
    print(f"   Overhead: {((advanced_time - basic_time) / basic_time * 100):.1f}%")
    
    # 오버헤드가 너무 크지 않은지 확인
    overhead_ratio = advanced_time / basic_time
    assert overhead_ratio < 3.0, f"고급 기능 오버헤드가 너무 큼: {overhead_ratio:.1f}x"
    
    print("✅ Performance test passed!")


if __name__ == "__main__":
    print("🌟 Reality Stone MNIST Integration Test Suite")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"💻 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    tests = [
        test_basic_mnist,
        test_advanced_mnist,
        test_nan_detection,
        test_performance_comparison,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 MNIST Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All MNIST integration tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some MNIST tests failed.")
        sys.exit(1) 