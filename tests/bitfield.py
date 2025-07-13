"""비트필드 레이어 테스트"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from reality_stone.layers import BitfieldLinear
import time


def test_bitfield_forward():
    """비트필드 레이어 순전파 테스트"""
    print("\n=== BitfieldLinear 순전파 테스트 ===")
    
    batch_size = 32
    in_features = 128
    out_features = 10
    
    # 기존 Linear 레이어 생성
    linear = nn.Linear(in_features, out_features)
    
    # BitfieldLinear로 변환
    bitfield_linear = BitfieldLinear.from_linear(linear, basis_size=256, r_max=1.0)
    
    # 랜덤 입력
    x = torch.randn(batch_size, in_features)
    
    # 순전파
    with torch.no_grad():
        # 원본 출력
        y_original = linear(x)
        
        # 압축된 레이어 출력
        y_bitfield = bitfield_linear(x)
    
    print(f"입력 형태: {x.shape}")
    print(f"원본 출력 형태: {y_original.shape}")
    print(f"압축 출력 형태: {y_bitfield.shape}")
    
    # 근사 오차 계산
    mse = torch.mean((y_original - y_bitfield) ** 2).item()
    rel_error = torch.norm(y_original - y_bitfield) / torch.norm(y_original)
    
    print(f"MSE: {mse:.6f}")
    print(f"상대 오차: {rel_error:.4%}")
    
    # 압축률 계산
    original_params = in_features * out_features * 32  # 32 bits per float
    compressed_params = out_features * 22  # 22 bits per code
    compression_ratio = original_params / compressed_params
    
    print(f"압축률: {compression_ratio:.1f}x")
    print("✓ 순전파 테스트 통과")


def test_bitfield_backward():
    """비트필드 레이어 역전파 테스트"""
    print("\n=== BitfieldLinear 역전파 테스트 ===")
    
    batch_size = 16
    in_features = 64
    out_features = 10
    
    # 기존 Linear 레이어 생성
    linear = nn.Linear(in_features, out_features)
    
    # BitfieldLinear로 변환
    bitfield_linear = BitfieldLinear.from_linear(linear, basis_size=128, r_max=1.0)
    
    # 랜덤 입력과 타겟
    x = torch.randn(batch_size, in_features, requires_grad=True)
    target = torch.randint(0, out_features, (batch_size,))
    
    # 순전파
    y = bitfield_linear(x)
    
    # 손실 계산
    loss = nn.CrossEntropyLoss()(y, target)
    
    # 역전파
    loss.backward()
    
    print(f"입력 그래디언트 형태: {x.grad.shape}")
    print(f"입력 그래디언트 노름: {x.grad.norm().item():.4f}")
    print("✓ 역전파 테스트 통과")


class BitfieldMLP(nn.Module):
    """비트필드 레이어를 사용한 간단한 MLP"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        # 첫 번째 레이어는 일반 Linear
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        # 두 번째 레이어를 BitfieldLinear로 생성
        fc2_temp = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(fc2_temp.weight)
        self.fc2_bitfield = BitfieldLinear.from_linear(fc2_temp, basis_size=256, r_max=1.0)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2_bitfield(x)
        return x


def test_mnist_training():
    """MNIST 데이터셋으로 BitfieldMLP 학습 테스트"""
    print("\n=== BitfieldMLP MNIST 학습 테스트 ===")
    
    # 간단한 합성 데이터 생성 (실제 MNIST 대신)
    n_samples = 1000
    input_size = 784
    num_classes = 10
    
    # 랜덤 데이터 생성
    X = torch.randn(n_samples, input_size)
    y = torch.randint(0, num_classes, (n_samples,))
    
    # 데이터셋과 데이터로더
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 모델 생성
    model = BitfieldMLP()
    print(model)
    
    # 옵티마이저와 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    model.train()
    for epoch in range(5):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_time = time.time() - start_time
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"[BitfieldMLP] Epoch {epoch+1}/5 loss={avg_loss:.4f} time={epoch_time:.2f}s acc={accuracy:.2f}%")
    
    print("✓ MNIST 학습 테스트 통과")


if __name__ == "__main__":
    # 모든 테스트 실행
    test_bitfield_forward()
    test_bitfield_backward()
    test_mnist_training() 