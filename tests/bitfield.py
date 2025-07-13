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


class DeepBitfieldNet(nn.Module):
    """더 깊은 비트필드 네트워크"""
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # 히든 레이어들
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                # 첫 레이어는 일반 Linear
                layers.append(nn.Linear(prev_size, hidden_size))
            else:
                # 나머지는 BitfieldLinear
                fc_temp = nn.Linear(prev_size, hidden_size)
                nn.init.xavier_uniform_(fc_temp.weight)
                bitfield = BitfieldLinear.from_linear(fc_temp, basis_size=256, r_max=2.0)
                layers.append(bitfield)
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # 출력 레이어도 BitfieldLinear
        fc_out = nn.Linear(prev_size, num_classes)
        nn.init.xavier_uniform_(fc_out.weight)
        layers.append(BitfieldLinear.from_linear(fc_out, basis_size=256, r_max=2.0))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


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


def test_deep_network():
    """깊은 비트필드 네트워크 테스트"""
    print("\n=== DeepBitfieldNet 테스트 ===")
    
    # 데이터 생성
    n_samples = 500
    input_size = 784
    num_classes = 10
    
    X = torch.randn(n_samples, input_size)
    y = torch.randint(0, num_classes, (n_samples,))
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 모델 생성
    model = DeepBitfieldNet()
    print(f"모델 구조:\n{model}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    
    # 간단한 학습
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    start_time = time.time()
    
    for epoch in range(100):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/3: acc={accuracy:.2f}%")
    
    total_time = time.time() - start_time
    print(f"총 학습 시간: {total_time:.2f}s")
    print("✓ 깊은 네트워크 테스트 통과")


def test_riemannian_functions():
    """리만 기하학 함수들의 비트필드 인코딩 테스트"""
    print("\n=== 리만 기하학 함수 비트필드 테스트 ===")
    
    # 다양한 크기의 행렬로 테스트
    sizes = [(10, 8), (64, 32), (128, 64)]
    
    for m, n in sizes:
        print(f"\n행렬 크기: {m}x{n}")
        
        # 다양한 패턴의 가중치 생성
        weights = []
        
        # 1. 가우시안 분포
        w1 = torch.randn(m//4, n) * 0.1
        weights.append(("Gaussian", w1))
        
        # 2. 희소 행렬
        w2 = torch.randn(m//4, n) * 0.5
        mask = torch.rand(m//4, n) > 0.8
        w2 = w2 * mask.float()
        weights.append(("Sparse", w2))
        
        # 3. 구조화된 패턴
        w3 = torch.zeros(m//4, n)
        for i in range(m//4):
            for j in range(n):
                w3[i, j] = 0.3 * torch.sin(torch.tensor(i * j * 0.1))
        weights.append(("Structured", w3))
        
        # 4. 극단적 값
        w4 = torch.randn(m//4, n)
        w4 = torch.sign(w4) * (torch.abs(w4) ** 0.3)
        weights.append(("Extreme", w4))
        
        # 전체 가중치 결합
        full_weights = torch.cat([w for _, w in weights], dim=0)
        
        # Linear 레이어 생성
        linear = nn.Linear(n, m, bias=False)
        linear.weight.data = full_weights
        
        # BitfieldLinear로 변환
        bitfield = BitfieldLinear.from_linear(linear, basis_size=256, r_max=2.0)
        
        # 테스트 입력
        x = torch.randn(32, n)
        
        with torch.no_grad():
            y_original = linear(x)
            y_bitfield = bitfield(x)
        
        # 각 부분별 오차 분석
        start_idx = 0
        for name, w in weights:
            end_idx = start_idx + w.shape[0]
            
            y_orig_part = y_original[:, start_idx:end_idx]
            y_bit_part = y_bitfield[:, start_idx:end_idx]
            
            mse = torch.mean((y_orig_part - y_bit_part) ** 2).item()
            rel_error = torch.norm(y_orig_part - y_bit_part) / (torch.norm(y_orig_part) + 1e-8)
            
            print(f"  {name}: MSE={mse:.6f}, 상대오차={rel_error:.2%}")
            
            start_idx = end_idx
        
        # 전체 압축률
        original_bits = m * n * 32
        compressed_bits = m * 22
        compression_ratio = original_bits / compressed_bits
        print(f"  압축률: {compression_ratio:.1f}x")


def test_hyperbolic_mlp():
    """하이퍼볼릭 기하학을 활용한 MLP 테스트"""
    print("\n=== 하이퍼볼릭 MLP 테스트 ===")
    
    class HyperbolicMLP(nn.Module):
        def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
            super().__init__()
            
            layers = []
            prev_size = input_size
            
            for i, hidden_size in enumerate(hidden_sizes):
                # 첫 레이어는 일반 Linear
                if i == 0:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.LayerNorm(hidden_size))
                    layers.append(nn.ReLU())
                else:
                    # 나머지는 BitfieldLinear
                    fc = nn.Linear(prev_size, hidden_size)
                    nn.init.xavier_uniform_(fc.weight)
                    # 더 큰 r_max로 다양한 함수 활용
                    bitfield = BitfieldLinear.from_linear(fc, basis_size=256, r_max=3.0)
                    layers.append(bitfield)
                    layers.append(nn.LayerNorm(hidden_size))
                    layers.append(nn.GELU())  # GELU가 하이퍼볼릭과 잘 맞음
                
                layers.append(nn.Dropout(0.3))
                prev_size = hidden_size
            
            # 출력 레이어
            fc_out = nn.Linear(prev_size, num_classes)
            nn.init.xavier_uniform_(fc_out.weight, gain=0.1)  # 작은 초기화
            layers.append(BitfieldLinear.from_linear(fc_out, basis_size=256, r_max=2.0))
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.model(x)
    
    # 데이터 생성
    n_samples = 2000
    X = torch.randn(n_samples, 784) * 0.5
    # 더 복잡한 타겟 생성
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        # 입력의 특정 패턴에 따라 클래스 결정
        pattern = X[i].view(28, 28).sum(dim=0)[:10].sum().item()
        y[i] = int(abs(pattern) * 10) % 10
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 모델 생성
    model = HyperbolicMLP()
    print(model)
    
    # 학습
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    best_acc = 0
    
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        accuracy = 100. * correct / total
        best_acc = max(best_acc, accuracy)
        
        print(f"Epoch {epoch+1}/10: loss={total_loss/len(train_loader):.4f}, "
              f"acc={accuracy:.2f}%, lr={scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\n최고 정확도: {best_acc:.2f}%")
    print("✓ 하이퍼볼릭 MLP 테스트 통과")


if __name__ == "__main__":
    # 모든 테스트 실행
    test_bitfield_forward()
    test_bitfield_backward()
    test_riemannian_functions()
    test_mnist_training()
    test_hyperbolic_mlp()
    test_deep_network() 