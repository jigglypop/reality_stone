"""비트필드 레이어 테스트"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    for epoch in range(20):
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
        
        print(f"[BitfieldMLP] Epoch {epoch+1}/20 loss={avg_loss:.4f} time={epoch_time:.2f}s acc={accuracy:.2f}%")
    
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
    full_epoch = 20
    
    model.train()
    start_time = time.time()
    
    for epoch in range(full_epoch):
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
        print(f"Epoch {epoch+1}/{full_epoch}: acc={accuracy:.2f}%")
    
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    best_acc = 0
    
    for epoch in range(20):
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
        
        print(f"Epoch {epoch+1}/20: loss={total_loss/len(train_loader):.4f}, "
              f"acc={accuracy:.2f}%, lr={scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\n최고 정확도: {best_acc:.2f}%")
    print("✓ 하이퍼볼릭 MLP 테스트 통과")


def test_3d_tensor_support():
    """3D 텐서 지원 테스트"""
    print("\n=== 3D 텐서 지원 테스트 ===")
    
    batch_size = 16
    seq_len = 32
    in_features = 64
    out_features = 32

    
    # 기존 Linear 레이어 생성
    linear = nn.Linear(in_features, out_features)
    
    # BitfieldLinear로 변환
    bitfield_linear = BitfieldLinear.from_linear(linear, basis_size=128, r_max=1.0)
    
    # 3D 입력 텐서 생성: [batch_size, seq_len, in_features]
    x_3d = torch.randn(batch_size, seq_len, in_features)
    
    # 순전파
    with torch.no_grad():
        # 원본 Linear 레이어 출력
        y_original = linear(x_3d)
        
        # 비트필드 레이어 출력
        y_bitfield = bitfield_linear(x_3d)
    
    print(f"3D 입력 형태: {x_3d.shape}")
    print(f"원본 출력 형태: {y_original.shape}")
    print(f"비트필드 출력 형태: {y_bitfield.shape}")
    
    # 형태 검증
    assert y_original.shape == y_bitfield.shape, f"출력 형태가 다름: {y_original.shape} vs {y_bitfield.shape}"
    assert y_bitfield.shape == (batch_size, seq_len, out_features), f"예상 출력 형태와 다름: {y_bitfield.shape}"
    
    # 근사 오차 계산
    mse = torch.mean((y_original - y_bitfield) ** 2).item()
    rel_error = torch.norm(y_original - y_bitfield) / torch.norm(y_original)
    
    print(f"3D 텐서 MSE: {mse:.6f}")
    print(f"3D 텐서 상대 오차: {rel_error:.4%}")
    
    # 압축률 계산
    original_params = in_features * out_features * 32
    compressed_params = out_features * 22
    compression_ratio = original_params / compressed_params
    
    print(f"압축률: {compression_ratio:.1f}x")
    print("✓ 3D 텐서 지원 테스트 통과")


def test_3d_tensor_gradients():
    """3D 텐서 그래디언트 테스트"""
    print("\n=== 3D 텐서 그래디언트 테스트 ===")
    
    batch_size = 8
    seq_len = 16
    in_features = 32
    out_features = 16
    
    # 기존 Linear 레이어 생성
    linear = nn.Linear(in_features, out_features)
    
    # BitfieldLinear로 변환
    bitfield_linear = BitfieldLinear.from_linear(linear, basis_size=64, r_max=1.0)
    
    # 3D 입력 텐서 생성 (requires_grad=True)
    x_3d = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
    
    # 순전파
    y_3d = bitfield_linear(x_3d)
    
    # 간단한 손실 계산
    loss = y_3d.mean()
    
    # 역전파
    loss.backward()
    
    print(f"3D 입력 형태: {x_3d.shape}")
    print(f"3D 출력 형태: {y_3d.shape}")
    print(f"3D 입력 그래디언트 형태: {x_3d.grad.shape}")
    print(f"3D 입력 그래디언트 노름: {x_3d.grad.norm().item():.6f}")
    
    # 그래디언트 검증
    assert x_3d.grad.shape == x_3d.shape, f"그래디언트 형태가 다름: {x_3d.grad.shape} vs {x_3d.shape}"
    assert x_3d.grad.norm().item() > 0, "그래디언트가 0입니다"
    
    print("✓ 3D 텐서 그래디언트 테스트 통과")


def test_multidimensional_performance():
    """다차원 텐서 정확도 및 속도 비교"""
    print("\n=== 다차원 텐서 정확도 및 속도 비교 ===")
    
    in_features = 256
    out_features = 128
    
    # 기존 Linear 레이어 생성
    linear = nn.Linear(in_features, out_features)
    bitfield_linear = BitfieldLinear.from_linear(linear, basis_size=256, r_max=4.0)
    
    # 테스트 케이스
    test_cases = [
        ("2D", (64, in_features)),
        ("3D", (16, 32, in_features)),
        ("4D", (8, 4, 16, in_features)),
        ("5D", (4, 2, 4, 8, in_features)),
        ("6D", (2, 2, 2, 4, 8, in_features))
    ]
    
    print(f"\n{'차원':<4} {'형태':<20} {'MSE':<12} {'상대오차':<10} {'코사인유사도':<12} {'원본시간':<10} {'압축시간':<10} {'속도비':<8}")
    print("-" * 90)
    
    for name, shape in test_cases:
        # 입력 텐서 생성
        x = torch.randn(shape)
        
        # 정확도 측정 (3회 평균)
        mse_list = []
        rel_error_list = []
        cosine_sim_list = []
        
        for _ in range(3):
            with torch.no_grad():
                y_original = linear(x)
                y_bitfield = bitfield_linear(x)
            
            mse = torch.mean((y_original - y_bitfield) ** 2).item()
            rel_error = (torch.norm(y_original - y_bitfield) / torch.norm(y_original)).item()
            
            # 코사인 유사도 계산
            original_flat = y_original.view(-1)
            bitfield_flat = y_bitfield.view(-1)
            
            cosine_sim = torch.cosine_similarity(original_flat.unsqueeze(0), bitfield_flat.unsqueeze(0), dim=1).item()
            
            mse_list.append(mse)
            rel_error_list.append(rel_error)
            cosine_sim_list.append(cosine_sim)
        
        avg_mse = sum(mse_list) / len(mse_list)
        avg_rel_error = sum(rel_error_list) / len(rel_error_list)
        avg_cosine_sim = sum(cosine_sim_list) / len(cosine_sim_list)
        
        # 속도 측정 (워밍업 후 5회 평균)
        for _ in range(2):  # 워밍업
            with torch.no_grad():
                _ = linear(x)
                _ = bitfield_linear(x)
        
        # 원본 속도
        start_time = time.time()
        for _ in range(5):
            with torch.no_grad():
                _ = linear(x)
        original_time = (time.time() - start_time) / 5
        
        # 압축 속도
        start_time = time.time()
        for _ in range(5):
            with torch.no_grad():
                _ = bitfield_linear(x)
        bitfield_time = (time.time() - start_time) / 5
        
        speed_ratio = bitfield_time / original_time
        
        # 결과 출력
        shape_str = str(shape) if len(str(shape)) <= 18 else str(shape)[:15] + "..."
        print(f"{name:<4} {shape_str:<20} {avg_mse:<12.6f} {avg_rel_error:<10.1%} {avg_cosine_sim:<12.4f} "
              f"{original_time*1000:<10.3f} {bitfield_time*1000:<10.3f} {speed_ratio:<8.2f}x")
    
    print("\n* 시간 단위: ms")
    print("* 속도비 > 1.0 = 압축이 더 느림, < 1.0 = 압축이 더 빠름")
    print("* 코사인 유사도 = 원본과 비트필드 출력 간 방향 유사도")
    print("✓ 다차원 텐서 성능 비교 완료")


def test_classification_accuracy():
    """실제 분류 작업에서의 정확도 비교"""
    print("\n=== 분류 정확도 비교 테스트 ===")
    
    # 간단한 분류 모델 정의
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size=784, hidden_size=128, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    class BitfieldClassifier(nn.Module):
        def __init__(self, fc1_to_convert: nn.Linear, fc2_to_convert: nn.Linear):
            super().__init__()
            # 비트필드로 변환
            self.fc1 = BitfieldLinear.from_linear(fc1_to_convert, basis_size=256, r_max=4.0)
            self.fc2 = BitfieldLinear.from_linear(fc2_to_convert, basis_size=128, r_max=4.0)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # 모델 생성
    original_model = SimpleClassifier()
    bitfield_model = BitfieldClassifier(original_model.fc1, original_model.fc2)
    
    # 테스트 데이터 생성
    batch_size = 100
    test_data = torch.randn(batch_size, 784) * 0.5
    test_labels = torch.randint(0, 10, (batch_size,))
    
    # 모델 평가
    original_model.eval()
    bitfield_model.eval()
    
    with torch.no_grad():
        # 원본 모델 예측
        original_output = original_model(test_data)
        original_pred = torch.argmax(original_output, dim=1)
        original_accuracy = (original_pred == test_labels).float().mean().item()
        
        # 비트필드 모델 예측
        bitfield_output = bitfield_model(test_data)
        bitfield_pred = torch.argmax(bitfield_output, dim=1)
        bitfield_accuracy = (bitfield_pred == test_labels).float().mean().item()
        
        # 예측 일치도
        prediction_agreement = (original_pred == bitfield_pred).float().mean().item()
        
        # 출력 유사도
        output_cosine_sim = torch.cosine_similarity(
            original_output.view(-1).unsqueeze(0),
            bitfield_output.view(-1).unsqueeze(0),
            dim=1
        ).item()
        
        # 출력 MSE
        output_mse = torch.mean((original_output - bitfield_output) ** 2).item()
    
    print(f"원본 모델 정확도: {original_accuracy:.1%}")
    print(f"비트필드 모델 정확도: {bitfield_accuracy:.1%}")
    print(f"예측 일치도: {prediction_agreement:.1%}")
    print(f"출력 코사인 유사도: {output_cosine_sim:.4f}")
    print(f"출력 MSE: {output_mse:.6f}")
    
    # 압축률 계산
    original_params = sum(p.numel() for p in original_model.parameters()) * 32
    bitfield_params = sum(p.numel() for p in bitfield_model.parameters()) * 22
    compression_ratio = original_params / bitfield_params
    
    print(f"압축률: {compression_ratio:.1f}x")
    print("✓ 분류 정확도 비교 테스트 완료")


def test_multidimensional_training():
    """다차원 텐서 입력에 대한 분류 학습 테스트"""
    print("\n=== 다차원 텐서 분류 학습(ACC) 테스트 ===")

    # 모델 정의
    class MultiDimClassifier(nn.Module):
        def __init__(self, in_features, hidden_features, num_classes):
            super().__init__()
            fc1 = nn.Linear(in_features, hidden_features)
            self.layer1 = BitfieldLinear.from_linear(fc1, basis_size=128, r_max=4.0)
            self.relu = nn.ReLU()
            self.classifier = nn.Linear(hidden_features, num_classes)

        def forward(self, x):
            # x shape: [batch, dim1, dim2, in_features]
            hidden = self.relu(self.layer1(x))
            # 분류를 위해 차원 축소
            pooled = hidden.mean(dim=[1, 2])
            output = self.classifier(pooled)
            return output

    # 파라미터 설정
    n_samples = 500
    in_features = 32
    hidden_features = 64
    num_classes = 10
    shape = (n_samples, 5, 4, in_features)

    # 데이터 및 레이블 생성
    X = torch.randn(shape)
    y = (X.sum(dim=[1,2,3]) * 10).long() % num_classes

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델, 옵티마이저, 손실 함수
    model = MultiDimClassifier(in_features, hidden_features, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    print(f"학습 시작: 입력 형태 (batch, {shape[1]}, {shape[2]}, {shape[3]})")
    model.train()
    for epoch in range(20):
        correct, total, total_loss = 0, 0, 0
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
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/20: loss={avg_loss:.4f}, acc={accuracy:.2f}%")

    print("✓ 다차원 텐서 분류 학습(ACC) 테스트 통과")


if __name__ == "__main__":
    # 기존 테스트들 실행
    test_bitfield_forward()
    test_bitfield_backward()
    test_riemannian_functions()
    test_mnist_training()
    test_hyperbolic_mlp()
    test_deep_network()
    
    # 새로운 3D 텐서 테스트들 실행
    test_3d_tensor_support()
    test_3d_tensor_gradients()
    
    # 개선된 다차원 성능 분석
    test_multidimensional_performance()

    # 다차원 텐서 학습 테스트
    test_multidimensional_training()
    
    # 실제 분류 정확도 비교
    test_classification_accuracy() 