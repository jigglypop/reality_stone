"""다차원 비트필드 레이어 테스트"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset
from reality_stone.layers import BitfieldLinear

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
    test_3d_tensor_support()
    test_3d_tensor_gradients()
    test_multidimensional_performance()
    test_multidimensional_training() 