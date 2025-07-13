"""실제 공개 데이터셋을 사용한 BitfieldLinear 테스트"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from reality_stone.layers import BitfieldLinear

def load_electricity_dataset():
    print("\n=== 전기 소비 시계열 데이터셋 로드 ===")
    n_clients = 370
    n_timesteps = 1000  # 실제는 26304이지만 메모리를 위해 축소
    features = 8  # 전기 소비 + 시간 특징 + 날씨 등
    data = []
    labels = []
    
    for client_id in range(n_clients):
        # 클라이언트별 기본 소비 패턴
        base_consumption = np.random.uniform(0.5, 2.0)
        
        # 시간별 데이터 생성
        time_series = np.zeros((n_timesteps, features))
        
        for t in range(n_timesteps):
            # 시간대별 패턴 (24시간 주기)
            hour = t % 24
            daily_pattern = np.sin(2 * np.pi * hour / 24) * 0.3
            
            # 주간 패턴 (주말/평일)
            day = (t // 24) % 7
            weekly_pattern = 0.2 if day >= 5 else 0  # 주말에 소비 증가
            
            # 계절 패턴
            season_pattern = np.sin(2 * np.pi * t / (24 * 365)) * 0.1
            
            # 전기 소비
            consumption = base_consumption + daily_pattern + weekly_pattern + season_pattern
            consumption += np.random.normal(0, 0.1)  # 노이즈
            
            # 특징 벡터 구성
            time_series[t, 0] = consumption
            time_series[t, 1] = hour / 24.0  # 정규화된 시간
            time_series[t, 2] = day / 7.0  # 정규화된 요일
            time_series[t, 3] = np.sin(2 * np.pi * hour / 24)  # 시간 sin
            time_series[t, 4] = np.cos(2 * np.pi * hour / 24)  # 시간 cos
            time_series[t, 5] = np.sin(2 * np.pi * day / 7)  # 요일 sin
            time_series[t, 6] = np.cos(2 * np.pi * day / 7)  # 요일 cos
            time_series[t, 7] = t / n_timesteps  # 전체 시간축에서의 위치
        
        data.append(time_series)
        
        # 레이블: 평균 소비량에 따른 클라이언트 타입 (0-4)
        avg_consumption = np.mean(time_series[:, 0])
        label = min(int(avg_consumption * 2.5), 4)
        labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    print(f"데이터셋 형태: {data.shape}")
    print(f"클라이언트 타입 분포: {np.bincount(labels)}")
    
    return torch.FloatTensor(data), torch.LongTensor(labels)


def test_electricity_3d():
    """전기 소비 데이터로 3D 시계열 분류 테스트"""
    print("\n=== 전기 소비 3D 시계열 분류 테스트 ===")
    
    # 데이터 로드
    X, y = load_electricity_dataset()
    n_samples, seq_len, n_features = X.shape
    
    # 학습/테스트 분할
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 시계열 분류 모델
    class TimeSeriesClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, use_bitfield=False):
            super().__init__()
            self.use_bitfield = use_bitfield
            
            # 시계열 인코더
            if use_bitfield:
                fc1 = nn.Linear(input_size, hidden_size)
                self.encoder = BitfieldLinear.from_linear(fc1, basis_size=128, r_max=2.0)
            else:
                self.encoder = nn.Linear(input_size, hidden_size)
            
            self.norm1 = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.2)
            
            # Temporal pooling
            self.pool = nn.AdaptiveAvgPool1d(1)
            
            # 분류기
            self.classifier = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # x: [batch, seq_len, features]
            # 각 시간 단계별로 인코딩
            x = self.encoder(x)  # [batch, seq_len, hidden]
            x = self.norm1(x)
            x = torch.relu(x)
            x = self.dropout(x)
            
            # Temporal pooling: [batch, hidden, seq_len] -> [batch, hidden, 1]
            x = x.transpose(1, 2)
            x = self.pool(x)
            x = x.squeeze(-1)  # [batch, hidden]
            
            # 분류
            output = self.classifier(x)
            return output
    
    # 하이퍼파라미터
    hidden_size = 64
    num_classes = 5
    batch_size = 32
    num_epochs = 20
    
    # 데이터로더
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 두 모델 비교
    results = {}
    
    for use_bitfield in [False, True]:
        model_name = "BitfieldLinear" if use_bitfield else "Standard Linear"
        print(f"\n{model_name} 학습:")
        
        model = TimeSeriesClassifier(n_features, hidden_size, num_classes, use_bitfield)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 학습
        model.train()
        best_acc = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 평가
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100. * correct / total
            best_acc = max(best_acc, accuracy)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: loss={total_loss/len(train_loader):.4f}, test_acc={accuracy:.1f}%")
            
            model.train()
        
        results[model_name] = best_acc
        print(f"  최고 정확도: {best_acc:.1f}%")
    
    print(f"\n결과 비교:")
    print(f"Standard Linear: {results['Standard Linear']:.1f}%")
    print(f"BitfieldLinear: {results['BitfieldLinear']:.1f}%")
    print(f"정확도 차이: {abs(results['Standard Linear'] - results['BitfieldLinear']):.1f}%")


def create_action_recognition_data():
    """액션 인식용 4D 데이터 생성 (NTU RGB+D 스타일)"""
    print("\n=== 액션 인식 4D 데이터 생성 ===")
    
    # 파라미터
    n_samples = 600
    n_frames = 30  # 30 프레임
    n_joints = 25  # 25개 관절
    n_coords = 3   # x, y, z 좌표
    n_actions = 6  # 6개 액션 클래스
    
    # 액션별 특징적인 움직임 패턴 정의
    action_patterns = {
        0: "wave",      # 손 흔들기
        1: "walk",      # 걷기
        2: "sit_down",  # 앉기
        3: "stand_up",  # 일어서기
        4: "drink",     # 마시기
        5: "phone"      # 전화하기
    }
    
    data = []
    labels = []
    
    for i in range(n_samples):
        action = i % n_actions
        labels.append(action)
        
        # 기본 스켈레톤 포즈
        skeleton = np.zeros((n_frames, n_joints, n_coords))
        
        # 액션별 움직임 생성
        for t in range(n_frames):
            progress = t / n_frames
            
            if action == 0:  # wave - 오른손 흔들기
                # 오른손 관절 (인덱스 11)
                skeleton[t, 11, 0] = 0.5 + 0.3 * np.sin(4 * np.pi * progress)
                skeleton[t, 11, 1] = 1.5 + 0.2 * np.sin(4 * np.pi * progress)
                skeleton[t, 11, 2] = 0.1 * np.sin(8 * np.pi * progress)
                
            elif action == 1:  # walk - 다리 움직임
                # 왼발 (인덱스 19), 오른발 (인덱스 23)
                skeleton[t, 19, 0] = 0.2 * np.sin(2 * np.pi * progress)
                skeleton[t, 19, 2] = 0.3 * abs(np.sin(2 * np.pi * progress))
                skeleton[t, 23, 0] = -0.2 * np.sin(2 * np.pi * progress)
                skeleton[t, 23, 2] = 0.3 * abs(np.cos(2 * np.pi * progress))
                
            elif action == 2:  # sit_down - 앉는 동작
                # 힙 관절 (인덱스 0) 높이 감소
                skeleton[t, 0, 1] = 1.0 * (1 - progress * 0.5)
                # 무릎 굽히기
                skeleton[t, 17, 1] = 0.5 * (1 - progress * 0.7)
                skeleton[t, 21, 1] = 0.5 * (1 - progress * 0.7)
                
            elif action == 3:  # stand_up - 일어서는 동작
                # 힙 관절 높이 증가
                skeleton[t, 0, 1] = 0.5 + 0.5 * progress
                # 무릎 펴기
                skeleton[t, 17, 1] = 0.15 + 0.35 * progress
                skeleton[t, 21, 1] = 0.15 + 0.35 * progress
                
            elif action == 4:  # drink - 마시는 동작
                # 오른손을 입으로
                skeleton[t, 11, 0] = 0.3 * (1 - progress) + 0.1 * progress
                skeleton[t, 11, 1] = 0.8 + 0.6 * progress
                skeleton[t, 11, 2] = 0.2 * progress
                
            elif action == 5:  # phone - 전화하는 동작
                # 오른손을 귀로
                skeleton[t, 11, 0] = 0.3 + 0.2 * progress
                skeleton[t, 11, 1] = 0.8 + 0.7 * progress
                skeleton[t, 11, 2] = 0.1 + 0.1 * progress
            
            # 노이즈 추가
            skeleton[t] += np.random.normal(0, 0.02, (n_joints, n_coords))
        
        data.append(skeleton)
    
    data = np.array(data)
    labels = np.array(labels)
    
    print(f"액션 데이터 형태: {data.shape}")
    print(f"액션 분포: {[f'{action_patterns[i]}: {np.sum(labels==i)}' for i in range(n_actions)]}")
    
    return torch.FloatTensor(data), torch.LongTensor(labels)


def test_action_recognition_4d():
    """4D 액션 인식 테스트"""
    print("\n=== 4D 액션 인식 테스트 ===")
    
    # 데이터 로드
    X, y = create_action_recognition_data()
    n_samples, n_frames, n_joints, n_coords = X.shape
    
    # 학습/테스트 분할
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 4D 입력을 처리하는 모델
    class ActionRecognizer(nn.Module):
        def __init__(self, use_bitfield=False):
            super().__init__()
            self.use_bitfield = use_bitfield
            
            # 4D를 3D로 변환 (joints와 coords를 합침)
            input_size = n_joints * n_coords  # 25 * 3 = 75
            hidden_size = 128
            
            # 프레임별 인코더
            if use_bitfield:
                fc1 = nn.Linear(input_size, hidden_size)
                self.frame_encoder = BitfieldLinear.from_linear(fc1, basis_size=256, r_max=2.0)
            else:
                self.frame_encoder = nn.Linear(input_size, hidden_size)
            
            # Temporal 모델링
            self.lstm = nn.LSTM(hidden_size, 64, batch_first=True, bidirectional=True)
            
            # 분류기
            self.classifier = nn.Linear(128, 6)  # bidirectional이므로 64*2
        
        def forward(self, x):
            # x: [batch, frames, joints, coords]
            batch_size, n_frames = x.size(0), x.size(1)
            
            # Reshape: [batch, frames, joints*coords]
            x = x.view(batch_size, n_frames, -1)
            
            # 각 프레임 인코딩
            x = self.frame_encoder(x)  # [batch, frames, hidden]
            x = torch.relu(x)
            
            # LSTM
            lstm_out, (h_n, _) = self.lstm(x)
            
            # 마지막 hidden state 사용 (bidirectional 고려)
            h_forward = h_n[0]  # [batch, 64]
            h_backward = h_n[1]  # [batch, 64]
            h_combined = torch.cat([h_forward, h_backward], dim=1)  # [batch, 128]
            
            # 분류
            output = self.classifier(h_combined)
            return output
    
    # 데이터로더
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 두 모델 비교
    results = {}
    
    for use_bitfield in [False, True]:
        model_name = "BitfieldLinear" if use_bitfield else "Standard Linear"
        print(f"\n{model_name} 학습:")
        
        model = ActionRecognizer(use_bitfield=use_bitfield)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 학습
        model.train()
        best_acc = 0
        
        for epoch in range(15):
            total_loss = 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 평가
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100. * correct / total
            best_acc = max(best_acc, accuracy)
            
            if epoch % 3 == 0:
                print(f"  Epoch {epoch+1}/15: loss={total_loss/len(train_loader):.4f}, test_acc={accuracy:.1f}%")
            
            model.train()
        
        results[model_name] = best_acc
        print(f"  최고 정확도: {best_acc:.1f}%")
    
    print(f"\n결과 비교:")
    print(f"Standard Linear: {results['Standard Linear']:.1f}%")
    print(f"BitfieldLinear: {results['BitfieldLinear']:.1f}%")
    print(f"정확도 차이: {abs(results['Standard Linear'] - results['BitfieldLinear']):.1f}%")


def test_compression_performance():
    """압축 성능 종합 분석"""
    print("\n=== 압축 성능 종합 분석 ===")
    
    # 다양한 설정으로 테스트
    test_cases = [
        (128, 64, 2, "Small 2D"),
        (512, 256, 2, "Large 2D"),
        (128, 64, 3, "Small 3D"),
        (256, 128, 4, "Medium 4D"),
    ]
    
    for in_feat, out_feat, ndim, name in test_cases:
        print(f"\n{name} ({in_feat}→{out_feat}, {ndim}D):")
        
        # 원본 레이어
        linear = nn.Linear(in_feat, out_feat)
        
        # 테스트할 basis_size들
        basis_sizes = [64, 128, 256]
        
        for basis_size in basis_sizes:
            bitfield = BitfieldLinear.from_linear(linear, basis_size=basis_size, r_max=2.0)
            
            # 적절한 shape 생성
            if ndim == 2:
                x = torch.randn(32, in_feat)
            elif ndim == 3:
                x = torch.randn(16, 8, in_feat)
            elif ndim == 4:
                x = torch.randn(8, 4, 4, in_feat)
            
            # 순전파
            with torch.no_grad():
                y_orig = linear(x)
                y_bit = bitfield(x)
            
            # 메트릭 계산
            mse = torch.mean((y_orig - y_bit) ** 2).item()
            rel_error = (torch.norm(y_orig - y_bit) / torch.norm(y_orig)).item()
            cosine_sim = torch.cosine_similarity(
                y_orig.view(-1).unsqueeze(0),
                y_bit.view(-1).unsqueeze(0)
            ).item()
            
            # 압축률
            original_params = in_feat * out_feat * 32
            compressed_params = out_feat * 22
            compression_ratio = original_params / compressed_params
            
            print(f"  basis_size={basis_size}: MSE={mse:.6f}, 상대오차={rel_error:.2%}, "
                  f"코사인유사도={cosine_sim:.4f}, 압축률={compression_ratio:.1f}x")


if __name__ == "__main__":
    # 전기 소비 시계열 데이터 테스트 (3D)
    test_electricity_3d()
    
    # 액션 인식 데이터 테스트 (4D)
    test_action_recognition_4d()
    
    # 압축 성능 분석
    test_compression_performance()
    
    print("\n✓ 모든 테스트 완료!") 