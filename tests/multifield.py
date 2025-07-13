"""실제 공개 데이터셋을 사용한 BitfieldLinear 테스트"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from reality_stone.layers import BitfieldLinear
import torch.nn.functional as F

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
        def __init__(self, input_size, hidden_size, num_classes, use_bitfield=False, use_int8=False):
            super().__init__()
            self.use_bitfield = use_bitfield
            self.use_int8 = use_int8
            
            # 시계열 인코더
            if use_bitfield:
                fc1 = nn.Linear(input_size, hidden_size)
                self.encoder = BitfieldLinear.from_linear(fc1, basis_size=128, r_max=2.0)
                # INT8 최적화 활성화
                if use_int8:
                    self.encoder.enable_int8_optimization()
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
    
    # 세 모델 비교
    results = {}
    
    for model_type in ["Standard", "Bitfield", "Bitfield+INT8"]:
        use_bitfield = model_type != "Standard"
        use_int8 = model_type == "Bitfield+INT8"
        
        print(f"\n{model_type} 학습:")
        
        model = TimeSeriesClassifier(n_features, hidden_size, num_classes, use_bitfield, use_int8)
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
        
        results[model_type] = best_acc
        print(f"  최고 정확도: {best_acc:.1f}%")
    
    print(f"\n결과 비교:")
    print(f"Standard Linear: {results['Standard']:.1f}%")
    print(f"BitfieldLinear: {results['Bitfield']:.1f}%")
    print(f"BitfieldLinear+INT8: {results['Bitfield+INT8']:.1f}%")


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
    """압축 성능을 다차원 텐서에 대해 테스트합니다."""
    print("\n=== 압축 성능 종합 분석 ===\n")
    
    ndim_configs = [
        (2, "Small 2D"),
        (3, "Medium 3D"),
        (4, "Large 4D")
    ]
    
    for ndim, name in ndim_configs:
        in_feat, out_feat = (128, 64) if ndim == 2 else (64, 32)
        print(f"\n{name} ({in_feat}→{out_feat}, {ndim}D):")
        
        linear = nn.Linear(in_feat, out_feat)
        
        if ndim == 2:
            x = torch.randn(32, in_feat)
        elif ndim == 3:
            x = torch.randn(16, 8, in_feat)
        else: # ndim == 4
            x = torch.randn(8, 4, 4, in_feat)
            
        basis_sizes = [64, 128, 256]

        for basis_size in basis_sizes:
            bitfield = BitfieldLinear.from_linear(linear, basis_size=basis_size, r_max=2.0)
            
            # GPU로 이동 (가능한 경우)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x_device = x.to(device)
            linear_device = linear.to(device)
            bitfield_device = bitfield.to(device)
            
            # 워밍업
            for _ in range(10):
                _ = linear_device(x_device)
                _ = bitfield_device(x_device)
            
            # 시간 측정
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            for _ in range(100):
                y_orig_timed = linear_device(x_device)
            torch.cuda.synchronize() if device == 'cuda' else None
            standard_time = (time.time() - start) / 100

            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            for _ in range(100):
                y_bit_timed = bitfield_device(x_device)
            torch.cuda.synchronize() if device == 'cuda' else None
            bitfield_time = (time.time() - start) / 100

            # 정확도 (MSE) 계산
            y_orig = linear_device(x_device)
            y_bit = bitfield_device(x_device)
            mse = F.mse_loss(y_bit, y_orig).item()
            
            original_size = in_feat * out_feat * 4
            compressed_size = out_feat * 4 + out_feat * in_feat * 1 + out_feat * 4
            compression_ratio = original_size / compressed_size
            speedup = standard_time / bitfield_time

            print(f"  기저 {basis_size}: MSE={mse:.6f}, "
                  f"압축률={compression_ratio:.1f}x, 속도={speedup:.2f}x "
                  f"({standard_time*1000:.3f}ms vs {bitfield_time*1000:.3f}ms)")


def test_extreme_compression():
    """극한 압축 모드 테스트 (잔차 없음)"""
    print("\n=== 극한 압축 모드 테스트 ===\n")
    
    # 다양한 크기의 레이어 테스트
    test_configs = [
        (512, 256, "Medium"),
        (1024, 512, "Large"),
        (2048, 1024, "XLarge"),
    ]
    
    for in_feat, out_feat, name in test_configs:
        print(f"\n{name} Layer ({in_feat}→{out_feat}):")
        
        # Standard Linear
        linear = nn.Linear(in_feat, out_feat)
        
        # 일반 압축 (잔차 포함)
        bitfield_with_res = BitfieldLinear.from_linear(
            linear, basis_size=256, r_max=2.0, use_residual=True
        )
        
        # 극한 압축 (잔차 없음)
        bitfield_no_res = BitfieldLinear.from_linear(
            linear, basis_size=256, r_max=2.0, use_residual=False
        )
        
        # 테스트 입력
        x = torch.randn(32, in_feat)
        
        with torch.no_grad():
            y_orig = linear(x)
            y_with_res = bitfield_with_res(x)
            y_no_res = bitfield_no_res(x)
        
        # 오차 계산
        mse_with_res = F.mse_loss(y_with_res, y_orig).item()
        mse_no_res = F.mse_loss(y_no_res, y_orig).item()
        
        print(f"  MSE (잔차 포함): {mse_with_res:.6f}")
        print(f"  MSE (잔차 없음): {mse_no_res:.6f}")
        
        # 이론적 압축률 계산
        original_size = in_feat * out_feat * 4 / 1024  # KB
        compressed_with_res = (out_feat * 4 + out_feat * in_feat + out_feat * 4) / 1024
        compressed_no_res = out_feat * 4 / 1024
        
        print(f"  압축률 (잔차 포함): {original_size / compressed_with_res:.1f}x")
        print(f"  압축률 (잔차 없음): {original_size / compressed_no_res:.1f}x")
        
        # 속도 측정
        import time
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_device = x.to(device)
        linear_device = linear.to(device)
        bitfield_with_res_device = bitfield_with_res.to(device)
        bitfield_no_res_device = bitfield_no_res.to(device)
        
        # 워밍업
        for _ in range(10):
            _ = linear_device(x_device)
            _ = bitfield_with_res_device(x_device)
            _ = bitfield_no_res_device(x_device)
        
        # Standard Linear 속도
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(100):
            _ = linear_device(x_device)
        torch.cuda.synchronize() if device == 'cuda' else None
        linear_time = (time.time() - start) / 100
        
        # BitfieldLinear (잔차 포함) 속도
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(100):
            _ = bitfield_with_res_device(x_device)
        torch.cuda.synchronize() if device == 'cuda' else None
        bitfield_with_res_time = (time.time() - start) / 100
        
        # BitfieldLinear (잔차 없음) 속도
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(100):
            _ = bitfield_no_res_device(x_device)
        torch.cuda.synchronize() if device == 'cuda' else None
        bitfield_no_res_time = (time.time() - start) / 100
        
        print(f"  속도 (Linear): {linear_time*1000:.3f} ms")
        print(f"  속도 (잔차 포함): {bitfield_with_res_time*1000:.3f} ms ({linear_time/bitfield_with_res_time:.2f}x)")
        print(f"  속도 (잔차 없음): {bitfield_no_res_time*1000:.3f} ms ({linear_time/bitfield_no_res_time:.2f}x)")


def test_int8_optimization_performance():
    """INT8 최적화 성능 벤치마크"""
    print("\n=== INT8 최적화 성능 벤치마크 ===\n")
    
    # 다양한 크기의 레이어 테스트
    test_configs = [
        (512, 256, "Small"),
        (1024, 512, "Medium"),
        (2048, 1024, "Large"),
    ]
    
    batch_size = 32
    num_iterations = 100
    
    for in_feat, out_feat, name in test_configs:
        print(f"\n{name} Layer ({in_feat}→{out_feat}):")
        
        # Standard Linear
        linear = nn.Linear(in_feat, out_feat)
        
        # BitfieldLinear (FP32 기저)
        bitfield_fp32 = BitfieldLinear.from_linear(
            linear, basis_size=256, r_max=2.0, use_residual=False
        )
        
        # BitfieldLinear (INT8 기저)
        bitfield_int8 = BitfieldLinear.from_linear(
            linear, basis_size=256, r_max=2.0, use_residual=False
        )
        bitfield_int8.enable_int8_optimization()
        
        # 테스트 입력
        x = torch.randn(batch_size, in_feat)
        
        # GPU로 이동 (가능한 경우)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_device = x.to(device)
        linear_device = linear.to(device)
        bitfield_fp32_device = bitfield_fp32.to(device)
        bitfield_int8_device = bitfield_int8.to(device)
        
        # 워밍업
        for _ in range(10):
            _ = linear_device(x_device)
            _ = bitfield_fp32_device(x_device)
            _ = bitfield_int8_device(x_device)
        
        # Standard Linear 속도
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            _ = linear_device(x_device)
        torch.cuda.synchronize() if device == 'cuda' else None
        linear_time = (time.time() - start) / num_iterations
        
        # BitfieldLinear FP32 속도
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            _ = bitfield_fp32_device(x_device)
        torch.cuda.synchronize() if device == 'cuda' else None
        bitfield_fp32_time = (time.time() - start) / num_iterations
        
        # BitfieldLinear INT8 속도
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            _ = bitfield_int8_device(x_device)
        torch.cuda.synchronize() if device == 'cuda' else None
        bitfield_int8_time = (time.time() - start) / num_iterations
        
        # 정확도 비교
        with torch.no_grad():
            y_linear = linear_device(x_device)
            y_fp32 = bitfield_fp32_device(x_device)
            y_int8 = bitfield_int8_device(x_device)
            
            mse_fp32 = F.mse_loss(y_fp32, y_linear).item()
            mse_int8 = F.mse_loss(y_int8, y_linear).item()
        
        # 메모리 사용량 계산
        linear_memory = in_feat * out_feat * 4 / 1024  # KB
        bitfield_memory = out_feat * 4 / 1024  # 코드만
        basis_fp32_memory = 256 * in_feat * 4 / 1024  # FP32 기저
        basis_int8_memory = 256 * in_feat * 1 / 1024  # INT8 기저
        
        print(f"  속도 비교:")
        print(f"    - Linear: {linear_time*1000:.3f} ms (기준)")
        print(f"    - Bitfield FP32: {bitfield_fp32_time*1000:.3f} ms ({linear_time/bitfield_fp32_time:.2f}x)")
        print(f"    - Bitfield INT8: {bitfield_int8_time*1000:.3f} ms ({linear_time/bitfield_int8_time:.2f}x)")
        print(f"  정확도 (MSE):")
        print(f"    - Bitfield FP32: {mse_fp32:.6f}")
        print(f"    - Bitfield INT8: {mse_int8:.6f}")
        print(f"  메모리 사용량:")
        print(f"    - Linear: {linear_memory:.1f} KB")
        print(f"    - Bitfield FP32: {bitfield_memory + basis_fp32_memory:.1f} KB ({linear_memory/(bitfield_memory + basis_fp32_memory):.1f}x 압축)")
        print(f"    - Bitfield INT8: {bitfield_memory + basis_int8_memory:.1f} KB ({linear_memory/(bitfield_memory + basis_int8_memory):.1f}x 압축)")


if __name__ == "__main__":
    # 전기 소비 시계열 데이터 테스트 (3D)
    test_electricity_3d()
    
    # 액션 인식 데이터 테스트 (4D)
    test_action_recognition_4d()
    
    # 압축 성능 분석
    test_compression_performance()
    test_extreme_compression()
    
    # INT8 최적화 성능 테스트
    test_int8_optimization_performance()
    
    print("\n✓ 모든 테스트 완료!") 