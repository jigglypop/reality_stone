"""
Reality Stone 고급 기능 성능 테스트
동적 곡률, 체비셰프, 측지선 활성화 등이 실제로 정확도를 올리는지 확인
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
import os

# Reality Stone import
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))
    import reality_stone as rs
    import reality_stone.advanced as advanced
    HAS_REALITY_STONE = True
except ImportError as e:
    print(f"Warning: Reality Stone import failed: {e}")
    HAS_REALITY_STONE = False

@dataclass
class TestResult:
    """테스트 결과 저장"""
    model_name: str
    accuracy: float
    loss: float
    training_time: float
    nan_count: int

class AdvancedFeatureTest:
    """고급 기능 성능 테스트"""
    
    def __init__(self, device='cpu'):  # CPU 전용으로 변경
        self.device = device
        self.results = []
        
    def setup_data(self, batch_size=256):
        """MNIST 데이터 로더 설정"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('.', train=False, transform=transform)
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def train_and_evaluate(self, model, model_name, epochs=5):
        """모델 훈련 및 평가"""
        print(f"\n{'='*50}")
        print(f"Testing: {model_name}")
        print(f"{'='*50}")
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        best_accuracy = 0
        final_loss = float('inf')
        nan_count = 0
        
        for epoch in range(epochs):
            # 훈련
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if batch_idx >= 200:  # 빠른 테스트를 위해 제한
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten
                
                optimizer.zero_grad()
                
                try:
                    output = model(data)
                    loss = criterion(output, target)
                    
                    if torch.isnan(loss):
                        nan_count += 1
                        continue
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Training error: {e}")
                    nan_count += 1
                    
            avg_loss = epoch_loss / max(batch_count, 1)
            
            # 평가
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader):
                    if batch_idx >= 50:  # 빠른 테스트
                        break
                        
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    
                    try:
                        output = model(data)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                    except:
                        pass
                        
            accuracy = 100 * correct / total if total > 0 else 0
            best_accuracy = max(best_accuracy, accuracy)
            final_loss = avg_loss
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        training_time = time.time() - start_time
        
        result = TestResult(
            model_name=model_name,
            accuracy=best_accuracy,
            loss=final_loss,
            training_time=training_time,
            nan_count=nan_count
        )
        
        self.results.append(result)
        return result

# 테스트 모델들
class BaselineMLP(nn.Module):
    """기본 MLP"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

class SimpleHyperbolicMLP(nn.Module):
    """간단한 하이퍼볼릭 MLP"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)
        
    def forward(self, x):
        h1 = torch.tanh(self.linear1(x))  # 하이퍼볼릭 활성화
        h2 = torch.tanh(self.linear2(h1))
        return self.linear3(h2)

class DynamicCurvatureMLP(nn.Module):
    """동적 곡률 MLP"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)
        
        # 동적 곡률 예측을 위한 파라미터
        self.curvature_predictor = nn.Linear(256, 1)
        
    def forward(self, x):
        h1 = self.linear1(x)
        
        # 동적 곡률 예측
        curvature = torch.sigmoid(self.curvature_predictor(h1)) + 0.1  # [0.1, 1.1]
        
        # Reality Stone 동적 곡률 적용 (안전한 방식)
        if HAS_REALITY_STONE:
            try:
                # CPU로 이동해서 처리
                h1_cpu = h1.cpu()
                h1_normalized = torch.tanh(h1_cpu)
                
                # 배치 전체에 평균 곡률 적용 (더 안전)
                avg_curvature = curvature.mean().item()
                zero_tensor = torch.zeros_like(h1_normalized)
                
                # Möbius 변환 적용
                result = rs.mobius_add(h1_normalized, zero_tensor, avg_curvature)
                h1 = result.to(x.device)
            except Exception as e:
                print(f"Dynamic curvature fallback: {e}")
                h1 = torch.tanh(h1)
        else:
            h1 = torch.tanh(h1)
            
        h2 = torch.tanh(self.linear2(h1))
        return self.linear3(h2)

class ChebyshevMLP(nn.Module):
    """체비셰프 근사 MLP"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)
        
    def chebyshev_activation(self, x, order=5):
        """체비셰프 다항식 기반 활성화"""
        if not HAS_REALITY_STONE:
            return torch.tanh(x)
            
        try:
            # CPU에서 처리
            x_cpu = x.cpu()
            result = advanced.chebyshev_approximation_cpu(x_cpu, order, 1.0)
            return result.to(x.device)
        except Exception as e:
            print(f"Chebyshev fallback: {e}")
            # 폴백: 수동 체비셰프 근사
            x_clamped = torch.clamp(x, -0.99, 0.99)
            result = torch.zeros_like(x)
            for n in range(1, order+1, 2):  # 홀수 항만
                T_n = torch.cos(n * torch.acos(x_clamped))
                coeff = 4.0 / (np.pi * (n*n - 0.25))
                result += coeff * T_n
            return torch.clamp(result, -10, 10)
        
    def forward(self, x):
        h1 = self.chebyshev_activation(self.linear1(x))
        h2 = self.chebyshev_activation(self.linear2(h1))
        return self.linear3(h2)

class GeodesicMLP(nn.Module):
    """측지선 활성화 MLP"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)
        
        # 측지선 파라미터
        self.t_param = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        h1 = self.linear1(x)
        
        # Reality Stone 측지선 레이어 적용 (안전한 방식)
        if HAS_REALITY_STONE:
            try:
                # CPU에서 처리
                h1_cpu = torch.tanh(h1).cpu()
                u_cpu = torch.tanh(self.linear2(h1_cpu))
                t = torch.sigmoid(self.t_param).item()
                
                h2_cpu = rs.poincare_ball_layer(h1_cpu, u_cpu, 1.0, t)
                h2 = h2_cpu.to(x.device)
            except Exception as e:
                print(f"Geodesic fallback: {e}")
                h2 = torch.tanh(self.linear2(torch.tanh(h1)))
        else:
            h2 = torch.tanh(self.linear2(torch.tanh(h1)))
            
        return self.linear3(h2)

class FullAdvancedMLP(nn.Module):
    """모든 고급 기능을 사용한 MLP"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)
        
        # 동적 곡률 예측
        self.curvature_predictor = nn.Linear(256, 1)
        # 측지선 파라미터
        self.t_param = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, x):
        h1 = self.linear1(x)
        
        # 1. 동적 곡률 예측
        curvature = torch.sigmoid(self.curvature_predictor(h1)) + 0.1
        avg_curvature = curvature.mean().item()
        
        # 2. 체비셰프 활성화 (안전한 방식)
        if HAS_REALITY_STONE:
            try:
                h1_cpu = h1.cpu()
                h1_activated_cpu = advanced.chebyshev_approximation_cpu(h1_cpu, 5, avg_curvature)
                h1_activated = h1_activated_cpu.to(x.device)
            except Exception as e:
                print(f"Full advanced chebyshev fallback: {e}")
                h1_activated = torch.tanh(h1)
        else:
            h1_activated = torch.tanh(h1)
        
        # 3. 측지선 레이어 (안전한 방식)
        u = self.linear2(h1_activated)
        if HAS_REALITY_STONE:
            try:
                h1_cpu = h1_activated.cpu()
                u_cpu = torch.tanh(u).cpu()
                t = torch.sigmoid(self.t_param).item()
                
                h2_cpu = rs.poincare_ball_layer(h1_cpu, u_cpu, avg_curvature, t)
                h2 = h2_cpu.to(x.device)
            except Exception as e:
                print(f"Full advanced geodesic fallback: {e}")
                h2 = torch.tanh(u)
        else:
            h2 = torch.tanh(u)
            
        return self.linear3(h2)

def run_advanced_feature_tests():
    """고급 기능 성능 테스트 실행"""
    print("="*80)
    print("Reality Stone Advanced Features Performance Test")
    print("="*80)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Reality Stone Available: {HAS_REALITY_STONE}")
    
    if HAS_REALITY_STONE:
        try:
            features = rs.get_available_features()
            print(f"\nAvailable Features:")
            for feature, available in features.items():
                status = "✅" if available else "❌"
                print(f"  {feature}: {status}")
        except:
            print("Could not check features")
    
    # 테스터 초기화
    tester = AdvancedFeatureTest()
    tester.setup_data(batch_size=128)
    
    # 테스트할 모델들
    models = [
        ("Baseline MLP", BaselineMLP()),
        ("Simple Hyperbolic", SimpleHyperbolicMLP()),
        ("Dynamic Curvature", DynamicCurvatureMLP()),
        ("Chebyshev Activation", ChebyshevMLP()),
        ("Geodesic Activation", GeodesicMLP()),
        ("Full Advanced", FullAdvancedMLP()),
    ]
    
    # 각 모델 테스트
    for model_name, model in models:
        try:
            tester.train_and_evaluate(model, model_name, epochs=5)
        except Exception as e:
            print(f"ERROR testing {model_name}: {e}")
    
    # 결과 요약
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Loss':<10} {'Time(s)':<10} {'NaN Count':<10}")
    print("-"*80)
    
    for result in tester.results:
        print(f"{result.model_name:<25} "
              f"{result.accuracy:>10.2f}% "
              f"{result.loss:>9.4f} "
              f"{result.training_time:>9.2f} "
              f"{result.nan_count:>9d}")
    
    # 최고 성능 분석
    if tester.results:
        print("\n" + "="*50)
        print("PERFORMANCE ANALYSIS")
        print("="*50)
        
        best_model = max(tester.results, key=lambda x: x.accuracy)
        baseline = next((r for r in tester.results if "Baseline" in r.model_name), None)
        
        print(f"Best Model: {best_model.model_name} ({best_model.accuracy:.2f}%)")
        
        if baseline:
            improvement = best_model.accuracy - baseline.accuracy
            print(f"Baseline: {baseline.model_name} ({baseline.accuracy:.2f}%)")
            print(f"Improvement: {improvement:+.2f}%")
            
            if improvement > 1.0:
                print("✅ Advanced features show significant improvement!")
            elif improvement > 0:
                print("⚠️ Advanced features show modest improvement")
            else:
                print("❌ Advanced features did not improve performance")
        
        # 각 고급 기능별 분석
        print(f"\nFeature Analysis:")
        for result in tester.results:
            if "Dynamic Curvature" in result.model_name and baseline:
                improvement = result.accuracy - baseline.accuracy
                print(f"  Dynamic Curvature: {improvement:+.2f}%")
            elif "Chebyshev" in result.model_name and baseline:
                improvement = result.accuracy - baseline.accuracy
                print(f"  Chebyshev Activation: {improvement:+.2f}%")
            elif "Geodesic" in result.model_name and baseline:
                improvement = result.accuracy - baseline.accuracy
                print(f"  Geodesic Activation: {improvement:+.2f}%")
    
    return tester.results

if __name__ == "__main__":
    results = run_advanced_feature_tests()
    
    print("\n" + "="*80)
    print("Test completed! Check if advanced features improve accuracy.")
    print("="*80) 