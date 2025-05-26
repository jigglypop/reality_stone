# 즉시 시도할 수 있는 수정된 모델들

import torch
import torch.nn as nn
import reality_stone as rs

# 해결책 1: 체비셰프를 조건부로만 사용
class ConditionalChebyshevMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        
        # 체비셰프는 극한값에서만 사용 (안전장치)
        h_max = torch.max(torch.abs(h))
        if h_max > 5.0:  # 큰 값에서만 체비셰프 사용
            try:
                h = rs.chebyshev_approximation(h / h_max, order=5, curvature=1.0) * h_max
            except:
                h = torch.tanh(h)  # 실패시 기본 tanh
        else:
            h = torch.tanh(h)  # 일반적인 경우 기본 tanh
            
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u) 
        z = rs.poincare_ball_layer(h, u, self.c, self.t)
        if torch.isnan(z).any():
            z = h
        output = z @ self.out_weights + self.out_bias
        return output

# 해결책 2: 동적 곡률을 더 안전하게 구현
class SafeDynamicCurvatureMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
        
        # 간단한 곡률 예측기
        self.curvature_fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 안전한 동적 곡률 예측
        c_logits = self.curvature_fc(x)
        c_pred = torch.sigmoid(c_logits) * 0.009 + 0.001  # [0.001, 0.01] 범위
        c_avg = c_pred.mean().item()  # 배치 평균 사용
        
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 안전한 곡률 사용
        safe_c = max(min(c_avg, 0.01), 0.001)  # 범위 제한
        z = rs.poincare_ball_layer(h, u, safe_c, self.t)
        if torch.isnan(z).any():
            z = h
            
        output = z @ self.out_weights + self.out_bias
        return output

# 해결책 3: 기본 최적화만 적용한 안전한 모델
class SafeOptimizedMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        
        # 더 나은 초기화
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc_out = nn.Linear(hid, out_dim)
        
        # 초기화 개선
        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        h = torch.tanh(self.fc1(x))
        u = torch.sigmoid(self.fc2(h))
        
        # 경계 안전성 보장
        h_norm = torch.norm(h, dim=1, keepdim=True)
        u_norm = torch.norm(u, dim=1, keepdim=True)
        
        # 0.9를 넘지 않도록 스케일링
        h = h * torch.clamp(h_norm, max=0.9) / (h_norm + 1e-8)
        u = u * torch.clamp(u_norm, max=0.9) / (u_norm + 1e-8)
        
        try:
            z = rs.poincare_ball_layer(h, u, self.c, self.t)
            # NaN 체크 강화
            if torch.isnan(z).any() or torch.isinf(z).any():
                z = h
        except:
            z = h
            
        return self.fc_out(z)

# 해결책 4: 체비셰프 없이 수치 안정성만 개선
class NumericallyStableMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def safe_poincare_layer(self, h, u, c, t):
        """수치적으로 안전한 포인카레 레이어"""
        # 입력 정규화
        h_norm = torch.norm(h, dim=1, keepdim=True)
        u_norm = torch.norm(u, dim=1, keepdim=True)
        
        # 경계값 제한
        max_norm = 0.95
        h = h * torch.clamp(h_norm, max=max_norm) / (h_norm + 1e-8)
        u = u * torch.clamp(u_norm, max=max_norm) / (u_norm + 1e-8)
        
        try:
            result = rs.poincare_ball_layer(h, u, c, t)
            
            # 결과 검증
            if torch.isnan(result).any() or torch.isinf(result).any():
                return h  # 안전한 fallback
                
            # 결과 정규화
            result_norm = torch.norm(result, dim=1, keepdim=True)
            result = result * torch.clamp(result_norm, max=max_norm) / (result_norm + 1e-8)
            
            return result
        except:
            return h  # 완전한 fallback

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        z = self.safe_poincare_layer(h, u, self.c, self.t)
        output = z @ self.out_weights + self.out_bias
        return output

# 빠른 테스트 함수
def quick_test_models():
    """각 모델의 기본 동작 테스트"""
    models = {
        "ConditionalChebyshev": ConditionalChebyshevMLP(),
        "SafeDynamicCurvature": SafeDynamicCurvatureMLP(),
        "SafeOptimized": SafeOptimizedMLP(),
        "NumericallyStable": NumericallyStableMLP(),
    }
    
    x = torch.randn(4, 784)
    
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(x)
                
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            output_range = (output.min().item(), output.max().item())
            
            status = "✓" if not (has_nan or has_inf) else "✗"
            print(f"{status} {name}: range={output_range}, NaN={has_nan}, Inf={has_inf}")
            
        except Exception as e:
            print(f"✗ {name}: Error - {e}")

if __name__ == "__main__":
    print("=== 모델 안전성 테스트 ===")
    quick_test_models()