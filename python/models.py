import torch
import torch.nn as nn
import reality_stone as rs

class LorentzMLP(nn.Module):
    """
    로렌츠 모델을 사용한 다층 퍼셉트론
    
    로렌츠 모델은 (n+1)차원 민코프스키 공간에서 초곡면 상의 점들을 사용합니다.
    특징:
    - 수치적으로 안정적
    - 곡률 계산이 정확함
    - 차원이 n+1로 증가함 (시간 성분 추가)
    """
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        
        # 입력 차원 -> 은닉 차원 투영
        # 로렌츠 모델은 시간 성분이 1차원 추가됨
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        
        # 로렌츠 공간 내부의 변환
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        
        # 출력 차원으로 투영
        self.out_weights = nn.Parameter(torch.randn(hid+1, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # 입력 형태 변환
        x = x.view(x.size(0), -1)
        
        # 첫 번째 레이어 (일반 선형 변환)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        
        # 두 번째 레이어 (다시 일반 선형 변환)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 로렌츠 공간으로 변환 (3D → 4D)
        # 푸앵카레 → 로렌츠 좌표계 변환
        lorentz_h = rs.poincare_to_lorentz(h, self.c)
        lorentz_u = rs.poincare_to_lorentz(u, self.c)
        
        # 로렌츠 측지선 연산
        lorentz_z = rs.lorentz_layer(lorentz_h, lorentz_u, self.c, self.t)
        
        # NaN 처리
        if torch.isnan(lorentz_z).any():
            lorentz_z = lorentz_h
        
        # 출력 레이어
        output = lorentz_z @ self.out_weights + self.out_bias
        return output

class KleinMLP(nn.Module):
    """
    클라인 모델을 사용한 다층 퍼셉트론
    
    클라인 모델은 쌍곡 공간을 유클리드 공간의 원판으로 표현합니다.
    특징:
    - 측지선이 직선 세그먼트로 표현됨
    - 모델 간 변환이 비교적 용이함
    - 구현이 단순함
    """
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        
        # 입력 차원 -> 은닉 차원 투영
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        
        # 클라인 공간 내부의 변환
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        
        # 출력 차원으로 투영
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # 입력 형태 변환
        x = x.view(x.size(0), -1)
        
        # 첫 번째 레이어
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        
        # 두 번째 레이어
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        
        # 푸앵카레에서 클라인 좌표계로 변환
        klein_h = rs.poincare_to_klein(h, self.c)
        klein_u = rs.poincare_to_klein(u, self.c)
        
        # 클라인 측지선 연산
        klein_z = rs.klein_layer(klein_h, klein_u, self.c, self.t)
        
        # NaN 처리
        if torch.isnan(klein_z).any():
            klein_z = klein_h
        
        # 출력 레이어
        output = klein_z @ self.out_weights + self.out_bias
        return output

class HybridHyperbolicMLP(nn.Module):
    """
    여러 하이퍼볼릭 모델을 조합한 하이브리드 MLP
    
    각 모델의 장점을 활용하여 성능을 최적화할 수 있습니다.
    """
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7, mode='auto'):
        super().__init__()
        self.c = c
        self.t = t
        self.mode = mode
        
        # 입력 차원 -> 은닉 차원
        self.fc1 = nn.Linear(in_dim, hid)
        self.bn1 = nn.BatchNorm1d(hid)
        
        # 하이퍼볼릭 변환 레이어
        self.fc2 = nn.Linear(hid, hid)
        self.bn2 = nn.BatchNorm1d(hid)
        
        # 출력 레이어 - 로렌츠 모델은 차원이 하나 더 크므로 입력 차원을 조정
        self.lorentz_out = nn.Linear(hid+1, out_dim)
        self.poincare_out = nn.Linear(hid, out_dim)
        self.klein_out = nn.Linear(hid, out_dim)

    def forward(self, x):
        device = x.device
        
        # 입력 형태 변환
        x = x.view(x.size(0), -1)
        
        # 첫 번째 레이어
        h = self.fc1(x)
        h = self.bn1(h)
        h = torch.tanh(h)
        
        # 두 번째 레이어
        u = self.fc2(h)
        u = self.bn2(u)
        u = torch.sigmoid(u)
        
        # 모드에 따라 다른 하이퍼볼릭 모델 사용
        if self.mode == 'poincare' or (self.mode == 'auto' and h.size(1) < 64):
            # 푸앵카레 볼 모델 (차원이 적을 때 효율적)
            z = rs.poincare_ball_layer(h, u, self.c, self.t)
            # NaN 처리
            if torch.isnan(z).any():
                z = h
            output = self.poincare_out(z)
            
        elif self.mode == 'lorentz' or (self.mode == 'auto' and h.size(1) >= 64):
            # 로렌츠 모델 (높은 차원에서 수치적으로 안정적)
            lorentz_h = rs.poincare_to_lorentz(h, self.c)
            lorentz_u = rs.poincare_to_lorentz(u, self.c)
            
            lorentz_z = rs.lorentz_layer(lorentz_h, lorentz_u, self.c, self.t)
            # NaN 처리
            if torch.isnan(lorentz_z).any():
                lorentz_z = lorentz_h
                
            output = self.lorentz_out(lorentz_z)
            
        elif self.mode == 'klein':
            # 클라인 모델 (직선 측지선)
            klein_h = rs.poincare_to_klein(h, self.c)
            klein_u = rs.poincare_to_klein(u, self.c)
            
            klein_z = rs.klein_layer(klein_h, klein_u, self.c, self.t)
            # NaN 처리
            if torch.isnan(klein_z).any():
                klein_z = klein_h
                
            output = self.klein_out(klein_z)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return output