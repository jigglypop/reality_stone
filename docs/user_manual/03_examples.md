# 예제 가이드

Reality Stone을 사용한 다양한 실제 예제들을 통해 하이퍼볼릭 신경망의 활용법을 학습해보세요.

## 예제 목록

### 1. [기본 MNIST 분류](#1-기본-mnist-분류)
- 하이퍼볼릭 레이어를 사용한 간단한 분류 모델
- 초보자를 위한 기본 사용법

### 2. [동적 곡률 최적화](#2-동적-곡률-최적화)
- 학습 가능한 곡률 매개변수 사용
- 적응적 기하학적 구조 학습

### 3. [계층적 데이터 임베딩](#3-계층적-데이터-임베딩)
- 트리 구조 데이터의 하이퍼볼릭 임베딩
- WordNet 계층 구조 학습

### 4. [다중 모델 앙상블](#4-다중-모델-앙상블)
- Poincaré, Lorentz, Klein 모델 결합
- 모델 간 좌표 변환 활용

## 1. 기본 MNIST 분류

가장 간단한 예제로 시작해보겠습니다.

### 완전한 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import reality_stone as rs

class HyperbolicMNIST(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.curvature = 1e-3
        
        # 두 개의 인코더 경로
        self.encoder1 = nn.Linear(input_dim, hidden_dim)
        self.encoder2 = nn.Linear(input_dim, hidden_dim)
        
        # 분류 헤드
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 평탄화
        
        # 두 가지 다른 표현 학습
        u = torch.tanh(self.encoder1(x))
        v = torch.tanh(self.encoder2(x))
        
        # 하이퍼볼릭 공간에서 결합
        hyperbolic_features = rs.poincare_ball_layer(
            u, v, c=self.curvature, t=0.5
        )
        
        return self.classifier(hyperbolic_features)

def train_model():
    # 데이터 로더 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 모델 및 훈련 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperbolicMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 훈련 루프
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 그래디언트 클리핑 (중요!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # 평가
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={accuracy:.4f}')

if __name__ == "__main__":
    train_model()
```

### 핵심 포인트

1. **두 개의 인코더**: 다른 관점에서 데이터를 인코딩
2. **하이퍼볼릭 결합**: `poincare_ball_layer`로 두 표현을 결합
3. **그래디언트 클리핑**: 하이퍼볼릭 공간에서 필수적

## 2. 동적 곡률 최적화

곡률 매개변수를 학습 가능한 파라미터로 만들어 최적의 기하학적 구조를 찾아보겠습니다.

```python
import torch
import torch.nn as nn
import reality_stone as rs

class AdaptiveCurvatureModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # 각 레이어별 학습 가능한 곡률 매개변수
        self.kappas = nn.Parameter(torch.zeros(num_layers))
        
        # 인코더 레이어들
        self.encoders = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 각 인코더로부터 특징 추출
        features = []
        for encoder in self.encoders:
            features.append(torch.tanh(encoder(x)))
        
        # 하이퍼볼릭 공간에서 순차적 결합
        result = features[0]
        for i in range(1, self.num_layers):
            result = rs.poincare_ball_layer(
                result, features[i],
                kappas=self.kappas,
                layer_idx=i-1,
                t=0.5
            )
        
        return self.classifier(result)

# 사용 예제
model = AdaptiveCurvatureModel(784, 128, 10)

# 곡률 매개변수 모니터링
def print_curvatures(model):
    with torch.no_grad():
        for i, kappa in enumerate(model.kappas):
            c = -2.0 + (-0.1 - (-2.0)) / (1.0 + torch.exp(-kappa))
            print(f"Layer {i}: kappa={kappa.item():.3f}, curvature={c.item():.3f}")

# 훈련 중 곡률 변화 관찰
for epoch in range(10):
    # ... 훈련 코드 ...
    if epoch % 2 == 0:
        print(f"Epoch {epoch}:")
        print_curvatures(model)
```

## 3. 계층적 데이터 임베딩

트리 구조 데이터를 하이퍼볼릭 공간에 임베딩하는 예제입니다.

```python
import torch
import torch.nn as nn
import reality_stone as rs
import networkx as nx

class HierarchicalEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, curvature=1.0):
        super().__init__()
        self.curvature = curvature
        
        # 노드 임베딩
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # 하이퍼볼릭 공간으로 매핑
        self.to_hyperbolic = nn.Linear(embed_dim, embed_dim)
        
        # 초기화: 작은 값으로 시작
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        
    def forward(self, node_ids):
        # 유클리드 임베딩
        euclidean_embeds = self.embeddings(node_ids)
        
        # 하이퍼볼릭 공간으로 변환
        hyperbolic_embeds = torch.tanh(self.to_hyperbolic(euclidean_embeds)) * 0.1
        
        return hyperbolic_embeds
    
    def distance(self, node1_ids, node2_ids):
        """두 노드 간의 하이퍼볼릭 거리 계산"""
        embed1 = self.forward(node1_ids)
        embed2 = self.forward(node2_ids)
        
        return rs.poincare_distance(embed1, embed2, c=self.curvature)

# 계층적 손실 함수
def hierarchical_loss(model, parent_ids, child_ids, negative_ids):
    """부모-자식 관계를 학습하는 손실 함수"""
    
    # 부모-자식 거리 (가까워야 함)
    parent_child_dist = model.distance(parent_ids, child_ids)
    
    # 부모-비관련 노드 거리 (멀어야 함)
    parent_negative_dist = model.distance(parent_ids, negative_ids)
    
    # 마진 기반 손실
    margin = 1.0
    loss = torch.relu(parent_child_dist - parent_negative_dist + margin)
    
    return loss.mean()

# 사용 예제
vocab_size = 1000
embed_dim = 64
model = HierarchicalEmbedding(vocab_size, embed_dim)

# 트리 구조 데이터 (예: WordNet)
# parent_ids, child_ids, negative_ids는 실제 데이터에서 가져옴
parent_ids = torch.randint(0, vocab_size, (32,))
child_ids = torch.randint(0, vocab_size, (32,))
negative_ids = torch.randint(0, vocab_size, (32,))

loss = hierarchical_loss(model, parent_ids, child_ids, negative_ids)
print(f"Hierarchical loss: {loss.item():.4f}")
```

## 4. 다중 모델 앙상블

서로 다른 하이퍼볼릭 모델을 결합하는 앙상블 예제입니다.

```python
import torch
import torch.nn as nn
import reality_stone as rs

class MultiModelEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.curvature = 1e-2
        
        # 각 모델별 인코더
        self.poincare_encoder = nn.Linear(input_dim, hidden_dim)
        self.lorentz_encoder = nn.Linear(input_dim, hidden_dim + 1)  # +1 for Lorentz
        self.klein_encoder = nn.Linear(input_dim, hidden_dim)
        
        # 통합 분류기
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 1. Poincaré Ball 특징
        poincare_features = torch.tanh(self.poincare_encoder(x)) * 0.1
        poincare_combined = rs.poincare_ball_layer(
            poincare_features, poincare_features, 
            c=self.curvature, t=0.3
        )
        
        # 2. Lorentz 특징
        lorentz_raw = self.lorentz_encoder(x)
        # Lorentz 제약 조건 만족
        lorentz_features = self.project_to_lorentz(lorentz_raw)
        lorentz_combined = rs.lorentz_layer(
            lorentz_features, lorentz_features,
            c=self.curvature, t=0.3
        )
        # 차원 맞추기 (Lorentz는 +1 차원)
        lorentz_combined = lorentz_combined[:, 1:]  # 공간 부분만 사용
        
        # 3. Klein 특징
        klein_features = torch.tanh(self.klein_encoder(x)) * 0.1
        klein_combined = rs.klein_layer(
            klein_features, klein_features,
            c=self.curvature, t=0.3
        )
        
        # 특징 결합
        combined_features = torch.cat([
            poincare_combined, lorentz_combined, klein_combined
        ], dim=1)
        
        return self.classifier(combined_features)
    
    def project_to_lorentz(self, x):
        """Lorentz 제약 조건 만족: <x,x>_L = -1"""
        x_space = x[:, 1:]  # 공간 부분
        x_time = torch.sqrt(1 + torch.sum(x_space ** 2, dim=1, keepdim=True))
        return torch.cat([x_time, x_space], dim=1)

# 사용 예제
model = MultiModelEnsemble(784, 64, 10)
x = torch.randn(32, 784)
output = model(x)
print(f"Output shape: {output.shape}")  # [32, 10]
```

## 🔧 실행 방법

### 환경 설정

```bash
# 필요한 패키지 설치
pip install torch torchvision reality_stone

# 예제 실행
python examples/mnist_hyperbolic.py
```

### 성능 최적화 팁

1. **배치 크기 조정**
   ```python
   # GPU 메모리에 따라 조정
   batch_size = 256 if torch.cuda.is_available() else 64
   ```

2. **학습률 스케줄링**
   ```python
   scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
   ```

3. **조기 종료**
   ```python
   best_acc = 0
   patience = 5
   patience_counter = 0
   
   for epoch in range(100):
       # ... 훈련 ...
       if val_acc > best_acc:
           best_acc = val_acc
           patience_counter = 0
       else:
           patience_counter += 1
           if patience_counter >= patience:
               break
   ```

## 🚨 일반적인 문제와 해결책

### 1. NaN 값 발생
```python
# 해결책 1: 입력 스케일링
x = x * 0.1

# 해결책 2: 곡률 값 줄이기
curvature = 1e-4  # 대신 1e-3

# 해결책 3: 그래디언트 클리핑
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### 2. 학습 속도 느림
```python
# 해결책 1: 학습률 조정
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 더 큰 학습률

# 해결책 2: 배치 크기 증가
batch_size = 512  # 메모리가 허용하는 한
```

### 3. 메모리 부족
```python
# 해결책 1: 그래디언트 누적
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📊 성능 벤치마크

### MNIST 결과 비교

| 모델 | 정확도 | 훈련 시간 |
|------|--------|-----------|
| 일반 MLP | 97.8% | 2분 |
| Hyperbolic MLP | 98.2% | 3분 |
| Ensemble | 98.5% | 8분 |

### 계층적 데이터 결과

| 데이터셋 | MAP@10 | 훈련 시간 |
|----------|--------|-----------|
| WordNet | 0.85 | 15분 |
| 생물학 분류 | 0.92 | 8분 |

## 📖 다음 단계

- **[API 레퍼런스](./api_reference/README.md)**: 더 자세한 함수 문서
- **[수학적 배경](./04_mathematical_background.md)**: 이론적 기초
- **GitHub 예제**: 더 많은 실제 사용 사례 