# 시작하기

이 가이드는 Reality Stone을 설치하고 첫 번째 하이퍼볼릭 신경망 모델을 만드는 방법을 안내합니다.

## 시스템 요구사항

### 필수 요구사항
- **Python**: 3.8 이상 (3.10 권장)
- **PyTorch**: 2.0.0 이상
- **NumPy**: 1.21.0 이상

### 선택적 요구사항
- **CUDA Toolkit**: 11.0 이상 (GPU 가속 사용 시)
- **NVIDIA GPU**: Compute Capability 7.0 이상 (Tesla V100, RTX 20xx 시리즈 이상)

## 설치

### 방법 1: pip를 사용한 설치 (권장)

```bash
pip install reality_stone
```

### 방법 2: 소스에서 설치

```bash
# 저장소 클론
git clone https://github.com/jigglypop/reality_stone.git
cd reality_stone

# 가상환경 생성 (선택사항)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 개발 모드로 설치
pip install -e .
```

### 방법 3: maturin을 사용한 개발 설치

```bash
# Rust 및 maturin 설치
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# 프로젝트 빌드 및 설치
maturin develop --release
```

## 설치 확인

```python
import reality_stone as rs
import torch

print(f"Reality Stone 버전: {rs.__version__ if hasattr(rs, '__version__') else 'dev'}")
print(f"Rust 확장 사용 가능: {rs._has_rust_ext}")
print(f"CUDA 사용 가능: {rs._has_cuda}")
```

예상 출력:
```
Reality Stone 버전: 0.2.0
Rust 확장 사용 가능: True
CUDA 사용 가능: True
```

## 첫 번째 예제: MNIST 분류

Reality Stone을 사용하여 간단한 MNIST 분류 모델을 만들어보겠습니다.

### 1. 필요한 라이브러리 import

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import reality_stone as rs
```

### 2. 하이퍼볼릭 MLP 모델 정의

```python
class HyperbolicMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, curvature=1e-3):
        super().__init__()
        self.curvature = curvature
        
        # 일반적인 선형 레이어들
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 입력을 평탄화
        x = x.view(x.size(0), -1)
        
        # 첫 번째 레이어
        h1 = torch.tanh(self.linear1(x))
        
        # 두 번째 레이어
        h2 = torch.tanh(self.linear2(h1))
        
        # 하이퍼볼릭 레이어 적용
        # h1과 h2를 하이퍼볼릭 공간에서 결합
        hyperbolic_features = rs.poincare_ball_layer(
            h1, h2, 
            c=self.curvature, 
            t=0.5  # 보간 비율
        )
        
        # 출력 레이어
        output = self.output_layer(hyperbolic_features)
        return output
```

### 3. 데이터 로더 설정

```python
# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터셋 로드
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

# 데이터 로더
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
```

### 4. 모델 훈련

```python
# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화
model = HyperbolicMLP(curvature=1e-3).to(device)

# 옵티마이저 및 손실 함수
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 훈련 루프
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(loader)

# 평가 함수
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total

# 훈련 실행
epochs = 5
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # 훈련
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # 평가
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
```

## 핵심 개념 이해

### 1. Poincaré Ball Layer

`rs.poincare_ball_layer(u, v, c, t)`는 두 벡터 `u`와 `v`를 하이퍼볼릭 공간에서 결합합니다:

- **u, v**: 입력 텐서 (같은 크기)
- **c**: 곡률 매개변수 (양수, 작을수록 평평한 공간)
- **t**: 보간 비율 (0~1, 0.5는 균등 혼합)

### 2. 곡률 매개변수 (c)

- **c = 0**: 유클리드 공간 (평면)
- **c > 0**: 하이퍼볼릭 공간 (음의 곡률)
- **일반적 값**: 1e-3 ~ 1e-1

### 3. 보간 비율 (t)

- **t = 0**: 첫 번째 입력만 사용
- **t = 0.5**: 두 입력을 균등하게 혼합
- **t = 1**: 두 번째 입력만 사용

## 문제 해결

### 자주 발생하는 오류

#### 1. "Rust extension not found"
```bash
# maturin을 사용하여 다시 빌드
maturin develop --release
```

#### 2. CUDA 오류
```python
# CPU 모드로 강제 설정
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

#### 3. NaN 값 발생
```python
# 곡률 값을 줄여보세요
model = HyperbolicMLP(curvature=1e-4)  # 기본값보다 작게

# 또는 그래디언트 클리핑 사용
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 다음 단계

- **[API 레퍼런스](./api_reference/README.md)**: 모든 함수와 클래스의 상세 문서
- **[예제 가이드](./03_examples.md)**: 더 복잡한 사용 예제들
- **[수학적 배경](./04_mathematical_background.md)**: 하이퍼볼릭 기하학 이론

---

이제 Reality Stone의 기본 사용법을 익혔습니다! 다음 섹션에서는 더 자세한 API 문서를 확인해보세요. 