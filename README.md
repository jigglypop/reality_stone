# HyperButterfly: PyTorch를 위한 효율적인 하이퍼볼릭 기하학 라이브러리

[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![라이선스](https://img.shields.io/badge/%EB%9D%BC%EC%9D%B4%EC%84%A0%EC%8A%A4-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🌟 개요

**HyperButterfly**는 하이퍼볼릭 공간에서의 기하학적 딥러닝을 위한 고성능 PyTorch 확장 라이브러리입니다. 리만 다양체, 특히 하이퍼볼릭 공간에서의 효율적인 연산과 Butterfly 팩터를 통한 효율적인 변환 구현을 제공합니다.

## ✨ 주요 기능

- 🚀 **포인카레 볼 모델**: 하이퍼볼릭 공간의 지수 맵, 로그 맵, 측지 거리 계산을 위한 최적화된 C++ 구현
- 🧮 **Butterfly 팩터**: O(N log N) 복잡도로 행렬 변환을 근사하는 효율적인 알고리즘
- 🔄 **Hyper-Butterfly 레이어**: 하이퍼볼릭 공간에서의 효율적인 신경망 레이어
- 🔍 **수치적 안정성**: 유한 조건수와 역전파 안정성 보장
- 📊 **시각화 도구**: 하이퍼볼릭 공간에서의 데이터 시각화

## 📦 설치 방법

```bash
git clone https://github.com/username/hyper_butterfly.git
cd hyper_butterfly
pip install -e .
```

## 🚀 빠른 시작

```python
import torch
import riemannian_manifold

# 포인카레 볼 모델에서 연산 예제
x = torch.zeros(1, 2)  # 포인카레 볼의 원점
v = torch.tensor([[0.3, 0.4]])  # 접벡터

# 지수 사상 적용
y = riemannian_manifold.exp_map(x, v)
print("원점으로부터의 지수 맵 결과:", y)

# 거리 계산
dist = riemannian_manifold.distance(x, y)
print(f"리만 거리: {dist.item():.4f}")

# Hyper-Butterfly 레이어 사용
layer = riemannian_manifold.HyperButterflyLayer(dim=8, num_layers=3, curvature=0.5)
input_data = torch.randn(8) * 0.3  # 반지름이 작은 점들
output = layer(input_data)
```

## 🧪 테스트 실행

라이브러리의 주요 기능을 테스트하려면:

```bash
python test.py
```

## 📚 주요 구현 내용

### 포인카레 볼 모델

포인카레 볼 모델은 하이퍼볼릭 공간의 등각 모델로, 다음과 같은 핵심 연산이 구현되어 있습니다:

1. **지수 맵 (Exponential Map)**:
   ```python
   # 원점에서의 지수 맵
   y = riemannian_manifold.exp_map(torch.zeros_like(x), v, c=1.0)
   ```

2. **로그 맵 (Logarithmic Map)**:
   ```python
   # 원점으로의 로그 맵
   v = riemannian_manifold.log_map(torch.zeros_like(y), y, c=1.0)
   ```

3. **측지 거리 (Geodesic Distance)**:
   ```python
   dist = riemannian_manifold.distance(x, y, c=1.0)
   ```

### Butterfly 팩터

Butterfly 팩터는 행렬을 효율적으로 표현하기 위한 방법으로, 다음과 같이 구현되어 있습니다:

```python
# 버터플라이 변환 레이어 적용
output = riemannian_manifold.butterfly_transform(input_data, params, layer=0)
```

### Hyper-Butterfly 레이어

Hyper-Butterfly 레이어는 하이퍼볼릭 공간에서 효율적인 신경망 레이어를 구현합니다:

```python
layer = riemannian_manifold.HyperButterflyLayer(dim=8, num_layers=3, curvature=0.5)
output = layer(input_data)
```

## 📄 논문 참조

이 구현은 "Hyper-Butterfly 네트워크: 계산적 하이퍼볼릭 기하학의 수학적 분석" 논문을 기반으로 합니다. 자세한 수학적 이론과 증명은 `hyper_butterfly.md` 문서를 참조하세요.

## 🤝 기여하기

기여는 언제나 환영합니다! 버그 리포트, 기능 요청, 풀 리퀘스트 모두 가능합니다.

## 📝 라이선스

MIT 라이선스에 따라 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.