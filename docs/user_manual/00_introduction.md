# Reality Stone 사용자 매뉴얼

## 소개

Reality Stone은 **하이퍼볼릭 기하학**을 활용한 고성능 신경망 라이브러리입니다. Rust와 CUDA로 구현된 핵심 계산 엔진과 PyTorch 친화적인 Python API를 제공하여, 계층적 데이터와 트리 구조 표현 학습에 특화되어 있습니다.

## 대상 독자

- **머신러닝 연구자**: 하이퍼볼릭 신경망을 활용한 새로운 모델 개발
- **데이터 사이언티스트**: 계층적 데이터(지식 그래프, 소셜 네트워크, 생물학적 분류) 분석
- **딥러닝 엔지니어**: 기존 PyTorch 모델에 하이퍼볼릭 레이어 통합

## 주요 특징

### 다양한 하이퍼볼릭 모델 지원
- **Poincaré Ball**: 포인카레 디스크 모델 기반 레이어
- **Lorentz Model**: 하이퍼볼로이드 모델 기반 레이어  
- **Klein Model**: Klein 디스크 모델 기반 레이어
- **Mobius Operations**: 뫼비우스 변환 기반 핵심 연산

### 고성능 계산
- **Rust 코어**: 메모리 안전성과 성능을 보장하는 Rust 구현
- **CUDA 가속**: GPU를 활용한 대규모 배치 처리
- **자동 최적화**: CPU/GPU 자동 선택 및 메모리 관리

### PyTorch 완벽 연동
- **네이티브 통합**: `torch.autograd`와 완전 호환
- **배치 처리**: 표준 PyTorch 배치 연산 지원
- **그래디언트 계산**: 자동 미분 및 역전파 지원

## 수학적 배경

Reality Stone은 **음의 곡률**을 갖는 하이퍼볼릭 공간에서 동작합니다. 이는 계층적 구조를 자연스럽게 표현할 수 있는 기하학적 특성을 제공합니다.

### 핵심 개념
- **하이퍼볼릭 공간**: 음의 곡률을 갖는 리만 다양체
- **측지선**: 하이퍼볼릭 공간에서의 최단 경로
- **지수/로그 매핑**: 접선 공간과 다양체 간의 변환

## 목차

1. **[시작하기](./01_getting_started.md)** - 설치 및 기본 사용법
2. **[API 레퍼런스](./api_reference/README.md)** - 상세한 함수 및 클래스 문서
3. **[예제 가이드](./03_examples.md)** - 실제 사용 예제와 튜토리얼
4. **[수학적 배경](./04_mathematical_background.md)** - 이론적 기초

## 빠른 시작 예제

```python
import torch
import reality_stone as rs

# 간단한 Poincaré Ball 레이어 사용
x = torch.randn(32, 64) * 0.1  # 배치 크기 32, 차원 64
y = torch.randn(32, 64) * 0.1

# 하이퍼볼릭 공간에서의 보간
result = rs.poincare_ball_layer(x, y, c=1.0, t=0.5)
print(f"Output shape: {result.shape}")  # [32, 64]
```

## 도움이 필요하다면

- **GitHub Issues**: [버그 리포트 및 기능 요청](https://github.com/jigglypop/reality_stone/issues)
- **문서**: 이 사용자 매뉴얼의 다른 섹션들을 참조하세요
- **예제**: `examples/` 디렉토리의 다양한 사용 사례를 확인하세요

---

다음: **[시작하기](./01_getting_started.md)**에서 Reality Stone을 설치하고 첫 번째 모델을 만들어보세요. 