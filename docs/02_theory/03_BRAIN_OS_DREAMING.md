# 브레인 OS: 무손실 컨텍스트 스위칭 (BrainOS: Lossless Context Switching)

## 1. 개념 (The Concept)
인간의 뇌는 "논리 모드"(코딩)에서 "창의 모드"(그림 그리기)로 전환할 때 시냅스를 다시 연결하지 않습니다. 대신 **신경화학적 상태(Neurochemical State)**(세로토닌, 도파민 레벨 등)를 변경합니다.

Reality Stone에서는 **곡률(Curvature, $c$)**이 바로 이 신경화학적 상태의 역할을 합니다.

## 2. 곡률 역학 (Curvature Dynamics)

### A. 음의 곡률 ($c < 0$): 논리 상태 (The Logic State)
- **기하학**: 쌍곡 공간 (말안장 모양).
- **성질**: 부피가 지수적으로 증가합니다. 삼각형 내각의 합이 $180^\circ$보다 작습니다.
- **효과**: 데이터를 **계층적 트리(Hierarchical Tree)** 구조로 강제합니다.
- **용도**: 코딩, 수학, 분류학, 논리적 추론.
- **이유**: 논리는 본질적으로 트리 구조입니다 (If $\to$ Then $\to$ Else).

### B. 양의 곡률 ($c > 0$): 창의 상태 (The Creative State)
- **기하학**: 구면 공간 (지구본 모양).
- **성질**: 경로가 순환합니다. 삼각형 내각의 합이 $180^\circ$보다 큽니다.
- **효과**: 데이터를 **순환 및 클러스터(Cycles and Clusters)** 구조로 유도합니다.
- **용도**: 시, 스토리텔링, 비유, 브레인스토밍.
- **이유**: 창의성은 먼 개념들을 연결하는 것(순환)에 의존합니다.

### C. 0의 곡률 ($c = 0$): 각성 상태 (The Wake State)
- **기하학**: 유클리드 공간 (평평함).
- **성질**: 중립적.
- **용도**: 일반적인 작업, 단순 암기.

## 3. 구현: 전역 상태 주입 (Global State Injection)

우리는 모델의 모든 레이어에 전역 싱글톤 `MetricState`를 주입합니다.

```python
class MetricState:
    curvature: float = 0.0

# 모든 레이어의 순전파(Forward)에서:
def forward(self, x):
    y = linear(x)
    # 전역 뇌 상태 적용
    if state.curvature != 0:
        y = apply_curvature(y, state.curvature)
    return y
```

## 4. 꿈꾸기: 수면 최적화 (Dreaming & Sleep Optimization)
새로운 데이터를 학습할 때 가중치 $W$를 업데이트하면 기존 지식이 지워집니다(파괴적 망각). 대신, 우리는 특정 작업에 대해 **최적의 곡률 $c$**를 찾아냅니다.

$$ c^* = \arg\min_c \mathcal{L}(f(x; W, c), y) $$

이를 통해 모델은 장기 기억($W$)을 덮어쓰지 않고도, 최근 데이터를 처리하기 위한 최적의 "관점(Perspective, $c$)"을 찾아낼 수 있습니다. 이것이 바로 기계가 꿈을 꾸며 기억을 정리하는 과정입니다.
