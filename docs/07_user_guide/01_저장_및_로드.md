# Reality Stone 모델 저장 및 로드 시스템

Reality Stone 모델은 `extract_riemannian_metric`을 통해 구조가 변경되었으므로,
일반적인 `AutoModel.from_pretrained`로는 로드할 수 없습니다.

다음 두 가지 방법으로 저장/로드를 관리해야 합니다.

## 1. 전체 모델 저장 (Pickle 기반)
가장 쉬운 방법이지만, 파이썬 버전/경로 의존성이 있습니다.
`safe_serialization=False` 옵션으로 `pytorch_model.bin`을 통째로 저장합니다.

### 저장
```python
model.save_pretrained("./my-model", safe_serialization=False)
tokenizer.save_pretrained("./my-model")
```

### 로드
로드할 때도 구조를 먼저 잡아야 합니다. (중요!)
그냥 로드하면 바닐라 모델 구조에 리만 가중치를 억지로 끼워 넣으려다 실패합니다.

```python
# 1. 바닐라 모델 로드 (구조 뼈대)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# 2. 리만 수술 집도 (빈 껍데기 생성)
model = extract_riemannian_metric(model, target_dim=64)

# 3. 학습된 가중치 주입
from transformers.modeling_utils import load_state_dict
state_dict = torch.load("./my-model/pytorch_model.bin")
model.load_state_dict(state_dict)
```

---

## 2. 메트릭 텐서만 저장 (Efficient)
원본 모델(수 기가바이트)은 허깅페이스에서 받고,
우리가 학습한 **작은 메트릭 파일(수십 메가바이트)**만 따로 저장하는 방식입니다. (권장)

### 저장 (State Dict 필터링)
```python
metric_state = {k: v for k, v in model.state_dict().items() if "metric_g" in k or "basis" in k}
torch.save(metric_state, "./my-model/reality_stone_metrics.pt")
```

### 로드
```python
# 1. 원본 모델 로드
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# 2. 수술
model = extract_riemannian_metric(model, target_dim=64)

# 3. 메트릭 로드
metrics = torch.load("./my-model/reality_stone_metrics.pt")
model.load_state_dict(metrics, strict=False) # strict=False로 겹치는 부분만 로드
```

## 개선된 파이프라인 제안
매번 `extract_riemannian_metric`을 돌리면 시간이 오래 걸리므로(SVD 계산 등),
한 번 추출된 구조는 캐싱하거나, **초기화 없는(Empty) 수술** 함수를 만들어야 합니다.

`python/reality_stone/metric_extraction.py`에 `apply_metric_structure_only` 함수를 추가하면,
SVD 계산 없이 빈 껍데기만 0.1초 만에 만들고 저장된 가중치를 로드할 수 있습니다.

