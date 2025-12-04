## RS-ULF / Reality Stone 운영 관점 설계서 (v0.1)

### 0. 목표

- **운영 목표**
  - 기존 HF GPT-2/GPT-Neo/Qwen 계열 모델을 **RS-ULF + HyperMetric + Symplectic** 경로로 변환·서빙하면서,
  - **추론 품질(로그확률, 응답 스타일)**을 유지하고,
  - **레이어 기준 ≥ 10x, 모델 기준 3–5x 수준의 파라미터/메모리 절감**,  
  - **토큰당 레이턴시 손실 ≤ 1.2x (또는 동급)** 범위로 운영 가능한 파이프라인 구축.

- **운영상 필수 요구사항**
  - 동일 프롬프트에 대해 **Original / RS-ULF / SVD-RS-ULF**의 출력을 항상 비교 가능.
  - 변환·압축 실패 시 **항상 원본 경로로 안전하게 폴백**.
  - 하나의 설정 파일/명령으로 **재현 가능한 변환·벤치마크 러너** 제공.

---

## 1. 시스템 구성 개요

### 1.1 핵심 컴포넌트

- **Python 상위 레이어**
  - `reality_stone.models.transformer_converter.RSULFTransformerConverter`
    - HF `PreTrainedModel` → `RSULFModel` 변환 엔진.
    - Rust 바인딩(`PyRSULFLayer`, `extract_global_basis`, `analyze_layer`) 호출.
  - `reality_stone.layers.rsulf_cuda.RSULFLayerCUDA`, `RSULFWrapperCUDA`
    - PyTorch 모듈 래퍼. Rust RS-ULF 레이어를 PyTorch 레이어로 노출.
  - `experiments/test_gpt2_conversion.py`
    - 현재 GPT-2 전용 데모/실험 스크립트.
    - 향후 **공통 벤치마크 러너**의 프로토타입.

- **Rust 코어**
  - `src/layers/rsulf.rs`
    - RS-ULF 메트릭/FFN 폴딩, 심플렉틱 업데이트, 파라미터 카운트 로직.
  - `src/bindings/rsulf.rs`
    - PyO3 바인딩, CUDA FFI, `extract_global_basis`, `create_compression_plan` 등.

- **운영용 래퍼 (타깃)**
  - `RSULFModel` + `RSULFLMHeadCUDA` 조합을  
    HF `PreTrainedModel` 스타일 API로 감싼 **고수준 어댑터 클래스** (미구현).

---

## 2. 변환·압축 파이프라인 (운영 플로우)

### 2.1 변환 플로우 요약

1. **원본 모델 로딩**
   - HF `AutoModelForCausalLM.from_pretrained(...)` 혹은 GPT2 계열 `GPT2LMHeadModel`.
   - 추론용 `eval()` 상태에서 고정.

2. **레이어 가중치 수집**
   - `RSULFTransformerConverter.extract_weights(layer)`:
     - `WQ`, `WK`, `W1`, `W2`, `ln_1` 등 레이어별 핵심 파라미터 추출.
   - `verify_weights`로 NaN/Inf 체크, 실패 시 해당 레이어는 `Skip`.

3. **전역 분석·플랜 생성 (압축 모드에서만)**
   - `analyze_layer(wq, wk, w1, w2, idx, r)`
     - 스펙트럴 디케이, 컨디션 넘버, 추천 랭크, 예상 정확도 계산.
   - `create_compression_plan(analyses, target_ratio)`
     - 전체 레이어 기준 **목표 압축률**과 **최소 예상 정확도**를 만족하는 랭크 플랜 계산.

4. **Global Basis 추출 (압축 모드)**
   - `extract_global_basis(valid_wq, valid_wk, r)`
     - 모든 레이어의 Q/K를 모아 공통 메트릭 기저 `U_global` 추출.

5. **레이어 변환**
   - **Exact 모드 (`exact=True`)**
     - 레이어별 `RSULFLayerCUDA(..., r=d_model, global_basis=None)` 생성.
     - 압축보다는 **이론 검증 / 동작 검증용**.
   - **압축 모드 (`exact=False`)**
     - `rank_by_idx[idx]`에 따라 레이어별 `r`를 조정.
     - `RSULFLayerCUDA(..., r=best_r, global_basis=global_basis)`로 생성.

6. **`RSULFModel` 구성**
   - Rust 레이어 리스트 → `RSULFWrapperCUDA` 모듈리스트로 감싸서  
     `RSULFModel.wrappers`에 저장.
   - 이후 고수준 어댑터에서 이 `wrappers`를 사용하여 forward 구성.

---

## 3. 추론 디코더 설계 (현행 vs 타깃)

### 3.1 현행 실험 디코더 (`test_gpt2_conversion.py`)

- **Original**
  - HF `model.generate` 사용.
  - `attention_mask`, `pad_token_id`, `repetition_penalty=1.2` 적용.

- **Structural RS-ULF / RS-ULF Rust / RS-ULF SVD**
  - 공통 커스텀 디코더 함수 `rsulf_generate_text`:
    - GPT-2 임베딩/포지션/`ln_f`/`lm_head`는 그대로 사용.
    - 중간 블록만 `structural_model` 또는 `RSULFModel.wrappers`로 교체.
    - 샘플러 `_sample_next_token`:
      - repetition penalty, temperature, top-k, top-p 적용.

- **문제점**
  - HF `generate`의 모든 옵션/캐시/로짓 래퍼를 그대로 따라가지는 않음.
  - 실험에는 충분하지만, **운영용 인터페이스로는 부족**.

### 3.2 타깃 운영 디코더 설계

- **목표**
  - HF `generate`를 가능한 한 그대로 사용하면서, **블록만 RS-ULF로 바꾼 모델 클래스**를 제공.

- **타깃 클래스 구조 (예시)**
  - `class RSULFLLMAdapter(PreTrainedModel):`
    - 내부에:
      - `self.transformer = RSULFModel` (RS-ULF 블록 스택)
      - `self.lm_head = nn.Linear(...)` (원본 `lm_head` 재사용)
      - 임베딩/포지션/`ln_f`도 원본 GPT-2에서 복사.
    - `forward(input_ids, attention_mask, past_key_values, use_cache, ...)` 시그니처를
      **GPT2LMHeadModel와 동일하게** 구현.
  - 이렇게 하면 **HF `generate`가 그대로 동작**하고,
    - RS-ULF 블록은 HF 프레임워크 내부에서 "원래 블록처럼" 취급.

- **운영 단계 목표**
  - `transformers.AutoModelForCausalLM.from_pretrained("...rsulf-r64")` 호출 시
    - RS-ULF 바디 + 원본 헤드/토크나이저를 가진 모델이 로드.
  - 동일 `generate` API / 동일 Config / 동일 토크나이저로 교체 가능.

---

## 4. 지표 및 모니터링 플랜

### 4.1 변환 시점 지표

- **레이어 단위**
  - `RS-ULF] Layer-wise similarity`:
    - `cos`, `rel_l2` (PyTorch vs RS-ULF 레이어 출력 비교).
    - 운영 기준:
      - 대부분 레이어: `cos ≥ 0.999`, `rel_l2 ≤ 1e-3`.
      - 마지막 몇 레이어는 모델 특성 상 약간 느슨한 기준 허용.
  - `RSULF` 변환 로그:
    - 레이어별 `ratio`, `curvature`, `r`, `compression_plan`의 예상 정확도.

- **모델 단위**
  - 파라미터 카운트:
    - `original_params`, `compressed_params`, `expected_compression_ratio`.
  - 저장:
    - 변환 완료 후 JSON/ YAML 요약 파일에 기록 (모델과 함께 아티팩트로 저장).

### 4.2 추론 시점 지표

- **성능**
  - 토큰당 레이턴시 (P50, P95).
  - QPS / 동시 요청 수 대비 스루풋.

- **품질**
  - 샘플 프롬프트 집합에 대한:
    - 로그확률 차이 (Original vs RS-ULF).
    - BLEU/ROUGE 등은 부차적, 우선순위는 **per-token logit L2, cosine**.

- **안정성**
  - NaN/Inf 감지:
    - Rust forward, PyTorch wrapper 양쪽에서 한 번씩 체크.
  - 예외/타임아웃:
    - 변환 실패 레이어, FFI 에러, CUDA 에러를 별도 카운터로 모니터링.

---

## 5. 운영 시나리오별 플랜

### 5.1 오프라인 변환 + 온라인 서빙

- **오프라인**
  - 변환 스크립트:
    - `python -m reality_stone.tools.convert --model gpt2 --rank 64 --output gpt2-rsulf-r64`
  - 산출물:
    - RS-ULF 전용 체크포인트 (파이토치 weight + Rust 구성 파라미터).
    - 변환 리포트(JSON): 레이어별 r, ratio, expected_accuracy, similarity stats.

- **온라인**
  - 서빙 인스턴스에서:
    - `from_pretrained("gpt2-rsulf-r64")` 로 로드.
    - 환경 변수 또는 설정으로:
      - **폴백 전략**: RS-ULF 실패 시 원본 GPT-2로 전환.
      - **모드 선택**: `exact`, `svd-r64`, `svd-r32` 등.

### 5.2 점진 롤아웃

- **단계**
  1. **Shadow 모드**: 동일 요청에 Original / RS-ULF를 둘 다 돌려 로깅만.
  2. **A/B 테스트**: 트래픽 일부 RS-ULF로 전환, 품질/레이턴시 모니터링.
  3. **Gradual rollout**: 10% → 50% → 100% 순차 전환.
  4. **Fallback 정책**: 품질/레이턴시 임계값 넘으면 자동 리버트.

---

## 6. 200B급 확장 플랜 (고수준)

### 6.1 병렬화 전략

- **층 병렬 (Pipeline / Tensor Parallel)과 RS-ULF의 결합**
  - RS-ULF는 **레이어별 공통 메트릭 기저**를 쓰기 때문에,
  - 큰 모델일수록:
    - Global Basis 추출을 **GPU/다중 노드**에서 수행 (faer + CUDA + sharded SVD).
    - HyperMetric/ManifoldLearner를 별도 학습 잡으로 분리 가능.

- **데이터 플로우**
  - 큰 모델에서:
    - Q/K/FFN 가중치를 **shard 단위**로 수집 → RS-ULF 변환기에게 순차 공급.
    - 변환기는 shard별로 RSULF 레이어를 생성하고, 최종적으로 다시 합쳐서 체크포인트 생성.

### 6.2 운영 관점 체크리스트 (200B 타깃)

- **메모리**
  - 변환 시 peak 메모리 예측:
    - 원본 모델 + RS-ULF 중간 행렬(G, SVD 버퍼)을 포함한 상한 계산.

- **시간**
  - 변환 시간 예측:
    - 레이어 수, d_model, r, seq_len, window, 샘플 수에 따른 러닝타임 모델 만들기.

- **신뢰도**
  - 층별 similarity 통과 기준:
    - 예: 모든 레이어 `cos ≥ 0.995` 이상이 아니면 RS-ULF 체크포인트 reject.

---

## 7. 단기 구현 TODO (운영 관점)

- **1단계 (이미 일부 구현됨)**
  - `RSULFTransformerConverter`에:
    - 변환 리포트(JSON) export 기능 추가.
    - 변환 실패/성공 레이어 목록, 압축 통계 자동 기록.

- **2단계**
  - `RSULFLLMAdapter` (HF 스타일 어댑터 클래스) 설계 및 구현.
  - `test_gpt2_conversion.py`를 일반화한 `benchmark_runner.py` 작성:
    - 프롬프트 리스트, max_new_tokens, 샘플링 설정을 받아  
      Original vs RS-ULF vs RS-ULF-SVD 비교.

- **3단계**
  - CLI/Config 기반 변환 툴 (`python -m reality_stone.tools.convert ...`).
  - Shadow 모드 및 간단한 A/B 테스트 스크립트 추가.


