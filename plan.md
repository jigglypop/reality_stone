관세음보살. 먼저 `experiments/test_gpt2_conversion.py`를 지정한 커맨드로 실행해서 현재 붕괴 상태를 확인하고, 동시에 `08_geodesic_compression` 문서 전체와 핵심 Rust/Python 구현을 읽어 논문 설계 대비 구현 완료율을 항목별로 정리한 표를 만들겠습니다. 그다음 추론 경로가 지오데식 전용 경로인지, 아니면 디폴트/섞인 경로인지 실제 호출 스택을 코드 기준으로 추적해서 “왜 이렇게까지 깨지는지”를 설명하겠습니다.

[11 tools called]

지금 문서와 핵심 Python 코드는 읽었고, Rust 쪽 레이어/메모리 구현을 더 봐서 청사진 대비 구현율을 정교하게 매핑하겠습니다. 추론 경로가 어디서 어떻게 RSULF/지오데식 레이어를 타는지도 Rust 바인딩과 Python 래퍼를 따라가서 정확히 짚어볼게요.

[14 tools called]

지금 RSULF/메모리/스플라인/디퓨전까지 Rust와 Python 인터페이스를 다 짚었고, 이제 `experiments/test_gpt2_conversion.py`와 실제 추론 경로를 확인해서 “디폴트 경로를 타는지 vs RS-ULF 경로를 타는지”를 확정하겠습니다. 그 다음 문서 항목별 구현율을 표로 정리하고, 왜 이렇게까지 출력이 깨지는지(레이어 정규화/지오데식 스텝/디퓨전 상호작용 관점)로 정리해서 드릴게요.

[5 tools called]

관세음보살.

이 환경에서는 `experiments/test_gpt2_conversion.py`를 직접 재실행하지 못해서, **당신이 이전에 남긴 실행 로그 + 현재 Rust/Python 코드 + 논문 문서(01~04장)**를 기준으로 정밀 매핑했습니다.

---

### 1. 논문(08_geodesic_compression) 항목별 구현 현황 요약

#### 1-1. 01_THEORY_AND_PROOFS.md

| 장/섹션                           | 항목                                    | 구현 위치                                                                                                                                    | 구현율(대략) | 설명                                                                                                                                                                                               |
| --------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2. Equivalence Theorem            | 트랜스포머 ↔ 지오데식 동역학            | `RSULFLayer::forward` (지오데식 스텝, 메트릭/게이지, 포텐셜, 그래프 확산), `geodesic.rs`, `diffusion.rs`                                     | **70%**      | B = WQᵀWK → g_sym, a_antisym, FFN→포텐셜, 지오데식/디퓨전까지 구조는 구현. 다만 완전한 심플렉틱 통합기·일반 다양체 MetricType과의 결합은 RS-ULF용 특화 구현에 한정.                                |
| 3. Compression Error Bounds       | Folding 에러 상계, 스펙트럼 기반 에너지 | `fold_dimension_svd`, `fold_with_global_basis`, `compute_curvature`, `verify_fold_consistency`, `adaptive_rank_svd`, `nystrom_approximation` | **80%**      | SVD 기반 에너지 캡처 비율, `fold_accuracy`, `reconstruction_error`, `curvature`(잔차 에너지)까지 포함. 논문 수식 그대로의 d_g 왜곡 계산은 없지만, 필요한 통계량은 대부분 구현.                     |
| 4. Curvature Compensation Theorem | 곡률로 오차 보상                        | `RSULFLayer::forward`(Christoffel/curvature 스텝), `SplineCache`(curvature로 중간 보정)                                                      | **60%**      | 잔차→`curvature` 계산 후, 지오데식 스텝에서 곡률 스케일링·Christoffel 유사 항을 추가하고, 스플라인에서도 곡률 기반 보정. 다만 이론의 “ε로부터 κ(ε)을 역산해 조정” 구조는 단순한 휴리스틱으로 구현. |

---

#### 1-2. 02_COMPRESSION_MECHANISM.md

| 섹션                            | 항목                                                 | 구현 위치                                                                                                                                           | 구현율  | 설명                                                                                                                                                                                    |
| ------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2. Metric-Gauge Decomposition   | B=WQᵀWK, g/A 분해, Global Basis, g_core/A_core, 곡률 | `RSULFLayer::from_transformer*`, `fold_dimension_svd`, `fold_with_global_basis`, `extract_global_basis`, `compute_curvature`                        | **90%** | B 계산 → g_sym, a_antisym 분리, GlobalBasis U 추출, g_core/A_core 투영, residual로 곡률 산출까지 문서 흐름 그대로 구현. 데이터 구조는 `RSULFComponents`가 `LayerParams` 역할을 수행.    |
| 3. FFN Helmholtz Compression    | 포텐셜 증류, 에너지/Force 매칭                       | `fold_ffn_svd`, `fold_ffn_random_projection`, `calibrate_eta_alpha`                                                                                 | **60%** | W1/W2를 SVD 또는 랜덤 프로젝션으로 접고, `calibrate_eta_alpha`로 eta/alpha를 데이터 기반 추정. 논문처럼 별도 `NeuralPotential` MLP를 학습시키는 부분과 에너지 적분항 최적화는 미구현.   |
| 4. Geodesic Spline for KV Cache | 제어점 선택, 스플라인 복원                           | `GeodesicMemory`(제어점 검출·Hermite 스플라인), `SplineCache`(곡률 보정 스플라인), `PyGeodesicMemory`, `PySplineCache`, `RSULFWrapperCUDA`에서 사용 | **80%** | 가속도 임계치 기반 제어점 선택, Hermite 스플라인 복원, 곡률 기반 보정, 압축 통계까지 구현. Q/K/V 구조 전체를 대상으로 한 완전한 KV 캐시 교체보다는, 현재는 `x` 상태 궤적 중심으로 통합. |

---

#### 1-3. 03_COMPLEXITY_AND_MEMORY.md

| 항목                        | 구현 위치                                                                                                                                                          | 구현율  | 설명                                                                                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| O(N²D) → O(NDR) 시간 복잡도 | `RSULFLayer::forward`(지금은 CPU O(N²) 메트릭 어텐션), `create_causal_laplacian`(희소 라플라시안), `geodesic_topk_attention.cu` / `geodesic_topk_attention` 바인딩 | **40%** | 알고리즘 구성요소(라플라시안, 지오데식 top-k CUDA 커널)는 존재하지만, RS-ULF 경로에서 CUDA 커널로 실제 교체·사용되지는 않음. 현재 RSULF forward는 CPU에서 O(N²) 어텐션을 돌려 “이론상의 O(N)”은 아직 미달.          |
| 파라미터 압축 수식, 압축률  | RSULF 접기 로직 전반, `RSULFLayer::param_count`, `create_compression_plan`                                                                                         | **80%** | Global Basis + 레이어별 core들로 파라미터 축소, 원본/압축 파라미터를 카운트해 ratio 계산. 논문 수식과 거의 같은 형태로 구현되어 있고, 실제로 변환 시 압축률 로그도 출력.                                            |
| KV 캐시 메모리 압축         | `GeodesicMemory::get_compression_stats`, `SplineCache`                                                                                                             | **70%** | 제어점 개수·커버 길이·압축비를 산출하고, 스플라인 기반 복원으로 “적은 포인트로 궤적 복원” 구조를 구현. 하지만 아직 전체 HF KV 캐시를 대체하는 수준의 통합은 아니라서, 논문 표의 10x 절감은 구조상 가능한 상태 정도. |
| 대역폭 최적화               | Global Basis 공유, `RSULFComponents` 구조, `geodesic_topk_attention`                                                                                               | **50%** | U 공유·코어 행렬 작은 구조·SPD Cholesky CUDA 커널로 설계는 갖췄지만, 실제 PyTorch 런타임에서 geodesic top-k를 불러 써서 대역폭 병목을 제거하는 통합은 아직 부족.                                                    |

---

#### 1-4. 04_IMPLEMENTATION_BLUEPRINT.md

##### (A) 변환 파이프라인

| 섹션                        | 항목                                  | 구현 위치                                                                                                        | 구현율   | 설명                                                                                                                                                                                                                   |
| --------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.0.1 Spectrum Analyzer     | 레이어별 스펙트럼, Tail, rank         | `analyze_weight_matrix`, `analyze_layer`, `analyze_layer_py`, `RSULFTransformerConverter.convert_model`에서 호출 | **90%**  | g = WQᵀWK에 대해 SVD, 조건수, 추천 rank, expected_accuracy 계산. Tail 에너지·스펙트럼 decay를 모두 집계.                                                                                                               |
| 2.0.2 Rank Planner          | RankPlan, global/group rank           | `create_compression_plan`, `create_compression_plan_py`                                                          | **60%**  | 전체 레이어의 param_count/expected_accuracy/추천 rank를 모아 compression plan 생성·요약. 하지만 **실제 레이어 생성 시 plan을 사용하지 않고** 여전히 `best_r = min(d_model, self.r)`로만 결정해서 정책이 반영되지 않음. |
| 2.1.1 Loader                | HF 가중치 로드                        | `RSULFTransformerConverter.extract_weights`                                                                      | **90%**  | GPT-2 Conv1D, LLaMA 계열 등 다양한 구조에서 WQ/WK/W1/W2 + LayerNorm 파라미터를 안전하게 추출, NaN/Inf 검사까지 수행.                                                                                                   |
| 2.1.2 Analyzer              | 메트릭 통계 수집                      | 위 Analyzer·Rank Planner 조합                                                                                    | **90%**  | 레이어 전 범위를 스캔하여 spectral_decay, condition_number, 추천 rank, 예상 정확도 등을 수집.                                                                                                                          |
| 2.1.3 Folder                | 레이어 가중치 폴딩, Global Basis 적용 | `RSULFLayer::from_transformer*`, `fold_dimension_svd`, `fold_with_global_basis`, `extract_global_basis`          | **100%** | Global Basis U를 추출하고, 레이어별 g_core, a_core, laplacian, FoldedFFN 생성까지 청사진과 거의 일치.                                                                                                                  |
| 2.1.4 Distiller (Potential) | FFN 포텐셜 증류                       | `fold_ffn_*`, `calibrate_eta_alpha`                                                                              | **30%**  | SVD/RandomProj 기반 구조 압축과 eta/alpha 보정까지 구현. 하지만 문서의 “작은 MLP 포텐셜을 SGD로 학습”은 전혀 없음.                                                                                                     |
| 2.1.5 Packer (.rsu)         | 전용 바이너리 저장                    | `_save_checkpoint`, `RSULFLayer::export_components`, `PyRSULFLayer.export_components`                            | **20%**  | `.rsu` 헤더·바이너리 포맷은 미구현. 대신 JSON으로 RSULFComponents를 내보내는 체크포인트 기능이 있어, 구조만 맞춘 상태.                                                                                                 |

##### (B) 런타임 커널 & 메모리 계층

| 섹션                    | 항목                                        | 구현 위치                                                                        | 구현율  | 설명                                                                                                                                                                                                                                    |
| ----------------------- | ------------------------------------------- | -------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.1 FusedGeodesicStep   | 지오데식 스텝, 포텐셜, 라플라시안, 재트랙션 | `RSULFLayer::forward`                                                            | **70%** | Attention(메트릭) + Gauge + FFN 포텐셜 + Graph Diffusion + 곡률 기반 지오데식 스텝을 **한 함수 안에서** 계산해 논리상 fused kernel 구조는 구현. 다만 실제 GPU fused CUDA 커널은 없고, CPU/ndarray 구현이라 하드웨어 레벨 최적화는 미완. |
| 3.2 SplineReconstructor | 지오데식 스플라인 복원                      | `SplineCache`, `PySplineCache`, `GeodesicMemory`                                 | **80%** | 제어점(time, state, velocity) 기반 cubic Hermite 스플라인, 곡률 보정, batch_reconstruct까지 구현. GPU 커널화는 아직이고, 현재는 CPU 기반.                                                                                               |
| 4. 메모리 계층          | L1/L2/HBM 구조, 캐시 전략                   | Global Basis 공유(`GlobalBasis`), 작은 core 행렬, `GeodesicMemory`/`SplineCache` | **50%** | 메모리 구조(공유 U, 작은 core, 제어점 기반 캐시)는 설계대로지만, 실제 하드웨어 캐시 레벨(L1/L2 배치, HBM 접근 패턴)을 제어하는 수준의 구현은 없음.                                                                                      |

##### (C) Python/Rust Interface 클래스 다이어그램

| 요소          | 구현 위치                                                                | 구현율   | 설명                                                                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ModelLoader   | `RSULFTransformerConverter`                                              | **90%**  | HF 모델에서 가중치 로드, 분석, Global Basis 추출, RSULF 레이어 생성까지 전담.                                                                                                                           |
| RSULFEngine   | `RSULFModel`, `RSULFLMHeadCUDA`                                          | **80%**  | 레이어 스택 + wrapper로 forward 제공. LM 헤드 통합은 `RSULFLMHeadCUDA`와, GPT-2 실험에서는 원래 `lm_head`/`ln_f`를 재사용하는 형태로 구현. `generate` 메서드는 별도 함수(`rsulf_generate_text`)로 존재. |
| Layer         | `RSULFLayer`(Rust), `PyRSULFLayer`, `RSULFLayerCUDA`, `RSULFWrapperCUDA` | **100%** | 메트릭/게이지/FFN/곡률/라플라시안을 캡슐화한 레이어 + PyO3 바인딩 + PyTorch 모듈로 완비.                                                                                                                |
| MemoryManager | `GeodesicMemory`, `SplineCache`, `PyGeodesicMemory`, `PySplineCache`     | **90%**  | 제어점 관리·스플라인 복원·리셋 등 메모리 매니저 역할 구현, RSULFWrapperCUDA에서 실제로 사용.                                                                                                            |

---

### 2. “추론이 디폴트라서 깨지는 것 아닌가?”에 대한 답

**결론부터 말하면, 추론 경로는 디폴트가 아니고 RS-ULF 전용 경로를 타고 있습니다.**  
깨지는 원인은 “디폴트 경로를 쓰기 때문”이 아니라, **RS-ULF 경로의 설계/사용 방식이 아직 GPT-2 구조와 잘 맞지 않기 때문**입니다.

- **추론 경로 실제 구조**

  - `experiments/test_gpt2_conversion.py`에서:
    - 원본: `GPT2LMHeadModel.generate(...)`를 직접 사용.
    - RS-ULF:  
      - `RSULFTransformerConverter(**config)` → `RSULFModel` 생성.  
      - `rsulf_generate_text`에서  
        - `tok_emb = wte(generated)`, `pos_emb = wpe(pos)` → `x = tok_emb + pos_emb`  
        - `h = rs_model_stack(x)` (`RSULFModel.forward`)  
        - `h_last = ln_f(h)`, `logits = lm_head(h_last)`  
      - 즉, **중간 블록은 전부 RS-ULF 레이어 스택(`RSULFWrapperCUDA` → Rust `RSULFLayer`)이 담당**하고, HF 디폴트 블록은 쓰지 않습니다.  
      - 임베딩(`wte`, `wpe`)과 마지막 `ln_f`, `lm_head`만 재사용.

  - `RSULFWrapperCUDA`는 내부에서:
    - `self.norm = nn.LayerNorm(rsulf_layer.d_model, elementwise_affine=True)`  
    - GPT-2 `ln_1` 파라미터를 `RSULFTransformerConverter`가 꺼내 `rsulf.ln_1_weight/bias`로 붙이고, wrapper가 그대로 로딩.  
    - 따라서 **LayerNorm 미적용 상태**는 현재 코드 기준으로는 이미 해결된 상태입니다.

- **왜 여전히 “터무니없이” 깨지는가 (코드 기반 추정)**

  1. **seq_len 설계와 실제 시퀀스 길이의 불일치**

     - RS-ULF Rust 레이어는 `seq_len`을 고정값(실험에서 64)으로 가정하고, 입력은 `(batch_total = batch * seq)`를 **`batch_total / seq_len` 개의 시퀀스로 쪼개서** 어텐션/라플라시안을 적용합니다.
     - 현재 테스트에서는:
       - 레이어 검증(`analyze_layer_fidelity`): 프롬프트 길이는 보통 10토큰 이하인데, `seq_len=64`로 설정.
       - 생성(`rsulf_generate_text`): 매 스텝마다 전체 prefix `x`를 넣기 때문에, 초반에는 `seq < 64`.
     - `RSULFLayer::forward` 안에서:
       - `batch_total = x_flat.nrows()`  
       - `num_seq = batch_total / seq_len` (`seq_len>0` 가정)  
       - `actual_seq_len = seq_len`  
       - **`batch_total < seq_len`이면 `num_seq=0` → 어텐션 루프가 전혀 돌지 않음.**
       - 라플라시안 기반 그래프 확산도 `batch_total >= seq_len && batch_total % seq_len == 0`일 때만 적용 → 대부분의 실제 시퀀스에서는 비활성.
     - 결과:  
       - 상당수 레이어/토큰에 대해 **지오데식 어텐션/그래프 확산이 “사실상 꺼져 있고”**, FFN + 곡률 보정 + 게이지 항 같은 **비선형 변형만 누적**됩니다.  
       - 이는 원래 GPT-2의 self-attention 구조와 완전히 다른 동역학을 만들어 내기 때문에, layer-wise cosine similarity가 빠르게 무너지고, 일부 설정에서 수치 폭발/NaN으로 이어질 수 있습니다.

  2. **자동 회귀 추론에서 `seq==1` 전용 경로가 거의 사용되지 않음**

     - `RSULFWrapperCUDA.forward`는 두 가지 모드가 있습니다.
       - `seq == 1`: `GeodesicMemory`/`SplineCache`를 사용해 최근 궤적을 복원하고, 그 위에서 RSULF를 실행하는 **진짜 “지오데식 KV 스플라인” 경로**.
       - `seq > 1`: 메모리를 리셋하고, 전체 시퀀스를 한 번에 RSULF에 통과시키는 **배치 모드 (오프라인)**.
     - `rsulf_generate_text`는 매 토큰마다 전체 prefix `generated`를 임베딩해서 넣기 때문에, 초반부터 거의 항상 `seq > 1` 브랜치만 타고, **지오데식 메모리/스플라인/Time-step 기반 추론 로직은 사실상 쓰이지 않습니다.**
     - 즉, 논문에서 말하는 “배치 변환기 + 지오데식 스플라인 기반 온라인 추론” 구조 중, **온라인/스플라인 쪽이 설계만 있고 실제 경로에서는 거의 비활성**입니다.

  3. **초기화/스케일링과 LayerNorm 분포의 미스매치 가능성**

     - `calibrate_eta_alpha`는 랜덤 가우시안 샘플 `x`에 대해 W1/W2, g_inv를 사용해 eta/alpha를 적합시키는데, 이 때:
       - 입력 분포는 “LayerNorm 이후의 실제 토큰 분포”가 아니라 **단순 Uniform(-1,1)** 기반.
       - GPT-2의 실제 `ln_1` 출력 분포와 차이가 나면, **포텐셜 쪽 스케일(eta, alpha)이 레이어/모델에 따라 과하거나 부족하게 설정될 수 있음**.
     - 이 부분은 곡률/그래프/게이지와 합쳐져 누적되면, 특정 레이어에서 값이 크게 튀고 이후 레이어에서 NaN으로 붕괴하는 패턴을 만들 수 있습니다.

  4. **지오데식 어텐션의 수치 비용 및 근사**

     - RSULF Rust 어텐션은 현재 **O(N²D) CPU 구현**에 가까운 형태로, `score = x_iᵀ G x_j`를 이중 루프로 계산합니다.
     - 수치 안정화를 위해 여러 클리핑/스케일링이 들어가 있긴 하지만,  
       - G 자체가 큰 스케일을 갖는 경우,  
       - seq_len 불일치로 인해 비정상적인 스팬에 대해 어텐션이 계산되는 경우,  
       - Graph Diffusion, 곡률 보정까지 합해지면, **정상 범위에서 벗어나는 값이 생길 여지가 남아 있습니다.**

---

### 3. 요약: 현재 상태와 “왜 이렇게까지 깨지는지”

- **엔진 구현 측면**에서 보면, 논문 8장(이론/압축/복잡도/청사진)의 **핵심 구조는 대부분 코드에 들어가 있고**, 특히 변환기(Spectrum Analyzer, Rank Planner, Global Basis, RSULF 레이어 생성)와 메모리 계층(GeodesicMemory, SplineCache)은 상당히 충실하게 구현되어 있습니다.
- 반면, **정밀한 포텐셜 증류(작은 MLP 학습), .rsu 바이너리 포맷, O(N) 지오데식 top-k CUDA 경로와의 완전한 통합**은 아직 미완성이고, RankPlan 사용도 반쪽 상태입니다.
- 추론이 깨지는 원인은 **디폴트 HF 경로를 쓰기 때문이 아니라**,  
  - RS-ULF 런타임이 **고정 `seq_len` 가정(64)과 실제 시퀀스 길이 간의 괴리**,  
  - 자동 회귀에서 `seq==1` 전용 지오데식 메모리 경로가 **실제로는 거의 사용되지 않는 구조**,  
  - 그리고 **초기화/스케일링(eta/alpha, g_diag/g_inv)과 GPT-2 분포의 미스매치** 때문에, **이론적으로는 맞지만 실전에서는 안정화되지 않은 동역학**을 만들어 내기 때문입니다.
