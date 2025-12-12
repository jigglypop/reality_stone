# GPT-2 RS-ULF 중간 보고 (2025-12-12)

이 문서는 “현재 구현이 어디까지 왔는지”를 기록하는 **실험 스냅샷/수치 리포트**다.
PEC의 의미/수학/근본 설계는 `docs/06_research_lab/04_gpt2_decoder_stabilization.md`에 분리해 둔다.

---

## 1) 현재 실험 파이프라인(기준 구현)

- **Teacher**: HF `gpt2` (`GPT2LMHeadModel`)
- **RS-ULF 변환**: `RSULFTransformerConverter(exact=True)`
- **BACD(Decoder)**: `fit_riemannian_decoder(... data_mode="teacher_sample")`
  - RS-ULF hidden → teacher hidden으로 선형 맵핑 후 teacher의 `lm_head`로 logits 생성
- **추론 안정화(샘플링 prior 제어)**: `rsulf_generate_text(...)`
  - `teacher_guidance`: RS logits와 teacher logits를 blending
  - `teacher_topk_mask`: teacher top-k 밖 토큰을 마스킹(희귀 토큰 폭주 방지)
- **바인딩 보강**: entity anchor memory + similarity/context/warmup gating
- **PEC(PFC) (Path Error/Curvature Correction)**:
  - `pfc_mode="accel"`: trajectory-only(범용)
  - (옵션) `pfc_mode="bilinear"`: FFN 저랭크 factor 의존(레거시)

관련 코드:
- `experiments/gpt2/main.py`
- `experiments/gpt2/decoder.py`
- `experiments/gpt2/trainer.py`
- `python/reality_stone/models/transformer_converter.py`
- `python/reality_stone/layers/rsulf_cuda.py`

---

## 2) 지표 정의(속도/압축률/정확도)

- **압축률(Compression ratio)**: 변환 로그의 `ratio=K.x` (대략 original/ compressed)
- **정확도(Accuracy proxy)**:
  - **Decoder fit (final_logits)**: BACD 학습 샘플에서 teacher logits과의 유사도(코사인/rel_l2)
  - **Layer-wise similarity**: 레이어별 hidden 유사도(코사인/rel_l2)
- **속도(Speed)**:
  - **Generate time**: 생성 1회 wall time
  - **BACD collect/fit time**: 데이터 수집 + solver/SVD 시간

---

## 3) 최신 실행 스냅샷(2025-12-12, pfc_mode="accel")

실행: `python experiments/gpt2/main.py`

| Rank r | 변환 압축률(로그 ratio) | Decoder fit(final_logits) cos | Decoder fit rel_l2 | BACD collect (s) | BACD fit (s) | Generate time (s) | 관찰 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 356 | ~2.9x | 0.9981 | 0.0485 | 4.51 | 2.41 | 0.95 | 문장 구조 회복. 다만 따옴표/기호 꼬임 일부 잔존 |
| 178 | ~5.7x | 1.0000 | 0.0357 | 4.47 | 2.47 | 0.79 | 문장 진행 OK. 숫자/기호 조각 토큰이 가끔 끼는 현상 |
| 89  | ~11.4x | 0.9961 | 0.0586 | 4.12 | 2.34 | 0.92 | 자연어 비율 증가, 상대적으로 안정적 |

현재 실험 주요 설정(요약):
- teacher-guided: `teacher_guidance=0.25`, `teacher_topk_mask=200`
- entity: `entity_memory=48`, `entity_beta=0.20→0.45(warmup)`, `entity_min_sim=0.40`
- PEC(PFC): `pfc_mode="accel"`, `pfc_curvature=3e-4`, `pfc_window=8`, `pfc_layers=6`, `pfc_speed_gate=1.0`

---

## 4) “개선 상황” 해석(유도: 어떤 변경이 어떤 개선을 만들었나)

1) **Decoder(BACD) 도입/학습 안정화**
   - RS hidden → teacher hidden 복원 경로가 생기면서 logits 붕괴가 완화됨.

2) **Teacher-guided decoding 도입**
   - “희귀 토큰/고정 스트링 폭주”가 크게 줄고 문장성이 회복됨.

3) **Entity memory gating 정교화**
   - 바인딩에 도움이 되는 컨텍스트에서만 작동하도록 제한하면서 오염을 줄임.

4) **PEC(PFC) (accel) 도입**
   - 의미 경로의 급격한 굽힘(2차 차분)을 제한하여, 폭주로 이어지는 drift를 완화하는 방향.
   - 현 단계에서는 “완전 해결”이 아니라 안정화/폭주 방지 성격이 강함.


