# 인간형 디코더 설계 문서

## 1. 목표

- 기존 GPT-2 스타일의 확률 샘플링(decoding)을 폐기하고, **Skeleton → Relation → Object → Commit**로 이어지는 인간형 디코딩 절차를 공학적으로 정의한다.
- 디코더는 “가중치”가 아니라 **가중치가 만들어내는 기능 응답**을 계층별로 모사하도록 설계한다.

## 2. 표기

- 시점 \(t\) 의 모델 은닉 상태: \(h_t \in \mathbb{R}^d\)
- 어휘 임베딩 행렬: \(E \in \mathbb{R}^{V \times d}\)
- 출력 projection: \(W_o \in \mathbb{R}^{V \times d}\)
- 기본 로짓: \(z_t = W_o h_t\), 확률 \(p_t = \mathrm{softmax}(z_t)\)

## 3. 어휘 기능 분해

어휘 집합 \(\mathcal{V}\) 를 기능별로 나눈다.

| 기호 | 내용 |
| ---- | ---- |
| \(\mathcal{V}_{\text{skel}}\) | 문장 골격(조사, 어미, 구두점, 접속어) |
| \(\mathcal{V}_{\text{rel}}\) | 동사, 관계어, 인과 연결 표현 |
| \(\mathcal{V}_{\text{obj}}\) | 목적어/명사/구체 대상 |
| \(\mathcal{V}_{\text{rest}}\) | 기타 |

분해는 POS 태그, 빈도 기반 규칙, 수작업 리스트 등을 조합해 고정한다.

## 4. 3단계 디코딩

### 4.1 Skeleton 단계 (저주파)

- 마스킹:  
  \( z^{\text{skel}}_t[i] = z_t[i] \) if \(v_i \in \mathcal{V}_{\text{skel}}\), otherwise \(-\infty\)
- 결정 규칙 (temperature=0, top-k=1):  
  \( y^{\text{skel}}_t = \arg\max_i z^{\text{skel}}_t[i] \)

### 4.2 Relation 단계 (중주파)

1. 후보 집합  
   \( \mathcal{C}^{\text{rel}}_t = \{ v_i \in \mathcal{V}_{\text{rel}} \mid z_t[i] \text{ 상위 } K_{\text{rel}} \} \)
2. 컨텍스트  
   \( c^{\text{rel}}_t = f_{\text{rel}}(h_{\le t}, \text{topic}, \text{tree}, \text{metric}) \)
3. 점수  
   \( s^{\text{rel}}_t(i) = \alpha_{\text{logit}} z_t[i] + \alpha_{\text{cos}} \cos(E_i, c^{\text{rel}}_t) \)  
   (또는 리만 구조 사용 시 geodesic 거리 항 추가)
4. 결정  
   \( y^{\text{rel}}_t = \arg\max_{i \in \mathcal{C}^{\text{rel}}_t} s^{\text{rel}}_t(i) \)

### 4.3 Object 단계 (고주파, WTA)

1. 후보 집합  
   \( \mathcal{C}^{\text{obj}}_t = \{ v_i \in \mathcal{V}_{\text{obj}} \mid z_t[i] \text{ 상위 } K_{\text{obj}} \} \)
2. 컨텍스트  
   \( c^{\text{obj}}_t = f_{\text{obj}}(h_{\le t}, \text{paragraph}, \text{topic}, \text{metric}) \)
3. 점수  
   \( s^{\text{obj}}_t(i) = \beta_{\text{logit}} z_t[i] + \beta_{\text{cos}} \cos(E_i, c^{\text{obj}}_t) - \beta_{\text{geo}} d_{\mathcal{M}}(\phi(E_i), \phi(c^{\text{obj}}_t)) \)
4. Winner-Takes-All  
   \( y^{\text{obj}}_t = \arg\max_{i \in \mathcal{C}^{\text{obj}}_t} s^{\text{obj}}_t(i) \)

## 5. Commit 단계

- 최종 출력: \( y_t = y^{\text{obj}}_t \)
- 선택 후 재샘플링 금지 (softmax sampling, temperature 조정, 재호출 등 없음)
- 다음 은닉 상태로 진행: 입력 임베딩 \(x_{t+1} = E_{y_t}\), 모델 forward 진행

## 6. KD 설계 (저/중/고주파)

### 6.1 저주파 (Skeleton 대응)

- 교사 은닉을 SVD 혹은 저주파 projector \(P_{\text{low}}\)로 분해
  \( H^{(l)} = U\Sigma V^\top,\quad H^{(l)}_{\text{low}} = U_{:, :r}\Sigma_{:r,:r} \)
- 학생 은닉 \(S^{(l)}_{\text{low}}\) 과의 loss
  \( \mathcal{L}_{\text{low}} = \sum_l \| H^{(l)}_{\text{low}} - S^{(l)}_{\text{low}} \|_2^2 \)

### 6.2 중주파 (Relation 대응)

- 레이어 차분
  \( \Delta H^{(l)} = H^{(l+1)} - H^{(l)} \)
- 학생과의 차분 일치
  \( \mathcal{L}_{\text{mid}} = \sum_l \| \Delta H^{(l)}_T - \Delta H^{(l)}_S \|_2^2 \)
- Attention 이동 패턴을 쓰는 경우
  \( \mathcal{L}_{\text{flow}} = \sum_l \mathrm{KL}(A_T^{(l)} \,\|\, A_S^{(l)}) \)

### 6.3 고주파 (Object 대응)

- 교사 로짓 \(z^T_t\), 학생 로짓 \(z^S_t\)
- Top-K 집합: \( \mathcal{C}^T_t = \text{TopK}(z^T_t, K) \)
- 순위/에너지 차: \( \Delta z^T_{ij} = z^T_t[i] - z^T_t[j] \)
- Ranking loss  
  \( \mathcal{L}_{\text{rank}}(t) = \sum_{i,j \in \mathcal{C}^T_t} \max(0, 1 - \Delta z^T_{ij}\Delta z^S_{ij}) \)
- Top-1 일치 loss  
  \( \mathcal{L}_{\text{top1}}(t) = \mathbf{1}[\arg\max_i z^T_t[i] \neq \arg\max_i z^S_t[i]] \)
- 고주파 총 loss  
  \( \mathcal{L}_{\text{high}} = \lambda_{\text{rank}}\sum_t \mathcal{L}_{\text{rank}}(t) + \lambda_{\text{top1}}\sum_t \mathcal{L}_{\text{top1}}(t) \)

## 7. 전체 알고리즘

1. 로짓 계산: \(z_t = W_o h_t\)
2. Skeleton 단계로 기본 골격 확정
3. Relation 단계로 동사/관계 확정
4. Object 단계로 목적어 WTA 선택
5. Commit: \(y_t\) 확정, 다음 시점으로 진행

이 절차는 “가중치 기반 KD → 기능 응답 KD → 기하 KD” 의 순서를 디코딩 레벨에서 그대로 반영하며, 인간형 WTA를 그대로 구현한다.

## 8. 컨텍스트 함수 정의

### 8.1 Skeleton

- \( f_{\text{skel}} \) 은 기본적으로 현재 히스토리의 평균 혹은 마지막 문장 임베딩.
- 구현 상 `HierarchicalSentenceTopicLLM.encode_tokens_to_sentences` 출력의 저주파 성분을 사용.

### 8.2 Relation

- \( c^{\text{rel}}_t = \gamma_1 \cdot h_t + \gamma_2 \cdot \text{topic}_t + \gamma_3 \cdot \text{metric}_t \)
- topic 은 `SentenceTopicHead` 의 \(P_{\text{topic}}\) 기대값.
- metric 은 `MetricContextRouter` 의 SPD 슬롯을 flatten 한 후 선형 변환.

### 8.3 Object

- \( c^{\text{obj}}_t = \delta_1 \cdot \text{paragraph\_embedding} + \delta_2 \cdot \text{sentence\_embedding}_t + \delta_3 \cdot \text{retrieved\_support} \)
- paragraph embedding 은 `paragraph_aggregator` 출력.
- retrieved support 는 QA 파이프라인에서 선택된 문장 평균.

## 9. 실제 통합 지점

| 모듈 | 역할 | 접점 |
| --- | --- | --- |
| `experiments/gpt2/decoder.py` | GPT-2 + RSULF inference | Skeleton/Relation/Object 단계를 그대로 넣어 기존 sampling 대체 |
| `python/reality_stone/models/hierarchical_sentence_topic_llm.py` | 계층 디코더 | `infer_hierarchical_llm_on_text` 의 토큰 선택 루틴을 인간형 규칙으로 교체 |
| `RSULFStudentAdapter` | KD | 6장 loss 추가 및 ΔH/Top-K 로그 잔차 계산 |

## 10. 의사코드

```
for t in range(max_len):
    z = W_o h_t
    y_skel = argmax(mask(z, V_skel))
    C_rel = topk(mask(z, V_rel), K_rel)
    y_rel = argmax_i score_rel(i, h_<=t, topic, metric)
    C_obj = topk(mask(z, V_obj), K_obj)
    y_obj = argmax_i score_obj(i, paragraph, sentence, metric)
    y_t = y_obj
    if y_t == eos: break
    h_{t+1} = transformer_step(y_t)
```

## 11. 파라미터 설정 가이드

- \(K_{\text{rel}}\): 8~16, 짧은 텍스트는 8, 장문은 16
- \(K_{\text{obj}}\): 20~32, 목적어 다양성 확보
- \(\alpha_{\text{logit}}, \alpha_{\text{cos}}, \alpha_{\text{geo}}\): logit 1.0 기준, cosine 0.5, geodesic 0.3에서 시작
- \(\gamma\), \(\delta\): normalize 된 벡터 가중치, 합 1 되도록 softmax 사용

## 12. 검증 체크리스트

1. Skeleton 단계에서 허용 토큰 외가 선택되지 않는지 (마스킹 확인)
2. Relation 단계에서 metric 컨텍스트가 존재하지 않을 때 fallback 동작
3. Object 단계에서 cos/geo 점수 계산 시 NaN 방지 (정규화, epsilon)
4. Commit 이후에는 temperature/top-p와 무관하게 결과 고정되는지
5. KD loss 전체가 기존 로짓 MSE 대신 해당 계층 loss만 쓰는지


