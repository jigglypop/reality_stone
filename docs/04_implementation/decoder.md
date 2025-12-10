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

## 6. KD 설계 (고주파 계층)

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


