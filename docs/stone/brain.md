좋아. “리얼리티 스톤” 가설(공유 위상=보존, 리만메트릭=문맥별 변형)의 **1차 역추론·오차점검**을 실제 논문 수치로 바로 해봤어. 우선 대표적이고 정량 수치가 있는 고전/최신 결과들만 뽑아 빠르게 크로스체크했고, 그 결과를 표로 정리해 놨어. (더 깔아볼 수도 있어.)

**요약 결론(1차 패스)**

* 격자세포 집단활동의 **토러스 위상**은 환경/수면 상태가 바뀌어도 보존됨 → “위상 보존, 메트릭만 변환” 축에 **합치**. ([Nature][1])
* 환경/맥락 변동 시 \*\*메트릭에 해당하는 양(격자 스케일/위상/발화률 분배)\*\*이 체계적으로 바뀜(스케일 팽창·위상 쉬프트·율재매핑) → **합치**. ([PMC][2], [SAS][3])
* 인간 fMRI/뇌파에서 **육방향(60°주기) 신호 세기**와 수행도 사이 **양의 상관(r≈0.43)** → “좋은 메트릭 정렬 ↔ 성능↑” 예측과 **합치**(95% CI ≈ \[0.07, 0.69]). ([PMC][4])
* 에너지 쪽은 기본 **시냅스·스파이크 에너지 예산**이 명확(스파이크 47%, 시냅스 34% 등)하지만, **‘메트릭 변화량 ↔ 에너지 변경량’의 직접 숫자결합**은 아직 표본 추출 추가 필요(가능성 높음, 미정량). ([PubMed][5], [SAGE Journals][6], [ScienceDirect][7])

표는 여기 있어: **Reverse-inference check (Batch 1)** — 각 연구의 정량지표, 값, 예측 일치여부(PASS/부분)까지 정리됨.
(표가 보일 거야. 더 넣어달라면 즉시 추가해줄게.)

---

# 이번에 실제로 뽑아낸 핵심 수치와 “오차 검증”

* **토러스 위상 보존(격자세포 집단)**:
  대규모 동시기록(총 7,671 유닛, 6 모듈) + **퍼시스턴트 코호몰로지**로 각 모듈의 집단활동이 **토러스**에 놓임을 확인. 깨어있음/수면/다른 과제에서도 모듈별 토러스 좌표가 유지됨 → “공유 위상은 고정” 예측과 정합. (정량: 모듈 전반에서 H¹ 두 고리/H² 한 공극 신호; 논문 본문 및 바코드 그림 기반) ([Nature][1])

* **맥락 변화 시 메트릭 성분 변화(격자 스케일/위상/발화률)**

  * \*\*새로움(노벨티)\*\*에서 **격자 스케일 팽창** 및 규칙성 저하 → 익숙해지며 복귀(정성·정량 보고). 이는 “부피요소(√det g)↑” **부호 예측**과 합치(숫자%는 PDF 그림에서 추출 필요). ([국립과학원 회의록][8])
  * **환경 변형**에서 모듈별 **리스케일 팩터**와 **경계-격자 상호작용**에 따른 **위상(phase) 쉬프트** 보고 → 문맥특이적 메트릭 변화와 합치. ([PMC][2])
  * **형상/색 변화**에서도 격자 **장소는 유지**하지만 **발화률 분배**가 바뀜(≈40%의 격자세포가 레이트 변화) → 좌표계(위상) 보존 + 메트릭만 변환 패턴과 합치. ([PMC][9])
  * **해마 CA1 재매핑 정량**(MEC LII 흥분 유도 시): **장소 이동 27%**, **On/Off 14%**, **율만 변화 15%** (합계 56%), 나머지 안정(≈44%) → “지도 위상은 크게 보존하되 가중치/율 재분배”와 정합. ([PMC][9])

* **육방향 코드 세기 ↔ 수행도**
  개념공간 과제에서 **vmPFC/EC 등 육방향 신호 Z≈4–5** 보고, 특히 **vmPFC 육방향 신호 세기와 수행 정확도 r=0.432, p=0.039**. (n=28 가정 시 Fisher 변환 95% CI ≈ **\[0.07, 0.69]**: 유의한 양의 상관) → “더 잘 정렬된 메트릭 ↔ 더 낮은 왜곡/오차” 예측과 정합. ([PMC][4])

* **에너지(대사) 쪽 기준선**
  회색질 시냅스·스파이크 에너지 예산: **스파이크 47%**, **시냅스후 34%**, **막 휴지 13%**, **글루탐산 재활용 3%**. 이건 우리 모델의 “메트릭 부피요소 변화 ↔ 발화/시냅스 부하 변화 ↔ **Δ에너지**”로 매핑할 때 기준선으로 사용. (직접 같은 동물·같은 세션에서 **BOLD/CMRO₂/EEG파워**와 **격자 스케일/위상/율 변화**를 동시정량한 데이터셋이 필요) ([PubMed][5], [SAGE Journals][6])

---

# 간단 수식으로 본 ‘예측–데이터’ 체크(부호/크기)

* **예측 A(위상 보존)**: 환경/상태 변화에도 **위상(토폴로지)** 불변, 좌표계는 동일 위상군(torus).
  **데이터**: 모듈 6/6 토러스 분류, 수면/각성/과제 간 유지 → **오차 0(부호)**. ([Nature][1])

* **예측 B(메트릭 변화)**: 문맥 변경 시 \*\*격자 스케일 s, 위상 φ, 지역 발화률 w(x)\*\*가 바뀐다.
  **데이터**: 노벨티→ s>1(팽창), 변형→ 모듈별 rescale/phase-shift, 형상/색→ 장소 고정·율 재분배(≈40%) → **오차 0(부호)**, \*\*크기(%)\*\*는 논문 그림치수 추가추출 필요. ([국립과학원 회의록][8], [PMC][2])

* **예측 C(정렬 품질 ↔ 성능)**: 육방향 코딩 세기↑ ↔ 성능↑.
  **데이터**: r≈0.43(유의) → **오차 0(부호)**, **크기**는 중간효과(95% CI \[0.07, 0.69]). ([PMC][4])

* **예측 D(Δ메트릭 ↔ Δ에너지)**: 대략적으로 **ΔE/E ∝ (s²−1)**(스케일 변화가 면적요소에 반영된다는 가정) 부호 예측 가능.
  **데이터**: 노벨티에서 \*\*BOLD/대사↑\*\*가 다수 보고되지만(작업/자극 의존) **동일 세션 동시정량 수치**가 필요 → **부분 일치(부호)**, **크기 오차 미산출**. ([Rotman Baycrest][10], [PMC][11])

---

# 다음 번 정밀 오차검증(바로 추출 가능한 TODO)

아래는 **추가 정량 추출이 쉬운** 것들부터 확장하면 돼.

1. **격자 스케일 변화량(%) 수치화**

   * Barry 2012(PNAS)에서 신유도 시 **격자 스케일 팽창 비율**(세션별 평균/SEM) 숫자 추출 → 예측 D의 **ΔE/E 부호·크기**와 매칭. ([국립과학원 회의록][8])

2. \*\*Keinath 2018(eLife)\*\*의 **모듈별 rescaling factor** 테이블화

   * 변형 조건별 **rescale factor/phase shift**를 모듈 단위로 벡터화 → “하나의 공간에 겹쳐 저장” 가설에서 말하는 **메트릭 장의 국소변형 패턴**과 1:1 매핑. ([PMC][2])

3. **율재매핑 분율** 고정밀 집계

   * Latuske 2018이 요약한 **27%/14%/15%** 외에, 원자료(도수/SE) 더 발굴 → “정보 손실 없는 중복제거 압축” 관점에서 **활성량 재분배 통계** 정교화. ([PMC][9])

4. **인간 데이터에서 에너지 프록시**

   * **육방향 iEEG 전력 증가율**(정렬 vs 비정렬) 평균%를 숫자로 추출 → 메트릭 정렬 품질 ↔ 에너지(전력) 예측과 대조. ([Cell][12])

5. **기준 에너지 예산과 결합**

   * Attwell & Laughlin의 분해(47/34/13/3%)를 이용해, **예상 Δ발화률/Δ시냅스활성**에서 **ΔE 예상치**를 산출(오차: 보고된 BOLD/CMRO₂ vs 예상 ΔE). ([PubMed][5])

---

# 한 줄 평(현재까지의 검증 상태)

* “**한 우주(공유 위상) 위에 여러 우주(메트릭 변화) 포개어 저장**”이라는 네 프레이밍과 **대표 논문 수치**는 **서로 잘 맞는다**.
* **직접적인 ‘메트릭 변화량 ↔ 에너지 변화량’의 크기 상관**은 **동일 세션 동시 기록 수치**가 더 필요(부호는 대체로 일치).

---

### 인용한 핵심 근거

* **격자세포 토러스 위상(대규모 동시기록 + TDA)**: Gardner et al., *Nature* 2022. ([Nature][1])
* **노벨티에서 격자 스케일 팽창**: Barry et al., *PNAS* 2012. ([국립과학원 회의록][8])
* **환경 변형 → 모듈별 리스케일/경계-격자 상호작용**: Keinath et al., *eLife* 2018. ([PMC][2])
* **형상/색 변화 시 격자 ‘위치 유지·율 변화(≈40%)’**: 리뷰 요약(Latuske 2018) 및 원자료. ([PMC][9])
* **인간 개념공간 육방향 코드, Z점수 & 수행도 상관 r=0.432**: Constantinescu et al., *Science* 2016. ([PMC][4])
* **에너지 예산(스파이크 47%, 시냅스 34% …)**: Attwell & Laughlin 2001(+업데이트 리뷰). ([PubMed][5], [ScienceDirect][7])

---

원하면 바로 **2차 배치**로 더 많은 논문에서 \*\*정량치(퍼센트/효과크기/CI)\*\*를 추가로 긁어와 표를 확장하고, 네 수학 가설의 **수식 예측값(예: ΔE/E ≈ k(s²−1))과 수치 오차**까지 계산해줄게.

[1]: https://www.nature.com/articles/s41586-021-04268-7 "Toroidal topology of population activity in grid cells | Nature"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6203432/?utm_source=chatgpt.com "Environmental deformations dynamically shift the grid cell ..."
[3]: https://www.sas.upenn.edu/psych/epsteinlab/pdfs/Keinath%20Epstein%20Balasubramanian%20eLife%202018.pdf?utm_source=chatgpt.com "Environmental deformations dynamically shift the grid cell spatial ..."
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5248972/ "
            Organizing Conceptual Knowledge in Humans with a Grid-like Code - PMC
        "
[5]: https://pubmed.ncbi.nlm.nih.gov/11598490/?utm_source=chatgpt.com "An energy budget for signaling in the grey matter of the brain"
[6]: https://journals.sagepub.com/doi/full/10.1097/00004647-200110000-00001?utm_source=chatgpt.com "An Energy Budget for Signaling in the Grey Matter of ..."
[7]: https://www.sciencedirect.com/science/article/pii/S0959438822001623?utm_source=chatgpt.com "Paying the brain's energy bill"
[8]: https://www.pnas.org/doi/pdf/10.1073/pnas.1209918109?utm_source=chatgpt.com "Grid cell firing patterns signal environmental novelty by ..."
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5758554/ "
            Hippocampal Remapping and Its Entorhinal Origin - PMC
        "
[10]: https://www.rotman-baycrest.on.ca/files/publicationmodule/%40random45f5724eba2f8/meltzer05_hipp_encode_recall.pdf?utm_source=chatgpt.com "Activation of human hippocampal formation reflects success in ..."
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3604647/?utm_source=chatgpt.com "Hippocampal networks habituate as novelty accumulates"
[12]: https://www.cell.com/current-biology/fulltext/S0960-9822%2818%2931260-0?utm_source=chatgpt.com "Hexadirectional Modulation of High-Frequency ..."
