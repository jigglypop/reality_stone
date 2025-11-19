# Reality Stone 문서

Reality Stone 프로젝트는 **인지적 흐름(Cognitive Flow)**에 따라 문서를 구성했습니다. 
처음 접하는 분들은 순서대로 읽으시는 것을 권장합니다.

## 1. Philosophy (Why)
> 왜 이 프로젝트가 필요한가? 역사적, 철학적 배경.

- [`01_WHY_RIEMANNIAN.md`](./01_philosophy/01_WHY_RIEMANNIAN.md) **[필독]**: 유클리드 함정과 역사적 필연성
- [`02_CORE_PRINCIPLES.md`](./01_philosophy/02_CORE_PRINCIPLES.md): 리만-라그랑주 모델의 핵심 원리

## 2. Theory (What)
> 어떤 수학적 이론을 사용하는가? 수식과 원리.

- [`01_SUMMARY.md`](./02_theory/01_SUMMARY.md): 핵심 아이디어 5분 요약
- [`02_CORE_EQUATIONS.md`](./02_theory/02_CORE_EQUATIONS.md): 완전한 수식 체계 (벨만, 리만, 라그랑지안)
- [`03_EQUATION_REFERENCE.md`](./02_theory/03_EQUATION_REFERENCE.md): 빠른 수식 참조
- [`04_COMPARISON.md`](./02_theory/04_COMPARISON.md): 기존 LLM vs Reality Stone 성능 비교

## 3. Architecture (System)
> 무엇을 만드는가? 전체 시스템 설계.

- [`01_COMPLETE_AGI.md`](./03_architecture/01_COMPLETE_AGI.md): 통합 AGI 시스템 조감도 (7개 계층)
- [`02_GEOMETRIC_DESIGN.md`](./03_architecture/02_GEOMETRIC_DESIGN.md): 기하학적 통합 설계
- [`03_HIERARCHICAL_LLM.md`](./03_architecture/03_HIERARCHICAL_LLM.md): 계층적 Sentence-Topic LLM 상세

## 4. Implementation (How)
> 어떻게 구현하는가? 상세 가이드와 코어.

- [`01_GUIDE.md`](./04_implementation/01_GUIDE.md): 모듈별 구현 가이드
- **Hyperbolic Core**:
  - [`POINCARE.md`](./04_implementation/hyperbolic_core/POINCARE.md)
  - [`LORENTZ.md`](./04_implementation/hyperbolic_core/LORENTZ.md)
  - [`KLEIN.md`](./04_implementation/hyperbolic_core/KLEIN.md)

## 5. Roadmap (When)
> 언제 완성되는가?

- [`ROADMAP.md`](./05_roadmap/ROADMAP.md): 단계별 구현 계획 및 마일스톤

---

## 빠른 실행 (Quick Start)

```bash
# 벨만-리만 AGI 데모
python examples/bellman_riemannian_demo.py

# 하이퍼볼릭 레이어 테스트
python -m tests.poincare --quick
```

## 문의 및 기여

프로젝트 저장소: https://github.com/jigglypop/reality_stone
