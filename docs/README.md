# Reality Stone Documentation

## 🌌 Reality Stone Engine (Integrated Flow)

**Current State of the Art (SOT)** for the project.

- **[05_unified_flow/00_개요.md](05_unified_flow/00_개요.md)**: **RS-ULF (Unified Riemannian Layer Flow)**. 현재 개발의 중심이 되는 통합 엔진 스펙입니다.
- **[00_seed_pack/00_시드_코어.md](00_seed_pack/00_시드_코어.md)**: **v0.8 Seed Engine**. 초기 핵심 컨셉과 압축 이론.

---

## 📚 Documentation Map

### 1. Foundations & Philosophy
> 왜 리만 기하학과 라그랑주 역학인가.

- **[01_philosophy/](01_philosophy/)**: 철학적 배경과 핵심 원리.
- **[02_theory/](02_theory/)**: 수학적 증명, Deep Injection, 리만 최적화(Adam).

### 2. Architecture & Design
> 시스템 전체 구조와 계층적 설계.

- **[03_architecture/](03_architecture/)**: Complete AGI, Hierarchical LLM, Bellman-Geodesic.

### 3. Implementation & API
> 개발자를 위한 구현 가이드와 API 명세.

- **[04_implementation/](04_implementation/)**:
    - [API Specs](04_implementation/03_api_specs/): Unified Layer API.
    - [Hyperbolic Core](04_implementation/hyperbolic_core/): Rust/CUDA Kernels.
    - [Integration](04_implementation/02_기존_LLM_통합.md): LLaMA/Mistral 통합 가이드.

### 4. Research Lab
> 미래 기술 및 심화 연구 주제.

- **[06_research_lab/](06_research_lab/)**:
    - **[New Ideas](06_research_lab/01_new_ideas/)**: 3D Physics Engine, Intent Classification.
    - **[Advanced Research](06_research_lab/02_advanced_research/)**: Manifold Diffusion, YDE.
    - **[Derivations](06_research_lab/03_derivations/)**: Scientific applications (Navier-Stokes, Protein Folding).

### 5. User Guide
> 사용자 매뉴얼.

- **[07_user_guide/](07_user_guide/)**: 저장/로드, 설정 등.

---

## 🛠 Quick Start

참조: `05_unified_flow/05_구현_체크리스트.md`

```bash
# Run tests (TBD)
python -m tests.rs_ulf.test_layer_norm
```

## 📂 Folder Structure

- `00_seed_pack/`: v0.8 Seed Engine Specs
- `01_philosophy/`: Fundamental Concepts
- `02_theory/`: Mathematical Theory
- `03_architecture/`: System Architecture
- `04_implementation/`: Code & API
- `05_unified_flow/`: **Main Engine Specs (RS-ULF)**
- `06_research_lab/`: R&D, Experiments
- `07_user_guide/`: Manuals
