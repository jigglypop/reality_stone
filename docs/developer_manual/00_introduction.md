# Reality Stone 개발자 매뉴얼

## 환영합니다!

Reality Stone 프로젝트에 기여하고 싶으신가요? 이 매뉴얼은 프로젝트의 내부 구조를 이해하고 효과적으로 기여할 수 있도록 도와드립니다.

## 대상 독자

- **오픈소스 기여자**: 버그 수정, 새 기능 추가, 문서 개선
- **연구자**: 새로운 하이퍼볼릭 연산 구현, 알고리즘 개선
- **시스템 개발자**: 성능 최적화, CUDA 커널 개발
- **패키지 관리자**: 빌드 시스템, CI/CD, 배포 관리

## 프로젝트 구조 개요

Reality Stone은 다음과 같은 계층적 아키텍처를 가지고 있습니다:

```
Reality Stone
├── Rust 코어 (src/)
│   ├── 핵심 수학 연산 구현
│   ├── CUDA 커널 통합
│   └── 메모리 관리 및 안전성
├── Python 바인딩 (python/)
│   ├── PyTorch 연동
│   ├── 사용자 친화적 API
│   └── 자동 미분 지원
└── 문서 및 예제 (docs/, examples/)
    ├── 사용자 가이드
    ├── API 문서
    └── 실제 사용 예제
```

## 주요 구성 요소

### 1. Rust 코어 (`src/`)
- **`src/layers/`**: 각 하이퍼볼릭 모델의 핵심 구현
- **`src/layers/cuda/`**: CUDA 가속 커널
- **`src/bindings/`**: Python 바인딩 생성
- **`src/ops/`**: 기본 수학 연산들

### 2. Python 패키지 (`python/`)
- **`python/reality_stone/layers/`**: 레이어별 Python 래퍼
- **`python/reality_stone/core/`**: 핵심 연산 래퍼
- **`python/reality_stone/__init__.py`**: 공개 API 정의

### 3. 빌드 시스템
- **`Cargo.toml`**: Rust 프로젝트 설정
- **`pyproject.toml`**: Python 패키지 설정
- **`build.rs`**: CUDA 빌드 스크립트

## 기여 방법

### 1. 코드 기여
- **버그 수정**: 이슈 트래커에서 버그 리포트 확인
- **새 기능**: 새로운 하이퍼볼릭 연산이나 레이어 추가
- **성능 개선**: 기존 코드의 최적화

### 2. 문서 기여
- **API 문서**: 새로운 함수나 클래스 문서화
- **튜토리얼**: 새로운 사용 사례 예제 작성
- **번역**: 다른 언어로 문서 번역

### 3. 테스트 기여
- **단위 테스트**: 새로운 기능에 대한 테스트 작성
- **통합 테스트**: 전체 워크플로우 테스트
- **성능 테스트**: 벤치마크 및 성능 회귀 테스트

## 개발 가이드 목차

1. **[개발 환경 설정](./01_setup_environment.md)** - Rust, CUDA, Python 환경 구축
2. **[시스템 아키텍처](./02_architecture.md)** - 내부 구조 상세 분석
3. **[새 레이어 추가](./03_adding_new_layers.md)** - 새로운 하이퍼볼릭 레이어 구현 가이드
4. **[코딩 스타일](./04_coding_style.md)** - 프로젝트 코딩 규칙 및 베스트 프랙티스
5. **[테스트 가이드](./05_testing.md)** - 테스트 작성 및 실행 방법

## 기여 프로세스

### 1. 이슈 확인
```bash
# GitHub 이슈 확인
https://github.com/jigglypop/reality_stone/issues

# 새로운 기능 제안
https://github.com/jigglypop/reality_stone/discussions
```

### 2. 포크 및 브랜치 생성
```bash
# 저장소 포크
git clone https://github.com/YOUR_USERNAME/reality_stone.git
cd reality_stone

# 새 브랜치 생성
git checkout -b feature/your-feature-name
```

### 3. 개발 및 테스트
```bash
# 개발 환경 설정
./scripts/setup_dev.sh

# 코드 작성
# ... 개발 작업 ...

# 테스트 실행
python tests/test_your_feature.py
```

### 4. 풀 리퀘스트
```bash
# 변경사항 커밋
git add .
git commit -m "Add: new hyperbolic layer implementation"

# 푸시
git push origin feature/your-feature-name

# GitHub에서 PR 생성
```

## 기여할 수 있는 영역

### 우선순위 높음
- **성능 최적화**: CUDA 커널 최적화
- **새로운 하이퍼볼릭 모델**: 새로운 기하학적 모델 구현
- **메모리 효율성**: 대규모 데이터 처리 개선

### 중간 우선순위
- **API 개선**: 더 직관적인 사용자 인터페이스
- **문서화**: 더 자세한 예제와 튜토리얼
- **테스트 커버리지**: 더 포괄적인 테스트 스위트

### 장기 목표
- **다른 프레임워크 지원**: TensorFlow, JAX 바인딩
- **분산 처리**: 멀티 GPU, 클러스터 지원
- **자동 최적화**: 하이퍼파라미터 자동 튜닝

## 기여자 인정

우리는 모든 기여자를 소중히 여기며 다음과 같은 방식으로 인정합니다:

- **README 기여자 목록**: 모든 기여자 이름 표시
- **릴리스 노트**: 주요 기여 내용 언급
- **특별 감사**: 중요한 기여에 대한 별도 감사 표시

## 도움이 필요하다면

- **GitHub Discussions**: 일반적인 질문과 토론
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **이메일**: 민감한 보안 문제나 개인적인 문의

## 행동 강령

우리는 모든 기여자가 존중받고 환영받는 환경을 만들기 위해 노력합니다:

- **존중**: 모든 의견과 기여를 존중합니다
- **포용**: 다양한 배경의 사람들을 환영합니다
- **건설적 피드백**: 비판적이지만 건설적인 피드백을 제공합니다
- **학습 지향**: 실수를 통해 배우는 것을 장려합니다

---

**다음 단계**: [개발 환경 설정](./01_setup_environment.md)에서 개발 환경을 구축해보세요! 