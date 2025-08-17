
# Reality Stone: 하이퍼볼릭 신경망을 위한 Rust/CUDA 가속 라이브러리

Reality Stone은 Poincaré/Lorentz/Klein 하이퍼볼릭 모델의 핵심 연산과 레이어를 Rust(+CUDA)로 구현하고, PyTorch Autograd로 노출하는 라이브러리입니다. 동적 곡률 최적화와 스플라인 기반 가중치 보간(압축)을 지원합니다.

현재 릴리스: 0.2.0, 라이선스: MIT, Python 3.8–3.12, PyTorch 2.0+ 지원.


## 주요 기능

- 하이퍼볼릭 모델 지원: Poincaré, Lorentz, Klein
- 기본 연산: 덧셈, 스칼라 곱, 거리, 모델 간 변환(Poincaré↔Lorentz/Klein)
- 레이어: `PoincareBallLayer`, `LorentzLayer`, `KleinLayer`
- 동적 곡률: 레이어별 `kappa`로 곡률 `c`를 학습 (정확한 backward 포함)
- 스플라인 압축: `SplineLinear`로 가중치 보간 + 잔차 보정
- 고성능 코어: Rust/ndarray 병렬화, CUDA 커널 가속


## 설치 및 빌드

사전 준비물
- Python 3.8 이상, pip
- Rust toolchain (stable), Cargo
- PyTorch 2.0 이상
- CUDA 사용 시: NVIDIA CUDA Toolkit 설치, 환경 변수 `CUDA_HOME` 또는 `CUDA_PATH` 설정

가상환경 생성 및 필수 패키지 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU 전용 빌드
```bash
maturin develop
python -c "import reality_stone as rs; print(rs._has_rust_ext, rs._has_cuda)"
```

CUDA 빌드
```bash
export CUDA_HOME=/usr/local/cuda   # Windows: set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x
maturin develop --features cuda --release
python -c "import reality_stone as rs; print(rs._has_rust_ext, rs._has_cuda)"
```

주의: 기본 CUDA 아키텍처 플래그는 `sm_70`입니다. 다른 GPU를 사용한다면 `build.rs`의 `-arch=sm_70`을 환경에 맞게 수정하세요.


## 빠른 예제

Poincaré 레이어
```python
import torch
import reality_stone as rs

u = torch.randn(4, 8)
v = torch.randn(4, 8)
y = rs.poincare_ball_layer(u, v, c=1e-3, t=0.7)
```

거리와 좌표 변환
```python
import torch
import reality_stone as rs

x = torch.randn(2, 4)
y = torch.randn(2, 4)
d = rs.poincare_distance(x, y, c=1e-3)
xL = rs.poincare_to_lorentz(x, c=1e-3)
xK = rs.poincare_to_klein(x, c=1e-3)
```

스플라인 기반 압축 레이어
```python
import torch
import torch.nn as nn
import reality_stone as rs

linear = nn.Linear(512, 256)
spline = rs.SplineLinear.from_linear(linear, k=8, learning_rate=0.01, steps=100, use_residual=True)
out = spline(torch.randn(2, 512))
```

모델을 하이퍼볼릭 레이어로 변환
```python
import torch.nn as nn
from reality_stone.layers import EquivalentHyperbolicLinear
from reality_stone.conversion import convert_to_hyperbolic

model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
convert_to_hyperbolic(model, {nn.Linear: EquivalentHyperbolicLinear}, c=1e-3)
```


## 아키텍처 개요

디렉터리 구조
```
src/            # Rust 코어 (layers, ops, bindings, cuda)
python/         # Python API 및 Autograd 레이어
tests/          # 예제/검증 스크립트
docs/           # 문서
```

데이터 플로우
- Python `torch.autograd.Function` → PyO3 모듈 `_rust` → Rust ndarray 구현 → (선택) CUDA 커널
- GPU 사용 시 PyTorch 텐서 포인터를 커널에 직접 전달하여 불필요한 복사를 최소화합니다.


## API 개요 (Python)

상위 함수
- `poincare_ball_layer(u, v, c, t)`
- `lorentz_layer(u, v, c, t)`
- `klein_layer(u, v, c, t)`

거리/변환
- `poincare_distance(x, y, c)`
- `poincare_to_lorentz(x, c)`, `poincare_to_klein(x, c)`
- `lorentz_to_poincare(x, c)`, `lorentz_to_klein(x, c)`
- `klein_to_poincare(x, c)`, `klein_to_lorentz(x, c)`

레이어
- `PoincareBallLayer`, `LorentzLayer`, `KleinLayer`
- 하이퍼볼릭 선형 변형: `HyperbolicLinear`, `GeodesicLinear`, `EquivalentHyperbolicLinear`
- 압축: `SplineLinear`

기타
- 투영: `project_to_ball(x, epsilon)`
- 메트릭 합성: `from reality_stone import metrikey` (SPD 메트릭/합성/암시적 변환 함수 제공)


## 테스트 실행

MNIST 간단 테스트
```bash
python -m tests.poincare --mode both --quick --epochs 2 --batch-size 256
python -m tests.lorentz  --quick --epochs 2 --batch-size 256
python -m tests.klein    --quick --epochs 2 --batch-size 256
```

공통 옵션: `--device {auto,cpu,cuda}`, `--data-dir tests/data`, `--epochs`, `--batch-size`, `--lr`, `--t`, `--c`, `--quick`, `--seed`


## 문제 해결

- Rust 확장 모듈을 찾지 못함: `maturin develop`로 빌드 후 다시 시도하세요.
- CUDA가 비활성으로 표시됨: `CUDA_HOME`/`CUDA_PATH` 확인, GPU 드라이버/Toolkit 설치 상태 점검, CUDA 피처로 빌드했는지 확인.
- Windows 빌드: Visual C++ Build Tools 설치 필요. PowerShell 대신 CMD/Developer Prompt 또는 Git Bash 사용 가능.
- CUDA 아키텍처 오류: `build.rs`의 `-arch=sm_70`을 환경에 맞게 수정.
- NumPy 2.x와의 호환성: 현재 `numpy>=1.21,<2.0`을 사용합니다.


## 변경 사항 요약 (v0.2.0)

- Poincaré/Lorentz/Klein 레이어 및 연산의 Python Autograd 경로 정비
- 동적/레이어별 곡률 API 추가: `poincare_ball_layer_layerwise_cpu` 및 정확한 backward
- 스플라인 압축 레이어 `SplineLinear` 추가 및 `from_linear` 최적화 파이프라인 도입
- 하이퍼볼릭 선형 변형 레이어군 추가: `HyperbolicLinear`, `GeodesicLinear`, `EquivalentHyperbolicLinear`, `CompactEquivalentHyperbolicLinear`
- `metrikey` 서브모듈 공개: SPD 메트릭 합성/적용, 암시적 변환 체인
- RBE 모듈은 현재 비활성화됨(코어에 잔존 코드와 문서 일부는 유지되나 런타임에서 사용되지 않음)


## 라이선스

MIT License

