# CUDA Setup Guide for Reality Stone

이 문서는 Reality Stone을 CUDA 지원 모드로 빌드하고 실행하는 방법을 설명합니다.

## 사전 요구사항

### 1. NVIDIA GPU 및 드라이버
- CUDA 지원 NVIDIA GPU 필요
- 최신 NVIDIA 드라이버 설치 필수
- 드라이버 확인:
  ```bash
  nvidia-smi
  ```

### 2. CUDA Toolkit
- CUDA 12.1 이상 권장
- 설치 경로: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
- 환경 변수 확인:
  ```bash
  echo $CUDA_PATH
  nvcc --version
  ```

### 3. Visual Studio (Windows)
- Visual Studio 2022 Community 이상
- "C++를 사용한 데스크톱 개발" 워크로드 설치
- Windows 10/11 SDK 포함

### 4. Rust 및 Python 환경
- Rust 1.70 이상
- Python 3.8 이상
- uv 패키지 매니저

## 설치 단계

### Step 1: 프로젝트 클론 및 환경 준비

```bash
git clone https://github.com/jigglypop/reality_stone.git
cd reality_stone
```

### Step 2: uv 캐시 디렉토리 설정 (선택사항)

C 드라이브 공간이 부족한 경우, uv 캐시를 다른 드라이브로 이동:

```bash
export UV_CACHE_DIR=E:/uv-cache
mkdir -p "$UV_CACHE_DIR"
```

이 설정을 영구적으로 적용하려면 `.bashrc` 또는 `.bash_profile`에 추가:

```bash
echo 'export UV_CACHE_DIR=E:/uv-cache' >> ~/.bashrc
```

### Step 3: PyTorch CUDA 버전 설치

**중요**: `pyproject.toml`에서 `torch` 의존성이 제거되어 있어야 합니다.

현재 `pyproject.toml`:
```toml
dependencies = [
    "numpy>=1.21.0,<2.0.0",
]
```

uv.lock 파일 제거 및 CUDA PyTorch 설치:

```bash
rm -f uv.lock
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

다른 CUDA 버전을 사용하는 경우:
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124`

### Step 4: PyTorch CUDA 설치 확인

```bash
uv run python -c "import torch; print('torch.__version__ =', torch.__version__); print('torch.version.cuda =', torch.version.cuda); print('torch.cuda.is_available() =', torch.cuda.is_available())"
```

예상 출력:
```
torch.__version__ = 2.5.1+cu121
torch.version.cuda = 12.1
torch.cuda.is_available() = True
```

만약 `torch.cuda.is_available() = False`인 경우:
- GPU 드라이버 재설치
- CUDA Toolkit 재설치
- `nvidia-smi` 명령으로 GPU 인식 확인

### Step 5: Reality Stone CUDA 확장 빌드

#### Windows에서 빌드

**방법 1: Visual Studio Native Tools Command Prompt 사용 (권장)**

1. 시작 메뉴에서 "x64 Native Tools Command Prompt for VS 2022" 실행
2. 프로젝트 디렉토리로 이동:
   ```cmd
   cd /d E:\reality_stone
   ```
3. venv 활성화:
   ```cmd
   .venv\Scripts\activate.bat
   ```
4. CUDA 기능 포함 빌드:
   ```cmd
   uv run maturin develop --features cuda
   ```

**방법 2: Git Bash 사용**

Git Bash에서도 빌드 가능하지만, `build.rs`가 자동으로 Windows SDK 경로를 감지합니다:

```bash
uv run maturin develop --features cuda
```

빌드 중 발생할 수 있는 경고:
- `Compiler family detection failed`: 무시 가능한 경고
- `non-local impl definition`: PyO3 매크로 관련 경고, 무시 가능

성공 메시지:
```
📦 Built wheel for CPython 3.12 to ...
✏️ Setting installed package as editable
🛠 Installed reality_stone-0.2.0
```

### Step 6: Reality Stone CUDA 기능 확인

```bash
uv run python -c "import torch, reality_stone as rs; print('cuda_available =', torch.cuda.is_available()); print('rs._has_cuda =', rs._has_cuda)"
```

예상 출력:
```
cuda_available = True
rs._has_cuda = True
```

`rs._has_cuda = True`는 다음을 의미합니다:
- PyTorch CUDA가 사용 가능
- Reality Stone Rust 확장이 CUDA 심볼을 포함하여 빌드됨
- 모든 CUDA 커널(Mobius, Poincaré, Lorentz, Klein, Geodesic Attention 등)이 사용 가능

### Step 7: CUDA 테스트 실행

```bash
uv run pytest tests/llm/test_bindings_cuda_symbols.py -v
uv run pytest tests/llm/test_poincare_cuda.py -v
uv run pytest tests/llm/test_lorentz_cuda.py -v
uv run pytest tests/llm/test_klein_cuda.py -v
```

CUDA가 올바르게 설정된 경우, 모든 테스트가 실행되며 `skip` 없이 통과합니다.

## 트러블슈팅

### 문제 1: `corecrt.h` 파일을 찾을 수 없음

```
fatal error C1083: Cannot open include file: 'corecrt.h': No such file or directory
```

**해결책**:
- Visual Studio Native Tools Command Prompt에서 빌드
- 또는 Windows SDK 재설치
- `build.rs`가 자동으로 SDK 경로를 감지하지만, 수동 설정이 필요한 경우:
  ```bash
  export INCLUDE="C:\Program Files (x86)\Windows Kits\10\Include\<version>\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\<version>\shared;C:\Program Files (x86)\Windows Kits\10\Include\<version>\um"
  ```

### 문제 2: 디스크 공간 부족 (os error 112)

```
디스크 공간이 부족합니다. (os error 112)
```

**해결책**:
- C 드라이브 공간 확보 (최소 5GB 이상 권장)
- 또는 uv 캐시를 다른 드라이브로 이동 (Step 2 참조)

### 문제 3: `torch.cuda.is_available() = False`

PyTorch는 설치되었지만 CUDA를 인식하지 못하는 경우:

**원인**:
- CPU 버전 PyTorch가 설치됨
- GPU 드라이버 문제
- CUDA Toolkit 버전 불일치

**해결책**:
1. PyTorch 버전 확인:
   ```bash
   uv run python -c "import torch; print(torch.__version__)"
   ```
   `+cpu` 또는 `+cu...` 없이 버전만 나오면 CPU 빌드입니다.

2. CUDA 버전 PyTorch 재설치 (Step 3 참조)

3. GPU 드라이버 확인:
   ```bash
   nvidia-smi
   ```

### 문제 4: `uv run`이 CPU 버전으로 되돌림

`uv pip install`로 CUDA PyTorch를 설치했는데, `uv run` 실행 시 CPU 버전으로 되돌아가는 경우:

**원인**: `pyproject.toml`에 `torch>=2.0.0` 의존성이 있으면 uv가 최신 CPU 빌드를 자동 설치

**해결책**: `pyproject.toml`에서 torch 의존성 제거 (이미 적용됨)

### 문제 5: 링커 에러 (LNK2019, LNK1120)

```
error LNK2019: unresolved external symbol ...
```

**원인**: CUDA 커널 함수 이름과 Rust FFI 선언 불일치

**해결책**: 이미 `#[link_name = "..."]` 속성으로 해결됨

## CUDA 기능 사용 예제

### Python에서 CUDA 커널 직접 호출

```python
import torch
import reality_stone as rs
import numpy as np

if rs._has_cuda and torch.cuda.is_available():
    # Poincaré distance (CUDA)
    x = np.random.randn(10, 128).astype(np.float32)
    y = np.random.randn(10, 128).astype(np.float32)
    c = 1.0
    
    dist = rs.poincare_distance_cuda(x, y, c)
    print("Poincaré distance (CUDA):", dist.shape)
    
    # Mobius addition (CUDA)
    result = rs.mobius_add_cuda(x, y, c)
    print("Mobius addition (CUDA):", result.shape)
```

### HierarchicalLLM에서 자동 CUDA 사용

```python
from reality_stone import HierarchicalLLM

model = HierarchicalLLM(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    manifold_type="poincare",
    curvature=-1.0
)

# torch.cuda.is_available()이 True면 자동으로 CUDA 사용
if torch.cuda.is_available():
    model = model.cuda()

output = model(input_ids, attention_mask)
```

## 성능 최적화

### CUDA 커널 사용 시 권장사항

1. **배치 크기 최적화**: GPU 메모리에 맞게 배치 크기 조정
2. **Mixed Precision**: FP16/BF16 사용으로 속도 향상
3. **메모리 관리**: `torch.cuda.empty_cache()` 주기적 호출
4. **프로파일링**: `torch.profiler` 사용

### 벤치마크

CUDA vs CPU 성능 비교 (예시):

| 연산 | CPU (ms) | CUDA (ms) | 속도 향상 |
|------|----------|-----------|-----------|
| Poincaré Distance (1024x512) | 45.2 | 2.1 | 21.5x |
| Geodesic Attention (512x512) | 123.5 | 8.7 | 14.2x |
| Mobius Addition (2048x768) | 67.8 | 3.4 | 19.9x |

## 참고 자료

- [PyTorch CUDA 설치 가이드](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
- [Visual Studio 다운로드](https://visualstudio.microsoft.com/downloads/)
- [Reality Stone GitHub](https://github.com/jigglypop/reality_stone)

## 문의

문제가 지속되면 GitHub Issues에 다음 정보와 함께 제보해 주세요:
- OS 및 버전
- CUDA Toolkit 버전
- PyTorch 버전
- 전체 에러 로그

