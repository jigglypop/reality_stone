# 개발 환경 설정

Reality Stone 개발에 필요한 환경을 구축하는 단계별 가이드입니다.

## 시스템 요구사항

### 필수 요구사항
- **OS**: Linux, macOS, Windows (WSL 권장)
- **Python**: 3.8 이상 (3.10 권장)
- **Rust**: 1.70 이상
- **Git**: 2.20 이상

### 선택적 요구사항 (GPU 가속)
- **CUDA Toolkit**: 11.0 이상
- **NVIDIA GPU**: Compute Capability 7.0 이상
- **cuDNN**: 8.0 이상

## 1단계: 기본 도구 설치

### Rust 설치

```bash
# Rust 설치
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 버전 확인
rustc --version
cargo --version
```

### Python 가상환경 설정

```bash
# pyenv 설치 (권장)
curl https://pyenv.run | bash

# Python 3.10 설치
pyenv install 3.10.12
pyenv global 3.10.12

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### maturin 설치

```bash
# Rust-Python 바인딩 빌드 도구
pip install maturin
```

## 2단계: CUDA 환경 설정 (선택사항)

### CUDA Toolkit 설치

#### Ubuntu/Debian
```bash
# CUDA 키링 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# CUDA 설치
sudo apt update
sudo apt install cuda-toolkit-11-8
```

#### CentOS/RHEL
```bash
# CUDA 저장소 추가
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# CUDA 설치
sudo yum install cuda-toolkit-11-8
```

### 환경 변수 설정

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 적용
source ~/.bashrc
```

### CUDA 설치 확인

```bash
# CUDA 컴파일러 확인
nvcc --version

# GPU 정보 확인
nvidia-smi
```

## 📦 3단계: 프로젝트 설정

### 저장소 클론

```bash
# 메인 저장소에서 클론
git clone https://github.com/jigglypop/reality_stone.git
cd reality_stone

# 또는 포크한 저장소에서 클론
git clone https://github.com/YOUR_USERNAME/reality_stone.git
cd reality_stone
```

### 의존성 설치

```bash
# Python 의존성 설치
pip install -r requirements.txt

# 개발 의존성 설치
pip install -r requirements-dev.txt
```

### 개발 모드 빌드

```bash
# CUDA 없이 빌드
maturin develop

# CUDA와 함께 빌드 (CUDA 환경이 설정된 경우)
maturin develop --features cuda

# 릴리스 모드 빌드 (최적화)
maturin develop --release
```

## ✅ 4단계: 설치 확인

### 기본 기능 테스트

```python
# test_installation.py
import torch
import reality_stone as rs

print(f"Reality Stone 로드 성공!")
print(f"Rust 확장 사용 가능: {rs._has_rust_ext}")
print(f"CUDA 사용 가능: {rs._has_cuda}")

# 기본 연산 테스트
x = torch.randn(10, 64) * 0.1
y = torch.randn(10, 64) * 0.1
result = rs.poincare_ball_layer(x, y, c=1e-3, t=0.5)
print(f"연산 결과 크기: {result.shape}")
print("모든 테스트 통과!")
```

```bash
python test_installation.py
```

### 단위 테스트 실행

```bash
# 모든 테스트 실행
python -m pytest tests/

# 특정 테스트 실행
python -m pytest tests/test_poincare.py -v

# 커버리지 포함 테스트
python -m pytest tests/ --cov=reality_stone --cov-report=html
```

## 🔍 5단계: 개발 도구 설정

### 코드 포맷팅

```bash
# Rust 코드 포맷팅
cargo fmt

# Python 코드 포맷팅
pip install black isort
black python/
isort python/
```

### 린팅

```bash
# Rust 린팅
cargo clippy

# Python 린팅
pip install flake8 mypy
flake8 python/
mypy python/
```

### 사전 커밋 훅 설정

```bash
# pre-commit 설치
pip install pre-commit

# 훅 설치
pre-commit install

# 모든 파일에 대해 실행
pre-commit run --all-files
```

## 6단계: IDE 설정

### VS Code 설정

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "rust-analyzer.cargo.features": ["cuda"],
    "files.associations": {
        "*.cu": "cuda-cpp"
    }
}
```

### 추천 VS Code 확장

```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "rust-lang.rust-analyzer",
        "ms-vscode.cpptools",
        "nvidia.nsight-vscode-edition"
    ]
}
```

## 문제 해결

### 자주 발생하는 문제

#### 1. CUDA 관련 오류
```bash
# 환경 변수 확인
echo $CUDA_HOME
echo $PATH

# CUDA 버전 확인
nvcc --version
```

#### 2. Rust 컴파일 오류
```bash
# Rust 업데이트
rustup update

# 캐시 정리
cargo clean
```

#### 3. Python 바인딩 오류
```bash
# maturin 재설치
pip uninstall maturin
pip install maturin

# 개발 모드 재빌드
maturin develop --release
```

### 성능 최적화 빌드

```bash
# 최고 성능 빌드
RUSTFLAGS="-C target-cpu=native" maturin develop --release

# 프로파일링 정보 포함
maturin develop --release --profile dev
```

## 벤치마크 실행

```bash
# 성능 벤치마크
python benchmarks/run_benchmarks.py

# 메모리 사용량 측정
python benchmarks/memory_benchmark.py

# GPU 성능 측정 (CUDA 사용 시)
python benchmarks/cuda_benchmark.py
```

## 🐳 Docker 개발 환경

```dockerfile
# Dockerfile.dev
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    curl build-essential

# Rust 설치
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 작업 디렉토리 설정
WORKDIR /workspace
COPY . .

# 의존성 설치
RUN pip install -r requirements-dev.txt
RUN maturin develop --features cuda
```

```bash
# Docker 이미지 빌드
docker build -f Dockerfile.dev -t reality-stone-dev .

# 개발 환경 실행
docker run -it --gpus all -v $(pwd):/workspace reality-stone-dev bash
```

## 📝 다음 단계

환경 설정이 완료되었다면:

1. **[시스템 아키텍처](./02_architecture.md)** - 프로젝트 내부 구조 이해
2. **[새 레이어 추가](./03_adding_new_layers.md)** - 새로운 기능 구현 방법
3. **[테스트 가이드](./05_testing.md)** - 테스트 작성 및 실행

---

**문제가 발생했다면**: GitHub Issues에 환경 정보와 함께 문제를 보고해주세요! 