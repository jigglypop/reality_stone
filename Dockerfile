# 빠른 빌드를 위한 최적화된 Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Rust 설치
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal

# Python 가상환경 생성 및 기본 패키지 설치
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip 업그레이드 및 uv 설치
RUN pip install --upgrade pip setuptools wheel && \
    pip install uv

# uv로 maturin 설치
RUN uv pip install maturin patchelf

# PyTorch 및 LibTorch 설치 (LibTorch 수동 설치 제거)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# tch가 Python의 PyTorch를 사용하도록 설정
ENV LIBTORCH_USE_PYTORCH=1

# CUDA 환경 설정
ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

# 의존성 파일 복사
COPY Cargo.toml Cargo.lock pyproject.toml ./

# 더미 src 디렉토리로 Rust 의존성 빌드
RUN mkdir src && \
    echo "fn main() {}" > src/lib.rs && \
    cargo fetch && \
    rm -rf src

# Python 의존성 설치
COPY requirements.txt* ./
RUN pip install numpy

# 개발 도구 설치
RUN pip install pytest ipython jupyter notebook

# Runtime stage
FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Python 환경 복사
COPY --from=builder /opt/venv /opt/venv

# 환경 변수 설정
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /workspace

CMD ["/bin/bash"]
