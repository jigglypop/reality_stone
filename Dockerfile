# Stage 1: Build the wheel in a full development environment
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Set non-interactive frontend to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies: build tools, Python, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3-pip \
    curl \
    ca-certificates \
    python3-venv \
    git \
    bash \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Rust toolchain using rustup and set CUDA_PATH
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/root/.local/bin:$PATH \
    CUDA_PATH=/usr/local/cuda
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Create a working directory inside the container
WORKDIR /app

# Copy the entire project context into the container
COPY . .

# Install maturin, the build tool for this project
RUN pip3 install maturin

# Build the Python wheel in release mode.
# The output will be placed in the `dist` directory.
RUN maturin build --release --out dist


# Stage 2: Create a lightweight final image with only the runtime dependencies
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the built wheel from the builder stage
COPY --from=builder /app/dist /tmp/dist

# Install Python dependencies using a more optimized method
# 1. Install PyTorch specifically for CUDA 12.1 from its official index.
# 2. Install numpy.
# 3. Install our wheel without its dependencies (as they are already installed).
# 4. Clean up the pip cache in the same layer to reduce final image size.
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install numpy && \
    pip3 install /tmp/dist/*.whl --no-deps && \
    rm -rf /root/.cache/pip

# Clean up temporary files
RUN rm -rf /tmp/dist

# Set a working directory
WORKDIR /app

# The final image is now ready and contains the installed package.
# You can run it and import 'reality_stone' in Python.
# Example: docker run -it --rm --gpus all <image_name> python3
CMD ["python3"]

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH 설정 (uv, cargo)
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# PyTorch 및 기타 Python 종속성 설치
# RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 