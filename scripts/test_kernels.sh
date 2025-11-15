#!/bin/bash
# CUDA 커널 단위 테스트 빌드 및 실행 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CUDA_DIR="$PROJECT_ROOT/src/layers/cuda"

echo "════════════════════════════════════════════════════════"
echo "   CUDA 커널 테스트"
echo "════════════════════════════════════════════════════════"
echo ""

# CUDA 경로 확인
if [ -z "$CUDA_HOME" ] && [ -z "$CUDA_PATH" ]; then
    echo "⚠️  Warning: CUDA_HOME/CUDA_PATH not set"
    CUDA_HOME="/usr/local/cuda"
    echo "   Using default: $CUDA_HOME"
fi

NVCC="${CUDA_HOME}/bin/nvcc"

if [ ! -f "$NVCC" ]; then
    echo "❌ Error: nvcc not found at $NVCC"
    echo "   Please set CUDA_HOME or CUDA_PATH"
    exit 1
fi

echo "✓ Found nvcc: $NVCC"
echo ""

# 빌드
echo "🔨 Building CUDA tests..."
cd "$CUDA_DIR"

# 아키텍처 감지 (sm_70 기본)
ARCH="${CUDA_ARCH:-sm_70}"
echo "   Target architecture: $ARCH"

$NVCC -std=c++11 -arch=$ARCH \
    test_kernels.cu \
    -o test_kernels \
    2>&1 | grep -v "warning: function" || true

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✓ Build complete"
echo ""

# 실행
echo "🚀 Running tests..."
echo ""
./test_kernels

TEST_RESULT=$?

# 정리
rm -f test_kernels

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "════════════════════════════════════════════════════════"
    echo "   모든 테스트 통과! ✅"
    echo "════════════════════════════════════════════════════════"
else
    echo "════════════════════════════════════════════════════════"
    echo "   테스트 실패 ❌"
    echo "════════════════════════════════════════════════════════"
fi

exit $TEST_RESULT

