#!/bin/bash
# 모든 커널 테스트 실행 (Rust + CUDA)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║         Reality Stone 커널 전체 테스트                   ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

RUST_PASSED=0
CUDA_PASSED=0

# Rust 테스트
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1: Rust CPU Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if bash "$SCRIPT_DIR/test_rust.sh"; then
    RUST_PASSED=1
fi

echo ""
echo ""

# CUDA 테스트
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2: CUDA GPU Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if bash "$SCRIPT_DIR/test_kernels.sh"; then
    CUDA_PASSED=1
fi

echo ""
echo ""

# 최종 결과
echo "╔════════════════════════════════════════════════════════╗"
echo "║                   최종 결과                             ║"
echo "╠════════════════════════════════════════════════════════╣"

if [ $RUST_PASSED -eq 1 ]; then
    echo "║  Rust CPU Tests:  ✅ PASS                              ║"
else
    echo "║  Rust CPU Tests:  ❌ FAIL                              ║"
fi

if [ $CUDA_PASSED -eq 1 ]; then
    echo "║  CUDA GPU Tests:  ✅ PASS                              ║"
else
    echo "║  CUDA GPU Tests:  ❌ FAIL                              ║"
fi

echo "╚════════════════════════════════════════════════════════╝"
echo ""

if [ $RUST_PASSED -eq 1 ] && [ $CUDA_PASSED -eq 1 ]; then
    echo "🎉 모든 커널 테스트 통과!"
    exit 0
else
    echo "⚠️  일부 테스트 실패"
    exit 1
fi

