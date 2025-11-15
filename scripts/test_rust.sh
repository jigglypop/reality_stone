#!/bin/bash
# Rust 단위 테스트 실행 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "════════════════════════════════════════════════════════"
echo "   Rust 단위 테스트 (CPU 구현)"
echo "════════════════════════════════════════════════════════"
echo ""

cd "$PROJECT_ROOT"

# CPU 전용 테스트
echo "🧪 Running CPU tests..."
cargo test --lib -- --nocapture

TEST_RESULT=$?

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

