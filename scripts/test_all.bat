@echo off
REM 모든 커널 테스트 실행 (Rust + CUDA) - Windows

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║                                                        ║
echo ║         Reality Stone 커널 전체 테스트                   ║
echo ║                                                        ║
echo ╚════════════════════════════════════════════════════════╝
echo.

set RUST_PASSED=0
set CUDA_PASSED=0

REM Rust 테스트
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Phase 1: Rust CPU Tests
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

call "%~dp0test_rust.bat"
if %ERRORLEVEL% equ 0 set RUST_PASSED=1

echo.
echo.

REM CUDA 테스트
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Phase 2: CUDA GPU Tests
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

call "%~dp0test_kernels.bat"
if %ERRORLEVEL% equ 0 set CUDA_PASSED=1

echo.
echo.

REM 최종 결과
echo ╔════════════════════════════════════════════════════════╗
echo ║                   최종 결과                             ║
echo ╠════════════════════════════════════════════════════════╣

if %RUST_PASSED% equ 1 (
    echo ║  Rust CPU Tests:  ✅ PASS                              ║
) else (
    echo ║  Rust CPU Tests:  ❌ FAIL                              ║
)

if %CUDA_PASSED% equ 1 (
    echo ║  CUDA GPU Tests:  ✅ PASS                              ║
) else (
    echo ║  CUDA GPU Tests:  ❌ FAIL                              ║
)

echo ╚════════════════════════════════════════════════════════╝
echo.

if %RUST_PASSED% equ 1 if %CUDA_PASSED% equ 1 (
    echo 🎉 모든 커널 테스트 통과!
    exit /b 0
) else (
    echo ⚠️  일부 테스트 실패
    exit /b 1
)

