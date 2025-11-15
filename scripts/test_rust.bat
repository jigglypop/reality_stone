@echo off
REM Rust 단위 테스트 실행 (Windows)

setlocal

echo ========================================================
echo    Rust 단위 테스트 (CPU 구현)
echo ========================================================
echo.

cd /d "%~dp0\.."

echo Running CPU tests...
cargo test --lib -- --nocapture

set TEST_RESULT=%ERRORLEVEL%

echo.
if %TEST_RESULT% equ 0 (
    echo ========================================================
    echo    모든 테스트 통과! ✅
    echo ========================================================
) else (
    echo ========================================================
    echo    테스트 실패 ❌
    echo ========================================================
)

exit /b %TEST_RESULT%

