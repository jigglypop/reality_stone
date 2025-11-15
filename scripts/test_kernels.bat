@echo off
REM CUDA 커널 단위 테스트 빌드 및 실행 (Windows)

setlocal enabledelayedexpansion

echo ========================================================
echo    CUDA 커널 테스트
echo ========================================================
echo.

REM CUDA 경로 확인
if "%CUDA_PATH%"=="" (
    if "%CUDA_HOME%"=="" (
        echo Warning: CUDA_PATH/CUDA_HOME not set
        set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
        echo Using default: !CUDA_PATH!
    ) else (
        set CUDA_PATH=%CUDA_HOME%
    )
)

set NVCC=%CUDA_PATH%\bin\nvcc.exe

if not exist "%NVCC%" (
    echo Error: nvcc not found at %NVCC%
    echo Please set CUDA_PATH or CUDA_HOME
    exit /b 1
)

echo Found nvcc: %NVCC%
echo.

REM 빌드
echo Building CUDA tests...
cd /d "%~dp0\..\src\layers\cuda"

if "%CUDA_ARCH%"=="" (
    set ARCH=sm_70
) else (
    set ARCH=%CUDA_ARCH%
)
echo Target architecture: %ARCH%

"%NVCC%" -std=c++11 -arch=%ARCH% test_kernels.cu -o test_kernels.exe 2>&1

if errorlevel 1 (
    echo Build failed
    exit /b 1
)

echo Build complete
echo.

REM 실행
echo Running tests...
echo.
test_kernels.exe

set TEST_RESULT=%ERRORLEVEL%

REM 정리
del test_kernels.exe 2>nul

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

