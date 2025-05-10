@echo off
:: build.bat

:: Visual Studio 2022 환경 설정
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

:: 필수 환경변수 설정
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=%CUDA_HOME%

:: 기존 빌드 제거
rmdir /s /q build
rmdir /s /q dist
rmdir /s /q *.egg-info

:: 빌드 실행
python setup.py build_ext --inplace
pip install -e .