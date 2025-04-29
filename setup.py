import os
import sys
from setuptools import setup, find_packages

# ① CUDA 탐지 함수 (환경변수나 표준 경로를 확인해 CUDA_HOME 설정)
def detect_cuda():
    if 'CUDA_HOME' in os.environ:
        return True
    for path in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    ):
        if os.path.isdir(path):
            os.environ['CUDA_HOME'] = path
            return True
    return False

# ② 기본값: CPU 전용
ext_modules = []
cmdclass    = {}

# ③ 빌드 시점에만 torch C++ 확장 모듈 import
if any(arg in sys.argv for arg in ('build', 'bdist_wheel')):
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        use_cuda = detect_cuda()
    except ImportError:
        use_cuda = False

    if use_cuda:
        ext_modules = [
            CUDAExtension(
                name="hyper_butterfly._C",
                sources=[
                    "hyper_butterfly/csrc/extension.cpp",
                    "hyper_butterfly/csrc/hyper_butterfly_cpu.cpp",
                    "hyper_butterfly/csrc/hyper_butterfly_cuda.cu",
                ],
                # ────────────────────────────────────
                # 헤더(.h) 검색 경로
                include_dirs=[
                    # 프로젝트 내 헤더
                    "hyper_butterfly/csrc",
                    # Windows SDK 헤더
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um",
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt",
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared",
                    # MSVC 표준 헤더
                    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\include",
                ],
                # ────────────────────────────────────
                # 라이브러리(.lib) 검색 경로
                library_dirs=[
                    r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64",
                    r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64",
                    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\lib\x64",
                ],
                define_macros=[("WITH_CUDA", None)],
                extra_compile_args={
                    # MSVC 최적화 플래그
                    "cxx": ["/O2"],
                    # NVCC 최적화 플래그
                    "nvcc": ["-O3", "--extended-lambda", "-Xcompiler", "/MD"],
                },
            )
        ]
        cmdclass = {'build_ext': BuildExtension}

# ④ setup() 호출
setup(
    name="hyper_butterfly",
    version="0.1.0",
    description="하이퍼볼릭 기하학을 위한 효율적인 PyTorch 라이브러리",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/username/hyper_butterfly",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    package_data={
        # csrc/*.h 를 배포패키지에 포함
        "hyper_butterfly": ["csrc/*.h"],
    },
    install_requires=["torch>=2.0.0"],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
