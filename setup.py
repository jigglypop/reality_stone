import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ['VS_NO_MANIFEST_CREATION'] = '1'

# CUDA 탐지 함수
def detect_cuda():
    if 'CUDA_HOME' in os.environ:
        print(f"CUDA_HOME 환경 변수: {os.environ['CUDA_HOME']}")
        return True
    for path in (
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    ):
        if os.path.isdir(path):
            os.environ['CUDA_HOME'] = path
            print(f"CUDA 경로 감지: {path}")
            return True
    print("CUDA를 찾을 수 없습니다.")
    return False

# 기본값: CPU 전용
ext_modules = []
cmdclass = {'build_ext': BuildExtension}

# 항상 C++ 확장 모듈 빌드
try:
    use_cuda = detect_cuda()
    if use_cuda:
        ext_modules = [
            CUDAExtension(
                # 모듈 이름을 hyper_butterfly._C로 설정
                name="hyper_butterfly._C",
                sources=[
                    # 메인 확장 파일
                    "hyper_butterfly/csrc/extension.cpp",
                    # 포앵카레 기하학 구현
                    "hyper_butterfly/csrc/geometry/poincare/forward_poincare_cpu.cpp",
                    "hyper_butterfly/csrc/geometry/poincare/forward_poincare_gpu.cu",
                    "hyper_butterfly/csrc/geometry/poincare/poincare_backward_gpu.cu",
                    # 버터플라이 변환 구현
                    "hyper_butterfly/csrc/ops/butterfly/forward_butterfly_cpu.cpp",
                    "hyper_butterfly/csrc/ops/butterfly/forward_butterfly_gpu.cu",
                    "hyper_butterfly/csrc/ops/butterfly/backward_butterfly_gpu.cu",
                ],
                # 헤더(.h) 검색 경로
                include_dirs=[
                    # 프로젝트 내 헤더
                    "hyper_butterfly/csrc",
                    "hyper_butterfly/csrc/utils",
                    "hyper_butterfly/csrc/geometry",
                    "hyper_butterfly/csrc/ops",
                    # Windows SDK 헤더
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um",
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt",
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared",
                ],
                # 라이브러리(.lib) 검색 경로
                library_dirs=[
                    r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64",
                    r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64",
                ],
                define_macros=[("WITH_CUDA", None)],
                extra_compile_args={
                    # MSVC 최적화 플래그
                    "cxx": ["/O2"],
                    # NVCC 최적화 플래그
                    "nvcc": ["-O3", "--extended-lambda", "-Xcompiler", "/MD"],
                },
                extra_link_args=["/MANIFEST:NO"],  # 매니페스트 생성 비활성화
            )
        ]
    else:
        # CPU 전용 확장 모듈
        from torch.utils.cpp_extension import CppExtension
        ext_modules = [
            CppExtension(
                name="hyper_butterfly._C",
                sources=[
                    "hyper_butterfly/csrc/extension.cpp",
                    "hyper_butterfly/csrc/geometry/poincare/forward_poincare_cpu.cpp",
                    "hyper_butterfly/csrc/ops/butterfly/forward_butterfly_cpu.cpp",
                    "hyper_butterfly/csrc/geometry/poincare/poincare_backward_cpu.cpp",
                ],
                include_dirs=[
                    "hyper_butterfly/csrc",
                    "hyper_butterfly/csrc/utils",
                    "hyper_butterfly/csrc/geometry",
                    "hyper_butterfly/csrc/ops",
                    # Windows SDK 헤더
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um",
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt",
                    r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared",
                ],
                library_dirs=[
                    r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64",
                    r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64",
                ],
                extra_compile_args={"cxx": ["/O2"]},
                extra_link_args=["/MANIFEST:NO"],  # 매니페스트 생성 비활성화
            )
        ]
except Exception as e:
    print(f"확장 모듈 설정 중 오류 발생: {e}")
    print("순수 Python 모듈로 설치합니다.")
    ext_modules = []

# setup() 호출
setup(
    name="hyper_butterfly_fft",
    version="0.1.0",
    description="하이퍼볼릭 기하학을 위한 효율적인 PyTorch 라이브러리",
    author="jigglypop",
    author_email="donghwanyeom@gmail.com",
    url="https://github.com/jigglypop/hyper_butterfly",
    packages=['hyper_butterfly'], 
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    package_data={
        # hyper_butterfly 패키지에 헤더 파일 포함
        "hyper_butterfly": [
            "csrc/*.h",
            "csrc/utils/*.h",
            "csrc/geometry/*.h",
            "csrc/geometry/poincare/*.h",
            "csrc/ops/butterfly/*.h",
        ],
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

