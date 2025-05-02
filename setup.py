import sys
import glob
from setuptools import setup, find_packages
import torch
import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def detect_cuda():
    if 'CUDA_HOME' in os.environ:
        cuda_home = os.environ['CUDA_HOME']
        print(f"기존 CUDA_HOME 환경변수 사용: {cuda_home}")
        return True
    cuda_paths = []
    for cuda_ver in ['12.1', '12.0', '11.8', '11.7']:
        path = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{cuda_ver}"
        if os.path.isdir(path):
            cuda_paths.append((cuda_ver, path))
    if cuda_paths:
        latest_ver, cuda_path = sorted(cuda_paths, reverse=True)[0]
        os.environ['CUDA_HOME'] = cuda_path
        return True
    return False

def build_cuda_extension():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        cuda_sources = glob.glob(os.path.join("hyper_butterfly", "csrc", "**", "*.cu"), recursive=True)
        cpu_sources = glob.glob("hyper_butterfly/csrc/extension.cpp") + glob.glob("hyper_butterfly/csrc/src/**/*.cpp", recursive=True)
        extra_compile_args = {}
        if sys.platform == 'win32':
            extra_compile_args = {
                "cxx": ["/O2", "/std:c++17"],
                "nvcc": ["-O3", "--extended-lambda", "-Xcompiler", "/MD"]
            }
        else:  # Linux/Mac
            extra_compile_args = {
                "cxx": ["-O3"],
                "nvcc": ["-O3"]
            }
        include_dirs = [
            os.path.join(PROJECT_ROOT, "hyper_butterfly", "csrc", "include"),
            os.path.join(PROJECT_ROOT, "hyper_butterfly", "csrc"),
        ]
        library_dirs = []
        if sys.platform == 'win32':
            vc_tools_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
            sdk_path = r"C:\Program Files (x86)\Windows Kits\10"
            if os.path.exists(vc_tools_path):
                msvc_versions = [d for d in os.listdir(vc_tools_path) if os.path.isdir(os.path.join(vc_tools_path, d))]
                if msvc_versions:
                    latest_msvc = sorted(msvc_versions)[-1]
                    include_dirs.append(os.path.join(vc_tools_path, latest_msvc, "include"))
                    library_dirs.append(os.path.join(vc_tools_path, latest_msvc, "lib", "x64"))
            if os.path.exists(sdk_path):
                include_path = os.path.join(sdk_path, "Include")
                lib_path = os.path.join(sdk_path, "Lib")
                
                if os.path.exists(include_path):
                    sdk_versions = [d for d in os.listdir(include_path) if os.path.isdir(os.path.join(include_path, d))]
                    if sdk_versions:
                        latest_sdk = sorted(sdk_versions)[-1]
                        include_dirs.extend([
                            os.path.join(include_path, latest_sdk, "ucrt"),
                            os.path.join(include_path, latest_sdk, "um"),
                            os.path.join(include_path, latest_sdk, "shared")
                        ])
                        library_dirs.extend([
                            os.path.join(lib_path, latest_sdk, "ucrt", "x64"),
                            os.path.join(lib_path, latest_sdk, "um", "x64")
                        ])
        
        ext_modules = [
            CUDAExtension(
                name="hyper_butterfly._C",
                sources=cpu_sources + cuda_sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                define_macros=[("WITH_CUDA", None)],
                extra_compile_args=extra_compile_args,
                extra_link_args=['/MANIFEST:NO'],
            )
        ]
        
        return ext_modules, {"build_ext": BuildExtension}
    except ImportError as e:
        print(f"CUDA 확장 모듈 설정 실패: {e}")
        return [], {}

# 메인 설정
if __name__ == "__main__":
    has_cuda = detect_cuda() and torch.cuda.is_available()
    ext_modules, cmdclass = [], {}
    if has_cuda:
        try:
            import torch
            ext_modules, cmdclass = build_cuda_extension()
        except ImportError:
            print("PyTorch를 찾을 수 없어 CPU 전용 버전으로 빌드합니다.")
    # setup 호출
    setup(
        name="hyper_butterfly",
        version="0.1.0",
        description="하이퍼볼릭 기하학을 위한 효율적인 PyTorch 라이브러리",
        author="jigglypop",
        author_email="donghwanyeom@gmail.com",
        url="https://github.com/jigglypop/hyper_butterfly",
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
        package_data={
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