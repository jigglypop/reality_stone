import os
import sys
import subprocess
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from pathlib import Path


def get_pytorch_cuda_version():
    """PyTorch가 사용하는 CUDA 버전 확인"""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA version: {cuda_version}")
        return cuda_version
    return None


def find_cuda_installation(preferred_version=None):
    """설치된 CUDA 찾기"""
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    
    if preferred_version:
        # PyTorch가 사용하는 버전 우선 확인
        cuda_path = os.path.join(cuda_base, f"v{preferred_version}")
        if os.path.exists(cuda_path):
            return cuda_path
    
    # 설치된 모든 CUDA 버전 확인
    if os.path.exists(cuda_base):
        versions = [d for d in os.listdir(cuda_base) if d.startswith('v')]
        if versions:
            # 최신 버전 사용
            latest = sorted(versions, reverse=True)[0]
            return os.path.join(cuda_base, latest)
    
    return None


def get_cuda_libraries():
    """PyTorch와 호환되는 CUDA 라이브러리 찾기"""
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    
    # PyTorch lib 폴더의 실제 파일 목록
    if os.path.exists(torch_lib_path):
        lib_files = [f for f in os.listdir(torch_lib_path) if f.endswith('.lib')]
        print(f"Available .lib files in PyTorch: {len(lib_files)} files")
    else:
        lib_files = []
    
    # 필요한 라이브러리와 가능한 변형들
    required_libs = {
        'cudart': ['cudart64_12', 'cudart64_121', 'cudart_static', 'cudart', 'cudart64'],
        'cublas': ['cublas64_12', 'cublas64_121', 'cublas', 'cublas64', 'cublasLt64_12', 'cublasLt64_121'],
        'cublasLt': ['cublasLt64_12', 'cublasLt64_121', 'cublasLt', 'cublasLt64'],
    }
    
    found_libs = []
    
    # PyTorch lib 폴더에서 실제 존재하는 라이브러리만 사용
    for lib_type, candidates in required_libs.items():
        found = False
        for candidate in candidates:
            if f"{candidate}.lib" in lib_files:
                found_libs.append(candidate)
                print(f"Found {lib_type}: {candidate}")
                found = True
                break
        
        if not found:
            print(f"Warning: No {lib_type} library found in PyTorch lib folder")
    
    # PyTorch 기본 라이브러리
    pytorch_libs = []
    for lib in ['c10', 'torch', 'torch_cpu', 'torch_python', 'c10_cuda', 'torch_cuda']:
        if f"{lib}.lib" in lib_files:
            pytorch_libs.append(lib)
    
    # 만약 CUDA 라이브러리를 찾지 못했다면, 기본 이름들로 시도
    if not found_libs:
        print("Warning: No CUDA libraries found in PyTorch lib folder, using default names")
        found_libs = ['cudart', 'cublas', 'cublasLt']
    
    return pytorch_libs + found_libs

def setup_windows_env():
    """Windows 개발 환경 설정"""
    if sys.platform != 'win32':
        return
    
    # Visual Studio 설정
    os.environ['DISTUTILS_USE_SDK'] = '1'
    
    # Visual Studio 찾기
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    ]
    
    for vs_path in vs_paths:
        vcvarsall = os.path.join(vs_path, r"VC\Auxiliary\Build\vcvarsall.bat")
        if os.path.exists(vcvarsall):
            print(f"Found Visual Studio: {vs_path}")
            cmd = f'"{vcvarsall}" x64 && set'
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()
            
            if proc.returncode == 0:
                for line in stdout.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            break
    
    # Windows SDK 설정
    sdk_base = r"C:\Program Files (x86)\Windows Kits\10"
    if os.path.exists(sdk_base):
        bin_path = os.path.join(sdk_base, "bin")
        if os.path.exists(bin_path):
            sdk_versions = [d for d in os.listdir(bin_path) 
                          if os.path.isdir(os.path.join(bin_path, d)) and d.startswith('10.')]
            if sdk_versions:
                latest_sdk = sorted(sdk_versions)[-1]
                print(f"Found Windows SDK: {latest_sdk}")
                
                # rc.exe 경로 추가
                rc_paths = [
                    os.path.join(bin_path, latest_sdk, "x64"),
                    os.path.join(bin_path, latest_sdk, "x86"),
                ]
                
                current_path = os.environ.get('PATH', '')
                new_paths = [p for p in rc_paths if os.path.exists(p)]
                if new_paths:
                    os.environ['PATH'] = ';'.join(new_paths) + ';' + current_path
                # SDK 환경 변수
                os.environ['WindowsSdkDir'] = sdk_base
                os.environ['WindowsSdkVersion'] = latest_sdk
                os.environ['WindowsSdkVerBinPath'] = os.path.join(bin_path, latest_sdk) + '\\'
    pytorch_cuda = get_pytorch_cuda_version()
    cuda_home = None
    if pytorch_cuda:
        cuda_home = find_cuda_installation(pytorch_cuda)
        if not cuda_home and '.' in pytorch_cuda:
            major_version = pytorch_cuda.split('.')[0]
            cuda_candidates = [
                f"v{pytorch_cuda}",
                f"v{major_version}.0",
                f"v{major_version}.1",
                f"v{major_version}.8",
            ]
            cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            for version in cuda_candidates:
                candidate_path = os.path.join(cuda_base, version)
                if os.path.exists(candidate_path):
                    cuda_home = candidate_path
                    break
    if not cuda_home:
        cuda_home = find_cuda_installation()
    if cuda_home:
        print(f"Using CUDA: {cuda_home}")
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['CUDA_PATH'] = cuda_home
        cuda_bin = os.path.join(cuda_home, 'bin')
        if cuda_bin not in os.environ['PATH']:
            os.environ['PATH'] = f"{cuda_bin};{os.environ['PATH']}"


def build_extension():
    """확장 모듈 빌드 설정"""
    try:
        setup_windows_env()
        project_root = Path(__file__).parent.absolute()
        src_dir = project_root / "src"
        cuda_sources = list(src_dir.rglob("*.cu"))
        cpp_sources = list(src_dir.rglob("*.cpp"))
        cuda_sources = [str(p) for p in cuda_sources]
        cpp_sources = [str(p) for p in cpp_sources]
        include_dirs = [
            str(project_root / "src" / "include"),
        ]
        library_dirs = []
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib_path):
            library_dirs.insert(0, torch_lib_path)
        if sys.platform == 'win32':
            sdk_base = r"C:\Program Files (x86)\Windows Kits\10"
            if os.path.exists(sdk_base):
                include_path = os.path.join(sdk_base, "Include")
                lib_path = os.path.join(sdk_base, "Lib")
                
                if os.path.exists(include_path) and os.path.exists(lib_path):
                    sdk_versions = [d for d in os.listdir(include_path) 
                                  if os.path.isdir(os.path.join(include_path, d)) and d.startswith('10.')]
                    if sdk_versions:
                        latest_sdk = sorted(sdk_versions)[-1]
                        include_dirs.extend([
                            os.path.join(include_path, latest_sdk, "ucrt"),
                            os.path.join(include_path, latest_sdk, "um"),
                            os.path.join(include_path, latest_sdk, "shared"),
                        ])
                        library_dirs.extend([
                            os.path.join(lib_path, latest_sdk, "ucrt", "x64"),
                            os.path.join(lib_path, latest_sdk, "um", "x64"),
                        ])
        
        # PyTorch include 경로
        include_dirs.extend(torch.utils.cpp_extension.include_paths())
        
        # CUDA 경로 (있으면 추가)
        cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH'))
        if cuda_home:
            include_dirs.append(os.path.join(cuda_home, 'include'))
            cuda_lib_path = os.path.join(cuda_home, 'lib', 'x64')
            if os.path.exists(cuda_lib_path):
                library_dirs.append(cuda_lib_path)
        
        # CUDA 사용 가능 여부 확인
        has_cuda = torch.cuda.is_available() and len(cuda_sources) > 0
        
        if has_cuda:
            print("Building with CUDA support")
            
            # 컴파일 옵션
            extra_compile_args = {
                "cxx": ["/O2", "/std:c++17", "/MD"],
                "nvcc": [
                    "-O3",
                    "--extended-lambda",
                    "-DWITH_CUDA",
                    "-Xcompiler", "/MD",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                ]
            }
            
            # CUDA 라이브러리
            libraries = get_cuda_libraries()
            print(f"Using libraries: {libraries}")
            
            # 추가 링크 옵션
            extra_link_args = []
            if torch_lib_path:
                extra_link_args.append(f'/LIBPATH:{torch_lib_path}')
            
            ext_modules = [
                CUDAExtension(
                    name="reality_stone._C",
                    sources=cpp_sources + cuda_sources,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    libraries=libraries,
                    define_macros=[("WITH_CUDA", None)],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                )
            ]
        else:
            print("Building CPU-only version")
            
            extra_compile_args = {
                "cxx": ["/O2", "/std:c++17", "/MD"],
            }
            
            ext_modules = [
                CppExtension(
                    name="reality_stone._C",
                    sources=cpp_sources,
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    extra_compile_args=extra_compile_args,
                )
            ]
        
        cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=False)}
        return ext_modules, cmdclass
        
    except Exception as e:
        print(f"Error in build_extension: {e}")
        import traceback
        traceback.print_exc()
        return [], {}


def main():
    """메인 setup 함수"""
    ext_modules, cmdclass = build_extension()
    
    setup(
        name="reality_stone",
        version="0.1.0",
        description="Hyperbolic neural network library for PyTorch",
        author="jigglypop",
        author_email="donghwanyeom@gmail.com",
        url="https://github.com/jigglypop/reality_stone",
        packages=['reality_stone'],
        package_dir={'reality_stone': 'python'},
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        install_requires=[
            "torch>=2.0.0",
            "numpy",
        ],
        python_requires=">=3.7",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
    )


if __name__ == "__main__":
    main()