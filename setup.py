# setup.py
import sys
import glob
from setuptools import setup, find_packages
import torch
import os

if sys.platform == 'win32':
    os.environ['DISTUTILS_USE_SDK'] = '1'
    os.environ['CUDA_HOME'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    os.environ['CUDA_PATH'] = os.environ['CUDA_HOME']
    os.environ['PATH'] = os.environ['CUDA_HOME'] + r'\bin;' + os.environ.get('PATH', '')
    print(f"CUDA_HOME = {os.environ['CUDA_HOME']}")
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))



# CUDA 경로 자동 감지
def detect_cuda_home():
    if 'CUDA_HOME' in os.environ:
        return os.environ['CUDA_HOME']
    
    if sys.platform == 'win32':
        cuda_paths = []
        for cuda_ver in ['12.8', '12.7', '12.6', '12.1', '12.0', '11.8', '11.7']:
            path = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{cuda_ver}"
            if os.path.isdir(path):
                cuda_paths.append((cuda_ver, path))
        
        if cuda_paths:
            latest_ver, cuda_path = sorted(cuda_paths, reverse=True)[0]
            os.environ['CUDA_HOME'] = cuda_path
            os.environ['CUDA_PATH'] = cuda_path
            print(f"자동 감지된 CUDA 경로: {cuda_path}")
            return cuda_path
    
    return None

def setup_visual_studio_env():
    """Visual Studio 환경 설정"""
    if sys.platform != 'win32':
        return
    
    # Visual Studio 2022의 vcvarsall.bat 찾기
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    
    vcvarsall = None
    for path in vs_paths:
        if os.path.exists(path):
            vcvarsall = path
            break
    
    if vcvarsall:
        import subprocess
        # vcvarsall.bat를 실행하여 환경변수 가져오기
        cmd = f'"{vcvarsall}" x64 && set'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)
        stdout, _ = proc.communicate()
        
        # 환경변수 설정
        for line in stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

def build_cuda_extension():
    try:
        # Visual Studio 환경 설정
        setup_visual_studio_env()
        
        # CUDA 경로 확인
        cuda_home = detect_cuda_home()
        if not cuda_home and torch.cuda.is_available():
            raise RuntimeError("CUDA가 설치되어 있지만 CUDA_HOME을 찾을 수 없습니다.")
        
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        
        # 소스 파일 수집
        cuda_sources = glob.glob(os.path.join("src", "**", "*.cu"), recursive=True)
        cpp_sources = glob.glob(os.path.join("src", "**", "*.cpp"), recursive=True)
        
        # extension.cpp를 제외한 모든 cpp 파일
        cpp_sources = [s for s in cpp_sources if not s.endswith('extension.cpp')]
        cpp_sources.insert(0, "src/extension.cpp")  # extension.cpp를 맨 앞에
        
        extra_compile_args = {}
        if sys.platform == 'win32':
            extra_compile_args = {
                "cxx": ["/O2", "/std:c++17"],
                "nvcc": ["-O3", "--extended-lambda", "-Xcompiler", "/MD"]
            }
        else:
            extra_compile_args = {
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--extended-lambda"]
            }
        
        include_dirs = [
            os.path.join(PROJECT_ROOT, "src", "include"),
        ]
        
        library_dirs = []
        
        # Windows SDK 경로 추가
        if sys.platform == 'win32':
            # Windows SDK
            sdk_path = r"C:\Program Files (x86)\Windows Kits\10"
            if os.path.exists(sdk_path):
                include_path = os.path.join(sdk_path, "Include")
                lib_path = os.path.join(sdk_path, "Lib")
                
                if os.path.exists(include_path):
                    sdk_versions = [d for d in os.listdir(include_path) 
                                  if os.path.isdir(os.path.join(include_path, d))]
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
            
            # Visual Studio MSVC
            vs_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
            if os.path.exists(vs_path):
                msvc_versions = [d for d in os.listdir(vs_path) 
                               if os.path.isdir(os.path.join(vs_path, d))]
                if msvc_versions:
                    latest_msvc = sorted(msvc_versions)[-1]
                    include_dirs.append(os.path.join(vs_path, latest_msvc, "include"))
                    library_dirs.append(os.path.join(vs_path, latest_msvc, "lib", "x64"))
        
        ext_modules = [
            CUDAExtension(
                name="RealityStone._C",
                sources=cpp_sources + cuda_sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                define_macros=[("WITH_CUDA", None)],
                extra_compile_args=extra_compile_args,
                extra_link_args=['/MANIFEST:NO'] if sys.platform == 'win32' else []
            )
        ]
        
        return ext_modules, {"build_ext": BuildExtension}
    except ImportError as e:
        print(f"CUDA 확장 모듈 설정 실패: {e}")
        return [], {}

# 메인 설정
if __name__ == "__main__":
    ext_modules, cmdclass = build_cuda_extension()
    
    setup(
        name="RealityStone",
        version="0.1.0",
        packages=find_packages(),
        package_dir={'': '.'},
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        package_data={
            "RealityStone": ["src/include/**/*.h"],
        },
        install_requires=["torch>=2.0.0"],
        python_requires=">=3.7",
    )
# import sys
# import glob
# from setuptools import setup, find_packages
# import torch
# import os
# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# 
# def detect_cuda():
#     if 'CUDA_HOME' in os.environ:
#         cuda_home = os.environ['CUDA_HOME']
#         print(f"기존 CUDA_HOME 환경변수 사용: {cuda_home}")
#         return True
#     cuda_paths = []
#     for cuda_ver in ['12.1', '12.0', '11.8', '11.7']:
#         path = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{cuda_ver}"
#         if os.path.isdir(path):
#             cuda_paths.append((cuda_ver, path))
#     if cuda_paths:
#         latest_ver, cuda_path = sorted(cuda_paths, reverse=True)[0]
#         os.environ['CUDA_HOME'] = cuda_path
#         return True
#     return False
# 
# def build_cuda_extension():
#     try:
#         from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#         cuda_sources = glob.glob(os.path.join("src", "**", "*.cu"), recursive=True)
#         cpu_sources = glob.glob("src/extension.cpp") + glob.glob("src/core/**/*.cpp", recursive=True)
#         extra_compile_args = {}
#         if sys.platform == 'win32':
#             extra_compile_args = {
#                 "cxx": ["/O2", "/std:c++17"],
#                 "nvcc": ["-O3", "--extended-lambda", "-Xcompiler", "/MD"]
#             }
#         else:  # Linux/Mac
#             extra_compile_args = {
#                 "cxx": ["-O3"],
#                 "nvcc": ["-O3"]
#             }
#         include_dirs = [
#             os.path.join(PROJECT_ROOT, "src", "include"),
#         ]
#         library_dirs = []
#         if sys.platform == 'win32':
#             vc_tools_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
#             sdk_path = r"C:\Program Files (x86)\Windows Kits\10"
#             if os.path.exists(vc_tools_path):
#                 msvc_versions = [d for d in os.listdir(vc_tools_path) if os.path.isdir(os.path.join(vc_tools_path, d))]
#                 if msvc_versions:
#                     latest_msvc = sorted(msvc_versions)[-1]
#                     include_dirs.append(os.path.join(vc_tools_path, latest_msvc, "include"))
#                     library_dirs.append(os.path.join(vc_tools_path, latest_msvc, "lib", "x64"))
#             if os.path.exists(sdk_path):
#                 include_path = os.path.join(sdk_path, "Include")
#                 lib_path = os.path.join(sdk_path, "Lib")
#                 
#                 if os.path.exists(include_path):
#                     sdk_versions = [d for d in os.listdir(include_path) if os.path.isdir(os.path.join(include_path, d))]
#                     if sdk_versions:
#                         latest_sdk = sorted(sdk_versions)[-1]
#                         include_dirs.extend([
#                             os.path.join(include_path, latest_sdk, "ucrt"),
#                             os.path.join(include_path, latest_sdk, "um"),
#                             os.path.join(include_path, latest_sdk, "shared")
#                         ])
#                         library_dirs.extend([
#                             os.path.join(lib_path, latest_sdk, "ucrt", "x64"),
#                             os.path.join(lib_path, latest_sdk, "um", "x64")
#                         ])
#         
#         ext_modules = [
#             CUDAExtension(
#                 name="src._C",
#                 sources=cpu_sources + cuda_sources,
#                 include_dirs=include_dirs,
#                 library_dirs=library_dirs,
#                 define_macros=[("WITH_CUDA", None)],
#                 extra_compile_args=extra_compile_args,
#                 extra_link_args=['/MANIFEST:NO'],
#             )
#         ]
#         
#         return ext_modules, {"build_ext": BuildExtension}
#     except ImportError as e:
#         print(f"CUDA 확장 모듈 설정 실패: {e}")
#         return [], {}
# 
# # 메인 설정
# if __name__ == "__main__":
#     has_cuda = detect_cuda() and torch.cuda.is_available()
#     ext_modules, cmdclass = [], {}
#     if has_cuda:
#         try:
#             import torch
#             ext_modules, cmdclass = build_cuda_extension()
#         except ImportError:
#             print("PyTorch를 찾을 수 없어 CPU 전용 버전으로 빌드합니다.")
#     # setup 호출
#     setup(
#         name="RealityStone",
#         version="0.1.0",
#         description="하이퍼볼릭 기하학을 위한 효율적인 PyTorch 라이브러리",
#         author="jigglypop",
#         author_email="donghwanyeom@gmail.com",
#         url="https://github.com/jigglypop/RealityStone",
#         packages=find_packages(),
#         ext_modules=ext_modules,
#         cmdclass=cmdclass,
#         include_package_data=True,
#         package_data={
#             "RealityStone": ["src/*.h"],
#         },
#         install_requires=["torch>=2.0.0"],
#         python_requires=">=3.7",
#         classifiers=[
#             "Development Status :: 3 - Alpha",
#             "Intended Audience :: Science/Research",
#             "Topic :: Scientific/Engineering :: Artificial Intelligence",
#             "License :: OSI Approved :: MIT License",
#             "Programming Language :: Python :: 3",
#         ],
#     )