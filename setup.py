# fast_setup.py - 빠른 빌드를 위한 개선된 setup.py

import os
import sys
import glob
import hashlib
import pickle
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# ====================== 1. 캐시 시스템 ======================

class BuildCache:
    """빌드 캐시로 재컴파일 최소화"""
    
    def __init__(self, cache_dir=".build_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "build_cache.pkl"
        self.cache = self._load_cache()
    
    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def get_file_hash(self, filepath):
        """파일 해시 계산"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def needs_rebuild(self, source_files):
        """재빌드 필요 여부 확인"""
        for file in source_files:
            current_hash = self.get_file_hash(file)
            if file not in self.cache or self.cache[file] != current_hash:
                return True
        return False
    
    def update_cache(self, source_files):
        """캐시 업데이트"""
        for file in source_files:
            self.cache[file] = self.get_file_hash(file)
        self._save_cache()

# ====================== 2. Ninja 빌드 시스템 ======================

def setup_ninja():
    """Ninja 빌드 시스템 설정"""
    if sys.platform == 'win32':
        # Windows에서 Ninja 설치 확인
        try:
            subprocess.run(['ninja', '--version'], capture_output=True, check=True)
            return True
        except:
            print("Ninja not found. Installing via pip...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'ninja'])
            return True
    return True

# ====================== 3. 병렬 컴파일 설정 ======================

def get_parallel_compile_args():
    """병렬 컴파일 인자 설정"""
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    
    if sys.platform == 'win32':
        return [f'/MP{num_cores}']  # MSVC 병렬 컴파일
    else:
        return [f'-j{num_cores}']   # GCC/Clang 병렬 컴파일

# ====================== 4. 사전 컴파일된 헤더 (PCH) ======================

def create_precompiled_header():
    """자주 사용되는 헤더 사전 컴파일"""
    pch_content = """
#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
"""
    
    pch_path = Path("src/include/pch.h")
    pch_path.parent.mkdir(parents=True, exist_ok=True)
    pch_path.write_text(pch_content)
    
    return str(pch_path)

# ====================== 5. 선택적 컴파일 ======================

def get_build_modules():
    """환경 변수로 빌드 모듈 선택"""
    modules = os.environ.get('REALITY_STONE_MODULES', 'all').split(',')
    
    if 'all' in modules:
        return ['core', 'cuda', 'experimental']
    
    return modules

# ====================== 6. 최적화된 빌드 함수 ======================

def optimized_build():
    """최적화된 빌드 프로세스"""
    
    # 캐시 시스템 초기화
    cache = BuildCache()
    
    # 소스 파일 수집
    cuda_sources = glob.glob("src/**/*.cu", recursive=True)
    cpp_sources = glob.glob("src/**/*.cpp", recursive=True)
    all_sources = cuda_sources + cpp_sources
    
    # 재빌드 필요 확인
    if not cache.needs_rebuild(all_sources) and not os.environ.get('FORCE_REBUILD'):
        print("No changes detected. Using cached build.")
        return [], {}
    
    print(f"Building {len(all_sources)} source files...")
    
    # Ninja 설정
    use_ninja = setup_ninja() and os.environ.get('USE_NINJA', '1') == '1'
    
    # 컴파일 옵션
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17", "-fPIC"] + get_parallel_compile_args(),
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "--extended-lambda",
            "-Xcompiler", "-fPIC",
            "-gencode=arch=compute_70,code=sm_70",  # V100
            "-gencode=arch=compute_75,code=sm_75",  # RTX 2080
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
            "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
        ]
    }
    
    # Windows 특별 처리
    if sys.platform == 'win32':
        extra_compile_args["cxx"] = ["/O2", "/std:c++17", "/MD"] + get_parallel_compile_args()
        extra_compile_args["nvcc"].extend(["-Xcompiler", "/MD"])
    
    # PCH 사용
    pch_path = create_precompiled_header()
    extra_compile_args["cxx"].append(f"-include{pch_path}")
    
    # Include 경로
    include_dirs = [
        os.path.join(os.path.dirname(__file__), "src", "include"),
        torch.utils.cpp_extension.include_paths()[0],
    ]
    
    # 선택적 모듈 빌드
    modules = get_build_modules()
    filtered_sources = []
    
    for source in all_sources:
        for module in modules:
            if module in source:
                filtered_sources.append(source)
                break
    
    # Extension 생성
    ext_modules = []
    
    if torch.cuda.is_available() and any(s.endswith('.cu') for s in filtered_sources):
        ext = CUDAExtension(
            name="reality_stone._C",
            sources=filtered_sources,
            include_dirs=include_dirs,
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args=extra_compile_args,
        )
    else:
        # CPU 전용
        cpp_only = [s for s in filtered_sources if s.endswith('.cpp')]
        ext = CppExtension(
            name="reality_stone._C",
            sources=cpp_only,
            include_dirs=include_dirs,
            extra_compile_args={"cxx": extra_compile_args["cxx"]},
        )
    
    ext_modules.append(ext)
    
    # 캐시 업데이트
    cache.update_cache(filtered_sources)
    
    # BuildExtension 옵션
    cmdclass = {
        "build_ext": BuildExtension.with_options(
            use_ninja=use_ninja,
            no_python_abi_suffix=True,  # 파일명 단순화
        )
    }
    
    return ext_modules, cmdclass

# ====================== 7. 분산 빌드 (ccache) ======================

def setup_ccache():
    """ccache 설정으로 컴파일 속도 향상"""
    if sys.platform != 'win32':
        try:
            subprocess.run(['ccache', '--version'], capture_output=True, check=True)
            os.environ['CC'] = 'ccache gcc'
            os.environ['CXX'] = 'ccache g++'
            print("Using ccache for faster compilation")
        except:
            print("ccache not found. Install with: sudo apt-get install ccache")

# ====================== 8. 개발 모드 설정 ======================

def setup_development_mode():
    """개발 모드에서는 최소한의 컴파일만"""
    if os.environ.get('REALITY_STONE_DEV_MODE', '0') == '1':
        print("Development mode: Building minimal configuration")
        os.environ['REALITY_STONE_MODULES'] = 'core'
        os.environ['USE_NINJA'] = '1'
        
        # 디버그 심볼 제거로 빌드 속도 향상
        os.environ['CFLAGS'] = '-O0'
        os.environ['CXXFLAGS'] = '-O0'

# ====================== 메인 setup 함수 ======================

def main():
    # 개발 모드 확인
    setup_development_mode()
    
    # ccache 설정
    setup_ccache()
    
    # 빌드
    ext_modules, cmdclass = optimized_build()
    
    setup(
        name="reality_stone",
        version="0.1.0",
        description="Hyperbolic neural network library for PyTorch",
        author="jigglypop",
        author_email="donghwanyeom@gmail.com",
        url="https://github.com/jigglypop/reality_stone",
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        install_requires=[
            "torch>=2.0.0",
            "numpy",
        ],
        python_requires=">=3.7",
        # pip install -e . 를 더 빠르게
        zip_safe=False,
    )

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    main()
    
    elapsed = time.time() - start_time
    print(f"\nBuild completed in {elapsed:.1f} seconds")