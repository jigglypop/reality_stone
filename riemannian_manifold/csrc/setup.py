from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 현재 디렉토리 기준으로 소스 파일 경로 확인
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"현재 디렉토리: {current_dir}")
print(f"파일 존재 여부: butterfly_cuda.cu - {os.path.exists(os.path.join(current_dir, 'butterfly_cuda.cu'))}")

# CUDA 버전 불일치 오류 무시
os.environ['TORCH_ALLOW_CUDA_VERSION_MISMATCH'] = '1'

setup(
    name='riemannian_cuda',
    ext_modules=[
        CUDAExtension(
            name='riemannian_cuda',
            sources=[
                os.path.join(current_dir, 'butterfly_cuda.cu'),
                os.path.join(current_dir, 'butterfly_gpu.cu'),
                os.path.join(current_dir, 'extension.cpp')
            ],
            extra_compile_args={
                'cxx': ['-std=c++14', '-O3', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': [
                    '--expt-relaxed-constexpr',
                    '--use_fast_math',
                    '-Xptxas=-v',
                    '--expt-relaxed-constexpr',
                    '--allow-unsupported-compiler',
                    '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '-gencode=arch=compute_89,code=compute_89',
                    '-gencode=arch=compute_89,code=sm_89',
                    '-std=c++17',
                    '--use-local-env'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 