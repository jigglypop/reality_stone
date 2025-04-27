from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import torch
import os

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Base C++ extension
ext_modules = [
    CppExtension(
        name="riemannian_manifold._C",
        sources=["riemannian_manifold/csrc/extension.cpp"],
        extra_compile_args=['-O3', '-std=c++14']
    )
]

# Add CUDA extension if available
if cuda_available:
    print("CUDA is available. Building with CUDA extensions.")
    cuda_ext = CUDAExtension(
        name="riemannian_manifold.riemannian_cuda",
        sources=[
            "riemannian_manifold/csrc/butterfly_cuda.cu",
            "riemannian_manifold/csrc/butterfly_gpu.cu"
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': ['-O3', '--use_fast_math', '--expt-extended-lambda']
        }
    )
    ext_modules.append(cuda_ext)
else:
    print("CUDA is not available. Building CPU-only extensions.")

setup(
    name="riemannian_manifold",
    version="0.1",
    description="Efficient Hyper-Butterfly Network for PyTorch",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.7.0',
        'numpy>=1.18.0',
    ]
)