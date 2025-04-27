from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch

# CUDA 버전 불일치로 인해 CPU 전용 확장만 사용
print("CUDA와 PyTorch 버전 불일치로 인해 CPU 확장만 빌드합니다.")
ext_modules = [
    CppExtension(
        name="riemannian_manifold._C",
        sources=["riemannian_manifold/csrc/extension.cpp"],
        extra_compile_args=['-O3', '-std=c++14']
    )
]

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