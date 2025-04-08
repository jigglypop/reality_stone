from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="riemannian_extension",
    version="0.1",
    description="Riemannian manifold operations for PyTorch",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=[
        CppExtension(
            name="riemannian_extension._C",
            sources=["riemannian_extension/csrc/extension.cpp"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
)