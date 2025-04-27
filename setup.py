from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="riemannian_manifold",
    version="0.1",
    description="Efficient Hyper-Butterfly Network for PyTorch",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=[
        CppExtension(
            name="riemannian_manifold._C",
            sources=["riemannian_manifold/csrc/extension.cpp"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
)