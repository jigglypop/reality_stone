from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='riemannian_cuda',
    ext_modules=[
        CUDAExtension('riemannian_cuda', [
            'poincare_cuda.cu',
            'butterfly_gpu.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 