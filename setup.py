from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 컴파일러 플래그 설정
extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3", 
        "--extended-lambda",
        "-Xcompiler", 
        "/MD" 
    ]
}

setup(
    name="riemutils",
    ext_modules=[
        CUDAExtension(
            name="riemutils._C",
            sources=[
                "riemutils/csrc/extension.cpp",
                "riemutils/csrc/hyper_butterfly_cpu.cpp",
                "riemutils/csrc/hyper_butterfly_cuda.cu",
            ],
            include_dirs=["riemutils/csrc"],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
