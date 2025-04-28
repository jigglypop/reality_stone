from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
            extra_compile_args={
                "cxx": ["-O3", "/std:c++17"],
                "nvcc": ["-O3", "-std=c++14"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
