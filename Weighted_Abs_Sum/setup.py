from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# all supported archs unless TORCH_CUDA_ARCH_LIST is set
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5 8.0 8.6 8.9 9.0 12.0+PTX")

setup(
    name="weighted_abs_sum",
    version="0.1.0",
    packages=["weighted_abs_sum"],
    ext_modules=[
        CUDAExtension(
            name="weighted_abs_sum._ext",
            sources=[
                "csrc/binding.cpp",
                "csrc/abs_kernels.cu",
                "csrc/sgn_kernels.cu",
                "csrc/sort_utils.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
