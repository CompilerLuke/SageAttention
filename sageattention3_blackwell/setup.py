import warnings
import os
import sysconfig
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = Path(this_dir)

PACKAGE_NAME = "sageattn4"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"
CHECK_GENERATED_TYPES = os.getenv("SAGEATTN4_CHECK_GENERATED_TYPES", "FALSE") == "TRUE"
NVCC_FAST_COMPILE = os.getenv("SAGEATTN4_NVCC_FAST_COMPILE", "")



def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


def python_nvidia_include_dirs():
    include_dirs = []
    roots = {
        Path(path)
        for path in (
            sysconfig.get_paths().get("purelib"),
            sysconfig.get_paths().get("platlib"),
        )
        if path
    }
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for include_dir in sorted(nvidia_root.glob("*/include")):
            if any((include_dir / header).exists() for header in ("cusparse.h", "cublas_v2.h", "cusolverDn.h")):
                include_dirs.append(include_dir)
    return include_dirs


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if (cc_major, cc_minor) == (10, 0):  # sm_100
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_100a,code=sm_100a")
    elif (cc_major, cc_minor) == (12, 0):  # sm_120
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120a,code=sm_120a")
    elif (cc_major, cc_minor) == (12, 1):  # sm_121
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_121a,code=sm_121a")
    else:
        raise RuntimeError("Unsupported GPU")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    cutlass_dir = repo_dir / "csrc" / "cutlass"
    (repo_dir / "csrc").mkdir(parents=True, exist_ok=True)
    if not cutlass_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", str(cutlass_dir)],
            check=True
        )
    nvcc_flags = [
        "-O3",
        # "-O0",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=-v",  # printing out number of registers
        "--ptxas-options=--verbose,--warn-on-local-memory-usage",  # printing out number of registers
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
    ]
    if CHECK_GENERATED_TYPES:
        nvcc_flags.append("-DSAGEATTN4_CHECK_GENERATED_TYPES=1")
    if NVCC_FAST_COMPILE:
        nvcc_flags.append(f"--Ofast-compile={NVCC_FAST_COMPILE}")
    cccl_candidates = [
        Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",
        Path(CUDA_HOME) / "include" / "cccl",
        *sorted(Path("/usr/local").glob("cuda-*/targets/x86_64-linux/include/cccl")),
    ]
    cccl_include_dirs = [path for path in cccl_candidates if (path / "cuda" / "std" / "utility").exists()]
    include_dirs = [
        repo_dir / "sageattn4",
        repo_dir / "sageattn4" / "blackwell",
        repo_dir / "sageattn4" / "quantization",
        Path(CUDA_HOME) / "include",
        *python_nvidia_include_dirs(),
        *cccl_include_dirs[:1],
        cutlass_dir / "include",
        cutlass_dir / "tools" / "util" / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name="sageattn4_fwd_cuda",
            sources=[
                "sageattn4/blackwell/api.cpp",
                "sageattn4/blackwell/fwd_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="sageattn4_quant_cuda",
            sources=["sageattn4/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )



class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        super().run()

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="FP4FlashAttention",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
