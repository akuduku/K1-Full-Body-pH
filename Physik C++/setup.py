"""Build the Python extension ph_k1_native (the K1 pH plant in C++/CUDA).

    cd "Physik C++" && python setup.py build_ext --inplace     # -> ph_k1_native*.so next to this file
    pip install --no-build-isolation "./Physik C++"            # or install it (needs the torch it runs with)

CUDA is built when nvcc is found (Kaggle, Colab); PH_K1_CUDA=0 forces CPU only, PH_K1_CUDA=1 requires CUDA.
Architectures: TORCH_CUDA_ARCH_LIST, default "7.5;8.0;8.6;8.9" (T4, A100, A10/RTX 30, L4/RTX 40).
"""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # sources are relative to this folder, wherever setup.py is started from

want = os.environ.get("PH_K1_CUDA", "auto")
has_nvcc = CUDA_HOME is not None and (Path(CUDA_HOME) / "bin" / "nvcc").exists()
use_cuda = has_nvcc if want == "auto" else want == "1"
if want == "1" and not has_nvcc:
    raise SystemExit("PH_K1_CUDA=1 but nvcc was not found (CUDA_HOME)")
if use_cuda:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9")

cxx = ["-O3", "-std=c++20", "-fopenmp"]  # torch >= 2.13 headers need C++20
nvcc = ["-O3", "-std=c++20", "-lineinfo"]  # "-std=" so torch does not add its own -std=c++17
sources = ["src/bindings.cpp", "src/cpu.cpp"] + (["src/cuda.cu"] if use_cuda else [])
Ext = CUDAExtension if use_cuda else CppExtension
ext = Ext(
    name="ph_k1_native",
    sources=sources,
    include_dirs=[str(HERE / "include")],
    define_macros=[("PH_K1_WITH_CUDA", "1")] if use_cuda else [],
    extra_compile_args={"cxx": cxx, "nvcc": nvcc} if use_cuda else {"cxx": cxx},
    extra_link_args=["-fopenmp"],
)
print(f"ph_k1_native: building with {'CUDA (' + os.environ['TORCH_CUDA_ARCH_LIST'] + ')' if use_cuda else 'CPU only'}")

setup(
    name="ph_k1_native",
    version="0.1.0",
    description="K1 port-Hamiltonian plant in C++/CUDA",
    ext_modules=[ext],
    # the repo path contains spaces, which ninja build files do not survive
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
