import os
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Verify CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install a CUDA-enabled PyTorch.")

# Common and CUDA sources as per the original nvdiffrast repository structure
cpp_sources = [
    # Torch bindings
    "nvdiffrast/torch/torch_bindings.cpp",
    "nvdiffrast/torch/torch_bindings_gl.cpp",
    "nvdiffrast/torch/torch_rasterize.cpp",
    "nvdiffrast/torch/torch_rasterize_gl.cpp",
    "nvdiffrast/torch/torch_texture.cpp",
    "nvdiffrast/torch/torch_antialias.cpp",
    "nvdiffrast/torch/torch_interpolate.cpp",

    # Common C++ sources
    "nvdiffrast/common/common.cpp",
    "nvdiffrast/common/glutil.cpp",

     #This error occurs because both RasterImpl.cpp and RasterImpl.cu produce the same output object file name RasterImpl.obj.
    #  On Windows, by default, both a .cpp and .cu file with the same base name (e.g. RasterImpl) 
    # will compile to the same .obj file, causing a conflict. So, keep the following line comented-out:
    # "nvdiffrast/common/texture.cpp",

    # CudaRaster CPU C++ implementation
    "nvdiffrast/common/cudaraster/impl/Buffer.cpp",
    "nvdiffrast/common/cudaraster/impl/CudaRaster.cpp",
    #This error occurs because both RasterImpl.cpp and RasterImpl.cu produce the same output object file name RasterImpl.obj.
    #  On Windows, by default, both a .cpp and .cu file with the same base name (e.g. RasterImpl) 
    # will compile to the same .obj file, causing a conflict. So, keep the following line comented-out:
    # "nvdiffrast/common/cudaraster/impl/RasterImpl.cpp",
]

cuda_sources = [
    # Common CUDA sources
    "nvdiffrast/common/rasterize.cu",
    "nvdiffrast/common/interpolate.cu",
    "nvdiffrast/common/antialias.cu",
    "nvdiffrast/common/texture.cu",

    # CudaRaster CUDA implementation
    "nvdiffrast/common/cudaraster/impl/RasterImpl.cu",
]

all_sources = cpp_sources + cuda_sources

include_dirs = [
    "nvdiffrast/common",
    "nvdiffrast/common/cudaraster",
    "nvdiffrast/common/cudaraster/impl",
    "nvdiffrast/torch",
    os.path.join(os.path.dirname(torch.__file__), 'include'),
    os.path.join(os.path.dirname(torch.__file__), 'include', 'torch', 'csrc', 'api', 'include'),
    os.path.join(os.path.dirname(torch.__file__), 'include', 'TH'),  # Add this
    os.path.join(os.path.dirname(torch.__file__), 'include', 'THC'),  # Add this
]

# Determine compiler flags
extra_cxx_flags = []
if os.name == 'nt':
    # On Windows use MSVC flags
    extra_cxx_flags.append('/std:c++17')
else:
    extra_cxx_flags.append('-std=c++17')

# CUDA architectures
nvcc_flags = [
    '--use_fast_math',
    '-O3',
    '--expt-relaxed-constexpr',
    '--allow-unsupported-compiler',  #else my VisualStudio 2022 isn't recognized.
    '-DNVDR_USE_TORCH',
    '-DNVDR_CTX_ARGS',
    '-DNVDR_CTX_PARAMS',
    # Add arch for Pascal, Turing, Ampere, Ada
    '-gencode=arch=compute_61,code=sm_61',
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_89,code=sm_89',
    '-gencode=arch=compute_89,code=compute_89'
]

# Read version from __init__.py
version = None
with open("nvdiffrast/__init__.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("'").strip('"')
            break
if version is None:
    raise RuntimeError("Could not find __version__ in nvdiffrast/__init__.py")

# Long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nvdiffrast",
    version=version,
    author="Samuli Laine",
    author_email="slaine@nvidia.com",
    description="nvdiffrast - modular primitives for high-performance differentiable rendering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVlabs/nvdiffrast",
    packages=setuptools.find_packages(),
    package_data={
        'nvdiffrast': [
            'common/*.h',
            'common/*.inl',
            'common/*.cu',
            'common/*.cpp',
            'common/cudaraster/*.hpp',
            'common/cudaraster/impl/*.cpp',
            'common/cudaraster/impl/*.hpp',
            'common/cudaraster/impl/*.inl',
            'common/cudaraster/impl/*.cu',
            'lib/*.h',
            'torch/*.h',
            'torch/*.inl',
            'torch/*.cpp',
            'tensorflow/*.cu',
        ] + (['lib/*.lib'] if os.name == 'nt' else [])
    },
    include_package_data=True,
    install_requires=['numpy'],
    python_requires='>=3.6',
    ext_modules=[
        CUDAExtension(
            'nvdiffrast._C',
            sources=all_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': extra_cxx_flags,
                'nvcc': nvcc_flags
            },
            extra_link_args=[ #must pass those args for windows.
                'opengl32.lib',
                'user32.lib',
                'gdi32.lib',
                'glu32.lib',
            ] if os.name == 'nt' else [],
            define_macros=[
                ('TORCH_EXTENSION_NAME', '"_C"'),
                ('NVDR_USE_TORCH', None),
                ('NVDR_CTX_ARGS', None),
                ('NVDR_CTX_PARAMS', None),
                ('TORCH_API_INCLUDE_EXTENSION_H', None),
                ('USE_PYTHON', None),
                ('NVDR_TORCH', None), # Add this
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
