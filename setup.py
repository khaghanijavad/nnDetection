from setuptools import setup, find_packages
from pathlib import Path
import os
import sys
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


# TODO -> Where to pu2t the bulding stuff?

def clean():
    """Custom clean command to tidy up the project root."""
    os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz')


def get_extensions():
    """
    Adapted from https://github.com/pytorch/vision/blob/master/setup.py
    and https://github.com/facebookresearch/detectron2/blob/master/setup.py
    """
    print("Build csrc")
    print("Building with {}".format(sys.version_info))

    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    extensions_dir = this_dir / 'nndet' / 'csrc'

    main_file = list(extensions_dir.glob('*.cpp'))
    source_cpu = []  # list((extensions_dir/'cpu').glob('*.cpp')) temporary until I added header files ...
    source_cuda = list((extensions_dir / 'cuda').glob('*.cu'))
    print("main_file {}".format(main_file))
    print("source_cpu {}".format(source_cpu))
    print("source_cuda {}".format(source_cuda))

    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []
    extra_compile_args = {"cxx": []}

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv('FORCE_CUDA', '0') == '1':
        print("Adding CUDA csrc to build")
        print("CUDA ARCH {}".format(os.getenv("TORCH_CUDA_ARCH_LIST")))
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [str(extensions_dir)]

    ext_modules = [
        extension(
            'nndet._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name='nndet',
    version="v0.1",
    packages=find_packages(),
    long_description_content_type='text/markdown',
    tests_require=["coverage"],
    python_requires="==3.12.6",
    author="Division of Medical Image Computing, German Cancer Research Center",
    maintainer_email='m.baumgartner@dkfz-heidelberg.de',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension,
        'clean': clean,
    },
    entry_points={},
)
