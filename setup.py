import sys
from glob import glob
from os.path import join

from pkg_resources import VersionConflict, require
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":

    module_name = 'pyronn_torch_cpp'

    cuda_sources = glob(join('generated_files', '*.cu'))

    generated_file = join('generated_files', 'pyronn_torch.cpp')

    setup(use_pyscaffold=True,
          ext_modules=[
              CUDAExtension(module_name,
                            [generated_file] + cuda_sources,
                            extra_compile_args={'cxx': [],
                                                'nvcc': ['-arch=sm_35', '-O3', '-allow-unsupported-compiler']})
          ],
          cmdclass={
              'build_ext': BuildExtension
          })
