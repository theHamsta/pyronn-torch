import sys
from glob import glob
from os import makedirs
from os.path import basename, dirname, join
from shutil import copyfile, copytree, rmtree

from pkg_resources import VersionConflict, require
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":

    object_cache = dirname(__file__)
    module_name = 'pyronn_torch_cpp'

    source_files = glob(join(dirname(__file__), 'src', 'pyronn_torch', 'PYRO-NN-Layers', '*.cu.cc'))

    generated_file = join('src', 'pyronn_torch', 'pyronn_torch.cpp')

    cuda_sources = []
    makedirs(join(object_cache, module_name), exist_ok=True)
    rmtree(join(object_cache, module_name, 'helper_headers'), ignore_errors=True)
    copytree(join(dirname(__file__), 'src', 'pyronn_torch',  'PYRO-NN-Layers', 'helper_headers'),
             join(object_cache, module_name, 'helper_headers'))

    for s in source_files:
        dst = join(object_cache, module_name, basename(s).replace('.cu.cc', '.cu'))
        copyfile(s, dst)  # Torch only accepts *.cu as CUDA
        cuda_sources.append(join(module_name, basename(s).replace('.cu.cc', '.cu')))

    setup(use_pyscaffold=True,
          ext_modules=[
              CUDAExtension(module_name,
                            [generated_file] + cuda_sources,
                            extra_compile_args={'cxx': ['--std=c++14'],
                                                'nvcc': ['-arch=sm_35']})
          ],
          cmdclass={
              'build_ext': BuildExtension
          })
