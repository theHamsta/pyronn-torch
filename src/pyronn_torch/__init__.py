# -*- coding: utf-8 -*-

import sys
from os.path import dirname

from pkg_resources import DistributionNotFound, get_distribution

from pyronn_torch.conebeam import ConeBeamProjector

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'pyronn-torch'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


try:
    sys.path.append(dirname(__file__))
    cpp_extension = __import__('pyronn_torch_cpp')
except Exception as e:
    import warnings
    warnings.warn(str(e))
    import pyronn_torch.codegen
    cpp_extension = pyronn_torch.codegen.compile_shared_object()

__all__ = ['ConeBeamProjector', 'cpp_extension']
