# -*- coding: utf-8 -*-

from os.path import dirname, join

import torch
from pkg_resources import DistributionNotFound, get_distribution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'pyronn-torch'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


try:
    cpp_extension = torch.ops.load_library(join(dirname(__file__), 'pyronn_torch.so'))
except Exception:
    import pyronn_torch.codegen
    cpp_extension = pyronn_torch.codegen.compile_shared_object()

