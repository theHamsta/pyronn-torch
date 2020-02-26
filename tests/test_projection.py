#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest

import pyronn_torch


def test_init():
    assert pyronn_torch.cpp_extension


@pytest.mark.parametrize('with_texture', ('with_texture', False))
def test_projection(with_texture):
    projector = pyronn_torch.ConeBeamProjector.from_conrad_config()

    volume = projector.new_volume_tensor()

    volume += 1.
    result = projector.project_forward(volume, use_texture=False)

    print(result)
