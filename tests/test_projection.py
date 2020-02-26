#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import os

import numpy as np
import pytest

import pyronn_torch


def test_init():
    assert pyronn_torch.cpp_extension


@pytest.mark.parametrize('with_texture', ('with_texture', False))
@pytest.mark.parametrize('with_backward', ('with_backward', False))
def test_projection(with_texture, with_backward):
    projector = pyronn_torch.ConeBeamProjector(
        (128, 128, 128),
        (2.0, 2.0, 2.0),
        (-127.5, -127.5, -127.5),
        (2, 480, 620),
        [1.0, 1.0],
        (0, 0),
        np.array([[[-3.10e+2, -1.20e+03,  0.00e+00,  1.86e+5],
                   [-2.40e+2,  0.00e+00,  1.20e+03,  1.44e+5],
                   [-1.00e+00,  0.00e+00,  0.00e+00,  6.00e+2]],
                  [[-2.89009888e+2, -1.20522754e+3, -1.02473585e-13,
                    1.86000000e+5],
                   [-2.39963440e+2, -4.18857765e+0,  1.20000000e+3,
                    1.44000000e+5],
                   [-9.99847710e-01, -1.74524058e-2,  0.00000000e+0,
                    6.00000000e+2]]])
    )

    volume = projector.new_volume_tensor(requires_grad=True if with_backward else False)

    volume += 1.
    result = projector.project_forward(volume, use_texture=with_texture)

    assert result is not None
    if with_backward:
        assert volume.requires_grad
        assert result.requires_grad

        loss = result.mean()
        loss.backward()


@pytest.mark.parametrize('with_texture', ('with_texture', False))
@pytest.mark.parametrize('with_backward', ('with_backward', False))
def test_projection_backward(with_texture, with_backward):
    projector = pyronn_torch.ConeBeamProjector(
        (128, 128, 128),
        (2.0, 2.0, 2.0),
        (-127.5, -127.5, -127.5),
        (2, 480, 620),
        [1.0, 1.0],
        (0, 0),
        np.array([[[-3.10e+2, -1.20e+03,  0.00e+00,  1.86e+5],
                   [-2.40e+2,  0.00e+00,  1.20e+03,  1.44e+5],
                   [-1.00e+00,  0.00e+00,  0.00e+00,  6.00e+2]],
                  [[-2.89009888e+2, -1.20522754e+3, -1.02473585e-13,
                    1.86000000e+5],
                   [-2.39963440e+2, -4.18857765e+0,  1.20000000e+3,
                    1.44000000e+5],
                   [-9.99847710e-01, -1.74524058e-2,  0.00000000e+0,
                    6.00000000e+2]]])
    )

    projection = projector.new_projection_tensor(requires_grad=True if with_backward else False)

    projection += 1.
    result = projector.project_backward(projection, use_texture=with_texture)

    assert result is not None
    if with_backward:
        assert projection.requires_grad
        assert result.requires_grad

        loss = result.mean()
        loss.backward()


@pytest.mark.skipif('CI' in os.environ, reason="No conrad config on CI")
@pytest.mark.parametrize('with_backward', ('with_backward', False))
def test_conrad_config(with_backward, with_texture=True):
    projector = pyronn_torch.ConeBeamProjector.from_conrad_config()

    volume = projector.new_volume_tensor(requires_grad=True if with_backward else False)

    volume += 1.
    result = projector.project_forward(volume, use_texture=with_texture)

    assert result is not None
    if with_backward:
        assert volume.requires_grad
        assert result.requires_grad

        loss = result.mean()
        loss.backward()
