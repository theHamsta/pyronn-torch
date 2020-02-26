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
@pytest.mark.parametrize('with_backward', ('with_backward', False))
def test_projection(with_texture, with_backward):
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
