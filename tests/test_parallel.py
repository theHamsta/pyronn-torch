#
# Copyright © 2021 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

import torch

from pyronn_torch.parallel import ParallelProjector

try:
    import pyconrad.autoinit
except Exception:
    from unittest.mock import MagicMock
    pyconrad = MagicMock()


def test_parallel():
    vol = torch.randn(200, 1, 256, 256)
    projector = ParallelProjector(volume_shape=vol.shape[-2:])

    projection = projector.project_forward(vol)
    reco = projector.project_backward(projection)

    pyconrad.imshow(projection)
    pyconrad.imshow(reco)


def test_parallel_other_direction():
    angles = torch.linspace(0, 360, 360 - 1)
    volume_shape = [255, 255]
    sino = torch.randn(len(angles), volume_shape[1])
    projector = ParallelProjector(volume_shape=volume_shape)

    vol = projector.project_backward(sino)
    reco = projector.project_forward(vol)

    pyconrad.imshow(reco)
    pyconrad.imshow(sino)


def test_parallel_grad():
    vol = torch.randn(200, 1, 256, 256, requires_grad=True)
    projector = ParallelProjector(volume_shape=vol.shape[-2:])

    projection = projector.project_forward(vol)
    reco = projector.project_backward(projection)
    reco.mean().backward()

    import pyconrad.autoinit
    pyconrad.imshow(projection)
    pyconrad.imshow(reco)
