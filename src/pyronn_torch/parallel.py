#
# Copyright © 2021 Mayank Patwari <mayank.patwari@fau.de>
#
# Distributed under terms of the GPLv3 license.
"""

"""
import numpy as np
import pyronn_torch
import torch


class State:
    def __init__(self, detector_origin, detector_spacing, projection_shape, ray_vectors,
                 volume_origin, volume_shape, volume_spacing):
        self._volume_shape = volume_shape
        self._volume_origin = volume_origin
        self._volume_spacing = volume_spacing
        self._projection_shape = projection_shape
        self._detector_origin = detector_origin
        self._detector_spacing = detector_spacing
        self._ray_vectors = ray_vectors


class _ForwardProjection(torch.autograd.Function):

    @staticmethod
    def forward(self, volume, state=None):

        if state is None:
            state = self.state
            return_none = True
        else:
            return_none = False

        projection = torch.zeros(state._projection_shape,
                                 requires_grad=volume.requires_grad,
                                 device='cuda').float().contiguous()
        pyronn_torch.cpp_extension.call_Parallel_Projection2D_Kernel_Launcher(
            state._detector_origin,
            state._detector_spacing,
            projection,
            state._ray_vectors,
            state._volume_origin[0],
            state._volume_origin[1],
            volume,
            state._volume_spacing[0],
            state._volume_spacing[1]
        )

        self.state = state
        if return_none:
            return projection, None
        else:
            return projection

    @staticmethod
    def backward(self, projection_grad, state=None, *args):

        if state is None:
            state = self.state
            return_none = True
        else:
            return_none = False

        volume_grad = torch.zeros(state._volume_shape, requires_grad=projection_grad.requires_grad).cuda()
        pyronn_torch.cpp_extension.call_Parallel_Backprojection2D_Kernel_Launcher(
            state._detector_origin,
            state._detector_spacing,
            projection_grad,
            state._ray_vectors,
            state._volume_origin[0],
            state._volume_origin[1],
            volume_grad,
            state._volume_spacing[0],
            state._volume_spacing[1]
        )

        self.state = state
        if return_none:
            return volume_grad, None
        else:
            return volume_grad


class _BackwardProjection(torch.autograd.Function):
    backward = staticmethod(_ForwardProjection.forward)
    forward = staticmethod(_ForwardProjection.backward)


class ParallelProjector:

    def __init__(self, detector_origin=None, detector_spacing=1, angles=torch.linspace(0, 360, 360 - 1),
                 volume_origin=None, volume_shape=[256, 256], volume_spacing=[1, 1]):
        self._volume_shape = volume_shape
        self._volume_origin = volume_origin or [-v/2 for v in reversed(volume_shape)]
        self._volume_spacing = volume_spacing
        self._projection_shape = [np.shape(angles)[0], volume_shape[1]]
        self._detector_origin = detector_origin or volume_shape[0] / 2
        self._detector_spacing = detector_spacing
        self._calc_ray_vectors(angles)

    def _calc_ray_vectors(self, angles):
        self._ray_vectors = torch.zeros(angles.shape[0], 2)
        self._ray_vectors[:, 0] = torch.cos(angles)
        self._ray_vectors[:, 1] = torch.sin(angles)

    def project_forward(self, volume):
        volume = volume.float().cuda().contiguous()
        if len(volume.shape) == 3:
            volume = volume[:, None, :, :]

        if len(volume.shape) != 4:
            raise ValueError('4D input expected! [batch, channel (only 1), dim1, dim2]')
        elif volume.shape[1] != 1:
            raise ValueError('Only channel dimension of 1 is currently supported!')

        projs = torch.zeros(volume.shape[0],
                            self._projection_shape[0],
                            self._projection_shape[1], device='cuda')

        for i, slice in enumerate(volume):
            projs[i] = _ForwardProjection().apply(slice[0], State(
                self._detector_origin,
                self._detector_spacing,
                self._projection_shape,
                self._ray_vectors,
                self._volume_origin,
                self._volume_shape,
                self._volume_spacing
            ))
        return projs

    def project_backward(self, projection):
        projection = projection.float().contiguous().cuda()

        if len(projection.shape) == 2:
            projection = projection[None, ...]
        if len(projection.shape) != 3:
            raise ValueError('3D input expected! [batch, number_of_views, image_dim]')

        volume = torch.zeros(projection.shape[0],
                             1,
                             self._volume_shape[0],
                             self._volume_shape[1]).cuda()

        for i, proj in enumerate(projection):
            volume[i] = _BackwardProjection().apply(proj, State(
                self._detector_origin,
                self._detector_spacing,
                self._projection_shape,
                self._ray_vectors,
                self._volume_origin,
                self._volume_shape,
                self._volume_spacing
            ))
        return volume
