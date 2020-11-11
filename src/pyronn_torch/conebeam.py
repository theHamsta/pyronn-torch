#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.
"""

"""
import numpy as np
import sympy as sp
import torch

import pyronn_torch


class State:
    def __init__(self,
                 projection_shape,
                 volume_shape,
                 source_points,
                 inverse_matrices,
                 projection_matrices,
                 volume_origin,
                 volume_spacing,
                 projection_multiplier,
                 step_size=1.,
                 with_texture=True):
        self.projection_shape = projection_shape
        self.volume_shape = volume_shape
        self.source_points = source_points
        self.inverse_matrices = inverse_matrices
        self.projection_matrices = projection_matrices
        self.volume_origin = volume_origin
        self.volume_spacing = volume_spacing
        self.projection_multiplier = projection_multiplier
        self.with_texture = with_texture
        self.step_size = step_size


class _ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(self, volume, state=None):
        if state is None:
            state = self.state
            return_none = True
        else:
            return_none = False

        volume = volume.float().cuda().contiguous()
        projection = torch.zeros(state.projection_shape,
                                 device='cuda',
                                 requires_grad=volume.requires_grad).float().contiguous()

        assert pyronn_torch.cpp_extension
        if state.with_texture:
            pyronn_torch.cpp_extension.call_Cone_Projection_Kernel_Tex_Interp_Launcher(
                inv_matrices=state.inverse_matrices,
                projection=projection,
                source_points=state.source_points,
                step_size=state.step_size,
                volume=volume,
                volume_spacing_x=state.volume_spacing[0],
                volume_spacing_y=state.volume_spacing[1],
                volume_spacing_z=state.volume_spacing[2])
        else:
            pyronn_torch.cpp_extension.call_Cone_Projection_Kernel_Launcher(
                inv_matrices=state.inverse_matrices,
                projection=projection,
                source_points=state.source_points,
                step_size=state.step_size,
                volume=volume,
                volume_spacing_x=state.volume_spacing[0],
                volume_spacing_y=state.volume_spacing[1],
                volume_spacing_z=state.volume_spacing[2])

        self.state = state
        if return_none:
            return projection, None
        else:
            return projection,

    @staticmethod
    def backward(self, projection_grad, state=None, *args):
        if state is None:
            state = self.state
            return_none = True
        else:
            return_none = False

        projection_grad = projection_grad.float().cuda().contiguous()
        volume_grad = torch.zeros(state.volume_shape,
                                  device='cuda',
                                  requires_grad=projection_grad.requires_grad)

        assert pyronn_torch.cpp_extension
        pyronn_torch.cpp_extension.call_Cone_Backprojection3D_Kernel_Launcher(
            state.projection_matrices, projection_grad,
            state.projection_multiplier, volume_grad, *state.volume_origin,
            *state.volume_spacing)

        self.state = state
        if return_none:
            return volume_grad, None
        else:
            return volume_grad,


class _BackwardProjection(torch.autograd.Function):
    backward = staticmethod(_ForwardProjection.forward)
    forward = staticmethod(_ForwardProjection.backward)


class ConeBeamProjector:
    def __init__(self, volume_shape, volume_spacing, volume_origin,
                 projection_shape, projection_spacing, projection_origin,
                 projection_matrices):
        self._volume_shape = volume_shape
        self._volume_origin = volume_origin
        self._volume_spacing = volume_spacing
        self._projection_shape = projection_shape
        self._projection_matrices_numpy = projection_matrices
        self._projection_spacing = projection_spacing
        self._projection_origin = projection_origin
        self._calc_inverse_matrices()

    @classmethod
    def from_conrad_config(cls):
        import pyconrad.autoinit
        import pyconrad.config
        volume_shape = pyconrad.config.get_reco_shape()
        volume_spacing = pyconrad.config.get_reco_spacing()
        volume_origin = pyconrad.config.get_reco_origin()
        projection_shape = pyconrad.config.get_sino_shape()
        projection_spacing = [
            pyconrad.config.get_geometry().getPixelDimensionX(),
            pyconrad.config.get_geometry().getPixelDimensionY(),
        ]
        projection_origin = [
            pyconrad.config.get_geometry().getDetectorOffsetU(),
            pyconrad.config.get_geometry().getDetectorOffsetV(),
        ]
        projection_matrices = pyconrad.config.get_projection_matrices()

        obj = cls(volume_shape=volume_shape,
                  volume_spacing=volume_spacing,
                  volume_origin=volume_origin,
                  projection_shape=projection_shape,
                  projection_spacing=projection_spacing,
                  projection_origin=projection_origin,
                  projection_matrices=projection_matrices)
        return obj

    def new_volume_tensor(self, requires_grad=False):
        return torch.zeros(self._volume_shape,
                           requires_grad=requires_grad).cuda()

    def new_projection_tensor(self, requires_grad=False):
        return torch.zeros(self._projection_shape,
                           requires_grad=requires_grad).cuda()

    def project_forward(self, volume, step_size=1., use_texture=True):
        return _ForwardProjection().apply(
            volume,
            State(projection_shape=self._projection_shape,
                  volume_shape=self._volume_shape,
                  source_points=self._source_points,
                  inverse_matrices=self._inverse_matrices,
                  projection_matrices=self._projection_matrices,
                  volume_origin=self._volume_origin,
                  volume_spacing=self._volume_spacing,
                  projection_multiplier=self._projection_multiplier,
                  step_size=step_size,
                  with_texture=use_texture))[0]

    def project_backward(self,
                         projection_stack,
                         step_size=1.,
                         use_texture=True):
        return _BackwardProjection().apply(
            projection_stack,
            State(projection_shape=self._projection_shape,
                  volume_shape=self._volume_shape,
                  source_points=self._source_points,
                  inverse_matrices=self._inverse_matrices,
                  projection_matrices=self._projection_matrices,
                  volume_origin=self._volume_origin,
                  volume_spacing=self._volume_spacing,
                  projection_multiplier=self._projection_multiplier,
                  step_size=step_size,
                  with_texture=use_texture))[0]

    def _calc_inverse_matrices(self):
        if self._projection_matrices_numpy is None:
            return
        self._projection_matrices = torch.stack(
            tuple(
                torch.from_numpy(p.astype(np.float32))
                for p in self._projection_matrices_numpy)).cuda().contiguous()

        inv_spacing = np.array([1 / s for s in self._volume_spacing],
                               np.float32)

        camera_centers = list(map(
            lambda x: np.array(sp.Matrix(x).nullspace(), np.float32),
            self._projection_matrices_numpy))

        source_points = list(map(
            lambda x: (-x[0, :3, 0] / x[0, 3, 0] * inv_spacing
                       - np.array(list(self._volume_origin)) * inv_spacing).astype(np.float32), camera_centers))

        scaling_matrix = np.array([[inv_spacing[0], 0, 0], [0, inv_spacing[1], 0], [0, 0, inv_spacing[2]]])
        inv_matrices = list(map(
            lambda x:
            (scaling_matrix @ np.linalg.inv(x[:3, :3])).astype(np.float32),
            self._projection_matrices_numpy))

        self._inverse_matrices = torch.stack(
            tuple(map(torch.from_numpy, inv_matrices))).float().cuda().contiguous()
        self._source_points = torch.stack(
            tuple(map(torch.from_numpy, source_points))).float().cuda().contiguous()
        self._projection_multiplier = 1. / self._projection_matrices.shape[0]

    @property
    def projection_matrices(self):
        return self._projection_matrices_numpy

    @projection_matrices.setter
    def projection_matrices(self, numpy_matrices):
        self._projection_matrices_numpy = numpy_matrices
        self._calc_inverse_matrices()
