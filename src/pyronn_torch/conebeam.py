#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import numpy as np
import torch
from scipy.linalg import null_space

import pyronn_torch


class _ForwardProjection(torch.autograd.Function):

    def __init__(self, projection_shape,
                 source_points,
                 inverse_matrices,
                 projection_matrices,
                 volume_origin,
                 volume_spacing,
                 projection_multiplier,
                 step_size=1.,
                 with_texture=True):
        self.projection_shape = projection_shape
        self.source_points = source_points
        self.inverse_matrices = inverse_matrices
        self.projection_matrices = projection_matrices
        self.volume_origin = volume_origin
        self.volume_spacing = volume_spacing
        self.projection_multiplier = projection_multiplier
        self.with_texture = with_texture
        self.step_size = step_size

    def forward(self, volume):
        volume = volume.cuda().contiguous()
        projection = torch.zeros(self.projection_shape, device='cuda', requires_grad=volume.requires_grad)

        assert pyronn_torch.cpp_extension
        if self.with_texture:
            pyronn_torch.cpp_extension.call_Cone_Projection_Kernel_Tex_Interp_Launcher(
                self.inverse_matrices,
                projection,
                self.source_points,
                self.step_size,
                volume,
                *self.volume_spacing)
        else:
            pyronn_torch.cpp_extension.call_Cone_Projection_Kernel_Launcher(
                self.inverse_matrices,
                projection,
                self.source_points,
                self.step_size,
                volume,
                *self.volume_spacing)

        return projection,

    def backward(self, *projection_grad):
        projection_grad = projection_grad[0]
        self.projection_matrices
        volume_grad = torch.zeros(self.volume_shape, device='cuda')

        assert pyronn_torch.cpp_extension
        pyronn_torch.cpp_extension.call_Cone_Backprojection3D_Kernel_Launcher(
            self.projection_matrices,
            projection_grad,
            self.projection_multiplier,
            volume_grad,
            self.volume_origin,
            self.volume_spacing)

        return volume_grad,


class _BackwardProjection(torch.autograd.Function):
    __init__ = _ForwardProjection.__init__
    backward = _ForwardProjection.forward
    forward = _ForwardProjection.backward


class ConeBeamProjector:

    def __init__(self,
                 volume_shape,
                 volume_spacing,
                 volume_origin,
                 projection_shape,
                 projection_spacing,
                 projection_origin,
                 projection_matrices):
        assert type(projection_matrices) == np.ndarray and projection_matrices.shape == (projection_shape[0], 3, 4)
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
        projection_spacing = [pyconrad.config.get_geometry().getPixelDimensionY(),
                              pyconrad.config.get_geometry().getPixelDimensionX()]
        projection_origin = [pyconrad.config.get_geometry().getDetectorOffsetV(),
                             pyconrad.config.get_geometry().getDetectorOffsetU()]
        projection_matrices = pyconrad.config.get_projection_matrices()

        obj = cls(volume_shape,
                  volume_spacing,
                  volume_origin,
                  projection_shape,
                  projection_spacing,
                  projection_origin,
                  projection_matrices)
        return obj

    def new_volume_tensor(self, requires_grad=False):
        return torch.zeros(self._volume_shape, requires_grad=requires_grad).cuda()

    def new_projection_tensor(self, requires_grad=False):
        return torch.zeros(self._projection_shape, requires_grad=requires_grad).cuda()

    def project_forward(self, volume, step_size=1., use_texture=True):
        return _ForwardProjection(self._projection_shape,
                                  self._source_points,
                                  self._inverse_matrices,
                                  self._projection_matrices,
                                  self._volume_origin,
                                  self._volume_shape,
                                  self._projection_multiplier,
                                  step_size,
                                  use_texture).forward(volume)[0]

    def project_backward(self, projection_stack, step_size=1., use_texture=True):
        return _BackwardProjection(self._projection_shape,
                                   self._source_points,
                                   self._inverse_matrices,
                                   self._projection_matrices,
                                   self._volume_origin,
                                   self._volume_shape,
                                   self._projection_multiplier,
                                   step_size,
                                   use_texture).backward(projection_stack)[0]

    def _calc_inverse_matrices(self):
        """
        Explanation
        The projection matrix P consists of a camera intrinsic K, and the extrinsics
        rotation R and translation t as P = K @ [R|t]. An alternative form uses the
        camera center C in world coordinates as P = K @ [R|-RC] = [KR|-KRC].

        Given P, we can obtain C = (KR)^-1 @ -(-KRC) = P[:3, :3]^-1 @ -P[:, 3].
        This is equivalent to the nullspace form used above C = ker(P).

        Furthermore, the inverse matrix M maps a point u = (u, v, 1) on the detector
        onto a 3D ray direction r as r = Mu. It is defined as M = -(KR)^-1

        The projector starts at the camera position C and steps along the ray
        direction r for either forward or back projection. All points along the
        line L = C + s*r, where s is in (0-sdd) are integrated over for the line
        integral at detector position u.

        For details see here https://ksimek.github.io/2012/08/22/extrinsic/

        :param matrices: maps a homogenous voxel index x to a detector index u through u = Px. Shaped (p, 3, 4)
        :return: None
        """
        if self._projection_matrices_numpy is None:
            return

        with torch.no_grad():
            # if we assumed numpy arrays (not lists of arrays) things would be cleaner imo:
            self._projection_matrices = torch.from_numpy(self._projection_matrices_numpy).cuda().contiguous()
            self._volume_origin_tensor = torch.from_numpy(self._volume_origin).cuda()

            # volume_spacing was reversed earlier (--> flip)
            inv_scale = torch.diag(torch.from_numpy(1 / np.flip(self._volume_spacing))).cuda()

            # init inverse matrices and source point arrays
            p, _, _, = self._projection_matrices_numpy.shape
            self._inverse_matrices = torch.zeros((p, 3, 3), dtype=torch.float32, device='cuda')
            self._source_points = torch.zeros((p, 3), dtype=torch.float32, device='cuda')

            # calculate derived tensors
            M = torch.linalg.inv(self._projection_matrices[:, :3, :3])
            for i in range(p):
                self._source_points[i] = -M[i] @ self._projection_matrices[i, :, 3] @ inv_scale \
                                         - self._volume_origin_tensor @ inv_scale
                self._inverse_matrices[i] = inv_scale @ M[i]
            self._projection_multiplier = 1.
