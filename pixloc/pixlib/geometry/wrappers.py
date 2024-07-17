"""
Convenience classes for an SE3 pose and a pinhole Camera with lens distortion.
Based on PyTorch tensors: differentiable, batched, with GPU support.
"""

import functools
import inspect
import math
from typing import Union, Tuple, List, Dict, NamedTuple
import torch
import numpy as np

from .optimization import skew_symmetric, so3exp_map
from .utils import undistort_points, J_undistort_points
from pixloc.pixlib.geometry.interpolation import interpolate_tensor
# from pytorch3d.structures import Pointclouds, Volumes
# from pytorch3d.ops import add_pointclouds_to_volumes

EPS = 1e-7

def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
       if they are numpy arrays. Use the device and dtype of the wrapper.
    """
    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device('cpu')
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        else:
            return NotImplemented


class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        '''Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        '''
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_aa(cls, aa: torch.Tensor, t: torch.Tensor):
        '''Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            aa: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        '''
        assert aa.shape[-1] == 3
        assert t.shape[-1] == 3
        assert aa.shape[:-1] == t.shape[:-1]
        return cls.from_Rt(so3exp_map(aa), t)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        '''Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        '''
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    def from_colmap(cls, image: NamedTuple):
        '''Pose from a COLMAP Image.'''
        return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1]+(3, 3))

    @property
    def t(self) -> torch.Tensor:
        '''Underlying translation vector with shape (..., 3).'''
        return self._data[..., -3:]

    def inv(self) -> 'Pose':
        '''Invert an SE(3) pose.'''
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C.'''
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        '''
        assert p3d.shape[-1] == 3
        # assert p3d.shape[:-2] == self.shape  # allow broadcasting
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points: T_A2B * p3D_A -> p3D_B.'''
        return self.transform(p3D)

    def __matmul__(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C.'''
        return self.compose(other)

    @autocast
    def J_transform(self, p3d_out: torch.Tensor):
        # only care 5DOF, R, lon:Tx, lat:Ty, because sat is parra projection
        #   Tx Ty Rz
        # [[1,0,py],
        #  [0,1,-px],
        #  [0,0,0]]
        # J_t = torch.diag_embed(torch.ones_like(p3d_out))
        # J_rot = -skew_symmetric(p3d_out)
        # J = torch.cat([J_t[...,:2], J_rot[...,-1:]], dim=-1)
        # return J  # N x 3 x 3

        # [[1,0,0,0,-pz,py],
        #  [0,1,0,pz,0,-px],
        #  [0,0,1,-py,px,0]]
        J_t = torch.diag_embed(torch.ones_like(p3d_out))
        J_rot = -skew_symmetric(p3d_out)
        J = torch.cat([J_t, J_rot], dim=-1)
        return J  # N x 3 x 6

    @autocast
    def J_transform2(self, p3d_out: torch.Tensor):
        # only care 5DOF, R, lon:Tx, lat:Ty, because sat is parra projection
        #   Tx Ty Rz
        # [[1,0,py],
        #  [0,1,-px],
        #  [0,0,0]]
        J_t = torch.diag_embed(torch.ones_like(p3d_out))
        J_rot = -skew_symmetric(p3d_out)
        J = torch.cat([J_t[...,:2], J_rot[...,-1:]], dim=-1)
        return J  # N x 3 x 3

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> Tuple[torch.Tensor]:
        '''Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation anngle in degrees.
            dt: translation distance in meters.
        '''
        eps=1e-6
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        # cos = torch.clamp((trace - 1) / 2, -1, 1)
        cos = torch.clamp((trace - 1) / 2, -1+eps, 1-eps)
        dr = torch.acos(cos).abs() / math.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt
    def magnitude_latlong(self) -> Tuple[torch.Tensor]:
        '''Magnitude of the SE(3) transformation.
        self is in query coordinates
        Returns:
            laterral translation distance in meters.
            longitudinal translation distance in meters.
        '''
        return torch.abs(self.t[..., 0]), torch.abs(self.t[..., -1])
    def shift_NE(self) -> Tuple[torch.Tensor]:
        '''shift in north and east, in meter
        self is in satellite, reference coordinates
        Returns:
            north translation distance in meters.
            east translation distance in meters.
        '''
        return -self.t[..., 1], self.t[..., 0]

    def __repr__(self):
        return f'Pose: {self.shape} {self.dtype} {self.device}'


class Camera(TensorWrapper):
    eps = 1e-3

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6, 8, 10, 11}
        super().__init__(data)

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]):
        '''Camera from a COLMAP Camera tuple or dictionary.
        We assume that the origin (0, 0) is the center of the top-left pixel.
        This is different from COLMAP.
        '''
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera['model']
        params = camera['params']

        if model in ['OPENCV', 'PINHOLE']:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL']:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == 'SIMPLE_RADIAL':
                params = np.r_[params, 0.]
        else:
            raise NotImplementedError(model)

        data = np.r_[camera['width'], camera['height'],
                     fx, fy, cx-0.5, cy-0.5, params]
        return cls(data)

    @property
    def size(self) -> torch.Tensor:
        '''Size (width height) of the images, with shape (..., 2).'''
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        '''Focal lengths (fx, fy) with shape (..., 2).'''
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        '''Principal points (cx, cy) with shape (..., 2).'''
        return self._data[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        '''Distortion parameters, with shape (..., {0, 2, 4}).'''
        return self._data[..., 6:-1]

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        '''Update the camera parameters after resizing an image.'''
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        # data = torch.cat([
        #     self.size * s,
        #     self.f * s,
        #     (self.c + 0.5) * s - 0.5,
        #     self.dist], -1)
        data = torch.cat([
            self.size*s,
            self.f*s,
            self.c*s,
            self.dist, self._data[:,10:]], -1)
        return self.__class__(data)

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        '''Update the camera parameters after cropping an image.'''
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([
            size,
            self.f,
            self.c - left_top,
            self.dist, self._data[:,10:]], -1)
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        '''Check if 2D points are within the image boundaries.'''
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Project 3D points into the camera plane and check for visibility.'''
        if np.infty in self._data:
            z = torch.ones_like(p3d[..., -1])
        else:
            z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)

        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid

    @autocast
    def project3d(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Project 3D points into the camera plane and check for visibility.'''
        if np.infty in self._data:
            z = torch.ones_like(p3d[..., -1])
        else:
            z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)

        p3d = p3d / z.unsqueeze(-1)
        return p3d, valid

    def J_project(self, p3d: torch.Tensor):
        if np.infty in self._data:
            x, y = p3d[..., 0], p3d[..., 1]
            ones = torch.ones_like(x)
            zero = torch.zeros_like(x)
            J = torch.stack([
                ones, zero, zero,
                zero, ones, zero], dim=-1)
        else:
            x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
            zero = torch.zeros_like(z)
            z = z.clamp(min=self.eps)
            J = torch.stack([
                1 / z, zero, -x / z ** 2,
                zero, 1 / z, -y / z ** 2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    @autocast
    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Undistort normalized 2D coordinates
           and check for validity of the distortion model.
        '''
        assert pts.shape[-1] == 2
        # assert pts.shape[:-2] == self.shape  # allow broadcasting
        return undistort_points(pts, self.dist)

    def J_undistort(self, pts: torch.Tensor):
        return J_undistort_points(pts, self.dist)  # N x 2 x 2

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        '''Convert normalized 2D coordinates into pixel coordinates.'''
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    @autocast
    def denormalize3d(self, p3d: torch.Tensor) -> torch.Tensor:
        '''Convert normalized 2D coordinates into pixel coordinates.'''
        z_f = torch.ones([self.f.size(0), 1]).to(self.f.device)
        z_c = torch.zeros([self.c.size(0), 1]).to(self.f.device)
        f = torch.cat([self.f, z_f], dim=-1).unsqueeze(-2)
        c = torch.cat([self.c, z_c], dim=-1).unsqueeze(-2)

        return p3d * f, p3d * f + c

    def J_denormalize(self):
        return torch.diag_embed(self.f).unsqueeze(-3)  # 1 x 2 x 2

    @autocast
    def world2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Transform 3D points into 2D pixel coordinates.'''
        p2d, visible = self.project(p3d)
        p2d, mask = self.undistort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    @autocast
    def world2image3d(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Transform 3D points into 2D pixel coordinates.'''
        p3df, p3dfc = self.denormalize3d(p3d)
        valid = self.in_image(p3dfc.detach()[... , :-1])
        return p3df, valid

    def J_world2image(self, p3d: torch.Tensor):
        p2d_dist, valid = self.project(p3d)
        J = (self.J_denormalize()
             @ self.J_undistort(p2d_dist)
             @ self.J_project(p3d))
        return J, valid

    def image2world(self, p2d: torch.Tensor) -> torch.Tensor:
        '''Transform 2D pixel coordinates into 3D points, scale unknown .'''
        if p2d.dim() == 4:
            p3d_xy = (p2d - self.c[:, None, None, :]) / self.f[:, None, None, :]
        else:
            p3d_xy = (p2d - self.c.unsqueeze(-2)) / self.f.unsqueeze(-2)
        if np.infty in self._data:
            # para projection, z unknown
            p3d = torch.cat([p3d_xy, torch.zeros_like(p3d_xy[..., :1])], dim=-1)
        else:
            p3d = torch.cat([p3d_xy, torch.ones_like(p3d_xy[...,:1])], dim=-1)
        return p3d

    @autocast
    def world2image2(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Transform 3D points into 2D pixel coordinates.'''
        p2d, visible = self.project2(p3d)
        p2d, mask = self.undistort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid


    @autocast
    def project2(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Project 3D points into the camera plane and check for visibility.'''
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)

        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid


    def voxelize(self, xyz, feat, size, level):
        B, C, D, A, _ = size
        device = feat.device

        pad = torch.zeros([B, 1]).to(device)
        c = torch.cat([self.c, pad], dim=-1)
        center = torch.tensor([[A / 2, A / 2, 0]], dtype=torch.float32).to(device)
        translation = (c - center).detach()

        pcs = Pointclouds(points=xyz, features=feat)

        init_vol = Volumes(features=torch.zeros(size).to(device),
                           densities=torch.zeros((B, 1, D, A, A)).to(device),
                           # volume_translation=[0, 0, 0],
                           volume_translation=translation,  #[self.c[0, 0]-A/2, self.c[0, 1]-A/2, 0],
                           )
        updated_vol = add_pointclouds_to_volumes(pointclouds=pcs,
                                                 initial_volumes=init_vol,
                                                 mode='trilinear',
                                                 )

        features = updated_vol.features()
        # features = features.mean(dim=2)
        mask = (features != 0).sum(dim=2, keepdim=True).detach()
        features = (features * mask).sum(dim=2) / (mask.sum(dim=2)+1e-6)

        return features

    def voxelize_(self, xyz, feat, size, level):
        # for debugging
        B, C, D, A, _ = size
        device = feat.device

        pcs = Pointclouds(points=xyz, features=feat)

        init_vol = Volumes(features=torch.zeros(size).to(device),
                           densities=torch.zeros((B, 1, D, A, A)).to(device),
                           # volume_translation=[0, 0, 0],
                           volume_translation=[self.c[0] - A / 2, self.c[1] - A / 2, 0],
                           )
        updated_vol = add_pointclouds_to_volumes(pointclouds=pcs,
                                                 initial_volumes=init_vol,
                                                 mode='trilinear',
                                                 )

        features = updated_vol.features()
        features = features.mean(dim=2)

        return features

    def __repr__(self):
        return f'Camera {self.shape} {self.dtype} {self.device}'


    @autocast
    def world2image3d_(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Transform 3D points into 2D pixel coordinates.'''
        p3d = self.denormalize3d_(p3d)
        valid = self.in_image(p3d[... , :-1])
        return p3d, valid

    @autocast
    def denormalize3d_(self, p3d: torch.Tensor) -> torch.Tensor:
        '''Convert normalized 2D coordinates into pixel coordinates.'''
        z = torch.ones([1]).to(self.f.device)
        f = torch.cat([self.f, z], dim=-1).unsqueeze(-2)
        c = torch.cat([self.c, z], dim=-1).unsqueeze(-2)

        return p3d * f #+ c


def project_grd_to_map(T, cam_q, cam_ref, F_query, F_ref, data, depth=100):
    # g2s with GH and T
    b, c, a, a = F_ref.size()
    # b, c, h, w = F_query.size()
    device = F_ref.device
    crop_a = a // 2 # 640, 1280 - 2 * depth*0.2 # TODO

    vv, uu = torch.meshgrid(torch.arange(a, device=device), torch.arange(a, device=device), indexing='ij')
    uv = torch.stack([uu, vv], dim=-1)
    uv = uv[None, :, :, :].repeat(b, 1, 1, 1)  # shape = [b, h, w, 2]
    uv = center_crop(uv, (crop_a, crop_a), mode='bhwc')

    p3d_s = cam_ref.image2world(uv)
    p3d_s[..., -1] = torch.ones_like(p3d_s[..., -1])
    # p3d_q = torch.einsum('bij,bhwj->...bhwi', T.inv().R, p3d_c)  # query world coordinate
    p3d_s2g = T.inv() * p3d_s.reshape(-1, a * a, 3)

    p2d_s2g, mask_s2g = cam_q.world2image(data['query']['T_w2cam'] * p3d_s2g)
    # p2d_s2g = p2d_s2g.reshape(-1, a * a, 2)
    F_q2r, mask_q2r, _ = interpolate_tensor(F_query, p2d_s2g,
                                                   'linear', pad=4, return_gradients=False, out_shape='bcn')
    F_q2r, mask_q2r = F_q2r.reshape(-1, c, a, a), mask_q2r.reshape(-1, 1, a, a)

    return F_q2r, mask_q2r, p3d_s, p3d_s2g


def project_map_to_grd(T, cam_q, cam_ref, F_query, F_ref, data):
    # s2g with GH and T
    b, c, a, a = F_ref.size()
    b, c, h, w = F_query.size()
    device = F_query.device
    vv, uu = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    uv = torch.stack([uu, vv], dim=-1)
    uv = uv[None, :, :, :].repeat(b, 1, 1, 1)  # shape = [b, h, w, 2]

    p3d_c = cam_q.image2world(uv)  # [b, h, w, 3]
    p3d_c[..., -1] = torch.ones_like(p3d_c[..., -1])
    p3d_grd = camera_to_onground(p3d_c, data['query']['T_w2cam'], data['query']['camera_h'], data['normal'], max=100.)
    # p3d_g2s = torch.einsum('bij,bhwj->...bhwi', T.R, p3d_grd)  # query world coordinate
    p3d_g2s = T * p3d_grd.reshape(-1, h * w, 3)

    p2d_g2s, mask_g2s = cam_ref.world2image2(data['ref']['T_w2cam'] * p3d_g2s)
    # p2d_g2s = p2d_g2s.reshape(-1, h * w, 2)
    F_r2q, mask_r2q, _ = interpolate_tensor(F_ref, p2d_g2s,
                                                   'linear', pad=4, return_gradients=False, out_shape='bcn')
    F_r2q, mask_r2q = F_r2q.reshape(-1, c, h, w), mask_r2q.reshape(-1, 1, h, w)

    return F_r2q, mask_r2q, p3d_grd, p3d_g2s


def camera_to_onground(p3d_c, T_w2cam, camera_h, normal_grd, min=1E-8, max=200.):
    # normal from query to camera coordinate
    normal = torch.einsum('...ij,...cj->...ci', T_w2cam.R, normal_grd)
    normal = normal.squeeze(1)
    h = 0
    if p3d_c.dim() > 3:
        b,h,w,c = p3d_c.shape
        p3d_c = p3d_c.flatten(1,2)
    # normal[:, 0] = 0    # set roll, pitch as 0
    # normal[:, -1] = 0
    depth = camera_h[:,None] / torch.einsum('b...i,b...i->b...', p3d_c, normal)
    valid = (depth < max) & (depth >= min)
    depth = depth.clamp(min, max)
    p3d_grd = depth.unsqueeze(-1) * p3d_c
    # each camera coordinate to 'query' coordinate
    p3d_grd = T_w2cam.inv()*p3d_grd # camera to query

    # not valid set to far away
    p3d_grd[~valid] = torch.tensor(max).to(p3d_grd)
    if h > 0:
        p3d_grd = p3d_grd.reshape(b,h,w,c)
    return p3d_grd


def center_crop(feature_map, crop_size, mode='bchw'):
    if mode == 'bchw':
        B, C, H, W = feature_map.size()
    elif mode == 'bhwc':
        feature_map = feature_map.permute(0, 3, 1, 2)
        B, C, H, W = feature_map.size()

    crop_h, crop_w = crop_size

    if crop_h > H or crop_w > W:
        raise ValueError("Crop size should be smaller than the feature map size")

    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2

    end_h = start_h + crop_h
    end_w = start_h + crop_w

    mask = torch.zeros_like(feature_map)
    mask[:, :, start_h:end_h, start_w:end_w] = 1

    feature_map = feature_map * mask.detach()

    if mode == 'bhwc':
        feature_map = feature_map.permute(0, 2, 3, 1)

    return feature_map
