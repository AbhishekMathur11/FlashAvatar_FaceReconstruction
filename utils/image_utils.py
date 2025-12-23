#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.graphics_utils import fov2focal

def depth2normal(depth, mask, camera):
    """Convert depth map to normal map via gradient computation.
    
    Args:
        depth: depth map [1, H, W] or [H, W]
        mask: visibility mask [1, H, W] or [H, W]
        camera: camera object with FoVx, FoVy, image_width, image_height attributes
    Returns:
        normal: normal map [3, H, W]
    """
    # Ensure depth and mask are 3D tensors [1, H, W]
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    # Convert to camera space
    camD = depth.permute([1, 2, 0])  # [H, W, 1]
    mask_perm = mask.permute([1, 2, 0])  # [H, W, 1]
    shape = camD.shape
    device = camD.device
    h, w = torch.meshgrid(torch.arange(0, shape[0], device=device), 
                         torch.arange(0, shape[1], device=device), indexing='ij')
    h = h.to(torch.float32)
    w = w.to(torch.float32)
    p = torch.cat([w.unsqueeze(-1), h.unsqueeze(-1)], axis=-1)  # [H, W, 2]
    
    # Center the pixel coordinates
    p[..., 0:1] -= camera.image_width / 2.0
    p[..., 1:2] -= camera.image_height / 2.0
    p *= camD
    
    # Convert to camera space using focal length
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([[K00, 0], [0, K11]], device=device, dtype=torch.float32)
    Kinv = torch.inverse(K)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)  # [H, W, 3]
    
    # Pad for gradient computation
    camPos = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask_padded = torch.nn.functional.pad(mask_perm[None].to(torch.float32), 
                                          [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    
    # Compute gradients via cross products
    p_c = camPos[:, 1:-1, 1:-1, :] * mask_padded[:, 1:-1, 1:-1, :]
    p_u = (camPos[:, :-2, 1:-1, :] - p_c) * mask_padded[:, :-2, 1:-1, :]
    p_l = (camPos[:, 1:-1, :-2, :] - p_c) * mask_padded[:, 1:-1, :-2, :]
    p_b = (camPos[:, 2:, 1:-1, :] - p_c) * mask_padded[:, 2:, 1:-1, :]
    p_r = (camPos[:, 1:-1, 2:, :] - p_c) * mask_padded[:, 1:-1, 2:, :]
    
    # Compute normals from cross products
    n_ul = torch.cross(p_u, p_l, dim=-1)
    n_ur = torch.cross(p_r, p_u, dim=-1)
    n_br = torch.cross(p_b, p_r, dim=-1)
    n_bl = torch.cross(p_l, p_b, dim=-1)
    
    n = n_ul + n_ur + n_br + n_bl
    n = n[0]  # Remove batch dimension
    
    mask_final = mask_padded[0, 1:-1, 1:-1, :]
    n = torch.nn.functional.normalize(n, dim=-1)
    n = (n * mask_final).permute([2, 0, 1])  # [3, H, W]
    
    return n

