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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def huber_loss(network_output, gt, alpha):
    diff = torch.abs(network_output - gt)
    mask = (diff < alpha).float()
    loss = 0.5*diff**2*mask + alpha*(diff-0.5*alpha)*(1.-mask)
    return loss.mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def cos_loss(output, gt, thrsh=0, weight=1):
    """Cosine loss for normal alignment.
    
    Args:
        output: predicted normals [3, H, W] or [N, 3]
        gt: ground truth normals [3, H, W] or [N, 3]
        thrsh: threshold angle in radians (default 0, meaning no threshold)
        weight: weight mask [H, W] or [N] (default 1, meaning uniform weight)
    """
    import numpy as np
    if output.dim() == 3:  # [3, H, W] format
        cos = torch.sum(output * gt * weight, 0)
        return (1 - cos[cos < np.cos(thrsh)]).mean() if thrsh > 0 else (1 - cos).mean()
    else:  # [N, 3] format
        # Handle weight being int, tensor, or None
        if isinstance(weight, int) or isinstance(weight, float):
            weight_tensor = weight
        elif hasattr(weight, 'dim'):
            weight_tensor = weight.unsqueeze(-1) if weight.dim() == 1 else weight
        else:
            weight_tensor = 1.0
        cos = torch.sum(output * gt * weight_tensor, -1)
        return (1 - cos[cos < np.cos(thrsh)]).mean() if thrsh > 0 else (1 - cos).mean()

def normal_smoothness_loss(gaussians, k=8):
    """Compute normal smoothness loss using KNN.
    
    Encourages neighboring gaussians to have similar normals for surfel-based representation.
    
    Args:
        gaussians: GaussianModel instance
        k: number of nearest neighbors to consider
    Returns:
        loss: scalar tensor representing the smoothness loss
    """
    try:
        from pytorch3d.ops import knn_points
        xyz = gaussians.get_xyz
        normals = gaussians.get_normal
        
        # Find K nearest neighbors (including self, so k+1)
        nn_dist, nn_idx, _ = knn_points(xyz[None], xyz[None], K=k+1, return_nn=True)
        nn_idx = nn_idx[0, :, 1:]  # Remove self, shape: [N, k]
        nn_normals = normals[nn_idx]  # [N, k, 3]
        
        # Average neighbor normals
        nn_normal_avg = torch.nn.functional.normalize(nn_normals.mean(dim=1), dim=-1)
        
        # Cosine loss between each gaussian's normal and average of its neighbors
        loss = cos_loss(normals, nn_normal_avg, thrsh=0)
        return loss
    except ImportError:
        # Fallback: return zero loss if pytorch3d is not available
        # User can install pytorch3d for this functionality
        return torch.tensor(0.0, device=gaussians.get_xyz.device, requires_grad=False)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

