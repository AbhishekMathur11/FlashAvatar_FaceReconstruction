import os, sys 
import random
import numpy as np
import torch
import argparse
import cv2
import time
import datetime
import lpips

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss
from utils.general_utils import normalize_for_percep
from pytorch3d.transforms import matrix_to_rotation_6d


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_mesh_ply(vertices, faces, path):
    """Save mesh as PLY file.
    
    Args:
        vertices: [N, 3] numpy array of vertex positions
        faces: [M, 3] numpy array of face indices (0-indexed)
        path: output file path
    """
    with open(path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def save_mesh_obj(vertices, faces, path):
    """Save mesh as OBJ file.
    
    Args:
        vertices: [N, 3] numpy array of vertex positions
        faces: [M, 3] numpy array of face indices (1-indexed for OBJ)
        path: output file path
    """
    with open(path, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--logname', type=str, default='log', help='log name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    percep_module = lpips.LPIPS(net='vgg').to(args.device)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    ## dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    logdir = os.path.join(data_dir, args.logname)
    os.makedirs(logdir, exist_ok=True)
    scene = Scene_mica(data_dir, mica_datadir, train_type=1, white_background=lpt.white_background, device = args.device)
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_save_path = os.path.join(logdir, 'test.avi')
    out = cv2.VideoWriter(vid_save_path, fourcc, 25, (args.image_res*2, args.image_res), True)
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer at {vid_save_path}")

    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    # Initialize metric accumulators
    psnr_all = []
    l1_all = []
    lpips_all = []

    print(f"Processing {len(viewpoint)} frames...")
    for iteration in range(len(viewpoint)):
        viewpoint_cam = viewpoint[iteration]
        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image= render_pkg["render"]
        image = image.clamp(0, 1)

        gt_image = viewpoint_cam.original_image.to(args.device)
        
        # Compute metrics
        with torch.no_grad():
            # PSNR
            psnr_val = psnr(image, gt_image).mean().item()
            psnr_all.append(psnr_val)
            
            # L1 loss
            l1_val = l1_loss(image, gt_image).item()
            l1_all.append(l1_val)
            
            # LPIPS
            image_percep = normalize_for_percep(image)
            gt_image_percep = normalize_for_percep(gt_image)
            lpips_val = torch.mean(percep_module.forward(image_percep, gt_image_percep)).item()
            lpips_all.append(lpips_val)
        
        # Print metrics for current frame
        print(f"Frame {iteration+1}/{len(viewpoint)} (ID: {frame_id}): PSNR={psnr_val:.4f}, L1={l1_val:.6f}, LPIPS={lpips_val:.6f}")

        save_image = np.zeros((args.image_res, args.image_res*2, 3))
        gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy()
        image_np = (image*255.).permute(1,2,0).detach().cpu().numpy()

        save_image[:, :args.image_res, :] = gt_image_np
        save_image[:, args.image_res:, :] = image_np
        save_image = save_image.astype(np.uint8)
        save_image = save_image[:,:,[2,1,0]]

        out.write(save_image)
    out.release()
    
    # Print average metrics
    avg_psnr = np.mean(psnr_all)
    avg_l1 = np.mean(l1_all)
    avg_lpips = np.mean(lpips_all)
    print(f"\n{'='*60}")
    print(f"Average Metrics (over {len(viewpoint)} frames):")
    print(f"  PSNR:  {avg_psnr:.4f}")
    print(f"  L1:    {avg_l1:.6f}")
    print(f"  LPIPS: {avg_lpips:.6f}")
    print(f"{'='*60}")
    
    # Export FLAME mesh for shape validation
    print(f"\nExporting mesh...")
    with torch.no_grad():
        # Get FLAME mesh with neutral expression (using shape from scene)
        shape_param = scene.shape_param.to(args.device)
        # Ensure shape_param has batch dimension [1, N]
        if shape_param.dim() == 1:
            shape_code = shape_param.unsqueeze(0)
        else:
            shape_code = shape_param
        
        # Create identity rotations in 6D format (required by FLAME)
        I = matrix_to_rotation_6d(torch.eye(3, device=args.device).unsqueeze(0))  # [1, 6]
        
        default_expr = torch.zeros(1, 100, device=args.device)
        default_jaw = I.clone()  # 6D rotation for jaw
        default_eyes = torch.cat([I.clone(), I.clone()], dim=1)  # 12D (6D for each eye)
        default_eyelids = torch.zeros(1, 2, device=args.device)
        
        flame_geometry = DeformModel.flame_model.forward_geo(
            shape_code,
            expression_params=default_expr,
            jaw_pose_params=default_jaw,
            eye_pose_params=default_eyes,
            eyelid_params=default_eyelids,
        )
        
        # Get vertices and faces
        vertices = flame_geometry[0].detach().cpu().numpy()  # [N, 3]
        faces = DeformModel.flame_model.faces.cpu().numpy()  # [M, 3]
        
        # Save as PLY and OBJ
        mesh_ply_path = os.path.join(logdir, 'mesh.ply')
        mesh_obj_path = os.path.join(logdir, 'mesh.obj')
        
        save_mesh_ply(vertices, faces, mesh_ply_path)
        save_mesh_obj(vertices, faces, mesh_obj_path)
        
        print(f"Mesh saved to:")
        print(f"  PLY: {mesh_ply_path}")
        print(f"  OBJ: {mesh_obj_path}")
    
    
   
        

           