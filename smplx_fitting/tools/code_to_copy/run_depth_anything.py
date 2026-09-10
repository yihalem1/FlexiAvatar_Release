import cv2
import numpy as np
import torch
import os.path as osp
from glob import glob
from pytorch3d.io import load_ply
import os
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer)
import json
from tqdm import tqdm
import argparse
import shutil

# ----------------------------- helpers (kept but unused) -----------------------------
def ensure_symlink(src, dst):
    try:
        if osp.islink(dst) or osp.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def select_subset_and_symlink(frames_dir, subset_dir, start_idx, end_idx):
    os.makedirs(subset_dir, exist_ok=True)
    img_paths = sorted(glob(osp.join(frames_dir, "*.png")))
    kept = []
    for p in img_paths:
        stem = osp.splitext(osp.basename(p))[0]
        if stem.isdigit():
            idx = int(stem)
            if start_idx <= idx <= end_idx:
                ensure_symlink(p, osp.join(subset_dir, f"{idx}.png"))
                kept.append(idx)
    kept.sort()
    return kept
# -----------------------------------------------------------------------

def render_depthmap(mesh, face, cam_param, render_shape):
    mesh = mesh.cuda()[None,:,:]
    face = face.cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}

    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2)  # flip x/y for PyTorch3D
    mesh = Meshes(mesh, face)

    cameras = PerspectiveCameras(
        focal_length=cam_param['focal'], 
        principal_point=cam_param['princpt'], 
        device='cuda',
        in_ndc=False,
        image_size=torch.LongTensor(render_shape).cuda().view(1,2)
    )
    raster_settings = RasterizationSettings(
        image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()

    with torch.no_grad():
        fragments = rasterizer(mesh)
    depthmap = fragments.zbuf.cpu().numpy()[0,:,:,0]
    return depthmap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path', required=True)
    parser.add_argument('--start_frame', type=int, default=0)   # kept but unused
    parser.add_argument('--end_frame', type=int, default=-1)    # kept but unused
    return parser.parse_args()

# get path
args = parse_args()
root_path = args.root_path

frames_dir = osp.join(root_path, 'frames')
frames_subset_dir = frames_dir  # use all frames
out_path = osp.join(root_path, 'depthmaps_subset')
os.makedirs(out_path, exist_ok=True)
# --------------------------------------------------------------------------

# run DepthAnything-V2 (on ALL frames)
assert osp.isfile('./checkpoints/depth_anything_v2_vitl.pth'), 'Please download depth_anything_v2_vitl.pth'
cmd = 'python run.py --encoder vitl --img-path ' + frames_subset_dir + ' --outdir ' + out_path + ' --pred-only --grayscale'
os.system(cmd)

# accumulators
depthmap_save = 0
color_save = 0
is_bkg_save = 0

# depthmaps list
depthmap_path_list = glob(osp.join(out_path, '*.png'))
assert len(depthmap_path_list) > 0, "No depthmaps were produced."
frame_idx_list = sorted([int(osp.splitext(osp.basename(x))[0]) for x in depthmap_path_list])

first_depth_img = cv2.imread(depthmap_path_list[0])
assert first_depth_img is not None, f"Failed to read {depthmap_path_list[0]}"
img_height, img_width = first_depth_img.shape[:2]

video_save = cv2.VideoWriter(
    osp.join(root_path, 'depthmaps.mp4'),
    cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height)
)

last_cam_param = None  # guard so we don't write point cloud without a camera

for frame_idx in tqdm(frame_idx_list):
    # load image
    img_path = osp.join(root_path, 'frames', f'{frame_idx}.png')
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Missing image for frame {frame_idx}: {img_path}. Skipping.")
        continue
    img = img.astype(np.float32)

    # load depthmap
    depthmap_path = osp.join(out_path, f'{frame_idx}.png')
    depth_bgr = cv2.imread(depthmap_path)
    if depth_bgr is None:
        print(f"[WARN] Missing depthmap for frame {frame_idx}: {depthmap_path}. Skipping.")
        continue
    
    # preview panel
    frame = np.concatenate((img, depth_bgr.astype(np.float32)), 1)
    frame = cv2.putText(
        frame, str(frame_idx),
        (int(img_width*0.1), int(img_height*0.1)),
        cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3
    )
    video_save.write(frame.astype(np.uint8))
    
    # SMPL-X depth for normalization
    smplx_mesh_path = osp.join(root_path, 'smplx_optimized', 'meshes_smoothed', f'{frame_idx}_smplx.ply')
    if not osp.isfile(smplx_mesh_path):
        continue
    smplx_vert, smplx_face = load_ply(smplx_mesh_path)
    with open(osp.join(root_path, 'cam_params', f'{frame_idx}.json')) as f:
        cam_param = json.load(f)
    last_cam_param = cam_param  # store last valid cam_param for point cloud write-out

    smplx_depthmap = render_depthmap(smplx_vert, smplx_face, cam_param, (img_height, img_width))
    smplx_is_fg = smplx_depthmap > 0

    # align depth
    depth_gray = 255 - depth_bgr[:,:,0]
    scale = np.abs(depth_gray[smplx_is_fg] - depth_gray[smplx_is_fg].mean()).mean()
    scale_smplx = np.abs(smplx_depthmap[smplx_is_fg] - smplx_depthmap[smplx_is_fg].mean()).mean()
    if scale <= 1e-6:
        print(f"[WARN] Degenerate scale for frame {frame_idx}; skipping.")
        continue
    depth_aligned = depth_gray / scale * scale_smplx
    depth_aligned = depth_aligned - depth_aligned[smplx_is_fg].mean() + smplx_depthmap[smplx_is_fg].mean()

    # robust mask load
    mask_path = osp.join(root_path, 'masks', f'{frame_idx}.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARN] Mask missing/unreadable for frame {frame_idx}: {mask_path}. Using SMPLX-derived bg.")
        is_bkg = ~smplx_is_fg
    else:
        is_fg = mask >= 128  # adjust threshold if needed
        is_bkg = ~is_fg

    # lazy init accumulators
    if isinstance(depthmap_save, int):
        depthmap_save = np.zeros_like(depth_aligned, dtype=np.float32)
        is_bkg_save = np.zeros_like(is_bkg, dtype=np.float32)
        color_save = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)

    # accumulate background
    depthmap_save += depth_aligned * is_bkg
    color_save += img * is_bkg[:,:,None]
    is_bkg_save += is_bkg.astype(np.float32)

video_save.release()

# If no background pixels at all, bail out
if not np.any(is_bkg_save > 0):
    raise RuntimeError("No background pixels accumulated. Check mask paths/thresholds.")

den = is_bkg_save.astype(np.float32)
den_safe = den.copy()
den_safe[den_safe == 0.0] = 1.0  # prevent divide-by-zero

# Per-pixel averaging without boolean flattening
depthmap_save = depthmap_save / den_safe
color_save = color_save / den_safe[..., None]
# ------------------------------------------------------------------------

# make sure we have a camera to write the point cloud
if last_cam_param is None:
    raise RuntimeError("No valid SMPL-X camera parameters found; cannot write background point cloud.")

# save background point cloud
cam_param = last_cam_param
with open(osp.join(root_path, 'bkg_point_cloud.txt'), 'w') as f:
    for i in tqdm(range(img_height)):
        for j in range(img_width):
            if den[i, j] > 0:  # background seen at least once
                z = float(depthmap_save[i, j])
                x = (j - cam_param['princpt'][0]) / cam_param['focal'][0] * z
                y = (i - cam_param['princpt'][1]) / cam_param['focal'][1] * z
                rgb = color_save[i, j]
                f.write(f"{x} {y} {z} {rgb[0]} {rgb[1]} {rgb[2]}\n")
