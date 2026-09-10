import torch
from pytorch3d.ops import knn_points

from utils.smpl_x import smpl_x


OTSU_BINS = 256
MAX_LOW_CLASS_MEAN = 0.10   
MIN_SEPARABILITY = 0.60     
MIN_PRUNE_FRAC = 0.05
MAX_PRUNE_FRAC = 0.95
DILATE_RINGS = 1
VIS_WARMUP_ITR = 2000
REFRESH_ITR = 500
NEAR_Z = 0.2


def otsu(v, n_bins=OTSU_BINS):
  
    v = v.detach().flatten().float()
    hist = torch.histc(v, bins=n_bins, min=0.0, max=1.0)
    total = hist.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = hist / total
    centers = (torch.arange(n_bins, device=v.device, dtype=torch.float32) + 0.5) / n_bins

    w0 = torch.cumsum(p, 0)
    w1 = 1.0 - w0
    csum = torch.cumsum(p * centers, 0)
    mu_total = csum[-1]
    mu0 = csum / w0.clamp(min=1e-12)
    mu1 = (mu_total - csum) / w1.clamp(min=1e-12)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2

    peak = sigma_b.max()
    if peak <= 0:
        return 0.0, 0.0, 0.0, 0.0
   
    plateau = torch.nonzero(sigma_b >= peak * (1.0 - 1e-6), as_tuple=True)[0]
    k = int(plateau[len(plateau) // 2])

    var_total = float((p * (centers - mu_total) ** 2).sum())
    eta = float(peak) / var_total if var_total > 0 else 0.0
    return float(centers[k]), eta, float(mu0[k]), float(mu1[k])


@torch.no_grad()
def analytic_visibility(mean_3d, cam_param, img_shape, bbox=None, is_world_coord=True):
   
    img_height, img_width = int(img_shape[0]), int(img_shape[1])
    focal = cam_param['focal'].view(2).float()

    if is_world_coord:
        R = cam_param['R'].view(3, 3).float()
        t = cam_param['t'].view(1, 3).float()
        p_cam = mean_3d @ R.permute(1, 0) + t            # world -> camera
    else:
        p_cam = mean_3d
    z = p_cam[:, 2]
    in_front = z > NEAR_Z
    z_safe = z.clamp(min=1e-6)

    # symmetric frustum, identical to get_fov() + get_proj_matrix()
    tan_x = img_width / (2.0 * focal[0])
    tan_y = img_height / (2.0 * focal[1])
    px = (p_cam[:, 0] / (z_safe * tan_x) + 1.0) * img_width * 0.5
    py = (p_cam[:, 1] / (z_safe * tan_y) + 1.0) * img_height * 0.5

    if bbox is None:
        xmin, ymin, xmax, ymax = 0.0, 0.0, float(img_width), float(img_height)
    else:
        b = bbox.view(4).float()
        # same clamping RGBLoss applies before cropping
        xmin = float(max(b[0], 0.0))
        ymin = float(max(b[1], 0.0))
        xmax = float(min(b[0] + b[2], img_width))
        ymax = float(min(b[1] + b[3], img_height))

    inside = (px >= xmin) & (px < xmax) & (py >= ymin) & (py < ymax)
    return in_front & inside


@torch.no_grad()
def dilate(mask, neighbor_idxs, neighbor_weights, rings=DILATE_RINGS):
    if rings <= 0:
        return mask
    valid = neighbor_weights < 0
    for _ in range(rings):
        mask = mask | (mask[neighbor_idxs] & valid).any(1)
    return mask


@torch.no_grad()
def build_keep_mask(v_captured, is_face, neighbor_idxs, neighbor_weights,
                    v_generated=None):
 
    tau, eta, mu0, mu1 = otsu(v_captured)

    keep = v_captured.view(-1) > tau
    n_cap_keep = int(keep.sum())
    if v_generated is not None:
        keep = keep | (v_generated.view(-1) > tau)
    n_union_keep = int(keep.sum())

    keep = keep | is_face.view(-1)                        
    keep = dilate(keep, neighbor_idxs, neighbor_weights)

    V = keep.numel()
    prune_frac = 1.0 - float(keep.sum()) / V

    enabled, reason = True, 'ok'
    if mu0 >= MAX_LOW_CLASS_MEAN:
       
        enabled, reason = False, 'sequence is fully observed (mu0 %.3f >= %.2f)' % (
            mu0, MAX_LOW_CLASS_MEAN)
    elif eta < MIN_SEPARABILITY:
        enabled, reason = False, 'no clear split in v (eta %.3f < %.2f)' % (eta, MIN_SEPARABILITY)
    elif prune_frac < MIN_PRUNE_FRAC:
        enabled, reason = False, 'nothing to prune (%.1f%% < %.1f%%)' % (
            prune_frac * 100, MIN_PRUNE_FRAC * 100)
    elif prune_frac > MAX_PRUNE_FRAC:
        enabled, reason = False, 'degenerate (%.1f%% > %.1f%%)' % (
            prune_frac * 100, MAX_PRUNE_FRAC * 100)

    if not enabled:
        keep = torch.ones_like(keep)

    info = {
        'tau': tau, 'eta': eta, 'mu0': mu0, 'mu1': mu1,
        'prune_frac': prune_frac if enabled else 0.0,
        'kept': int(keep.sum()), 'total': V,
        'kept_captured_only': n_cap_keep,
        'kept_after_gen_union': n_union_keep,
        'enabled': enabled, 'reason': reason,
    }
    return keep, info


@torch.no_grad()
def posed_gaussian_means(human_gaussian, smplx_param, cache=None):
   
    if cache is None:
        cache = {}
    if 'mesh_neutral_pose' not in cache:
        mesh_neutral_pose, mesh_wo_upsample, _, transform_mat_neutral_pose = \
            human_gaussian.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        joint_zero_pose = human_gaussian.get_zero_pose_human()

        nn_vertex_idxs = knn_points(mesh_neutral_pose[None, :, :],
                                    mesh_wo_upsample[None, :, :],
                                    K=1, return_nn=True).idx[0, :, 0]
        nn_vertex_idxs = human_gaussian.lr_idx_to_hr_idx(nn_vertex_idxs)
        part = (human_gaussian.is_rhand + human_gaussian.is_lhand + human_gaussian.is_face) > 0
        nn_vertex_idxs[part] = torch.arange(smpl_x.vertex_num_upsampled).cuda()[part]

        cache['mesh_neutral_pose'] = mesh_neutral_pose
        cache['transform_mat_neutral_pose'] = transform_mat_neutral_pose
        cache['joint_zero_pose'] = joint_zero_pose
        cache['nn_vertex_idxs'] = nn_vertex_idxs

    mean_3d = cache['mesh_neutral_pose']
    mean_3d = mean_3d + (smplx_param['expr'][None, None, :] * human_gaussian.expr_dirs).sum(2)

    transform_mat_joint = human_gaussian.get_transform_mat_joint(
        cache['transform_mat_neutral_pose'], cache['joint_zero_pose'], smplx_param)
    transform_mat_vertex = human_gaussian.get_transform_mat_vertex(
        transform_mat_joint, cache['nn_vertex_idxs'])
    return human_gaussian.lbs(mean_3d, transform_mat_vertex, smplx_param['trans']), cache
