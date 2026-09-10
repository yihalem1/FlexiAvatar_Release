import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import eval_sh, RGB2SH, get_fov, get_view_matrix, get_proj_matrix
from utils.smpl_x import smpl_x
from utils.flame import flame
from smplx.lbs import batch_rigid_transform
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from nets.layer import make_linear_layers
from pytorch3d.structures import Meshes
from config import cfg
from utils import visibility
import copy

import math

class ContinuousTimeEncoder(nn.Module):
  
    def __init__(self,
                 mode='fourier',
                 n_freq=6,
                 f_min_hz=0.25,   # ~4s period
                 learn_time_scale=True,  
                 time2vec_dim=8):
        super().__init__()
        self.mode = mode
        self.n_freq = n_freq
        self.f_min_hz = f_min_hz
        self.learn_time_scale = learn_time_scale

        self.log_s = nn.Parameter(torch.zeros(())) if learn_time_scale else None

        if mode == 'fourier':
            k = torch.arange(n_freq, dtype=torch.float32)
            self.register_buffer('freqs_hz', f_min_hz * (2.0 ** k))  # (n_freq,)
        elif mode == 'time2vec':
            self.w0 = nn.Parameter(torch.randn(1))
            self.b0 = nn.Parameter(torch.zeros(1))
            self.Wp = nn.Parameter(torch.randn(time2vec_dim - 1))
            self.bp = nn.Parameter(torch.zeros(time2vec_dim - 1))
        else:
            raise ValueError("mode must be 'fourier' or 'time2vec'")

    @property
    def out_dim(self):
        if self.mode == 'fourier':
            return 1 + 2 * self.n_freq  
        else:
            return 1 + (self.Wp.numel())  

    def forward(self, frame_id: torch.Tensor, fps: float = None):
        """
        frame_id: shape (1,) int/float tensor
        fps     : float or None. If None, t_sec := frame_id * exp(log_s)
        returns : (out_dim,) float32
        """
        t = frame_id.to(dtype=torch.float32)
        if fps is not None:
            t_sec = t / float(fps)
            s = torch.exp(self.log_s) if self.learn_time_scale else 1.0
            t_sec = t_sec * s
        else:
            assert self.learn_time_scale, "Set learn_time_scale=True if fps is None."
            t_sec = t * torch.exp(self.log_s)

        if self.mode == 'fourier':
            angles = (2.0 * math.pi * self.freqs_hz) * t_sec 
            sin, cos = torch.sin(angles), torch.cos(angles)
            return torch.cat([t_sec, sin, cos], dim=0)
        else:
            # Time2Vec
            lin = self.w0 * t_sec + self.b0
            per = torch.sin(self.Wp * t_sec + self.bp)
            return torch.cat([lin.view(1), per], dim=0)


class RGBResidualInputCfg:
    def __init__(self,
                 use_triplane=True,
                 use_xyz=True,
                 use_time=True,
                 use_pose6d=False,
                 use_normal=False,
                 use_expr = True):
        self.use_triplane = use_triplane
        self.use_xyz = use_xyz
        self.use_time = use_time
        self.use_pose6d = use_pose6d
        self.use_normal = use_normal
        self.use_expr = use_expr

class ResidualOffsetMLP(nn.Module):

    def __init__(self, in_dim, hid=256, out_dim=3, use_gn=True, add_skip=True):
        super().__init__()
        def block(n_in, n_out, k=4):
            layers = []
            for i in range(k):
                layers.append(nn.Linear(n_in if i == 0 else n_out, n_out))
                if use_gn:
                    layers.append(nn.GroupNorm(4, n_out))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.add_skip = add_skip
        self.b1 = block(in_dim, hid)
        self.b2 = block(in_dim + hid if add_skip else hid, hid)
        self.head = nn.Sequential(nn.Linear(hid, out_dim), nn.Tanh())

    def forward(self, x):
        h1 = self.b1(x)
        if self.add_skip:
            h2 = self.b2(torch.cat([x, h1], dim=-1))
        else:
            h2 = self.b2(h1)
        return self.head(h2)

def cat_params_to_optimizer(params_new, optimizer):
    optimizable_params = {}
    for group in optimizer.param_groups:
        if group['name'] not in params_new:
            continue
        param_new = params_new[group['name']]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(param_new)))
            stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(param_new)))

            del optimizer.state[group['params'][0]]
            group['params'][0] = nn.Parameter(torch.cat((group['params'][0], param_new)).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_params[group['name']] = group['params'][0]
        else:
            group['params'][0] = nn.Parameter(torch.cat((group['params'][0], param_new)).requires_grad_(True))
            optimizable_params[group['name']] = group['params'][0]
    return optimizable_params

def prune_params_from_optimizer(param_names, is_valid, optimizer):
    optimizable_params = {}
    for group in optimizer.param_groups:
        if group['name'] not in param_names:
            continue
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state['exp_avg'] = stored_state['exp_avg'][is_valid]
            stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][is_valid]

            del optimizer.state[group['params'][0]]
            group['params'][0] = nn.Parameter((group['params'][0][is_valid].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_params[group['name']] = group['params'][0]
        else:
            group['params'][0] = nn.Parameter(group['params'][0][is_valid].requires_grad_(True))
            optimizable_params[group['name']] = group['params'][0]
    return optimizable_params

def replace_param_from_optimizer(param, name, optimizer):
    optimizable_params = {}
    for group in optimizer.param_groups:
        if group['name'] == name:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.zeros_like(param)
                stored_state['exp_avg_sq'] = torch.zeros_like(param)
                del optimizer.state[group['params'][0]]

            group['params'][0] = nn.Parameter(param.requires_grad_(True))
            if stored_state is not None:
                optimizer.state[group['params'][0]] = stored_state
            optimizable_params[group['name']] = group['params'][0]
    return optimizable_params

class SceneGaussian(nn.Module):
    def __init__(self):
        super(SceneGaussian, self).__init__()
    
    def init_from_point_cloud(self, xyz, rgb, cam_dist):
        # point cloud of a scene
        xyz, rgb = xyz.cuda(), rgb.cuda()
        point_num = xyz.shape[0]

        # initialize scale and rotation
        points = knn_points(xyz[None,:,:], xyz[None,:,:], K=4, return_nn=True)
        dist = torch.sum((xyz[:,None,:] - points.knn[0,:,1:,:])**2,2).mean(1) # average of distances to top-3 closest points (exactly same as https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/gaussian_model.py#L134)
        dist = torch.clamp_min(dist, 0.0000001)
        scale = torch.log(torch.sqrt(dist))[:,None].repeat(1,3)
        rotation = matrix_to_rotation_6d(torch.eye(3)[None,:,:].repeat(point_num,1,1).float().cuda())

        # initialize spherical harmonics of color
        feature = torch.zeros((point_num, (cfg.max_sh_degree+1)**2, 3)).float().cuda()
        feature[:,0,:] = RGB2SH(rgb) # first band

        # initialize opacity
        opacity = 0.1 * torch.ones((point_num,1)).float().cuda()
        opacity = torch.log(opacity / (1 - opacity)) # inverse of sigmoid
        
        # register parameters
        # Gaussians
        self.register_buffer('point_num', torch.LongTensor([point_num]).cuda())
        self.mean = nn.Parameter(xyz)
        self.scale = nn.Parameter(scale)
        self.rotation = nn.Parameter(rotation)
        self.feature_dc = nn.Parameter(feature[:,0:1,:])
        self.feature_rest = nn.Parameter(feature[:,1:,:])
        self.opacity = nn.Parameter(opacity)
        self.register_buffer('active_sh_degree', torch.zeros((1)).float().cuda())
        self.register_buffer('radius_max', torch.zeros((point_num)).float().cuda())
        self.register_buffer('xyz_grad_accum', torch.zeros((point_num,1)).float().cuda())
        self.register_buffer('track_cnt', torch.zeros((point_num,1)).float().cuda())
        # camera distribution of a scene
        self.register_buffer('cam_dist_trans', cam_dist['translate'].cuda())
        self.register_buffer('cam_dist_radius', cam_dist['radius'].cuda())

    def init_from_point_num(self, point_num):
        # register parameters
        # Gaussians
        point_num = int(point_num)
        self.register_buffer('point_num', torch.LongTensor([point_num]).cuda())
        self.mean = nn.Parameter(torch.zeros((point_num,3)).float().cuda())
        self.scale = nn.Parameter(torch.zeros((point_num,3)).float().cuda())
        self.rotation = nn.Parameter(torch.zeros((point_num,6)).float().cuda())
        self.feature_dc = nn.Parameter(torch.zeros((point_num,1,3)).float().cuda())
        self.feature_rest = nn.Parameter(torch.zeros((point_num,(cfg.max_sh_degree+1)**2-1,3)).float().cuda())
        self.opacity = nn.Parameter(torch.zeros((point_num,1)).float().cuda())
        self.register_buffer('active_sh_degree', torch.zeros((1)).float().cuda())
        self.register_buffer('radius_max', torch.zeros((point_num)).float().cuda())
        self.register_buffer('xyz_grad_accum', torch.zeros((point_num,1)).float().cuda())
        self.register_buffer('track_cnt', torch.zeros((point_num,1)).float().cuda())
        self.register_buffer('cam_dist_trans', torch.zeros((3)).float().cuda())
        self.register_buffer('cam_dist_radius', torch.zeros((1)).float().cuda())
  
    def get_optimizable_params(self):
        optimizable_params = [
            {'params': [self.mean], 'name': 'mean_scene', 'lr': cfg.position_lr_init * float(self.cam_dist_radius)},
            {'params': [self.feature_dc], 'name': 'feature_dc_scene', 'lr': cfg.feature_lr},
            {'params': [self.feature_rest], 'name': 'feature_rest_scene', 'lr': cfg.feature_lr / 20.0},
            {'params': [self.opacity], 'name': 'opacity_scene', 'lr': cfg.opacity_lr},
            {'params': [self.scale], 'name': 'scale_scene', 'lr': cfg.scale_lr},
            {'params': [self.rotation], 'name': 'rotation_scene', 'lr': cfg.rotation_lr}
        ]
        return optimizable_params

    def set_sh_degree(self, itr):
        self.active_sh_degree[:] = min(itr // cfg.increase_sh_degree_interval, cfg.max_sh_degree)

    def track_stats(self, mean_2d_grad, do_update):
        self.xyz_grad_accum[do_update,:] += torch.norm(mean_2d_grad[do_update,:2], dim=1, keepdim=True)
        self.track_cnt[do_update,:] += 1

    def densify_and_prune(self, screen_size_max, optimizer):
        grad_track = torch.nan_to_num(self.xyz_grad_accum / self.track_cnt)

        self.clone_points(grad_track, optimizer)
        self.split_points(grad_track, optimizer)

        do_prune = (torch.sigmoid(self.opacity) < cfg.opacity_min)[:,0]
        if screen_size_max:
            big_points_vs = self.radius_max > screen_size_max
            big_points_ws = torch.max(torch.exp(self.scale), 1).values > 0.1 * self.cam_dist_radius
            do_prune = torch.logical_or(torch.logical_or(do_prune, big_points_vs), big_points_ws)
        self.prune_points(do_prune, optimizer)

        torch.cuda.empty_cache()

    def clone_points(self, grad, optimizer):
        # extract points that satisfy the gradient condition
        mask = torch.where(grad[:,0] >= cfg.densify_grad_thr, True, False)
        mask = torch.logical_and(mask, torch.max(torch.exp(self.scale), 1).values <= cfg.dense_percent_thr*self.cam_dist_radius)
        mean_new = self.mean[mask]
        feature_dc_new = self.feature_dc[mask]
        feature_rest_new = self.feature_rest[mask]
        opacity_new = self.opacity[mask]
        scale_new = self.scale[mask]
        rotation_new = self.rotation[mask]
        self.densify(mean_new, feature_dc_new, feature_rest_new, opacity_new, scale_new, rotation_new, optimizer)

    def split_points(self, grad, optimizer, split_factor=2):
        # extract points that satisfy the gradient condition
        # pad tracked stats due to the changed point_num in clone_points
        padded_grad = torch.zeros((int(self.point_num))).float().cuda()
        padded_grad[:grad.shape[0]] = grad[:,0]
        mask = torch.where(padded_grad >= cfg.densify_grad_thr, True, False)
        mask = torch.logical_and(mask, torch.max(torch.exp(self.scale), 1).values > cfg.dense_percent_thr*self.cam_dist_radius)

        std = torch.exp(self.scale)[mask,:].repeat(split_factor,1)
        mean = torch.zeros((std.shape[0],3)).float().cuda()
        samples = torch.normal(mean=mean, std=std)
        rotation = rotation_6d_to_matrix(self.rotation[mask,:]).repeat(split_factor,1,1)

        mean_new = torch.bmm(rotation, samples[:,:,None])[:,:,0] + self.mean[mask,:].repeat(split_factor,1)
        scale_new = torch.log(torch.exp(self.scale)[mask,:].repeat(split_factor,1) / (0.8*split_factor))
        rotation_new = self.rotation[mask,:].repeat(split_factor,1)
        feature_dc_new = self.feature_dc[mask,:,:].repeat(split_factor,1,1)
        feature_rest_new = self.feature_rest[mask,:,:].repeat(split_factor,1,1)
        opacity_new = self.opacity[mask,:].repeat(split_factor,1)
        point_num_new = mean_new.shape[0]

        self.densify(mean_new, feature_dc_new, feature_rest_new, opacity_new, scale_new, rotation_new, optimizer)

        do_prune = torch.cat((mask, torch.zeros((point_num_new)).cuda().bool()))
        self.prune_points(do_prune, optimizer)

    def densify(self, mean, feature_dc, feature_rest, opacity, scale, rotation, optimizer):
        optimizable_params_new = {'mean_scene': mean, 'feature_dc_scene': feature_dc, 'feature_rest_scene': feature_rest, 'opacity_scene': opacity, 'scale_scene': scale, 'rotation_scene': rotation}

        optimizable_params = cat_params_to_optimizer(optimizable_params_new, optimizer)
        self.point_num[:] = optimizable_params['mean_scene'].shape[0]
        self.mean = optimizable_params['mean_scene']
        self.feature_dc = optimizable_params['feature_dc_scene']
        self.feature_rest = optimizable_params['feature_rest_scene']
        self.opacity = optimizable_params['opacity_scene']
        self.scale = optimizable_params['scale_scene']
        self.rotation = optimizable_params['rotation_scene']

        self.radius_max = torch.zeros((int(self.point_num))).float().cuda()
        self.xyz_grad_accum = torch.zeros((int(self.point_num),1)).float().cuda()
        self.track_cnt = torch.zeros((int(self.point_num),1)).float().cuda()

    def prune_points(self, do_prune, optimizer):
        param_names = ['mean_scene', 'feature_dc_scene', 'feature_rest_scene', 'opacity_scene', 'scale_scene', 'rotation_scene']
        is_valid = ~do_prune
        optimizable_params = prune_params_from_optimizer(param_names, is_valid, optimizer)
        
        self.point_num[:] = optimizable_params['mean_scene'].shape[0]
        self.mean = optimizable_params['mean_scene']
        self.feature_dc = optimizable_params['feature_dc_scene']
        self.feature_rest = optimizable_params['feature_rest_scene']
        self.opacity = optimizable_params['opacity_scene']
        self.scale = optimizable_params['scale_scene']
        self.rotation = optimizable_params['rotation_scene']

        self.radius_max = self.radius_max[is_valid]
        self.xyz_grad_accum = self.xyz_grad_accum[is_valid]
        self.track_cnt = self.track_cnt[is_valid]

    def reset_opacity(self, optimizer):
        opacity_new = torch.minimum(torch.sigmoid(self.opacity), torch.ones_like(self.opacity)*0.01)
        opacity_new = torch.log(opacity_new / (1 - opacity_new)) # inverse of sigmoid
        optimizable_params = replace_param_from_optimizer(opacity_new, 'opacity_scene', optimizer)
        self.opacity = optimizable_params['opacity_scene']
 
    def forward(self, cam_param):
        mean_3d = self.mean
        opacity = torch.sigmoid(self.opacity)
        scale = torch.exp(self.scale)
        rotation = matrix_to_quaternion(rotation_6d_to_matrix(self.rotation))
        sh = torch.cat((self.feature_dc, self.feature_rest),1)
        active_sh_degree = self.active_sh_degree
        
        # sh -> rgb
        sh = sh.permute(0,2,1)
        cam_pos = torch.matmul(torch.inverse(cam_param['R']), -cam_param['t'].view(3,1)).view(1,3)
        view_dir = F.normalize(mean_3d - cam_pos, p=2, dim=1)
        rgb = eval_sh(active_sh_degree, sh, view_dir)
        rgb = torch.clamp_min(rgb + 0.5, 0.0)

        return {'mean_3d': mean_3d, 
                'opacity': opacity, 
                'scale': scale, 
                'rotation': rotation, 
                'rgb': rgb}


USE_LOCAL_RGB = True  


class HumanGaussian(nn.Module):
    def __init__(self, rgb_res_cfg=None, residual_scale= 0.05):
        super(HumanGaussian, self).__init__()
        self.triplane = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        self.triplane_face = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())

        self.time_enc_mode = 'fourier'   # or 'time2vec'

        if cfg.subject_id in ["seattle", "citron"]:
            self.f_min_hz = 0.10
            self.n_freq = 6
            
        else:  # Bike, Jogging
            self.f_min_hz = 0.10
            self.n_freq = 8

        print("subject_id:", cfg.subject_id, "f_min_hz:", self.f_min_hz, "n_freq:", self.n_freq)

        self.time_embed = ContinuousTimeEncoder(
            mode=self.time_enc_mode,
            n_freq=self.n_freq,         
            f_min_hz=self.f_min_hz,    
            learn_time_scale=True,  
            time2vec_dim=16
        ).cuda()
        self.default_fps = getattr(cfg, 'fps', None)  


        self.geo_net = make_linear_layers([cfg.triplane_shape[0]*3, 128, 128, 128], use_gn=True)
        self.mean_offset_net = make_linear_layers([128, 3], relu_final=False)
        self.scale_net = make_linear_layers([128, 1], relu_final=False)
        self.geo_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(len(smpl_x.joint_part['body'])-1)*6, 128, 128, 128], use_gn=True)
        self.mean_offset_offset_net = make_linear_layers([128, 3], relu_final=False)
        self.scale_offset_net = make_linear_layers([128, 1], relu_final=False)
        feat_dim = cfg.triplane_shape[0] * 3 

      
        if USE_LOCAL_RGB:
            self.rgb_net_global = make_linear_layers(
                [feat_dim, 256, 256, 256, 3], relu_final=False, use_gn=True)

        

            self.pe_L_xyz = 10  
            self.pe_L_t = 8  

            PE_XYZ_DIM = 3 + 2 * self.pe_L_xyz * 3  # 63
            PE_T_DIM = 1 + 2 * self.pe_L_t * 1  # 11
            IN_DIM = PE_XYZ_DIM + PE_T_DIM  # 74

            self.rgb_res_cfg = rgb_res_cfg or RGBResidualInputCfg(
                use_triplane=False,  
                use_xyz=True,
                use_time=True,
                use_pose6d=False,
                use_normal=False,
                use_expr=False,
            )
            self.residual_scale = residual_scale  

            max_in_dim = self._compute_max_residual_in_dim()

            H = 256
            self.rgb_res_face = ResidualOffsetMLP(max_in_dim, hid=H, out_dim=3, use_gn=True, add_skip=True)
            self.rgb_res_lhand = ResidualOffsetMLP(max_in_dim, hid=H, out_dim=3, use_gn=True, add_skip=True)
            self.rgb_res_rhand = ResidualOffsetMLP(max_in_dim, hid=H, out_dim=3, use_gn=True, add_skip=True)
            self.rgb_res_rest = ResidualOffsetMLP(max_in_dim, hid=H, out_dim=3, use_gn=True, add_skip=True)

            pose_dim = (smpl_x.joint_num - 1) * 6
            feat_dim = cfg.triplane_shape[0] * 3
            in_dim = feat_dim + pose_dim + 3
            self.rgb_offset_net = make_linear_layers(
                [in_dim, 256, 256, 256, 3], relu_final=False, use_gn=True)

        else:
            self.rgb_net = make_linear_layers(
                [feat_dim, 128, 128, 128, 3], relu_final=False, use_gn=True)

        self._zero_init_residual(self.rgb_res_face)
        self._zero_init_residual(self.rgb_res_lhand)
        self._zero_init_residual(self.rgb_res_rhand)
        self._zero_init_residual(self.rgb_res_rest)

        feat_dim = cfg.triplane_shape[0] * 3  

        pose_dim = (smpl_x.joint_num - 1) * 6  
        in_dim = feat_dim + pose_dim + 3  
        self.rgb_offset_net = make_linear_layers(
            [in_dim, 256, 256, 256, 3],
            relu_final=False, use_gn=True)

        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender]).cuda()
        self.shape_param = nn.Parameter(smpl_x.shape_param.float().cuda())
        self.joint_offset = nn.Parameter(smpl_x.joint_offset.float().cuda())
     
    def init(self):
        xyz, _, _, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        skinning_weight = self.smplx_layer.lbs_weights.float()
        pose_dirs = self.smplx_layer.posedirs.permute(1,0).reshape(smpl_x.vertex_num,3*(smpl_x.joint_num-1)*9)
        expr_dirs = self.smplx_layer.expr_dirs.view(smpl_x.vertex_num,3*smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face, is_face_expr = torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda()
        is_rhand[smpl_x.rhand_vertex_idx], is_lhand[smpl_x.lhand_vertex_idx], is_face[smpl_x.face_vertex_idx], is_face_expr[smpl_x.expr_vertex_idx] = 1.0, 1.0, 1.0, 1.0
        is_cavity = torch.FloatTensor(smpl_x.is_cavity).cuda()[:,None]

        _, skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num,3)).float().cuda(), [skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity]) # upsample with dummy vertex

        pose_dirs = pose_dirs.reshape(smpl_x.vertex_num_upsampled*3,(smpl_x.joint_num-1)*9).permute(1,0) 
        expr_dirs = expr_dirs.view(smpl_x.vertex_num_upsampled,3,smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face, is_face_expr = is_rhand[:,0] > 0, is_lhand[:,0] > 0, is_face[:,0] > 0, is_face_expr[:,0] > 0
        is_cavity = is_cavity[:,0] > 0

        self.register_buffer('pos_enc_mesh', xyz)
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('pose_dirs', pose_dirs)
        self.register_buffer('expr_dirs', expr_dirs)
        self.register_buffer('is_rhand', is_rhand)
        self.register_buffer('is_lhand', is_lhand)
        self.register_buffer('is_face', is_face)
        self.register_buffer('is_face_expr', is_face_expr)
        self.register_buffer('is_cavity', is_cavity)
        V = smpl_x.vertex_num_upsampled
        self.register_buffer('vis_hits', torch.zeros((V,)).float().cuda())
        self.register_buffer('vis_hits_gen', torch.zeros((V,)).float().cuda())
        self.register_buffer('obs_frames', torch.zeros((1,)).float().cuda())
        self.register_buffer('obs_frames_gen', torch.zeros((1,)).float().cuda())
        self.register_buffer('vis_keep', torch.ones((V,)).bool().cuda())
        self.register_buffer('vis_ready', torch.zeros((1,)).float().cuda())
        self.register_buffer('vis_enabled', torch.zeros((1,)).float().cuda())
        self.register_buffer('vis_stats', torch.zeros((3,)).float().cuda())  # tau, eta, mu0

    def set_mesh_neighbors(self, neighbor_idxs, neighbor_weights):
      
        self._neighbor_idxs = neighbor_idxs
        self._neighbor_weights = neighbor_weights

    @torch.no_grad()
    def accumulate_visibility(self, render_is_vis, render_idx, mean_3d,
                              cam_param, img_shape, bbox, is_generated):
    
        if render_idx is None:
            v = render_is_vis.view(-1).float()
        else:
            v = visibility.analytic_visibility(mean_3d, cam_param, img_shape, bbox).float()
            v[render_idx] = render_is_vis.view(-1).float()

        if is_generated:
            self.vis_hits_gen += v
            self.obs_frames_gen += 1
        else:
            self.vis_hits += v
            self.obs_frames += 1

    @torch.no_grad()
    def update_keep_mask(self, cur_itr, logger=None):
      
        if float(self.obs_frames) < 1:
            return
        v_cap = self.vis_hits / self.obs_frames
        v_gen = (self.vis_hits_gen / self.obs_frames_gen) if float(self.obs_frames_gen) > 0 else None

        keep, info = visibility.build_keep_mask(
            v_cap, self.is_face, self._neighbor_idxs, self._neighbor_weights, v_gen)

        self.vis_keep.copy_(keep)
        self.vis_ready.fill_(1.0)
        self.vis_enabled.fill_(1.0 if info['enabled'] else 0.0)
        self.vis_stats.copy_(torch.tensor([info['tau'], info['eta'], info['mu0']],
                                          device=self.vis_stats.device))

        msg = ('[vis @ itr %d] tau*=%.3f eta=%.3f mu0=%.4f | keep %d/%d (prune %.1f%%) | %s'
               % (cur_itr, info['tau'], info['eta'], info['mu0'], info['kept'],
                  info['total'], info['prune_frac'] * 100,
                  'ENABLED' if info['enabled'] else 'DISABLED: ' + info['reason']))
        if v_gen is not None and info['enabled']:
            msg += ' | generated block re-admitted %d' % (
                info['kept_after_gen_union'] - info['kept_captured_only'])
        print(msg)
        if logger is not None:
            logger.info(msg)
        return info

    def visibility_active(self, apply_visibility, vis_itr):
        if not apply_visibility:
            return False
        if not (bool(self.vis_ready.item()) and bool(self.vis_enabled.item())):
            return False

        return (vis_itr is None) or (vis_itr >= visibility.VIS_WARMUP_ITR)

    def _zero_init_residual(self, mlp):
        last = None
        for m in reversed(list(mlp.modules())):
            if isinstance(m, nn.Linear):
                last = m
                break
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _compute_max_residual_in_dim(self):
        tri = cfg.triplane_shape[0] * 3
        xyz = 3 + 2 * self.pe_L_xyz * 3
        t = self.time_embed.out_dim

        pose6d = (smpl_x.joint_num - 1) * 6
        normal = 3
        expr = smpl_x.expr_param_dim
        return tri + xyz + t + pose6d + normal + expr

    def _build_residual_input(self, tri_feat, xyz_world, frame_id, pose6d_all, normals_world, expr):
        V = tri_feat.shape[0]
        max_in = self._compute_max_residual_in_dim()
        x = torch.zeros((V, max_in), device=tri_feat.device, dtype=tri_feat.dtype)
        ofs = 0
        if self.rgb_res_cfg.use_triplane:
            D = tri_feat.shape[1];
            x[:, ofs:ofs + D] = tri_feat;
            ofs += D
        if self.rgb_res_cfg.use_xyz:
            pe_xyz = self._pos_enc(xyz_world, L=self.pe_L_xyz);
            D = pe_xyz.shape[1]
            x[:, ofs:ofs + D] = pe_xyz;
            ofs += D
        if self.rgb_res_cfg.use_time:
            pe_t = self.time_embed(frame_id.view(1), fps=self.default_fps)  # (D_t,)
            pe_t = pe_t[None, :].expand(V, -1)  # broadcast to vertices
            D = pe_t.shape[1]
            x[:, ofs:ofs + D] = pe_t; ofs += D
        if self.rgb_res_cfg.use_pose6d:
            D = pose6d_all.shape[1];
            x[:, ofs:ofs + D] = pose6d_all;
            ofs += D
        if self.rgb_res_cfg.use_normal:
            D = 3;
            x[:, ofs:ofs + D] = normals_world;
            ofs += D
        if self.rgb_res_cfg.use_expr:
            expr_v = expr[None, :].expand(V, -1)  # (V, expr_dim)
            D = expr_v.shape[1];
            x[:, ofs:ofs + D] = expr_v;
            ofs += D
        return x

    def get_optimizable_params(self):
        optimizable_params = [
            {'params': [self.triplane], 'name': 'triplane_human', 'lr': cfg.lr},
            {'params': [self.triplane_face], 'name': 'triplane_face_human', 'lr': cfg.lr},
            {'params': list(self.geo_net.parameters()), 'name': 'geo_net_human', 'lr': cfg.lr},
            {'params': list(self.mean_offset_net.parameters()), 'name': 'mean_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.scale_net.parameters()), 'name': 'scale_net_human', 'lr': cfg.lr},
            {'params': list(self.geo_offset_net.parameters()), 'name': 'geo_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.mean_offset_offset_net.parameters()), 'name': 'mean_offset_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.scale_offset_net.parameters()), 'name': 'scale_offset_net_human', 'lr': cfg.lr},
            {'params': list(self.rgb_offset_net.parameters()), 'name': 'rgb_offset_net_human', 'lr': cfg.lr},
            {'params': [self.shape_param], 'name': 'shape_param_human', 'lr': cfg.lr},
            {'params': [self.joint_offset], 'name': 'joint_offset_human', 'lr': cfg.lr},
        ]

        if USE_LOCAL_RGB:
            optimizable_params += [
                {'params': list(self.rgb_net_global.parameters()),
                 'name': 'rgb_global_human', 'lr': cfg.lr},
           
                {'params': list(self.rgb_res_face.parameters()), 'name': 'rgb_res_face', 'lr': cfg.lr},
                {'params': list(self.rgb_res_lhand.parameters()), 'name': 'rgb_res_lhand', 'lr': cfg.lr},
                {'params': list(self.rgb_res_rhand.parameters()), 'name': 'rgb_res_rhand', 'lr': cfg.lr},
                {'params': list(self.rgb_res_rest.parameters()), 'name': 'rgb_res_rest', 'lr': cfg.lr},

            ]
        else:
            optimizable_params += [
                {'params': list(self.rgb_net.parameters()),
                 'name': 'rgb_net_human', 'lr': cfg.lr}
            ]
        return optimizable_params

    def _pos_enc(self, x: torch.Tensor, L: int = 10):
        """Standard NeRF‑style sin/cos positional encoding."""
        freqs = 2 ** torch.arange(L, device=x.device) * math.pi
        out = [x]
        for f in freqs:
            out += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(out, dim=-1)  # (N, 3+2·3L)

    @staticmethod
    def _time_enc(frame_id: torch.Tensor,
                  n_frames: int,
                  n_freq: int = 5) -> torch.Tensor:

        frame_id = torch.as_tensor(frame_id, dtype=torch.float32,
                                   device=frame_id.device).flatten()

        t_norm = 2.0 * frame_id / (n_frames - 1) - 1.0  # (1,)

        freqs = 2.0 ** torch.arange(n_freq, device=frame_id.device) * math.pi
        angles = freqs * t_norm  # (n_freq,)
        sin, cos = torch.sin(angles), torch.cos(angles)

        return torch.cat([t_norm, sin, cos])

    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info):
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smpl_x.neutral_body_pose.view(1,-1).cuda() # 大 pose
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth
        if use_id_info:
            shape_param = self.shape_param[None,:]
            face_offset = smpl_x.face_offset[None,:,:].float().cuda()
            joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        else:
            shape_param = torch.zeros((1,smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None
        output = self.smplx_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        mesh_neutral_pose = output.vertices[0] # 大 pose human
        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh(mesh_neutral_pose) # 大 pose human
        joint_neutral_pose = output.joints[0][:smpl_x.joint_num,:] # 大 pose human

        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(len(smpl_x.joint_part['body'])-1,3)
        zero_hand_pose = zero_hand_pose.view(len(smpl_x.joint_part['lhand']),3)
        neutral_body_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(neutral_body_pose)))
        jaw_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(jaw_pose)))
        pose = torch.cat((zero_pose, neutral_body_pose_inv, jaw_pose_inv, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(pose[None,:,:,:], joint_neutral_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_neutral_pose = transform_mat_neutral_pose[0]
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose

    def get_zero_pose_human(self, return_mesh=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_body_pose = torch.zeros((1,(len(smpl_x.joint_part['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        shape_param = self.shape_param[None,:]
        face_offset = smpl_x.face_offset[None,:,:].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        joint_zero_pose = output.joints[0][:smpl_x.joint_num,:] # zero pose human
        if not return_mesh:
            return joint_zero_pose
        else: 
            mesh_zero_pose = output.vertices[0] # zero pose human
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(mesh_zero_pose) # zero pose human
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

    def get_transform_mat_joint(self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        trans = smplx_param['trans'].view(1,3)

        # forward kinematics
        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]
        
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint
    
    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs):
        skinning_weight = self.skinning_weight[nn_vertex_idxs,:]
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(smpl_x.joint_num,16)).view(smpl_x.vertex_num_upsampled,4,4)
        return transform_mat_vertex

    def lbs(self, xyz, transform_mat_vertex, trans):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(smpl_x.vertex_num_upsampled,4)[:,:3]
        xyz = xyz + trans
        return xyz
    
    def extract_tri_feature(self):
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_shape_3d[2]/2)
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3

        ## 2. triplane features of face vertices
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_mesh[self.is_face,:]
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_face_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_face_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_face_shape_3d[2]/2)
        
        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane_face[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane_face[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane_face[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat_face = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # sum(self.is_face), cfg.triplane_shape[0]*3
        
        tri_feat[self.is_face] = tri_feat_face
        return tri_feat

    def forward_geo_network(self, tri_feat, smplx_param):
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)

        # combine pose with triplane feature
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).view(1,(len(smpl_x.joint_part['body'])-1)*6).repeat(smpl_x.vertex_num_upsampled,1) # without root pose
        feat = torch.cat((tri_feat, pose.detach()),1)

        # forward to geometry networks
        geo_offset_feat = self.geo_offset_net(feat)
        mean_offset_offset = self.mean_offset_offset_net(geo_offset_feat) # pose-dependent mean offset of Gaussians
        scale_offset = self.scale_offset_net(geo_offset_feat) # pose-dependent scale of Gaussians
        return mean_offset_offset, scale_offset
    
    def get_mean_offset_offset(self, smplx_param, mean_offset_offset):
        # poses from smplx parameters
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        pose = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) # without root pose

        # smplx pose-dependent vertex offset
        pose = (axis_angle_to_matrix(pose) - torch.eye(3)[None,:,:].float().cuda()).view(1,(smpl_x.joint_num-1)*9)
        smplx_pose_offset = torch.matmul(pose.detach(), self.pose_dirs).view(smpl_x.vertex_num_upsampled,3)

        mask = ((self.is_rhand + self.is_lhand + self.is_face_expr) > 0)[:,None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        smplx_pose_offset = smplx_pose_offset * mask
        output = mean_offset_offset + smplx_pose_offset
        return output, mean_offset_offset

    def forward_rgb_network(self, tri_feat, smplx_param, cam_param, xyz):

        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body']) - 1, 3)
        jaw_pose = smplx_param['jaw_pose'].view(1, 3)
        leye_pose = smplx_param['leye_pose'].view(1, 3)
        reye_pose = smplx_param['reye_pose'].view(1, 3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']), 3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']), 3)

        pose_all = torch.cat(
            (body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose), dim=0
        )  # (J‑1, 3)

        pose6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose_all))  # (J‑1, 6)
        pose6d = pose6d.view(1, -1).repeat(smpl_x.vertex_num_upsampled, 1)  # (V, 6*(J‑1))

        with torch.no_grad():
            normal = Meshes(
                verts=xyz[None],
                faces=torch.as_tensor(smpl_x.face_upsampled, device=xyz.device)[None]
            ).verts_normals_packed()  # (V,3)
            is_cavity = self.is_cavity[:, None].float()
            normal = normal * (1 - is_cavity) - normal * is_cavity

        rgb_feat = torch.cat((tri_feat, pose6d, normal), dim=1)
        rgb_offset = self.rgb_offset_net(rgb_feat)  # (V,3)
        return rgb_offset
    
    def lr_idx_to_hr_idx(self, idx):
        return idx

    def forward(self, smplx_param, cam_param, is_world_coord=False, cur_itr=None,
                vis_itr=None, apply_visibility=False):
      
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        joint_zero_pose = self.get_zero_pose_human()

        # extract triplane feature
        tri_feat = self.extract_tri_feature()
      
        # get Gaussian assets
        geo_feat = self.geo_net(tri_feat)
        mean_offset = self.mean_offset_net(geo_feat) # mean offset of Gaussians
        scale = self.scale_net(geo_feat) # scale of Gaussians
        mean_3d = mesh_neutral_pose + mean_offset # 大 pose
 
        # get pose-dependent Gaussian assets
        mean_offset_offset, scale_offset = self.forward_geo_network(tri_feat, smplx_param)
        scale, scale_refined = torch.exp(scale).repeat(1,3), torch.exp(scale+scale_offset).repeat(1,3)
        mean_combined_offset, mean_offset_offset = self.get_mean_offset_offset(smplx_param, mean_offset_offset)
        mean_3d_refined = mean_3d + mean_combined_offset # 大 pose

        # smplx facial expression offset
        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.expr_dirs).sum(2)
        mean_3d = mean_3d + smplx_expr_offset # 大 pose
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose
        
        # get nearest vertex
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(mean_3d[None,:,:], mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: smpl_x.vertex_num_upsampled
        nn_vertex_idxs = self.lr_idx_to_hr_idx(nn_vertex_idxs)
        mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
        nn_vertex_idxs[mask] = torch.arange(smpl_x.vertex_num_upsampled).cuda()[mask]

        # get transformation matrix of the nearest vertex and perform lbs
        transform_mat_joint = self.get_transform_mat_joint(transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_joint, nn_vertex_idxs)
        mean_3d = self.lbs(mean_3d, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param
        mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param


        # camera coordinate system -> world coordinate system
        if not is_world_coord:
            mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
            mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)

        if USE_LOCAL_RGB == True:
            rgb_base = self.rgb_net_global(tri_feat)  # global prediction
          
            rgb_offset_base = self.forward_rgb_network(tri_feat, smplx_param, cam_param, mean_3d_refined)

            body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body']) - 1, 3)
            jaw_pose = smplx_param['jaw_pose'].view(1, 3)
            leye_pose = smplx_param['leye_pose'].view(1, 3)
            reye_pose = smplx_param['reye_pose'].view(1, 3)
            lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']), 3)
            rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']), 3)
            pose_all = torch.cat((body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose), 0)
            pose6d = matrix_to_rotation_6d(axis_angle_to_matrix(pose_all)).view(1, -1)
            pose6d_all = pose6d.repeat(smpl_x.vertex_num_upsampled, 1)

            with torch.no_grad():
                normals_world = Meshes(
                    verts=mean_3d_refined[None],
                    faces=torch.as_tensor(smpl_x.face_upsampled, device=mean_3d_refined.device)[None]
                ).verts_normals_packed()
                is_cavity = self.is_cavity[:, None].float()
                normals_world = normals_world * (1 - is_cavity) - normals_world * is_cavity
            expr = smplx_param['expr']  # (expr_dim,)

            # compute ramp
            if cur_itr is None:
                ramp = 1.0
            else:
                s, L = cfg.rgb_offset_start_iter, cfg.rgb_offset_ramp_len
                if cur_itr < s:
                    ramp = 0.0
                else:
                    ramp = min(1.0, (cur_itr - s) / max(1, L))

            if ramp == 0.0:
                rgb_offset = torch.zeros_like(rgb_base)
            else:
               
                if 'frame_id' in smplx_param:
                    frame_id = smplx_param['frame_id']
                else:
                    frame_id = torch.tensor([float(getattr(cfg, 'ref_frame_id', 0))],
                                            device=mean_3d_refined.device)
                res_in_all = self._build_residual_input(tri_feat, mean_3d_refined, frame_id,
                                                        pose6d_all, normals_world, expr)

                rgb_residual = torch.zeros_like(rgb_offset_base)

                if self.is_face.any():
                    rgb_residual[self.is_face] = self.rgb_res_face(res_in_all[self.is_face])

                if self.is_lhand.any():
                    rgb_residual[self.is_lhand] = self.rgb_res_lhand(res_in_all[self.is_lhand])

                if self.is_rhand.any():
                    rgb_residual[self.is_rhand] = self.rgb_res_rhand(res_in_all[self.is_rhand])

                rgb_residual = rgb_residual * self.residual_scale
                rgb_offset = torch.tanh(rgb_offset_base + rgb_residual) * 0.5
                rgb_offset = ramp * rgb_offset

            rgb = (torch.tanh(rgb_base) + 1) / 2.0
            rgb_refined = (torch.tanh(rgb_base + rgb_offset) + 1) / 2.0

        else:
            rgb = self.rgb_net(tri_feat)  # rgb of Gaussians
            # forward to rgb network
            rgb_offset = self.forward_rgb_network(tri_feat, smplx_param, cam_param, mean_3d_refined)
            rgb, rgb_refined = (torch.tanh(rgb) + 1) / 2, (torch.tanh(rgb + rgb_offset) + 1) / 2  # normalize to [0,1]
            
        # Gaussians and offsets
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant rotation
        opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant opacity

 
        if self.visibility_active(apply_visibility, vis_itr):
            active_mask = self.vis_keep.float()[:,None]
            render_idx = torch.nonzero(self.vis_keep, as_tuple=False).squeeze(1)
        else:
            active_mask = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda()
            render_idx = None

        assets = {
                'mean_3d': mean_3d,
                'opacity': opacity,
                'scale': scale,
                'rotation': rotation,
                'rgb': rgb,
                'active_mask': active_mask,
                'render_idx': render_idx
                }
        assets_refined = {
                'mean_3d': mean_3d_refined,
                'opacity': opacity,
                'scale': scale_refined,
                'rotation': rotation,
                'rgb': rgb_refined,
                'active_mask': active_mask,
                'render_idx': render_idx
                }
        offsets = {
                'mean_offset': mean_offset,
                'mean_offset_offset': mean_offset_offset,
                'scale_offset': scale_offset,
                'rgb_offset': rgb_offset
                }
        return assets, assets_refined, offsets, mesh_neutral_pose
       
class GaussianRenderer(nn.Module):
    def __init__(self):
        super(GaussianRenderer, self).__init__()
    
    def forward(self, gaussian_assets, img_shape, cam_param, bg=torch.ones((3)).float().cuda()):
        # assets for the rendering
        mean_3d = gaussian_assets['mean_3d']
        opacity = gaussian_assets['opacity']
        scale = gaussian_assets['scale']
        rotation = gaussian_assets['rotation']
        rgb = gaussian_assets['rgb']

        # create rasterizer
        # permute view_matrix and proj_matrix following GaussianRasterizer's configuration following below links
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L55
        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3,:3]
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view_matrix, 
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()
        
        # rasterize visible Gaussians to image and obtain their radius (on screen). 
        render_img, radius, render_depthmap, render_mask = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scale,
            rotations=rotation,
            cov3D_precomp=None)
        
        return {'img': render_img,
                'depthmap': render_depthmap,
                'mask': render_mask,
                'mean_2d': mean_2d,
                'is_vis': radius > 0,
                'radius': radius}

class SMPLXParamDict(nn.Module):
    def __init__(self):
        super(SMPLXParamDict, self).__init__()

        self.register_buffer("_device_anchor", torch.empty(0))

    # initialize SMPL-X parameters of all frames
    def init(self, smplx_params):
        _smplx_params = {}
        for frame_idx in smplx_params.keys():
            _smplx_params[str(frame_idx)] = nn.ParameterDict({})
            for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose',
                               'rhand_pose', 'expr', 'trans']:
                if 'pose' in param_name:
                    _smplx_params[str(frame_idx)][param_name] = nn.Parameter(
                        matrix_to_rotation_6d(axis_angle_to_matrix(smplx_params[frame_idx][param_name].cuda())))
                else:
                    _smplx_params[str(frame_idx)][param_name] = nn.Parameter(smplx_params[frame_idx][param_name].cuda())

        self.smplx_params = nn.ParameterDict(_smplx_params)

    def get_optimizable_params(self):
        optimizable_params = []
        for frame_idx in self.smplx_params.keys():
          
            if int(frame_idx) >= cfg.gen_id_offset_base:
                continue
            for param_name in self.smplx_params[frame_idx].keys():
                optimizable_params.append({'params': [self.smplx_params[frame_idx][param_name]],
                                           'name': 'smplx_' + param_name + '_' + frame_idx, 'lr': cfg.smplx_param_lr})
        return optimizable_params

    def forward(self, frame_idxs):
        out = []
        for frame_idx in frame_idxs:
            frame_idx = str(int(frame_idx))
            smplx_param = {}
            for param_name in self.smplx_params[frame_idx].keys():
                if 'pose' in param_name:
                    smplx_param[param_name] = matrix_to_axis_angle(
                        rotation_6d_to_matrix(self.smplx_params[frame_idx][param_name]))
                else:
                    smplx_param[param_name] = self.smplx_params[frame_idx][param_name]
            dev = self._device_anchor.device
            smplx_param["frame_id"] = torch.tensor([int(frame_idx) % cfg.gen_id_offset_base], device=dev)

            out.append(smplx_param)
        return out
