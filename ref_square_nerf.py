import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_blender import load_blender_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def run_network_ms(inputs, fn, netchunk=1024*64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    outputs_flat = batchify(fn, netchunk)(inputs_flat)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs



def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., c2w_staticcam=None,
                  **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    # provide ray directions as input
    viewdirs = rays_d
    if c2w_staticcam is not None:
        # special case to visualize effect of viewdirs
        rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'rgb_vd', 'rgb_vd_noweight', 'rgb_vi', 'offset_raw', 'offset']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, rgb_vd, rgb_vd_noweight, rgb_vi, offset_raw, offset, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        if i==0:
            print(rgb.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            rgb_vi = rgb_vi.cpu().numpy()
            rgb8_vi = to8b(rgb_vi)
            filename = os.path.join(savedir, '{:03d}_vi.png'.format(i))
            imageio.imwrite(filename, rgb8_vi)
            rgb_vd = rgb_vd.cpu().numpy()
            rgb8_vd = to8b(rgb_vd)
            filename = os.path.join(savedir, '{:03d}_vd.png'.format(i))
            imageio.imwrite(filename, rgb8_vd)
            rgb_vd_noweight = rgb_vd_noweight.cpu().numpy()
            rgb8_vd_noweight = to8b(rgb_vd_noweight)
            filename = os.path.join(savedir, '{:03d}_vd_noweight.png'.format(i))
            imageio.imwrite(filename, rgb8_vd_noweight)



    rgbs = np.stack(rgbs, 0)

    return rgbs

def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    embed_fn_offset, input_ch_offset = get_embedder(10, args.i_embed)

    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        
    skips = [4]
    model = Ref_square_NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, skips=skips,
                 input_ch_views=input_ch_views, feature_dim=args.feature_dim).to(device)
    grad_vars = list(model.parameters())

    model_fine = Ref_square_NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                        input_ch=input_ch, skips=skips,
                        input_ch_views=input_ch_views, feature_dim=args.feature_dim).to(device)
    grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    network_query_ms = lambda inputs, network_fn : run_network_ms(inputs, network_fn,
                                                                netchunk=args.netchunk)
    
    offset = Offset(D=8, W=args.netwidth, input_ch=input_ch_offset, input_ch_views=input_ch_views,
                 skips=skips).to(device)
    grad_vars += list(offset.parameters())
    
    offset_fine = Offset(D=8, W=args.netwidth, input_ch=input_ch_offset, input_ch_views=input_ch_views,
                 skips=skips).to(device)
    grad_vars += list(offset_fine.parameters())
    

    decoder = Decoder(D=2, W=24).to(device)
    grad_vars += list(decoder.parameters())
    gate = Gate(D=2, W=24).to(device)
    grad_vars += list(gate.parameters())
    

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        offset.load_state_dict(ckpt['offset'])
        offset_fine.load_state_dict(ckpt['offset_fine'])
        decoder.load_state_dict(ckpt['decoder'])
        gate.load_state_dict(ckpt['gate'])
    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'network_query_ms' : network_query_ms,
        'perturb' : args.perturb,
        'N_importance_vi' : args.N_importance_vi,
        'N_importance_vd' : args.N_importance_vd,
        'N_importance_glass' : args.N_importance_glass,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'offset' : offset,
        'offset_fine' : offset_fine,
        'decoder' : decoder,
        'gate' : gate,
        'feature_dim' : args.feature_dim,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.


    optimizer.param_groups[0]['capturable'] = True
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer



def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std


    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)


    return rgb_map, disp_map, acc_map, weights, depth_map

def glass_raw2outputs(raw, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    xyz = raw[...,:3]  # [N_rays, N_samples, 3]

    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    offset_cumsum = weights[...,None] * xyz
    offset = torch.cumsum(offset_cumsum, dim=1)

    return offset, weights, xyz



def raw2features(raw, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    weights = []
    feature_maps = []
    alpha = raw[:,:,0]
    alpha = raw2alpha(alpha, dists)
    weights = (alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1])
    features = raw[...,1:]
    feature_map = torch.sum(weights[..., None] * features, -2)

    return weights, feature_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                network_query_ms,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance_vi=0,
                N_importance_vd=0,
                N_importance_glass=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                offset=None,
                offset_fine=None,
                decoder=None,
                gate=None,
                feature_dim=None,
                ):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples) 
    
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    
    glass_raw = network_query_fn(pts, viewdirs, offset)
    offset, weights_glass, offset_raw_0 = glass_raw2outputs(glass_raw, z_vals, rays_d)
    
    pts_added = pts + offset


#     raw = run_network(pts)
    raw = network_query_fn(pts_added, viewdirs, network_fn)
    raw_nerf, raw_vd = torch.split(raw, [4,1+feature_dim], dim=2)
    rgb_map_vi, disp_map, acc_map, weights_vi, depth_map = raw2outputs(raw_nerf, z_vals, rays_d, raw_noise_std, white_bkgd)
    weights_pts, feature_maps = raw2features(raw_vd, z_vals, rays_d)
    decode = network_query_ms(feature_maps, decoder)
    weights_sum = torch.sum(weights_pts, dim=1) 
    weights = torch.relu(network_query_ms(feature_maps, gate))
    rgb_0 = rgb_map_vi + (decode * weights)


    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

    z_samples_glass = sample_pdf(z_vals_mid, weights_glass[...,1:-1], N_importance_vi, det=(perturb==0.))
    z_samples_glass = z_samples_glass.detach()
    z_samples_vi = sample_pdf(z_vals_mid, weights_vi[...,1:-1], N_importance_vd, det=(perturb==0.))
    z_samples_vi = z_samples_vi.detach()
    z_samples_vd = sample_pdf(z_vals_mid, weights_pts[...,1:-1], N_importance_glass, det=(perturb==0.))
    z_samples_vd = z_samples_vd.detach()

    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples_glass,z_samples_vi, z_samples_vd], -1), -1)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    
    glass_raw = network_query_fn(pts, viewdirs, offset_fine)
    offset, weights_glass, offset_raw = glass_raw2outputs(glass_raw, z_vals, rays_d)

    pts_added = pts + offset

    raw = network_query_fn(pts_added, viewdirs, network_fine)
    raw_nerf, raw_vd = torch.split(raw, [4,1+feature_dim], dim=2)

    rgb_map_vi, disp_map, acc_map, weights_vi, depth_map = raw2outputs(raw_nerf, z_vals, rays_d, raw_noise_std, white_bkgd)
    
    weights_pts, feature_maps = raw2features(raw_vd, z_vals, rays_d)
    decode = network_query_ms(feature_maps, decoder)
    weights_sum = torch.sum(weights_pts, dim=1)
    weights = torch.relu(network_query_ms(feature_maps, gate))
    rgb_map_vd = decode * weights
    rgb_map = rgb_map_vi + rgb_map_vd

    ret = {'rgb_map' : rgb_map, 'rgb_vd' : rgb_map_vd, 'rgb_vd_noweight' : decode, 'rgb_vi' : rgb_map_vi, 'offset_raw' : offset_raw, 'offset' : offset}
    if retraw:
        ret['raw'] = raw
    ret['rgb0'] = rgb_0
    ret['offset_raw0'] = offset_raw_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--feature_dim", type=int, default=64, 
                        help='view-dependent feature vector dimension')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*8, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*16, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance_vi", type=int, default=0,
                        help='number of additional fine samples per ray : View-independent')
    parser.add_argument("--N_importance_vd", type=int, default=0,
                        help='number of additional fine samples per ray : View-Dependent')
    parser.add_argument("--N_importance_glass", type=int, default=0,
                        help='number of additional fine samples per ray : Glass estimation')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxelscd ')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=200000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=100, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None

    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 1.5
        far = 10.5

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    #args.render_only = True
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path 
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    print("NEAR : ", near, far)
    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)


        #####  Core optimization loop  #####
        rgb, rgb_vd, rgb_vd_noweight, rgb_vi, offset_raw, offset, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        

        
        optimizer.zero_grad()
        rgb_clip_bool = rgb > 0.9999
        target_clip_bool = target_s > 0.9999
        train_bool = torch.logical_and(rgb_clip_bool, target_clip_bool)
        new_target_s = torch.where(train_bool, rgb, target_s)
        img_loss = img2mse(rgb, new_target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
        
        if 'rgb0' in extras:
            rgb0 = extras['rgb0']
            rgb0_clip_bool = rgb0 > 0.9999
            train_bool = torch.logical_and(rgb0_clip_bool, target_clip_bool)
            new_target_s0 = torch.where(train_bool, rgb0, target_s)
            img_loss0 = img2mse(rgb0, new_target_s0)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        
        img_loss_vi = img2mse(rgb_vi, target_s)
        psnr_vi = mse2psnr(img_loss_vi)


        loss_offset_raw = torch.sqrt(torch.sum(offset_raw ** 2))
        loss_offset = torch.sqrt(torch.sum(offset ** 2))
        loss = loss + loss_offset_raw * 0.00001

        offset_raw0 = extras['offset_raw0']
        loss_offset_raw_0 = torch.sqrt(torch.sum(offset_raw0 ** 2))
        loss = loss + loss_offset_raw_0 * 0.00001
                
        loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time()-time0

        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'offset': render_kwargs_train['offset'].state_dict(),
                'offset_fine': render_kwargs_train['offset_fine'].state_dict(),
                'decoder': render_kwargs_train['decoder'].state_dict(),
                'gate': render_kwargs_train['gate'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} PSNR_VI: {psnr_vi.item()} loss_offset_raw: {loss_offset_raw.item()}")

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
