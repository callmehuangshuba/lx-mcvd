import torch

from functools import partial
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
import torch.nn as nn

def anneal_dsm_score_estimation(scorenet,scorenet_teacher, x, labels=None, loss_type='a', hook=None, cond=None, cond_mask=None,era5_cond=None, era5_cond_3d=None,gamma=False, L1=False, all_frames=False):

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    version = getattr(net, 'version', 'SMLD').upper()
    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    if all_frames:
        x = torch.cat([x, cond], dim=1)
        cond = None

    # z, perturbed_x
    if version == "SMLD":
        sigmas = net.sigmas
        if labels is None:
            labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        z = torch.randn_like(x)
        perturbed_x = x + used_sigmas * z
    elif version == "DDPM" or version == "DDIM" or version == "FPNDM":
        alphas = net.alphas
        if labels is None:
            labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
        used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        if gamma:
            used_k = net.k_cum[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            used_theta = net.theta_t[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            z = Gamma(used_k, 1 / used_theta).sample()
            z = (z - used_k*used_theta) / (1 - used_alphas).sqrt()
        else:
            z = torch.randn_like(x)
        perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z # b,5,128,128

    scorenet_teacher = partial(scorenet_teacher, cond=cond,era5_cond=era5_cond,era5_cond_3d=era5_cond_3d)
    scorenet = partial(scorenet, cond=cond,target=x)
    # (b,5,128,128) (b,384,16,16) (b,384,32,32)
    h_student, feat_16_stu,feat_32_stu,encoder_16_stu,encoder_32_stu,causal_loss = scorenet(
       perturbed_x, labels, cond_mask=cond_mask
    )

    # 前向：教师模型（含 ERA5）
    with torch.no_grad():
        # (b,5,128,128) (b,384,16,16) (b,384,32,32)
        h_teacher, feat_16_tea, feat_32_tea,encoder_16_tea,encoder_32_tea = scorenet_teacher(
           perturbed_x, labels, cond_mask=cond_mask
        )
    # Loss
    if L1:
        def pow_(x):
            return x.abs()
    else:
        def pow_(x):
            return 1 / 2. * x.square()
    # 和GT loss
    loss_score = pow_((z - h_student).reshape(len(x), -1)).sum(dim=-1).mean()  # pow_((z - h_student).reshape(len(x), -1)).sum(dim=-1).shape = (36)

    # Loss 2：输出蒸馏
    loss_distill = pow_((h_student- h_teacher.detach()).reshape(len(x), -1)).sum(dim=-1).mean() 

    # Loss 3：特征蒸馏
    loss_feat_32=0.0
    loss_feat_16=0.0
    loss_encoder_32=0.0
    loss_encoder_16=0.0
    for var in ['sst','msl','z','r']:
        feat_teacher = feat_32_tea[var].detach()  # ERA5
        feat_student = feat_32_stu[var]    
        loss_feat_32+=F.mse_loss(feat_teacher, feat_student)
    for var in ['sst','msl','z','r']:
        feat_teacher = feat_16_tea[var].detach()  # ERA5
        feat_student = feat_16_stu[var]        # cond
        loss_feat_16+=F.mse_loss(feat_teacher, feat_student)
    for var in ['sst','msl','z','r']:
        encoder_teacher = encoder_32_tea[var].detach()  # ERA5
        encoder_student = encoder_32_stu[var]    
        loss_encoder_32+=F.mse_loss(encoder_teacher, encoder_student)
    for var in ['sst','msl','z','r']:
        encoder_teacher = encoder_16_tea[var].detach()  # ERA5
        encoder_student = encoder_16_stu[var]    
        loss_encoder_16+=F.mse_loss(encoder_teacher, encoder_student)

    loss_feat_encoder = loss_feat_32 + loss_feat_16 + loss_encoder_32 + loss_encoder_16

    epsilon = 1e-6
    max_lambda = 100
    min_lambda = 0.01
    lambda_ = loss_score.item() /4* (abs(causal_loss.item()) + epsilon)
    lambda_adaptive = max(min_lambda, min(lambda_, max_lambda)) 


    loss = loss_score + loss_distill + 10*loss_feat_encoder+ lambda_adaptive*causal_loss
    if hook is not None:# None
        hook(loss, labels)

    return loss

def attention_map(f):  # f: (B, C, H, W)
    return F.normalize(f.pow(2).mean(1).view(f.size(0), -1))