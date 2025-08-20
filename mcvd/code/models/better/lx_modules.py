import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import numpy as np
import torch


class CrossAttention_Casual(nn.Module):
    
    """跨模态注意力模块（支持多变量权重分析）"""
    def __init__(self, dim=384, heads=6, dim_head=64, mask_ratio=0.7):
        super().__init__()
        self.heads = heads # 2
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.mask_ratio = mask_ratio
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)


    def forward(self, query, context):# q = era5  k,v = 卫星云图
        # 输入形状处理
        q = self.to_q(query)  # [B,16,H,W]
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 分解多头并重排维度
        B, C, H, W = q.shape # B,2*8,32,32
        # _, C2, _, _ = v.shape
        q = q.view(B, self.heads, C//self.heads, H* W).transpose(-2, -1)  # [B, heads, HW, dim]
        k = k.view(B, self.heads, C//self.heads, H* W)                    # [B, heads, dim, HW]
        v = v.view(B, self.heads, C//self.heads, H* W).transpose(-2, -1) # [B, heads, HW, dim]

        # 计算注意力权重
        # attention scores: [B, heads=2, H*W, H*W]
        attn = torch.matmul(q,k ) * self.scale
        attn = attn.softmax(dim=-1)
        
        B, heads, N, _ = attn.shape  # N = H*W
        attn_flat = attn.view(B * heads, -1)  # [B*heads, N*N]
        k_top = int(self.mask_ratio * attn_flat.shape[1])

        # top-k mask (per batch*head)
        _, indices = torch.topk(attn_flat, k_top, dim=1)
        binary_mask = torch.zeros_like(attn_flat)
        binary_mask.scatter_(1, indices, 1.0)
        binary_mask = binary_mask.view(B, heads, N, N)  # [B, heads, HW, HW]


        # 拆分参与反传（因果）与不参与反传（非因果）
        attn_causal = attn * binary_mask
        attn_noncausal = attn.detach() * (1 - binary_mask)

        # 合并为总 attention
        attn_final = attn_causal + attn_noncausal

        # --- Step 3: 注意力应用 ---
        out = torch.matmul(attn_final, v)  # [B, heads, HW, dim]
        out = out.transpose(-2, -1).contiguous().view(B, -1, H, W)


        # 用于可视化的平均注意力图
        attn_map = attn_final.mean(dim=1).mean(dim=1).view(B, H, W)

        return self.to_out(out), attn_map, attn_causal, attn_noncausal , binary_mask


class CrossAttention(nn.Module):
    
    """跨模态注意力模块（支持多变量权重分析）"""
    def __init__(self, dim=384, heads=6, dim_head=64):
        super().__init__()
        self.heads = heads # 2
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)


    def forward(self, query, context):# q = era5  k,v = 卫星云图
        # 输入形状处理
        q = self.to_q(query)  # [B,16,H,W]
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 分解多头并重排维度
        B, C, H, W = q.shape # B,2*8,32,32
        # _, C2, _, _ = v.shape
        q = q.view(B, self.heads, C//self.heads, H* W).transpose(-2, -1)  # [B, heads, HW, dim]
        k = k.view(B, self.heads, C//self.heads, H* W)                    # [B, heads, dim, HW]
        v = v.view(B, self.heads, C//self.heads, H* W).transpose(-2, -1) # [B, heads, HW, dim]

        # 计算注意力权重
        # attention scores: [B, heads=2, H*W, H*W]
        attn = torch.matmul(q,k ) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 注意力应用
        # attention output
        out = torch.matmul(attn,v)  # [B, heads, H*W, dim]
        out = out.transpose(-2, -1).contiguous().view(B, -1, H, W) # [B, inner_dim, H, W]
        
        attn_map = attn.mean(dim=1).mean(dim=1).view(B, H, W)
        return self.to_out(out), attn_map

class CausalMaskNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=384, mask_ratio=0.7):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 将空间压缩成 1x1，用于通道评分
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.mask_ratio = mask_ratio

    def forward(self, feat):
        """
        输入：
            feat: B x C x H x W
        返回：
            causal_feat: B x C x H x W（参与梯度）
            noncausal_feat: B x C x H x W（不参与梯度）
            binary_mask: B x 1 x H x W（可视化用）
        """
        B, C, H, W = feat.shape
        # 生成 soft mask: B x C x 1 x 1
        soft_mask = self.net(feat).squeeze(-1).squeeze(-1)  # -> B x C

        # Top-k 通道掩码
        k = int(self.mask_ratio * C)
        _, indices = torch.topk(soft_mask, k, dim=1)

        binary_mask = torch.zeros_like(soft_mask)
        binary_mask.scatter_(1, indices, 1.0)
        binary_mask = binary_mask.view(B, C, 1, 1)  # 用于广播到 H x 

        # 构造因果与非因果特征
        causal_feat = feat * binary_mask
        noncausal_feat = feat.detach() * (1 - binary_mask)

        return causal_feat, noncausal_feat, binary_mask,soft_mask


class CausalEffectMatrix(nn.Module):
    def __init__(self, era5_vars, in_channels=384, target_channels=5):
        super().__init__()
        self.era5_vars = era5_vars
        self.num_vars = len(era5_vars)
        
        # 因果效应矩阵A，初始化为可学习参数
        self.A = nn.Parameter(torch.zeros(self.num_vars, target_channels))
        
        # 用于计算DCE的网络
        self.dce_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, target_channels)
        )
        
        # 下采样目标到16x16
        self.target_downsample = nn.AdaptiveAvgPool2d((16, 16))
        self.target_channels=target_channels
        self.proj_layer = nn.Conv2d(in_channels, self.target_channels, kernel_size=1)
    def forward(self, var_feats, target, var_index, is_training=True):
        """
        计算因果效应并增强特征
        
        参数:
            var_feats: 当前变量的特征 [B, C, H, W]
            target: 未来卫星云图 [B, 5, 128, 128]
            var_index: 当前变量在era5_compositions中的索引
            is_training: 是否训练模式
        
        返回:
            enhanced_feat: 增强后的特征 [B, C, H, W]
            A_matrix: 因果效应矩阵 [num_vars, target_channels]
        """
        # 下采样目标到16x16减少计算量
        
        
        if is_training:
            # 计算直接因果效应(DCE)
            # 事实情况下的预测
            target_down = self.target_downsample(target)  # [B, 5, 16, 16]
            factual_pred = self.dce_net(var_feats)  # [B, target_channels=5]
            
            # 反事实干预 - 将特征置零
            cf_feats = torch.zeros_like(var_feats)
            counterfactual_pred = self.dce_net(cf_feats)  # [B, target_channels=5]
            
            # 计算DCE (事实与反事实的差异)
            target_flat = target_down.mean(dim=(2,3))  # [B, target_channels=5]
            dce = (factual_pred - counterfactual_pred) * target_flat  # [B, target_channels]
            dce = dce.mean(dim=0)  # [target_channels]
            
            # 更新因果效应矩阵中当前变量的行
            self.A.data[var_index] = 0.9 * self.A.data[var_index] + 0.1 * dce.detach()
        
        # 获取当前变量的因果效应向量
        var_effect = self.A[var_index]  # [target_channels]
        
        # 对特征进行因果增强
        B, C, H, W = var_feats.shape
        effect_weights = var_effect.view(1, -1, 1, 1)  # [ 1, target_channels=5, 1, 1]
        
        # 通过1x1卷积将特征映射到target_channels维度
        # feat_proj = nn.Conv2d(C, self.target_channels, kernel_size=1).to(var_feats.device)(var_feats) #   [B, 384,32,32] -> [B, target_channels=5,32,32]
        feat_proj = self.proj_layer(var_feats)
        # 应用因果增强
        enhanced_feat = var_feats + (feat_proj * effect_weights).sum(dim=1, keepdim=True)
        
        return enhanced_feat

class ConditionalMutualInfoLoss(nn.Module):
    def __init__(self, projection_dim=128, lambda_nc=0.5):
        super().__init__()
        self.lambda_nc = lambda_nc
        self.lambda_margin = 0.1
        # 投影头，将 Causal 和 Target 映射到同一维度空间
        self.causal_proj = nn.Sequential(
            nn.Conv2d(384, projection_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.noncausal_proj = nn.Sequential(
            nn.Conv2d(384, projection_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.target_proj = nn.Sequential(
            nn.Conv2d(5, projection_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, causal_feat, noncausal_feat, target):
        """
        causal_feat, noncausal_feat: B x C x H x W
        target: B x T x H' x W' -> 需要缩放或投影到特征空间
        """
        B = causal_feat.size(0)

        # 投影
        causal_vec = self.causal_proj(causal_feat).view(B, -1)       # B x D
        noncausal_vec = self.noncausal_proj(noncausal_feat).view(B, -1) # B x D (共享权重)
        target_vec = self.target_proj(target).view(B, -1)            # B x D

        # 计算余弦相似度：越接近 1 越相关
        sim_causal = F.cosine_similarity(causal_vec, target_vec, dim=1)  # B
        sim_noncausal = F.cosine_similarity(noncausal_vec, target_vec, dim=1)

        # 将 similarity 映射到 [0,1] 作为概率
        sim_causal = (sim_causal + 1) / 2
        sim_noncausal = (sim_noncausal + 1) / 2

        # 修复关键：防止 BCE 报错
        sim_causal = sim_causal.clamp(min=1e-7, max=1 - 1e-7)
        sim_noncausal = sim_noncausal.clamp(min=1e-7, max=1 - 1e-7)
        
        # 理想情况是 causal 越接近 1，noncausal 越接近 0
        loss_causal = F.binary_cross_entropy(sim_causal, torch.ones_like(sim_causal))
        loss_noncausal = F.binary_cross_entropy(sim_noncausal, torch.zeros_like(sim_noncausal))
        # 越接近说明 causal 和 noncausal 没分开
        margin = 0.3
        margin_loss = F.relu(sim_noncausal - sim_causal + margin).mean()

        # 条件互信息 loss（max causal info, suppress noncausal）
        cmi_loss = loss_causal + self.lambda_nc * loss_noncausal+self.lambda_margin * margin_loss

        return cmi_loss



class TemporalSelfAttention(nn.Module):
    def __init__(self, num_frames, embed_dim, downsample_size=32, num_heads=4):
        super().__init__()
        self.num_frames = num_frames  # 9 (cond + x concat后长度)
        self.embed_dim = embed_dim    # 32*32=1024 或 64*64=4096，建议不太大，视内存而定
        self.downsample_size = downsample_size

        # 下采样层，将128x128 -> downsample_size x downsample_size
        self.downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)  # 简单用4步长卷积减采样
        # 这里输入是单通道，可以后续扩展支持多通道

        # 线性层投影为Q,K,V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 多头注意力
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, T=9, H=128, W=128)
        """

        B, T, H, W = x.shape
        assert T == self.num_frames
        # 1. 下采样空间维度: 先 reshape合并T和B维，方便卷积
        x_ = x.view(B*T, 1, H, W)  # (B*T, 1, 128, 128)
        x_ = self.downsample(x_)   # (B*T, 1, downsample_size, downsample_size)
        x_ = x_.view(B, T, -1)     # (B, T, downsample_size*downsample_size)
        assert x_.shape[2] == self.embed_dim

        # 2. 计算 Q,K,V
        Q = self.q_proj(x_)  # (B, T, embed_dim)
        K = self.k_proj(x_)
        V = self.v_proj(x_)

        # 3. 多头拆分
        def split_heads(x):
            return x.view(B, T, self.num_heads, self.head_dim).transpose(1,2)  # (B, num_heads, T, head_dim)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # 4. 计算注意力权重，带 causal mask
        # Qh,Kh,Vh shape: (B, num_heads, T, head_dim)
        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)

        # causal mask，防止时间t看未来t+1及以后帧
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, num_heads, T, T)

        attn_out = torch.matmul(attn_probs, Vh)  # (B, num_heads, T, head_dim)

        # 5. 合并多头
        attn_out = attn_out.transpose(1,2).contiguous().view(B, T, self.embed_dim)  # (B, T, embed_dim)

        # 6. 输出线性层
        out = self.out_proj(attn_out)  # (B, T, embed_dim)

        # 7. 残差连接
        # out = out + x_  # (B, T, embed_dim)

        # 8. reshape回空间维度
        out = out.view(B, T, self.downsample_size, self.downsample_size)  # (B, T, H_ds, W_ds)

        # 如果需要，可以再上采样回原始分辨率，或者返回下采样结果
        out = F.interpolate(out.view(B*T, 1, self.downsample_size, self.downsample_size), size=(H, W), mode='bilinear', align_corners=False)
        out = out.view(B, T, H, W)  # (B, T, H, W)

        return out
    
class SPADE_NCSNpp(nn.Module):
  """NCSN++ model with SPADE normalization"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', get_sigmas(config))
    self.is3d = (config.model.arch in ["unetmore3d", "unetmorepseudo3d"])
    self.pseudo3d = (config.model.arch == "unetmorepseudo3d")
    if self.is3d:
      from . import layers3d

    self.channels = channels = config.data.channels
    self.num_frames = num_frames = config.data.num_frames
    self.num_frames_cond = num_frames_cond = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0)
    self.n_frames = num_frames

    self.nf = nf = config.model.ngf*self.num_frames if self.is3d else config.model.ngf # We must prevent problems by multiplying by num_frames
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = getattr(config.model, 'dropout', 0.0)
    resamp_with_conv = True
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = getattr(config.model, 'time_conditional', True)  # noise-conditional
    self.cond_emb = getattr(config.model, 'cond_emb', False)
    fir = True
    fir_kernel = [1, 3, 3, 1]
    self.skip_rescale = skip_rescale = True
    self.resblock_type = resblock_type = 'biggan'
    self.embedding_type = embedding_type = 'positional'
    init_scale = 0.0
    assert embedding_type in ['fourier', 'positional']

    self.spade_dim = spade_dim = getattr(config.model, "spade_dim", 128)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=16
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    temb_dim = None

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      temb_dim = nf * 4

      if self.cond_emb:
        modules.append(torch.nn.Embedding(num_embeddings=2, embedding_dim=nf // 2)) # makes it 8 times smaller (16 if ngf=32) since it should be small because there are only two possible values: 
        temb_dim += nf // 2

    if self.pseudo3d:
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.num_frames, act=self.act) # Activation here as per https://arxiv.org/abs/1809.04096
      conv1x1_cond = functools.partial(layers3d.ddpm_conv1x1_pseudo3d, n_frames=self.channels, act=self.act)
    elif self.is3d:
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.num_frames)
      conv1x1_cond = functools.partial(layers3d.ddpm_conv1x1_3d, n_frames=self.channels)
    else:
      conv3x3 = layerspp.conv3x3
      conv1x1 = conv1x1_cond = layerspp.conv1x1

    if self.is3d:
      AttnBlock = functools.partial(layers3d.AttnBlockpp3d,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale,
                                    n_head_channels=config.model.n_head_channels,
                                    n_frames=self.num_frames,
                                    act=None) # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
    else:
      AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale, n_head_channels=config.model.n_head_channels)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    ResnetBlockDDPM = layerspp.ResnetBlockDDPMppSPADE
    ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppSPADE

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=temb_dim,
                                      is3d=self.is3d,
                                      pseudo3d=self.pseudo3d,
                                      n_frames=self.num_frames,
                                      num_frames_cond=num_frames_cond,
                                      cond_ch=num_frames_cond*channels,
                                      spade_dim=spade_dim,
                                      act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=temb_dim,
                                      is3d=self.is3d,
                                      pseudo3d=self.pseudo3d,
                                      n_frames=self.num_frames,
                                      num_frames_cond=num_frames_cond,
                                      cond_ch=num_frames_cond*channels,
                                      spade_dim=spade_dim,
                                      act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    modules.append(conv3x3(channels*self.num_frames, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        hs_c.append(in_ch)

    # Middle Block
    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        in_ch_old = hs_c.pop()
        modules.append(ResnetBlock(in_ch=in_ch + in_ch_old,
                                     out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    modules.append(layerspp.get_act_norm(act=act, act_emb=act, norm='spade', ch=in_ch, is3d=self.is3d, n_frames=self.num_frames, num_frames_cond=num_frames_cond,
                                         cond_ch=num_frames_cond*channels, spade_dim=spade_dim, cond_conv=conv3x3, cond_conv1=conv1x1_cond))
    modules.append(conv3x3(in_ch, channels*self.num_frames, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond, cond=None, cond_mask=None):
    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0

    # if cond is not None:
    #   x = torch.cat([x, cond], dim=1) # B, (num_frames+num_frames_cond)*C, H, W

    if self.is3d: # B, N*C, H, W -> B, C*N, H, W : subtle but important difference!
      B, NC, H, W = x.shape
      CN = NC
      x = x.reshape(B, self.num_frames, self.channels, H, W).permute(0, 2, 1, 3, 4).reshape(B, CN, H, W)
      if cond is not None:
        B, NC, H, W = cond.shape
        CN = NC
        cond = cond.reshape(B, self.num_frames_cond, self.channels, H, W).permute(0, 2, 1, 3, 4).reshape(B, CN, H, W)

    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1
    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = layers.get_timestep_embedding(timesteps, self.nf)
    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb)) # b x k
      m_idx += 1
      if self.cond_emb:
        if cond_mask is None:
          cond_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.int32)
        temb = torch.cat([temb, modules[m_idx](cond_mask)], dim=1) # b x (k/8 + k)
        m_idx += 1
    else:
      temb = None

    # Downsampling block
    input_pyramid = None

    x = x.contiguous()
    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb, cond=cond)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1], cond=cond)
        else:
          h = modules[m_idx](hs[-1], temb, cond=cond)
        m_idx += 1
        hs.append(h)

    # Middle Block

    # ResBlock
    h = hs[-1]
    h = modules[m_idx](h, temb, cond=cond)
    m_idx += 1
    # AttnBlock
    h = modules[m_idx](h)
    m_idx += 1

    # ResBlock
    h = modules[m_idx](h, temb, cond=cond)
    m_idx += 1

    pyramid = None
    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1): 
        if self.is3d:
          # Get h and h_old
          B, CN, H, W = h.shape
          h = h.reshape(B, -1, self.num_frames, H, W)
          prev = hs.pop().reshape(B, -1, self.num_frames, H, W)
          # Concatenate
          h_comb = torch.cat([h, prev], dim=1) # B, C, N, H, W
          h_comb = h_comb.reshape(B, -1, H, W)
        else:
          prev = hs.pop()
          h_comb = torch.cat([h, prev], dim=1)
        h = modules[m_idx](h_comb, temb, cond=cond)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h, cond=cond)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb, cond=cond)
          m_idx += 1

    assert not hs
    # GroupNorm
    h = modules[m_idx](h, cond=cond)
    m_idx += 1

    # conv3x3_last
    h = modules[m_idx](h)
    m_idx += 1

    assert m_idx == len(modules)

    if self.is3d: # B, C*N, H, W -> B, N*C, H, W subtle but important difference!
      B, CN, H, W = h.shape
      NC = CN
      h = h.reshape(B, self.channels, self.num_frames, H, W).permute(0, 2, 1, 3, 4).reshape(B, NC, H, W)

    return h
def disabled_train(self):
    
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
class Transformer_v2(nn.Module):
  
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x
class PreNormattention(nn.Module):
  def __init__(self, dim, fn):
      super().__init__()
      self.norm = nn.LayerNorm(dim)
      self.fn = fn
  def forward(self, x, **kwargs):
      return self.fn(self.norm(x), **kwargs) + x
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class EmbedFC(nn.Module):
    
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)
class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 50,4,128,128
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class GradientEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filter 卷积核 (1个通道，提取x和y方向)
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                        [ 0,  0,  0],
                                        [ 1,  2,  1]], dtype=torch.float32)
        self.register_buffer('weight_x', sobel_kernel_x[None, None, :, :])  # (1,1,3,3)
        self.register_buffer('weight_y', sobel_kernel_y[None, None, :, :])

    def forward(self, x):
        """
        x: Tensor of shape (B, T, H, W), single-channel satellite frames
        Returns: Tensor of shape (B, T, H, W), gradient magnitude
        """
        B, T, H, W = x.shape
        x = x.reshape(B * T, 1, H, W)
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # avoid NaN
        grad_mag = grad_mag.view(B, T, H, W)
        return grad_mag
    