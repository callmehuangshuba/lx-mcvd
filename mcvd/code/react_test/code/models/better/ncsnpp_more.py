# adding conditional group-norm as per https://arxiv.org/pdf/2105.05233.pdf

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
from typing import Sequence, Union, Dict, Any, Optional, Callable
from . import layers, layerspp
from .. import get_sigmas
import torch.nn as nn
import functools
import torch
from torch import einsum
import numpy as np
from .taming.autoencoder_kl import AutoencoderKL
import os
import warnings
from omegaconf import OmegaConf
from einops import rearrange
from .taming.distributions import DiagonalGaussianDistribution
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppGN
get_act = layers.get_act
default_initializer = layers.default_init
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from timm.models.swin_transformer import SwinTransformerBlock

class NCSNpp(nn.Module):
  """NCSN++ model"""

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
    self.n_frames = num_frames + num_frames_cond

    self.nf = nf = config.model.ngf*self.n_frames if self.is3d else config.model.ngf # We must prevent problems by multiplying by n_frames
    self.numf = numf = config.model.ngf*self.num_frames if self.is3d else config.model.ngf # We must prevent problems by multiplying by n_frames
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
    self.era5_compositions=['sst','msl','z','r']
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
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.n_frames, act=self.act) # Activation here as per https://arxiv.org/abs/1809.04096
      conv3x3_last = functools.partial(layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.num_frames, act=self.act)
    elif self.is3d:
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.n_frames)
      conv3x3_last = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.num_frames)
    else:
      conv3x3 = layerspp.conv3x3
      conv3x3_last = layerspp.conv3x3

    if self.is3d:
      AttnBlockDown = functools.partial(layers3d.AttnBlockpp3d,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        n_head_channels=config.model.n_head_channels,
                                        n_frames = self.n_frames,
                                        act=None) # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
      AttnBlockUp = functools.partial(layers3d.AttnBlockpp3d,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      n_head_channels=config.model.n_head_channels,
                                      n_frames = self.num_frames,
                                      act=None) # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
    else:
      AttnBlockDown = AttnBlockUp = functools.partial(layerspp.AttnBlockpp,
                                                      init_scale=init_scale,
                                                      skip_rescale=skip_rescale, n_head_channels=config.model.n_head_channels)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if resblock_type == 'ddpm':
      ResnetBlockDown = functools.partial(ResnetBlockDDPM,
                                          act=act,
                                          dropout=dropout,
                                          init_scale=init_scale,
                                          skip_rescale=skip_rescale,
                                          temb_dim=temb_dim,
                                          is3d = self.is3d,
                                          n_frames = self.n_frames,
                                          pseudo3d = self.pseudo3d,
                                          act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096
      ResnetBlockUp = functools.partial(ResnetBlockDDPM,
                                        act=act,
                                        dropout=dropout,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        temb_dim=temb_dim,
                                        is3d = self.is3d,
                                        n_frames = self.num_frames,
                                        pseudo3d = self.pseudo3d,
                                        act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    elif resblock_type == 'biggan':
      ResnetBlockDown = functools.partial(ResnetBlockBigGAN,
                                          act=act,
                                          dropout=dropout,
                                          fir=fir,
                                          fir_kernel=fir_kernel,
                                          init_scale=init_scale,
                                          skip_rescale=skip_rescale,
                                          temb_dim=temb_dim,
                                          is3d = self.is3d,
                                          n_frames = self.n_frames,
                                          pseudo3d = self.pseudo3d,
                                          act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096
      ResnetBlockUp = functools.partial(ResnetBlockBigGAN,
                                        act=act,
                                        dropout=dropout,
                                        fir=fir,
                                        fir_kernel=fir_kernel,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        temb_dim=temb_dim,
                                        is3d = self.is3d,
                                        n_frames = self.num_frames,
                                        pseudo3d = self.pseudo3d,
                                        act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    modules.append(conv3x3(channels*self.n_frames, nf))
    # modules.append(conv3x3(self.num_frames, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlockDown(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlockDown(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlockDown(down=True, in_ch=in_ch))

        hs_c.append(in_ch)

    # Middle Block
    in_ch = hs_c[-1]
    modules.append(ResnetBlockDown(in_ch=in_ch))
    modules.append(AttnBlockDown(channels=in_ch))
    if self.is3d:
      # Converter
      modules.append(layerspp.conv1x1(self.n_frames, self.num_frames))
      in_ch =  int(in_ch * self.num_frames / self.n_frames)
    modules.append(ResnetBlockUp(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = numf * ch_mult[i_level]
        if self.is3d: # 1x1 self.num_frames + self.num_frames_cond -> self.num_frames
          modules.append(layerspp.conv1x1(self.n_frames, self.num_frames))
          in_ch_old = int(hs_c.pop() * self.num_frames / self.n_frames)
        else:
          in_ch_old = hs_c.pop()
        modules.append(ResnetBlockUp(in_ch=in_ch + in_ch_old,
                                     out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlockUp(channels=in_ch))

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlockUp(in_ch=in_ch, up=True))

    assert not hs_c

    modules.append(layerspp.get_act_norm(act=act, act_emb=act, norm='group', ch=in_ch, is3d=self.is3d, n_frames=self.num_frames))
    modules.append(conv3x3_last(in_ch, channels*self.num_frames, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)
    self.concat_channels = concat_channels = 8
    ### depth embedding
    if 'sst' in self.era5_compositions: # (b f) c h w
      self.sst_embedding1 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2, 1, 1), stride=(1, 1, 1)),                   # -> B, 384, 1, 32, 32
        nn.SiLU()
      )
      self.sst_embedding2 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1)),                   # -> B, 384, 1, 16, 16
        nn.SiLU()
      )
    if 'msl' in self.era5_compositions:# (b f) c h w
      self.msl_embedding1 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2, 1, 1), stride=(1, 1, 1)),                   # -> B, 384, 1, 32, 32
        nn.SiLU()
      )
      self.msl_embedding2 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1)),                   # -> B, 384, 1, 16, 16
        nn.SiLU()
      )
    if 'z' in self.era5_compositions:# (b f) c h w
      self.z_embedding1 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2, 1, 1), stride=(1, 1, 1)),                   # -> B, 384, 1, 32, 32
        nn.SiLU()
      )
      self.z_embedding2 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1)),                    # -> B, 384, 1, 16, 16
        nn.SiLU()
      )
    if 'r' in self.era5_compositions:# (b f) c h w
      self.r_embedding1 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2, 1, 1), stride=(1, 1, 1)),                   # -> B, 384, 1, 32, 32
        nn.SiLU()
      )
      self.r_embedding2 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1)),                  # -> B, 384, 1, 16, 16
        nn.SiLU()
      )
    if 'vo' in self.era5_compositions:
      self.vo_embedding1 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2, 1, 1), stride=(1, 1, 1)),                   # -> B, 384, 1, 32, 32
        nn.SiLU()
      )
      self.vo_embedding2 = nn.Sequential(
          # 输入: (B, 1, 4, 128, 128)   (B , C, T, H, W）
        nn.Conv3d(1, 96, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # -> B, 96, 3, 64, 64
        nn.SiLU(),
        nn.Conv3d(96, 192, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # -> B, 192, 2, 32, 32
        nn.SiLU(),
        nn.Conv3d(192, 384, kernel_size=(2,3,3), stride=(2,2,2), padding=(0,1,1)),                   # -> B, 384, 1, 16, 16
        nn.SiLU()
      )
    self.cross_attns16 = nn.ModuleDict({
        var: CrossAttention(dim=384)
        for var in self.era5_compositions
    })
    self.attention_maps16 = {}
    self.cross_attns32 = nn.ModuleDict({
        var: CrossAttention(dim=384)
        for var in self.era5_compositions
    })
    self.attention_maps32 = {}
    self.cross_attns64 = nn.ModuleDict({
        var: CrossAttention(dim=384)
        for var in self.era5_compositions
    })
    self.attention_map64 = {}


    self.var_causal16 = nn.ModuleDict({
        var:  CausalMaskNet(in_channels=384, mask_ratio=0.7)
        for var in self.era5_compositions
    })
    self.var_causal32 = nn.ModuleDict({
        var:  CausalMaskNet(in_channels=384, mask_ratio=0.7)
        for var in self.era5_compositions
    })
    self.var_causal64 = nn.ModuleDict({
        var:  CausalMaskNet(in_channels=384, mask_ratio=0.7)
        for var in self.era5_compositions
    })
    self.causal_effect = CausalEffectMatrix(era5_vars=self.era5_compositions, in_channels=384)
    self.var_weight_params32 = nn.Parameter(torch.ones(len(self.era5_compositions))) 
    self.var_weight_params16 = nn.Parameter(torch.ones(len(self.era5_compositions))) 
    self.cmi_criterion = ConditionalMutualInfoLoss()
    self.gamma  = nn.Parameter(torch.tensor(0.0))
    self.temporalAttention = TemporalSelfAttention(num_frames=9, embed_dim=32*32, downsample_size=32, num_heads=4)
  def forward(self, x, time_cond, cond=None, cond_mask=None,data_2d=None,data_3d=None,target=None,p_i=None):

    batch,fc,h,w = data_2d.shape
    sst = data_2d[:,0:4,:,:].unsqueeze(2)
    sst = torch.nan_to_num(sst, nan=0.0)
    msl = data_2d[:,4:,:,:].unsqueeze(2) # 50,4,1,128,128
    z = data_3d[:,0:4,0,:,:].unsqueeze(2) # 50,4,4,128,128    0=850pa
    r = data_3d[:,4:,-1,:,:].unsqueeze(2) # 50,4,4,128,128   -1=200pa
    vo=None
    self.alphas={}
    self.betas ={}
    self.casual_loss=0
    if sst is not None:
      ### DropPath mask
      sst = rearrange(sst, 'b f c h w -> b c f h w') # b ,4 ,1，128，128
      sst_32 = self.sst_embedding1(sst) # b,384,32,32
      sst_16 = self.sst_embedding2(sst) # b,384,16,16
      sst_32 = sst_32.squeeze(2) # B, 384, 32, 32
      sst_16 = sst_16.squeeze(2) # B, 384, 16, 16
    if msl is not None:
      ### DropPath mask
      msl = rearrange(msl, 'b f c h w -> b c f h w') # b ,4 ,1，128，128
      msl_32 = self.msl_embedding1(msl) # b,384,32,32
      msl_16 = self.msl_embedding2(msl) # b,384,16,16
      msl_32 = msl_32.squeeze(2) # B, 384, 32, 32
      msl_16 = msl_16.squeeze(2) # B, 384, 16, 16
    if z is not None:
      ### DropPath mask
      z = rearrange(z, 'b f c h w -> b c f h w') # b ,4 ,1，128，128
      z_32 = self.z_embedding1(z) # b,384,32,32
      z_16 = self.z_embedding2(z) # b,384,16,16
      z_32 = z_32.squeeze(2) # B, 384, 32, 32
      z_16 = z_16.squeeze(2) # B, 384, 16, 16
    if r is not None:
      ### DropPath mask
      r = rearrange(r, 'b f c h w -> b c f h w') # b ,4 ,1，128，128
      r_32 = self.r_embedding1(r) # b,384,32,32
      r_16 = self.r_embedding2(r) # b,384,16,16
      r_32 = r_32.squeeze(2) # B, 384, 32, 32
      r_16 = r_16.squeeze(2) # B, 384, 16, 16
    if vo is not None:
      ### DropPath mask
      vo = rearrange(vo, 'b f c h w -> b c f h w') # b ,4 ,1，128，128
      vo_32 = self.vo_embedding1(vo) # b,384,32,32
      vo_16 = self.vo_embedding2(vo) # b,384,16,16
      vo_32 = vo_32.squeeze(2) # B, 384, 32, 32
      vo_16 = vo_16.squeeze(2) # B, 384, 16, 16
    # 编码各变量
    encoded32 = {}
    encoded32['sst'] = sst_32
    encoded32['msl'] = msl_32
    encoded32['z'] = z_32
    encoded32['r'] = r_32

    encoded16 = {}
    encoded16['sst'] = sst_16
    encoded16['msl'] = msl_16
    encoded16['z'] = z_16
    encoded16['r'] = r_16
    # encoded['vo'] = vo


    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0

    if cond is not None:
      x = torch.cat([cond,x], dim=1) # B, (num_frames+num_frames_cond)*C, H, W

      out = self.temporalAttention(x)
      # x = x + 0.2 * out
      x = x + self.gamma * out
    if self.is3d: # B, N*C, H, W -> B, C*N, H, W : subtle but important difference!
      B, NC, H, W = x.shape
      CN = NC
      x = x.reshape(B, self.n_frames, self.channels, H, W).permute(0, 2, 1, 3, 4).reshape(B, CN, H, W)

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
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1

        hs.append(h)

    # Middle Block

    # ResBlock
    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    # AttnBlock
    h = modules[m_idx](h)
    m_idx += 1

    # Converter
    if self.is3d: # downscale time-dim, we decided to do it here, but could also have been done earlier or at the end
      # B, C*(num_frames+num_cond), H, W -> B, C, (num_frames+num_cond), H, W -----conv1x1-----> B, C, num_frames, H, W -> B, C*num_frames, H, W
      B, CN, H, W = h.shape
      h = h.reshape(-1, self.n_frames, H, W)
      h = modules[m_idx](h)
      m_idx += 1
      h = h.reshape(B, -1, H, W)

    # ResBlock
    h = modules[m_idx](h, temb)  # B 384 16 16
    m_idx += 1


    B,C,H,W = h.shape
    fused_feat = torch.zeros_like(h)
        # 计算 softmax 权重（确保所有权重和为 1）
    weights = torch.softmax(self.var_weight_params16, dim=0)  # [num_vars]
    for i, var in enumerate(self.era5_compositions):
      B,C,H,W=encoded16[var].shape
      var_feat = encoded16[var] # var_feat   =  [B,384,16,16]
      attn_feat, attn_weights = self.cross_attns16[var](query=var_feat,context=h) # [B,384,16,16] [B,H,W]
      fused_feat += attn_feat*weights[i]
      self.attention_maps16[var] = attn_weights.detach()  # 平均多头权重  36 H W
    h = h+fused_feat
    # visualize_attention16(attention_maps=self.attention_maps16)
    pyramid = None
    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1): 
        if self.is3d:
          # Get h and h_old
          B, CN, H, W = h.shape
          h = h.reshape(B, -1, self.num_frames, H, W)
          prev = hs.pop().reshape(-1, self.n_frames, H, W)
          # B, C*Nhs, H, W -> B, C, Nhs, H, W -----conv1x1-----> B, C, Nh, H, W -> B, C*Nh, H, W
          prev = modules[m_idx](prev).reshape(B, -1, self.num_frames, H, W)
          m_idx += 1
          # Concatenate
          h_comb = torch.cat([h, prev], dim=1) # B, C, N, H, W
          h_comb = h_comb.reshape(B, -1, H, W)
        else:
          prev = hs.pop()
          h_comb = torch.cat([h, prev], dim=1)
        h = modules[m_idx](h_comb, temb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

      if i_level == 3:
        B,C,H,W = h.shape
        fused_feat = torch.zeros_like(h)
            # 计算 softmax 权重（确保所有权重和为 1）
        weights = torch.softmax(self.var_weight_params32, dim=0)  # [num_vars]
        # visualize_data(data_2d=data_2d,data_3d=data_3d,cond = cond,save_dir="/opt/data/private/mcvd-pytorch/lx_2d+3d_era5_attn_visual/era5_cond_visual_6.3.png")
        for i, var in enumerate(self.era5_compositions):
          B,C,H,W=encoded32[var].shape
          var_feat = encoded32[var] # var_feat   =  [B,384,16,16]
          attn_feat, attn_weights= self.cross_attns32[var](query=var_feat,context=h) # [B,384,16,16] [B,H,W]
          causal_feat, noncausal_feat, binary_mask,soft_mask ,prune_ratio  = self.var_causal32[var](attn_feat)  # att_feat：因果掩码处理后的特征   mask_prob：soft 掩码（概率型 α）哪些区域可能是因果的”，也可用于可视化 
                                                                        # hard_mask：离散因果掩码（二值型）控制 feature * hard_mask 和 feature * (1 - hard_mask).detach() 的权重分配；
          if target is not None:
            is_training =True
          else:
            is_training = False
          enhance_causal_feat = self.causal_effect(causal_feat,target=target,var_index=self.era5_compositions.index(var),is_training=is_training)                     
          self.alphas[var] = enhance_causal_feat    # alpha：特征因果编码
          self.betas[var]  = noncausal_feat # 非因果部分
          if target is not None:
            cmi_loss = self.cmi_criterion(enhance_causal_feat, noncausal_feat, target,soft_mask,prune_ratio)
            self.casual_loss+=cmi_loss
          fused_feat += (enhance_causal_feat+noncausal_feat)*weights[i]
          self.attention_maps32[var] = attn_weights.detach()  # 平均多头权重  36 H W
        h = h+fused_feat
        # visualize_attention128(attention_maps=self.attention_maps32)
    assert not hs
    # GroupNorm
    h = modules[m_idx](h)
    m_idx += 1

    # conv3x3_last
    h = modules[m_idx](h)
    m_idx += 1

    assert m_idx == len(modules)

    if getattr(self.config.model, 'output_all_frames', False) and cond is not None: # we only keep the non-cond images (but we could use them eventually)
      _, h = torch.split(h, [self.num_frames_cond*self.config.data.channels,self.num_frames*self.config.data.channels], dim=1)

    if self.is3d: # B, C*N, H, W -> B, N*C, H, W subtle but important difference!
      B, CN, H, W = h.shape
      NC = CN
      h = h.reshape(B, self.channels, self.num_frames, H, W).permute(0, 2, 1, 3, 4).reshape(B, NC, H, W)
    if target is not None:
      return h,self.casual_loss
    else:
      return h



class UNetMore_DDPM(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.version = getattr(config.model, 'version', 'DDPM').upper()
    assert self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM", f"models/unet : version is not DDPM or DDIM! Given: {self.version}"

    self.config = config

    if getattr(config.model, 'spade', False):
      self.unet = SPADE_NCSNpp(config)
    else:
      self.unet = NCSNpp(config)
    # self.fusionNet = FusuionNET()
    self.schedule = getattr(config.model, 'sigma_dist', 'linear')
    if self.schedule == 'linear':
      self.register_buffer('betas', get_sigmas(config))
      self.register_buffer('alphas', torch.cumprod(1 - self.betas.flip(0), 0).flip(0))
      self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
    elif self.schedule == 'cosine':
      self.register_buffer('alphas', get_sigmas(config))
      self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
      self.register_buffer('betas', 1 - self.alphas/self.alphas_prev)
    self.gamma = getattr(config.model, 'gamma', False)
    if self.gamma:
        self.theta_0 = 0.001
        self.register_buffer('k', self.betas/(self.alphas*(self.theta_0 ** 2))) # large to small, doesn't match paper, match code instead
        self.register_buffer('k_cum', torch.cumsum(self.k.flip(0), 0).flip(0)) # flip for small-to-large, then flip back
        self.register_buffer('theta_t', torch.sqrt(self.alphas)*self.theta_0)

    self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)



  def forward(self, x, y, cond=None, cond_mask=None,era5_cond=None,era5_cond_3d=None,attn=None,target=None,p_i=None):
    if self.noise_in_cond and cond is not None: # We add noise to cond
      alphas = self.alphas
      # if labels is None:
      #     labels = torch.randint(0, len(alphas), (cond.shape[0],), device=cond.device)
      labels = y
      used_alphas = alphas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
      if self.gamma:
        used_k = self.k_cum[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
        used_theta = self.theta_t[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
        z = torch.distributions.gamma.Gamma(used_k, 1 / used_theta).sample()
        z = (z - used_k*used_theta)/(1 - used_alphas).sqrt()
      else:
        z = torch.randn_like(cond)
      cond = used_alphas.sqrt() * cond + (1 - used_alphas).sqrt() * z

    # visualize_data(data_2d=era5_cond,data_3d=era5_cond_3d,cond = cond,save_dir="/opt/data/private/mcvd-pytorch/lx_2d+3d_era5_attn_visual/era5_cond_visual.png")
    if target is not None:
       unet,causal_loss = self.unet(x, y, cond, cond_mask=cond_mask,data_2d = era5_cond,data_3d=era5_cond_3d,target=target)
       return unet,causal_loss
    else:
       return self.unet(x, y, cond, cond_mask=cond_mask,data_2d = era5_cond,data_3d=era5_cond_3d,p_i=p_i)

    

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
    def __init__(self, in_channels, hidden_channels=384, mask_ratio=0.7,tau=0.02, prune_thresh=0.1, max_prune_ratio=0.7, recovery_prob=0.002):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 将空间压缩成 1x1，用于通道评分
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.mask_ratio = mask_ratio
        self.sigmoid = nn.Sigmoid()
        self.tau = tau  # softmax temperature
        self.prune_thresh = prune_thresh  # soft_mask阈值
        self.max_prune_ratio = max_prune_ratio
        self.recovery_prob = recovery_prob  # 被裁剪通道恢复概率
    def forward(self, feat, training=True):
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
        raw_score  = self.net(feat).squeeze(-1).squeeze(-1)  # -> B x C

        soft_mask = F.softmax(raw_score / self.tau, dim=1)  # B x C
        # 阈值裁剪 + 随机恢复
        binary_mask = (soft_mask > self.prune_thresh).float()  # 初始裁剪掩码
        prune_ratio = 1 - binary_mask.sum(dim=1) / C  # 每个样本的裁剪率

        # 随机恢复部分被裁剪的通道（训练时）
        if training and self.recovery_prob > 0:
            recover_mask = (torch.rand_like(binary_mask) < self.recovery_prob).float()
            binary_mask = torch.max(binary_mask, recover_mask)

        # reshape 用于广播
        binary_mask_reshape = binary_mask.view(B, C, 1, 1)
        # 构造因果与非因果特征
        causal_feat = feat * binary_mask_reshape
        noncausal_feat = feat.detach() * (1 - binary_mask_reshape)

        return causal_feat, noncausal_feat,binary_mask_reshape,soft_mask, prune_ratio


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
        self.prune_lower_bound = 0.3  # 剪枝率过低的下限
        self.alpha = 1  # 剪枝引导loss的权重

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

    def forward(self, causal_feat, noncausal_feat, target,soft_mask=None, prune_ratio=None):
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

        # 理想情况是 causal 越接近 1，noncausal 越接近 0
        loss_causal = F.binary_cross_entropy(sim_causal, torch.ones_like(sim_causal))
        loss_noncausal = F.binary_cross_entropy(sim_noncausal, torch.zeros_like(sim_noncausal))
        # 越接近说明 causal 和 noncausal 没分开
        margin = 0.3
        margin_loss = F.relu(sim_noncausal - sim_causal + margin).mean()
        prune_penalty_loss = 0.0
        if (prune_ratio is not None) and (soft_mask is not None):
            avg_prune_ratio = prune_ratio.mean().item()
            if avg_prune_ratio < self.prune_lower_bound:
                # 强制 soft_mask 尽可能接近稀疏（趋向全0）
                sparsity_loss = soft_mask.mean()
                target_ratio = 0.3
                deviation_loss = (prune_ratio.mean() - target_ratio).pow(2)
                prune_penalty_loss = sparsity_loss + deviation_loss

        # 条件互信息 loss（max causal info, suppress noncausal）
        cmi_loss = loss_causal + self.lambda_nc * loss_noncausal+self.lambda_margin * margin_loss+  self.alpha * prune_penalty_loss

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

def visualize_data(data_2d, data_3d, cond, save_dir='visualizations', batch_idx=0, sample_num=9):
    os.makedirs(save_dir, exist_ok=True)

    sst = data_2d[:, 0:4, :, :].unsqueeze(2)  # (B,4,1,H,W)
    msl = data_2d[:, 4:, :, :].unsqueeze(2)   # (B,4,1,H,W)
    z = data_3d[:, 0:4, -1, :, :].unsqueeze(2)  # (B,4,1,H,W) @ 200hPa
    r = data_3d[:, 4:, 0, :, :].unsqueeze(2)
    # cond shape: (B,T,H,W) or (B,T,1,H,W)
    if cond.dim() == 4:
        cond_vis = cond.unsqueeze(2)  # (B,T,1,H,W)
    else:
        cond_vis = cond  # already (B,T,1,H,W)

    for i in range(min(sample_num, data_2d.shape[0]//10)):
        fig, axes = plt.subplots(5, 4, figsize=(12, 15))
        fig.suptitle(f'Batch {batch_idx} - Sample {i}', fontsize=16)

        for t in range(4):
            axes[0, t].imshow(sst[i, t, 0].cpu().numpy(), cmap='jet')
            axes[0, t].set_title(f'SST T{t}')
            axes[1, t].imshow(msl[i, t, 0].cpu().numpy(), cmap='jet')
            axes[1, t].set_title(f'MSL T{t}')
            axes[2, t].imshow(z[i, t, 0].cpu().numpy(), cmap='jet')
            axes[2, t].set_title(f'Z@850hPa T{t}')
            axes[3, t].imshow(r[i, t, 0].cpu().numpy(), cmap='jet')
            axes[3, t].set_title(f'RH@200hPa T{t}')
            axes[4, t].imshow(cond_vis[i, t, 0].detach().cpu().numpy(), cmap='gray')
            axes[4, t].set_title(f'Cond T{t}')

        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f'80000-batch{batch_idx}_sample{i}.png'))
        plt.close()


def visualize_attention16(attention_maps=None,save_path="/opt/data/private/mcvd-pytorch/lx_2d+3d_era5_attn_visual/16-new-attention-80000.png", samples_per_row=4):
    """
    可视化当前存储的所有样本的注意力图
    
    Args:
        save_path (str): 图片保存路径，如果为None则直接显示
        samples_per_row (int): 每行显示的样本数量
    """
    if attention_maps is None:
        print("No attention maps to visualize. Run forward pass first.")
        return
    
    vars = list(attention_maps.keys())
    
    for var in vars:
        # 获取当前变量的所有样本 [num_samples, H, W]
        attn_maps = attention_maps[var].cpu().numpy()  
        num_samples = attn_maps.shape[0]
        
        # 计算需要的行数
        num_rows = (num_samples + samples_per_row - 1) // samples_per_row
        
        # 创建子图画布
        fig, axes = plt.subplots(num_rows, samples_per_row, 
                              figsize=(3 * samples_per_row, 3 * num_rows))
        if num_samples == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        
        
        # 绘制每个样本
        for idx in range(num_samples//10):
            row = idx // samples_per_row
            col = idx % samples_per_row
            ax = axes[row, col]
            
            attn_map = attn_maps[idx]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            im = ax.imshow(attn_map, cmap='jet', vmin=0, vmax=1)
            ax.set_title(f'Sample {idx}')
            ax.axis('off')
        
        # 添加统一的colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), 
                    fraction=0.02, pad=0.01, 
                    label='Attention Value')
        
        plt.suptitle(f'Attention Maps for {var} (Total: {num_samples} samples)', 
                    y=1.02)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            var_save_path = save_path.replace('.png', f'_{var}.png')
            plt.savefig(var_save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
def visualize_attention32(attention_maps=None,save_path="/opt/data/private/mcvd-pytorch/lx_2d+3d_era5_attn_visual/32-new-attention-000.png", samples_per_row=4):
    """
    可视化当前存储的所有样本的注意力图
    
    Args:
        save_path (str): 图片保存路径，如果为None则直接显示
        samples_per_row (int): 每行显示的样本数量
    """
    if attention_maps is None:
        print("No attention maps to visualize. Run forward pass first.")
        return
    
    vars = list(attention_maps.keys())
    
    for var in vars:
        # 获取当前变量的所有样本 [num_samples, H, W]
        attn_maps = attention_maps[var].cpu().numpy()  
        num_samples = attn_maps.shape[0]
        
        # 计算需要的行数
        num_rows = (num_samples + samples_per_row - 1) // samples_per_row
        
        # 创建子图画布
        fig, axes = plt.subplots(num_rows, samples_per_row, 
                              figsize=(3 * samples_per_row, 3 * num_rows))
        if num_samples == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        
        
        # 绘制每个样本
        for idx in range(num_samples//10):
            row = idx // samples_per_row
            col = idx % samples_per_row
            ax = axes[row, col]
            
            attn_map = attn_maps[idx]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            im = ax.imshow(attn_map, cmap='jet', vmin=0, vmax=1)
            ax.set_title(f'Sample {idx}')
            ax.axis('off')
        
        # 添加统一的colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), 
                    fraction=0.02, pad=0.01, 
                    label='Attention Value')
        
        plt.suptitle(f'Attention Maps for {var} (Total: {num_samples} samples)', 
                    y=1.02)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            var_save_path = save_path.replace('.png', f'_{var}.png')
            plt.savefig(var_save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


def visualize_attention128(attention_maps=None,save_path="/opt/data/private/mcvd-pytorch/lx_2d+3d_era5_attn_visual/128-new-attention-128000-6-14_overlay-i=95.png", samples_per_row=4):
    """
    可视化当前存储的所有样本的注意力图
    
    Args:
        save_path (str): 图片保存路径，如果为None则直接显示
        samples_per_row (int): 每行显示的样本数量
    """
    npy_path = '/opt/data/private/mcvd-pytorch/TCGdiff_pred_traj-24-3-lx+era5-lx-causal+loss+temporal-random-idx-128000.npy'
    preds = torch.from_numpy(np.load(npy_path)).squeeze(-1).to(torch.device("cuda")).float()# shape: (84, 8, 128, 128)
    preds=(preds-preds.min())/(preds.max()-preds.min())

    if not attention_maps:
        print("No attention maps to visualize. Run forward pass first.")
        return
    
    vars = list(attention_maps.keys())
    
    for var in vars:
        attn_maps = attention_maps[var].unsqueeze(1)  # [N,1,H,W]
        attn_maps = attn_maps.detach().cpu().numpy()       # 转为 numpy [N,1,H,W]
        futures = preds[:,4].detach().cpu().numpy()  # (84,128,128)
        # 上采样每张 attention map 到 128x128
        attn_maps_upsampled = []
        for i in range(attn_maps.shape[0]):
            # shape: (1, H, W) -> (H, W)
            attn_map = attn_maps[i, 0]
            # resize to (128, 128)
            attn_map_resized = cv2.resize(attn_map, (128, 128), interpolation=cv2.INTER_CUBIC)
            attn_maps_upsampled.append(attn_map_resized)

        # [N,128,128]
        attn_maps_np = np.stack(attn_maps_upsampled, axis=0)

        num_samples = attn_maps_np.shape[0]
        # 计算需要的行数
        num_rows = (num_samples + samples_per_row - 1) // samples_per_row
    
        # 创建子图画布
        fig, axes = plt.subplots(num_rows, samples_per_row, 
                              figsize=(3 * samples_per_row, 3 * num_rows))
        if num_samples == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 绘制每个样本
        for idx in range(num_samples//10):
            row = idx // samples_per_row
            col = idx % samples_per_row
            ax = axes[row, col]
            future_img = futures[idx]
            attn_map = attn_maps_np[idx]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            ax.imshow(future_img, cmap='gray',  interpolation='nearest')
            im = ax.imshow(attn_map, cmap='jet', alpha=0.6, interpolation='nearest', vmin=0, vmax=1)
            ax.set_title(f'Sample {idx}')
            ax.axis('off')
        
        # 添加统一的colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), 
                    fraction=0.02, pad=0.01, 
                    label='Attention Value')
        
        plt.suptitle(f'Attention Maps for {var} (Total: {num_samples} samples)', 
                    y=1.02)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            var_save_path = save_path.replace('.png', f'_{var}.png')
            plt.savefig(var_save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

              

def compute_causal_loss(alpha, beta, y, delta=1.01):
    """
    alpha: [B, C, 32, 32] (causal feature)
    beta:  [B, C, 32, 32] (non-causal feature)
    y:     [B, 5, 128, 128] (target image)

    Returns:
    L_causal = -I(alpha; y | beta) + I(alpha; beta)
    """
    B = alpha.shape[0]

    # Step 1: Downsample y to 32x32
    y_down = F.adaptive_avg_pool2d(y, output_size=(32, 32))  # B, 5, 32, 32

    # Step 2: Flatten all features for Gram matrix
    def gram_matrix(x):
        B = x.size(0)
        x = x.view(B, -1)
        x = x - x.mean(dim=0, keepdim=True)  # 中心化：dim=0 更稳定
        K = torch.matmul(x, x.T)
        K = K / (x.size(1) + 1e-5)
        K = K + 1e-4 * torch.eye(B, device=x.device)  # 加正则项，防止退化
        return K


    def renyi_entropy(K, delta=1.01):
        eigvals = torch.linalg.eigvalsh(K)
        eigvals = torch.clamp(eigvals, min=1e-10)
        return (1 / (1 - delta)) * torch.log2(torch.sum(eigvals ** delta))

    def joint_entropy(K1, K2, K3=None, delta=1.01):
        if K3 is None:
            K = K1 * K2
        else:
            K = K1 * K2 * K3
        K = K / torch.trace(K)
        return renyi_entropy(K, delta)

    # Step 3: Compute Gram matrices
    Ka = gram_matrix(alpha)
    Kb = gram_matrix(beta)
    Ky = gram_matrix(y_down)

    # Step 4: Compute entropies
    H_a = renyi_entropy(Ka, delta)
    H_b = renyi_entropy(Kb, delta)
    H_ab = joint_entropy(Ka, Kb, delta=delta)
    H_yb = joint_entropy(Ky, Kb, delta=delta)
    H_ayb = joint_entropy(Ka, Ky, Kb, delta=delta)

    I_ab = H_a + H_b - H_ab
    I_a_y_given_b = H_ab + H_yb - H_b - H_ayb

    return -I_a_y_given_b + I_ab  # L_causal  # -I_a_y_given_b在知道 beta 的前提下，alpha 中还能提供多少关于 y 的额外信息   鼓励 alpha 对 y 有更多独立信息（通过最大化 I(alpha; y | beta)） 
                                  #              I_ab   鼓励 alpha 和 beta 解耦（通过最小化 I(alpha; beta)）


def visualize_attention_and_mask(attn_feat, soft_mask, binary_mask, save_path='/opt/data/private/mcvd-pytorch/lx_2d+3d_era5_attn_visual/mask_visual',sample_idx=0, var_name=None):
    """
    显示和保存 attn_feat、soft_mask、binary_mask 的可视化图像（并排）

    参数:
        attn_feat: Tensor[B, C, H, W]
        soft_mask: Tensor[B, H, W] （注意：不是 [B,1,H,W]）
        binary_mask: Tensor[B, 1, H, W]
        save_path: 保存文件夹路径
        var_name: 当前变量名（用于命名）
        epoch: 当前训练轮数
    """
    os.makedirs(save_path, exist_ok=True)

    # 取第一个样本
    attn_map = attn_feat[sample_idx].mean(dim=0).detach().cpu().numpy()  # [H, W]
    soft_map = soft_mask[sample_idx].detach().cpu().numpy()              # [H, W]
    binary_map = binary_mask[sample_idx, 0].detach().cpu().numpy()       # [H, W]

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(attn_map, cmap='viridis')
    axes[0].set_title("attn_feat[0, 0]")
    axes[0].axis("off")

    axes[1].imshow(soft_map, cmap='hot')
    axes[1].set_title("soft_mask[0]")
    axes[1].axis("off")

    axes[2].imshow(binary_map, cmap='gray')
    axes[2].set_title("binary_mask[0]")
    axes[2].axis("off")

    plt.suptitle(f"Variable: {var_name} ")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{var_name}_mask_compare-{sample_idx}-26000-new_attn_causal-6.3.png"))
    plt.close()


