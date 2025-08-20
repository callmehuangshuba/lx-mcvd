
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import numpy as np
import torch


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

# 可视化topk的特征图 图
def visualize_topk_attn_channels(attn_feat, indices, save_path="/opt/data/private/mcvd-pytorch2/mcvd-pytorch/channels_visual/channels_visual_topk.png", sample_idx=0, var_name='h', topk=30):
    """
    可视化 attn_feat 的 top-k 通道（每个通道为一个 H×W 热力图）

    参数:
        attn_feat: Tensor[B, C, H, W]
        indices: Tensor[B, topk] 或 Tensor[topk]，topk通道索引
        save_path: 保存图像的目录路径
        sample_idx: 使用第几个样本进行可视化
        var_name: 当前变量名（用于保存文件名）
        topk: 可视化的前多少个通道
    """
    os.makedirs(save_path, exist_ok=True)

    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    if indices.ndim == 2:
        indices = indices[sample_idx]  # B x topk -> 取第一个样本的通道索引

    attn_sample = attn_feat[sample_idx].detach().cpu()  # C x H x W
    H, W = attn_sample.shape[1:]

    # 准备 subplot
    n_cols = 5
    n_rows = int(np.ceil(topk / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))

    for i in range(topk):
        ch_idx = indices[i]
        ch_feat = attn_sample[ch_idx].numpy()  # H x W
        ch_feat_resized = cv2.resize(ch_feat, (128, 128), interpolation=cv2.INTER_CUBIC)
        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        im = ax.imshow(ch_feat_resized, cmap='viridis')
        ax.set_title(f"Channel {ch_idx}", fontsize=10)
        ax.axis("off")

    # 关闭多余子图
    for j in range(topk, n_cols * n_rows):
        row, col = divmod(j, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis("off")

    plt.suptitle(f"Top-{topk} Channels: {var_name}", fontsize=12)
    plt.tight_layout()
    save_file = os.path.join(save_path, f"{var_name}_top{topk}_channels_sample{sample_idx}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"✅ 已保存可视化图像到：{save_file}")

# 放大aug 增强前后特征图对比  只有保存aug的差异
def visualize_topk_channel_diff(before_feat, after_feat, indices, sample_idx=0, var_name="diff", save_path="/opt/data/private/mcvd-pytorch2/mcvd-pytorch/channels_visual/aug_diff", topk=10):
    """
    可视化增强前后特征图的差异（after - before）
    会对差异图归一化放大显示
    """
    os.makedirs(save_path, exist_ok=True)
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    if indices.ndim == 2:
        indices = indices[sample_idx]

    before = before_feat[sample_idx].detach().cpu().numpy()
    after = after_feat[sample_idx].detach().cpu().numpy()

    n_cols = 5
    n_rows = int(np.ceil(topk / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))

    for i in range(topk):
        ch_idx = indices[i]
        diff_map = after[ch_idx] - before[ch_idx]
        diff_map = cv2.resize(diff_map, (128, 128), interpolation=cv2.INTER_CUBIC)

        # 对差异图进行对称归一化（可正可负）
        abs_max = np.percentile(np.abs(diff_map), 99) + 1e-6
        diff_map = np.clip(diff_map / abs_max, -1, 1)

        row, col = divmod(i, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        im = ax.imshow(diff_map, cmap='seismic', vmin=-1, vmax=1)  # 红蓝对比
        ax.set_title(f"Δ Channel {ch_idx}", fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Top-{topk} Channel Differences: {var_name}", fontsize=12)
    plt.tight_layout()
    save_file = os.path.join(save_path, f"{var_name}_diff_top{topk}_sample{sample_idx}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"✅ 已保存差异图像到：{save_file}")
# 放大aug 增强前后特征图对比
def visualize_aug_diff_triplet(before_feat, after_feat, indices, sample_idx=0, topk=10, var_name="feat", save_path="/opt/data/private/mcvd-pytorch2/mcvd-pytorch/channels_visual/aug_diff_3"):
    """
    对比可视化增强前、增强后以及差异图（diff）的特征图

    参数:
        before_feat: Tensor[B, C, H, W]
        after_feat: Tensor[B, C, H, W]
        indices: Tensor[B, topk] or Tensor[topk]
        sample_idx: 第几个样本
        topk: 可视化前多少个通道
        var_name: 保存文件名用
        save_path: 图像保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    if indices.ndim == 2:
        indices = indices[sample_idx]  # B x topk -> topk

    before = before_feat[sample_idx].detach().cpu().numpy()  # C x H x W
    after = after_feat[sample_idx].detach().cpu().numpy()    # C x H x W

    n_cols = 3  # before / after / diff
    n_rows = topk
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    amplify = 1.0  # 控制差异图放大倍数

    for i in range(topk):
        ch_idx = indices[i]
        b = before[ch_idx]
        a = after[ch_idx]
        d = np.abs(a - b) * amplify

        # resize to (128, 128)
        b = cv2.resize(b, (128, 128), interpolation=cv2.INTER_CUBIC)
        a = cv2.resize(a, (128, 128), interpolation=cv2.INTER_CUBIC)
        d = cv2.resize(d, (128, 128), interpolation=cv2.INTER_CUBIC)

        # 归一化增强对比度（可选）
        def norm(x):
            x = x - np.min(x)
            if np.max(x) > 0:
                x = x / np.max(x)
            return x

        b = norm(b)
        a = norm(a)
        d = norm(d)

        axes[i, 0].imshow(b, cmap='viridis')
        axes[i, 0].set_title(f"Before - ch{ch_idx}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(a, cmap='viridis')
        axes[i, 1].set_title(f"After - ch{ch_idx}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(d, cmap='inferno')
        axes[i, 2].set_title(f"Diff ×{amplify} - ch{ch_idx}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    save_file = os.path.join(save_path, f"{var_name}_aug_diff_triplet_sample{sample_idx}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"✅ 增强对比图已保存：{save_file}")
def visualize_denoising(noise_x, x0_pred, target, num_samples=5, save_path="/opt/data/private/mcvd-pytorch2/mcvd-pytorch/denoise_img"):
  """
  可视化前 num_samples 个样本的噪声图像、去噪图像、以及目标图像。
  形状: (B, T, H, W)，T=帧数
  """
  noise_x = noise_x.detach().cpu()
  x0_pred = x0_pred.detach().cpu()
  target = target.detach().cpu()

  B, T, H, W = noise_x.shape
  num_samples = min(num_samples, B)

  for i in range(num_samples):
      fig, axes = plt.subplots(3, T, figsize=(T * 2.5, 3 * 2.5))

      for t in range(T):
          # 噪声图像
          axes[0, t].imshow(noise_x[i, t], cmap='gray')
          axes[0, t].set_title(f"Noise t={t}")
          axes[0, t].axis('off')

          # 去噪图像
          axes[1, t].imshow(x0_pred[i, t], cmap='gray')
          axes[1, t].set_title(f"Denoised t={t}")
          axes[1, t].axis('off')

          # 目标图像
          axes[2, t].imshow(target[i, t], cmap='gray')
          axes[2, t].set_title(f"GT t={t}")
          axes[2, t].axis('off')

      plt.tight_layout()
      if save_path:
          plt.savefig(f"{save_path}/sample_{i}.png")
      else:
          plt.show()
      plt.close()
import random
import matplotlib.patches as patches
# 可视化topk通道
def visualize_topk_indices(indices_dict, topk=30, save_path="/opt/data/private/mcvd-pytorch2/mcvd-pytorch/channels_visual/channels_visual_topk_num-1000.png"):
    """
    可视化不同变量的 top-k 通道排序编号（竖排颜色块+数字）

    Args:
        indices_dict: dict，key 是变量名，value 是通道索引数组（[topk] or [B, topk]）
        topk: 可视化前多少通道
        save_path: 保存路径（包含文件名）
    """
    variables = list(indices_dict.keys())
    n_vars = len(variables)

    # 构造一个 [topk, n_vars] 的 index 矩阵
    index_matrix = np.zeros((topk, n_vars), dtype=int)

    for i, var in enumerate(variables):
        indices = indices_dict[var]
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if indices.ndim == 2:
            indices = indices[0]
        index_matrix[:, i] = indices[:topk]

    # 画图：每列是一个变量，每行一个通道 index
    fig, ax = plt.subplots(figsize=(n_vars * 1.5, topk * 0.5))

    # 显示颜色块
    im = ax.imshow(index_matrix, cmap='tab20', aspect='auto')

    # 添加文字
    for i in range(topk):
        for j in range(n_vars):
            idx_val = int(index_matrix[i, j])
            ax.text(j, i, str(idx_val), ha='center', va='center', color='black', fontsize=8, fontweight='bold')

    # 设置轴
    ax.set_xticks(np.arange(n_vars))
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.set_yticks(np.arange(topk))
    ax.set_yticklabels([str(i) for i in range(topk)])

    ax.set_xlabel("Task / Variable")
    ax.set_ylabel("Top-k Channel Rank")
    ax.set_title("Top-k Channel Indices per Variable", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ 图像已保存到：{save_path}")

    plt.show()