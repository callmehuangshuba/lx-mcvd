#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-SNE可视化脚本：验证蒸馏模型的特征对齐效果
该脚本将提取教师和学生网络的特征，并使用T-SNE降维可视化特征分布
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import argparse
from tqdm import tqdm
import yaml
from matplotlib.colors import ListedColormap
import pandas as pd

# 导入模型相关模块
import sys
sys.path.append('.')
from models.better.ncsnpp_more import NCSNpp_stu
from datasets import get_dataset
import warnings
warnings.filterwarnings('ignore')

def load_model_and_config(config_path, ckpt_path):
    """加载配置文件和模型"""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建配置对象
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
    
    config = Config(config)
    
    # 加载模型
    model = NCSNpp_stu(config)
    
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        print(f"Loaded checkpoint from {ckpt_path}")
    
    return model, config

def extract_features_batch(model, dataloader, device, max_samples=1000):
    """批量提取模型特征"""
    model.eval()
    
    # 存储特征的列表
    features_dict = {
        'teacher_16': [], 'teacher_32': [],
        'student_16': [], 'student_32': [],
        'encoder_teacher_16': [], 'encoder_teacher_32': [],
        'encoder_student_16': [], 'encoder_student_32': [],
        'labels': [], 'era5_vars': []
    }
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if sample_count >= max_samples:
                break
                
            # 解析批次数据
            if len(batch) == 6:
                x, cond, era5_cond, era5_cond_3d, y, cond_mask = batch
            else:
                x, cond, era5_cond, era5_cond_3d, y = batch
                cond_mask = None
            
            x = x.to(device)
            cond = cond.to(device) if cond is not None else None
            era5_cond = era5_cond.to(device) if era5_cond is not None else None
            era5_cond_3d = era5_cond_3d.to(device) if era5_cond_3d is not None else None
            
            batch_size = x.shape[0]
            
            # 生成随机时间步
            time_steps = torch.randint(0, 1000, (batch_size,)).to(device)
            
            # 添加噪声
            sigmas = model.sigmas
            used_sigmas = sigmas[time_steps].reshape(batch_size, *([1] * len(x.shape[1:])))
            noise = torch.randn_like(x)
            perturbed_x = x + used_sigmas * noise
            
            # 教师网络前向传播（包含ERA5）
            try:
                h_teacher, feat_16_tea, feat_32_tea, encoder_16_tea, encoder_32_tea = model(
                    perturbed_x, time_steps, cond=cond, 
                    era5_cond=era5_cond, era5_cond_3d=era5_cond_3d,
                    cond_mask=cond_mask, return_feat=True
                )
                
                # 学生网络前向传播（不包含ERA5）
                h_student, feat_16_stu, feat_32_stu, encoder_16_stu, encoder_32_stu, _ = model(
                    perturbed_x, time_steps, cond=cond, 
                    era5_cond=None, era5_cond_3d=None,  # 学生网络不使用ERA5
                    cond_mask=cond_mask, return_feat=True, target=x
                )
                
                # 处理特征字典
                for var in ['sst', 'msl', 'z', 'r']:
                    if var in feat_16_tea and var in feat_32_tea:
                        # 教师特征 (16x16)
                        feat_tea_16 = feat_16_tea[var].mean(dim=[2, 3])  # B, C
                        features_dict['teacher_16'].append(feat_tea_16.cpu())
                        
                        # 教师特征 (32x32)  
                        feat_tea_32 = feat_32_tea[var].mean(dim=[2, 3])  # B, C
                        features_dict['teacher_32'].append(feat_tea_32.cpu())
                        
                        # 学生特征 (16x16)
                        feat_stu_16 = feat_16_stu[var].mean(dim=[2, 3])  # B, C
                        features_dict['student_16'].append(feat_stu_16.cpu())
                        
                        # 学生特征 (32x32)
                        feat_stu_32 = feat_32_stu[var].mean(dim=[2, 3])  # B, C
                        features_dict['student_32'].append(feat_stu_32.cpu())
                        
                        # 编码器特征
                        enc_tea_16 = encoder_16_tea[var].mean(dim=[2, 3])  # B, C
                        features_dict['encoder_teacher_16'].append(enc_tea_16.cpu())
                        
                        enc_tea_32 = encoder_32_tea[var].mean(dim=[2, 3])  # B, C
                        features_dict['encoder_teacher_32'].append(enc_tea_32.cpu())
                        
                        enc_stu_16 = encoder_16_stu[var].mean(dim=[2, 3])  # B, C
                        features_dict['encoder_student_16'].append(enc_stu_16.cpu())
                        
                        enc_stu_32 = encoder_32_stu[var].mean(dim=[2, 3])  # B, C
                        features_dict['encoder_student_32'].append(enc_stu_32.cpu())
                        
                        # 添加标签
                        features_dict['labels'].extend([f'Teacher_{var}'] * batch_size)
                        features_dict['labels'].extend([f'Student_{var}'] * batch_size)
                        features_dict['era5_vars'].extend([var] * batch_size * 2)
                
                sample_count += batch_size
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # 将列表转换为numpy数组
    for key in features_dict:
        if len(features_dict[key]) > 0 and key not in ['labels', 'era5_vars']:
            features_dict[key] = torch.cat(features_dict[key], dim=0).numpy()
    
    return features_dict

def create_tsne_visualization(features_dict, save_dir='tsne_results', perplexity=30):
    """创建T-SNE可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置图形样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 为不同尺度创建可视化
    scales = ['16', '32']
    feature_types = ['', 'encoder_']
    
    for scale in scales:
        for feat_type in feature_types:
            plt.figure(figsize=(15, 12))
            
            # 准备数据
            teacher_key = f'{feat_type}teacher_{scale}'
            student_key = f'{feat_type}student_{scale}'
            
            if teacher_key not in features_dict or student_key not in features_dict:
                continue
                
            teacher_feat = features_dict[teacher_key]
            student_feat = features_dict[student_key]
            
            if len(teacher_feat) == 0 or len(student_feat) == 0:
                continue
            
            # 合并特征
            all_features = np.concatenate([teacher_feat, student_feat], axis=0)
            
            # 创建标签
            n_teacher = teacher_feat.shape[0]
            n_student = student_feat.shape[0]
            labels = ['Teacher'] * n_teacher + ['Student'] * n_student
            
            # 标准化特征
            scaler = StandardScaler()
            all_features_scaled = scaler.fit_transform(all_features)
            
            # 执行T-SNE
            print(f"Running T-SNE for {feat_type}{scale} features...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(all_features)//4))
            features_2d = tsne.fit_transform(all_features_scaled)
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'T-SNE Visualization - {feat_type.title()}{scale}×{scale} Features', fontsize=16)
            
            # 1. 教师vs学生总体分布
            ax1 = axes[0, 0]
            colors = ['#FF6B6B', '#4ECDC4']
            for i, label in enumerate(['Teacher', 'Student']):
                mask = np.array(labels) == label
                ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=colors[i], label=label, alpha=0.6, s=50)
            ax1.set_title('Teacher vs Student Features')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 不同ERA5变量的分布
            ax2 = axes[0, 1]
            var_colors = {'sst': '#FF9999', 'msl': '#66B2FF', 'z': '#99FF99', 'r': '#FFCC99'}
            
            # 假设我们有ERA5变量信息（这里需要根据实际数据结构调整）
            era5_vars = ['sst', 'msl', 'z', 'r']
            samples_per_var = len(all_features) // (len(era5_vars) * 2)  # 2 for teacher and student
            
            for i, var in enumerate(era5_vars):
                start_idx = i * samples_per_var * 2
                end_idx = (i + 1) * samples_per_var * 2
                if end_idx <= len(features_2d):
                    ax2.scatter(features_2d[start_idx:end_idx, 0], 
                               features_2d[start_idx:end_idx, 1],
                               c=var_colors[var], label=var.upper(), alpha=0.6, s=50)
            ax2.set_title('Features by ERA5 Variables')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 距离分析
            ax3 = axes[1, 0]
            teacher_center = features_2d[:n_teacher].mean(axis=0)
            student_center = features_2d[n_teacher:].mean(axis=0)
            
            # 计算到中心的距离
            teacher_distances = np.linalg.norm(features_2d[:n_teacher] - teacher_center, axis=1)
            student_distances = np.linalg.norm(features_2d[n_teacher:] - student_center, axis=1)
            
            ax3.hist(teacher_distances, alpha=0.7, label='Teacher', bins=30, color='#FF6B6B')
            ax3.hist(student_distances, alpha=0.7, label='Student', bins=30, color='#4ECDC4')
            ax3.set_title('Distance to Cluster Center')
            ax3.set_xlabel('Distance')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 特征空间重叠度分析
            ax4 = axes[1, 1]
            
            # 计算教师和学生特征之间的最近邻距离
            from sklearn.neighbors import NearestNeighbors
            nn_model = NearestNeighbors(n_neighbors=1)
            nn_model.fit(features_2d[n_teacher:])  # 拟合学生特征
            distances, _ = nn_model.kneighbors(features_2d[:n_teacher])  # 查找教师特征的最近学生特征
            
            ax4.hist(distances.flatten(), bins=30, alpha=0.7, color='#9D4EDD')
            ax4.set_title('Teacher-Student Feature Alignment')
            ax4.set_xlabel('Distance to Nearest Student Feature')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_distance = distances.mean()
            ax4.axvline(mean_distance, color='red', linestyle='--', 
                       label=f'Mean Distance: {mean_distance:.3f}')
            ax4.legend()
            
            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join(save_dir, f'tsne_{feat_type}{scale}_features.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")
            
            # 计算并保存对齐指标
            alignment_metrics = {
                'mean_teacher_student_distance': mean_distance,
                'teacher_cluster_variance': np.var(teacher_distances),
                'student_cluster_variance': np.var(student_distances),
                'features_analyzed': len(all_features),
                'tsne_perplexity': tsne.perplexity
            }
            
            metrics_path = os.path.join(save_dir, f'alignment_metrics_{feat_type}{scale}.txt')
            with open(metrics_path, 'w') as f:
                for key, value in alignment_metrics.items():
                    f.write(f"{key}: {value}\n")
            
            plt.close()

def create_detailed_alignment_analysis(features_dict, save_dir='tsne_results'):
    """创建详细的特征对齐分析"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算不同尺度间的对齐度
    alignment_results = {}
    
    for scale in ['16', '32']:
        for feat_type in ['', 'encoder_']:
            teacher_key = f'{feat_type}teacher_{scale}'
            student_key = f'{feat_type}student_{scale}'
            
            if teacher_key in features_dict and student_key in features_dict:
                teacher_feat = features_dict[teacher_key]
                student_feat = features_dict[student_key]
                
                if len(teacher_feat) > 0 and len(student_feat) > 0:
                    # 计算特征相似度
                    teacher_mean = np.mean(teacher_feat, axis=0)
                    student_mean = np.mean(student_feat, axis=0)
                    
                    # 余弦相似度
                    cosine_sim = np.dot(teacher_mean, student_mean) / (
                        np.linalg.norm(teacher_mean) * np.linalg.norm(student_mean)
                    )
                    
                    # 欧氏距离
                    euclidean_dist = np.linalg.norm(teacher_mean - student_mean)
                    
                    alignment_results[f'{feat_type}{scale}'] = {
                        'cosine_similarity': cosine_sim,
                        'euclidean_distance': euclidean_dist,
                        'teacher_shape': teacher_feat.shape,
                        'student_shape': student_feat.shape
                    }
    
    # 保存对齐分析结果
    results_path = os.path.join(save_dir, 'feature_alignment_analysis.txt')
    with open(results_path, 'w') as f:
        f.write("=== Feature Alignment Analysis ===\n\n")
        for key, metrics in alignment_results.items():
            f.write(f"{key} Features:\n")
            f.write(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}\n")
            f.write(f"  Euclidean Distance: {metrics['euclidean_distance']:.4f}\n")
            f.write(f"  Teacher Shape: {metrics['teacher_shape']}\n")
            f.write(f"  Student Shape: {metrics['student_shape']}\n")
            f.write("\n")
    
    print(f"Detailed alignment analysis saved to: {results_path}")
    return alignment_results

def main():
    parser = argparse.ArgumentParser(description='T-SNE Visualization for Distillation Model')
    parser.add_argument('--config', type=str, 
                       default='configs/TCG.yml',
                       help='Path to config file')
    parser.add_argument('--ckpt', type=str, 
                       default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       default='/opt/data/private/datasets',
                       help='Path to dataset')
    parser.add_argument('--save_dir', type=str, 
                       default='tsne_results',
                       help='Directory to save results')
    parser.add_argument('--max_samples', type=int, 
                       default=1000,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--perplexity', type=int, 
                       default=30,
                       help='T-SNE perplexity parameter')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=== T-SNE Visualization for Distillation Model ===")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples}")
    
    # 加载模型和配置
    model, config = load_model_and_config(args.config, args.ckpt)
    model = model.to(args.device)
    
    # 创建数据加载器
    try:
        # 这里需要根据实际的数据集实现进行调整
        print("Loading dataset...")
        dataset, _ = get_dataset(args, config)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True, num_workers=2
        )
        print(f"Dataset loaded with {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataloader for demonstration...")
        # 创建dummy数据用于演示
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                x = torch.randn(5, 128, 128)  # 5 frames
                cond = torch.randn(4, 128, 128)  # 4 condition frames
                era5_cond = torch.randn(4, 4, 128, 128)  # ERA5 2D
                era5_cond_3d = torch.randn(4, 4, 128, 128)  # ERA5 3D
                y = torch.randint(0, 10, (1,))
                return x, cond, era5_cond, era5_cond_3d, y
        
        dataset = DummyDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 提取特征
    print("Extracting features...")
    features_dict = extract_features_batch(model, dataloader, args.device, args.max_samples)
    
    if len(features_dict['teacher_16']) > 0:
        print(f"Successfully extracted features from {len(features_dict['teacher_16'])} samples")
        
        # 创建T-SNE可视化
        print("Creating T-SNE visualizations...")
        create_tsne_visualization(features_dict, args.save_dir, args.perplexity)
        
        # 创建详细对齐分析
        print("Creating detailed alignment analysis...")
        alignment_results = create_detailed_alignment_analysis(features_dict, args.save_dir)
        
        print(f"\n=== Results Summary ===")
        for key, metrics in alignment_results.items():
            print(f"{key}: Cosine Similarity = {metrics['cosine_similarity']:.4f}")
        
        print(f"\nAll results saved to: {args.save_dir}")
        
    else:
        print("No features extracted. Please check the model and data configuration.")

if __name__ == "__main__":
    main()