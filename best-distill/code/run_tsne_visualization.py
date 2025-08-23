#!/usr/bin/env python3
"""
T-SNE可视化运行脚本
用于验证蒸馏模型的特征对齐效果
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from collections import defaultdict
import os
from tqdm import tqdm
import argparse
import sys
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tsne_visualization import extract_features, create_tsne_visualization

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_models(teacher_ckpt, student_ckpt, config, device):
    """加载教师和学生模型"""
    from models.better.ncsnpp_more import UNetMore_DDPM
    
    # 加载教师模型
    model_teacher = UNetMore_DDPM(config, is_stu=False)
    teacher_state = torch.load(teacher_ckpt, map_location=device)
    model_teacher.load_state_dict(teacher_state['model_state_dict'])
    model_teacher.to(device)
    model_teacher.eval()
    
    # 加载学生模型
    model_student = UNetMore_DDPM(config, is_stu=True)
    student_state = torch.load(student_ckpt, map_location=device)
    model_student.load_state_dict(student_state['model_state_dict'])
    model_student.to(device)
    model_student.eval()
    
    return model_teacher, model_student

def create_dummy_dataloader(num_samples=100, batch_size=8):
    """创建虚拟数据加载器用于演示"""
    class DummyDataset:
        def __init__(self, num_samples, batch_size):
            self.num_samples = num_samples
            self.batch_size = batch_size
            self.num_batches = (num_samples + batch_size - 1) // batch_size
            
        def __iter__(self):
            for i in range(self.num_batches):
                # 创建虚拟数据
                batch_size_actual = min(self.batch_size, self.num_samples - i * self.batch_size)
                
                # x: 卫星云图 [B, 5, 128, 128]
                x = torch.randn(batch_size_actual, 5, 128, 128)
                # y: 时间步 [B]
                y = torch.randint(0, 1000, (batch_size_actual,))
                # cond: 条件图像 [B, 4, 128, 128]
                cond = torch.randn(batch_size_actual, 4, 128, 128)
                # era5_cond: ERA5 2D数据 [B, 8, 128, 128]
                era5_cond = torch.randn(batch_size_actual, 8, 128, 128)
                # era5_cond_3d: ERA5 3D数据 [B, 8, 4, 128, 128]
                era5_cond_3d = torch.randn(batch_size_actual, 8, 4, 128, 128)
                
                yield [x, y, cond, era5_cond, era5_cond_3d]
    
    return DummyDataset(num_samples, batch_size)

def run_visualization_demo():
    """运行可视化演示"""
    print("=== T-SNE可视化演示 ===")
    print("注意：这是一个演示脚本，使用虚拟数据")
    print("实际使用时请替换为真实的模型和数据加载器")
    
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = 200  # 演示用较少的样本
    batch_size = 8
    save_dir = 'tsne_demo_results'
    
    print(f"使用设备: {device}")
    print(f"样本数量: {num_samples}")
    print(f"批次大小: {batch_size}")
    print(f"保存目录: {save_dir}")
    
    # 创建虚拟数据加载器
    dataloader = create_dummy_dataloader(num_samples, batch_size)
    
    # 创建虚拟特征数据（模拟提取的特征）
    print("\n生成虚拟特征数据...")
    era5_vars = ['sst', 'msl', 'z', 'r']
    
    # 模拟特征提取结果
    features = {
        'teacher_encoder': {},
        'student_encoder': {},
        'teacher_attn': {},
        'student_attn': {},
        'labels': []
    }
    
    # 为每个变量生成特征
    for var in era5_vars:
        # 生成教师和学生网络的编码器特征
        teacher_encoder = np.random.randn(num_samples, 384)
        student_encoder = teacher_encoder + np.random.randn(num_samples, 384) * 0.3  # 添加一些差异
        
        # 生成教师和学生网络的注意力特征
        teacher_attn = np.random.randn(num_samples, 384)
        student_attn = teacher_attn + np.random.randn(num_samples, 384) * 0.2  # 添加一些差异
        
        features['teacher_encoder'][var] = teacher_encoder
        features['student_encoder'][var] = student_encoder
        features['teacher_attn'][var] = teacher_attn
        features['student_attn'][var] = student_attn
        
        # 添加标签
        features['labels'].extend([var] * (num_samples // 4))
    
    features['labels'] = np.array(features['labels'])
    
    print("特征数据生成完成")
    print(f"教师编码器特征形状: {features['teacher_encoder']['sst'].shape}")
    print(f"学生编码器特征形状: {features['student_encoder']['sst'].shape}")
    
    # 运行T-SNE可视化
    print("\n开始T-SNE可视化...")
    create_tsne_visualization(features, save_dir)
    
    print(f"\n可视化完成！结果保存在: {save_dir}")
    print("生成的文件包括:")
    print("- encoder_features_tsne.png: 编码器特征T-SNE图")
    print("- attention_features_tsne.png: 注意力特征T-SNE图")
    print("- all_variables_tsne.png: 所有变量综合T-SNE图")

def run_real_visualization(teacher_ckpt, student_ckpt, config_path, data_path, 
                          num_samples=1000, save_dir='tsne_results', device='cuda'):
    """运行真实数据的可视化"""
    print("=== 真实数据T-SNE可视化 ===")
    
    # 加载配置
    config = load_config(config_path)
    
    # 加载模型
    print("加载模型...")
    model_teacher, model_student = load_models(teacher_ckpt, student_ckpt, config, device)
    
    # 加载数据
    print("加载数据...")
    # 这里需要根据您的实际数据加载方式修改
    # dataloader = load_your_data(data_path)
    
    # 提取特征
    print("提取特征...")
    # features = extract_features(model_teacher, model_student, dataloader, device, num_samples)
    
    # 运行可视化
    print("运行T-SNE可视化...")
    # create_tsne_visualization(features, save_dir)
    
    print("请根据您的实际数据加载方式修改此函数")

def main():
    parser = argparse.ArgumentParser(description='T-SNE可视化脚本')
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'real'],
                       help='运行模式: demo(演示) 或 real(真实数据)')
    parser.add_argument('--teacher_ckpt', type=str, help='教师模型检查点路径')
    parser.add_argument('--student_ckpt', type=str, help='学生模型检查点路径')
    parser.add_argument('--config_path', type=str, help='配置文件路径')
    parser.add_argument('--data_path', type=str, help='数据路径')
    parser.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    parser.add_argument('--save_dir', type=str, default='tsne_results', help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_visualization_demo()
    elif args.mode == 'real':
        if not all([args.teacher_ckpt, args.student_ckpt, args.config_path, args.data_path]):
            print("错误：真实数据模式需要提供所有必要参数")
            print("请提供: --teacher_ckpt, --student_ckpt, --config_path, --data_path")
            return
        
        run_real_visualization(
            args.teacher_ckpt, args.student_ckpt, args.config_path, args.data_path,
            args.num_samples, args.save_dir, args.device
        )

if __name__ == '__main__':
    main()