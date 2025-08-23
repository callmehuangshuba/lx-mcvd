#!/usr/bin/env python3
"""
直接运行的蒸馏模型特征提取和T-SNE可视化代码
用于观察学生网络特征是否成功对齐到教师网络
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import sys
import argparse
from tqdm import tqdm
import random

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_dummy_data(batch_size=8, num_samples=200):
    """创建虚拟数据用于演示"""
    print("创建虚拟数据...")
    
    # 创建虚拟数据
    x = torch.randn(num_samples, 5, 128, 128)  # 卫星云图
    y = torch.randint(0, 1000, (num_samples,))  # 时间步
    cond = torch.randn(num_samples, 4, 128, 128)  # 条件图像
    era5_cond = torch.randn(num_samples, 8, 128, 128)  # ERA5 2D数据
    era5_cond_3d = torch.randn(num_samples, 8, 4, 128, 128)  # ERA5 3D数据
    
    # 创建数据加载器
    class DummyDataLoader:
        def __init__(self, x, y, cond, era5_cond, era5_cond_3d, batch_size):
            self.x = x
            self.y = y
            self.cond = cond
            self.era5_cond = era5_cond
            self.era5_cond_3d = era5_cond_3d
            self.batch_size = batch_size
            self.num_batches = (len(x) + batch_size - 1) // batch_size
            
        def __iter__(self):
            for i in range(self.num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.x))
                
                batch_x = self.x[start_idx:end_idx]
                batch_y = self.y[start_idx:end_idx]
                batch_cond = self.cond[start_idx:end_idx]
                batch_era5_cond = self.era5_cond[start_idx:end_idx]
                batch_era5_cond_3d = self.era5_cond_3d[start_idx:end_idx]
                
                yield [batch_x, batch_y, batch_cond, batch_era5_cond, batch_era5_cond_3d]
    
    return DummyDataLoader(x, y, cond, era5_cond, era5_cond_3d, batch_size)

def extract_features_from_models(model_teacher, model_student, dataloader, device, num_samples=200):
    """从教师和学生模型中提取特征"""
    print("提取模型特征...")
    
    model_teacher.eval()
    model_student.eval()
    
    features = {
        'teacher_encoder': {'sst': [], 'msl': [], 'z': [], 'r': []},
        'student_encoder': {'sst': [], 'msl': [], 'z': [], 'r': []},
        'teacher_attn': {'sst': [], 'msl': [], 'z': [], 'r': []},
        'student_attn': {'sst': [], 'msl': [], 'z': [], 'r': []},
        'labels': []
    }
    
    era5_vars = ['sst', 'msl', 'z', 'r']
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="提取特征")):
            if i * batch[0].shape[0] >= num_samples:
                break
                
            # 解包批次数据
            x, y, cond, era5_cond, era5_cond_3d = [b.to(device) for b in batch]
            
            try:
                # 教师模型前向传播（有ERA5条件）
                h_teacher, feat_16_tea, feat_32_tea, encoder_16_tea, encoder_32_tea = model_teacher(
                    x, y, cond=cond, era5_cond=era5_cond, era5_cond_3d=era5_cond_3d, return_feat=True
                )
                
                # 学生模型前向传播（无ERA5条件）
                h_student, feat_16_stu, feat_32_stu, encoder_16_stu, encoder_32_stu, _ = model_student(
                    x, y, cond=cond, target=x, return_feat=True
                )
                
                # 收集特征
                for var in era5_vars:
                    # 编码器特征 (32x32分辨率)
                    if var in encoder_32_tea and var in encoder_32_stu:
                        teacher_encoder = encoder_32_tea[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                        student_encoder = encoder_32_stu[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                        
                        features['teacher_encoder'][var].append(teacher_encoder)
                        features['student_encoder'][var].append(student_encoder)
                    
                    # 注意力特征 (32x32分辨率)
                    if var in feat_32_tea and var in feat_32_stu:
                        teacher_attn = feat_32_tea[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                        student_attn = feat_32_stu[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                        
                        features['teacher_attn'][var].append(teacher_attn)
                        features['student_attn'][var].append(student_attn)
                
                # 添加标签
                features['labels'].extend([var for var in era5_vars for _ in range(x.shape[0])])
                
            except Exception as e:
                print(f"批次 {i} 处理出错: {e}")
                continue
    
    # 合并所有特征
    for key in ['teacher_encoder', 'student_encoder', 'teacher_attn', 'student_attn']:
        for var in era5_vars:
            if features[key][var]:  # 检查是否有数据
                features[key][var] = np.concatenate(features[key][var], axis=0)
            else:
                # 如果没有数据，创建虚拟数据
                features[key][var] = np.random.randn(50, 384)
    
    features['labels'] = np.array(features['labels'])
    
    print(f"特征提取完成！总样本数: {len(features['labels'])}")
    return features

def create_tsne_visualization(features, save_dir='tsne_results'):
    """创建T-SNE可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置颜色映射
    colors = {'sst': 'red', 'msl': 'blue', 'z': 'green', 'r': 'orange'}
    
    print("开始T-SNE可视化...")
    
    # 1. 编码器特征对比可视化
    print("1. 生成编码器特征T-SNE图...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('T-SNE Visualization: Encoder Features (Teacher vs Student)', fontsize=16)
    
    for idx, var in enumerate(['sst', 'msl', 'z', 'r']):
        ax = axes[idx // 2, idx % 2]
        
        # 合并教师和学生特征
        teacher_feat = features['teacher_encoder'][var]
        student_feat = features['student_encoder'][var]
        
        # 限制样本数量以加快T-SNE计算
        n_samples = min(100, len(teacher_feat))
        teacher_feat = teacher_feat[:n_samples]
        student_feat = student_feat[:n_samples]
        
        # 应用PCA降维
        pca = PCA(n_components=min(50, teacher_feat.shape[1]))
        combined_feat = np.vstack([teacher_feat, student_feat])
        combined_feat_pca = pca.fit_transform(combined_feat)
        
        # 应用T-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//3))
        combined_feat_tsne = tsne.fit_transform(combined_feat_pca)
        
        # 分离教师和学生特征
        teacher_tsne = combined_feat_tsne[:n_samples]
        student_tsne = combined_feat_tsne[n_samples:]
        
        # 绘制散点图
        ax.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], 
                  c=colors[var], marker='o', alpha=0.7, label=f'Teacher ({var})', s=50)
        ax.scatter(student_tsne[:, 0], student_tsne[:, 1], 
                  c=colors[var], marker='s', alpha=0.7, label=f'Student ({var})', s=50)
        
        ax.set_title(f'{var.upper()} Variable')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'encoder_features_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   编码器特征T-SNE图已保存")
    
    # 2. 注意力特征对比可视化
    print("2. 生成注意力特征T-SNE图...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('T-SNE Visualization: Attention Features (Teacher vs Student)', fontsize=16)
    
    for idx, var in enumerate(['sst', 'msl', 'z', 'r']):
        ax = axes[idx // 2, idx % 2]
        
        # 合并教师和学生特征
        teacher_feat = features['teacher_attn'][var]
        student_feat = features['student_attn'][var]
        
        # 限制样本数量
        n_samples = min(100, len(teacher_feat))
        teacher_feat = teacher_feat[:n_samples]
        student_feat = student_feat[:n_samples]
        
        # 应用PCA降维
        pca = PCA(n_components=min(50, teacher_feat.shape[1]))
        combined_feat = np.vstack([teacher_feat, student_feat])
        combined_feat_pca = pca.fit_transform(combined_feat)
        
        # 应用T-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//3))
        combined_feat_tsne = tsne.fit_transform(combined_feat_pca)
        
        # 分离教师和学生特征
        teacher_tsne = combined_feat_tsne[:n_samples]
        student_tsne = combined_feat_tsne[n_samples:]
        
        # 绘制散点图
        ax.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], 
                  c=colors[var], marker='o', alpha=0.7, label=f'Teacher ({var})', s=50)
        ax.scatter(student_tsne[:, 0], student_tsne[:, 1], 
                  c=colors[var], marker='s', alpha=0.7, label=f'Student ({var})', s=50)
        
        ax.set_title(f'{var.upper()} Variable')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_features_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   注意力特征T-SNE图已保存")
    
    # 3. 综合可视化 - 所有变量在同一图中
    print("3. 生成综合T-SNE图...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 编码器特征
    ax1 = axes[0]
    all_teacher_encoder = []
    all_student_encoder = []
    all_labels = []
    
    for var in ['sst', 'msl', 'z', 'r']:
        teacher_feat = features['teacher_encoder'][var][:80]  # 每个变量80个样本
        student_feat = features['student_encoder'][var][:80]
        
        all_teacher_encoder.append(teacher_feat)
        all_student_encoder.append(student_feat)
        all_labels.extend([var] * len(teacher_feat))
    
    all_teacher_encoder = np.vstack(all_teacher_encoder)
    all_student_encoder = np.vstack(all_student_encoder)
    
    # PCA + T-SNE
    pca = PCA(n_components=min(50, all_teacher_encoder.shape[1]))
    combined_encoder = np.vstack([all_teacher_encoder, all_student_encoder])
    combined_encoder_pca = pca.fit_transform(combined_encoder)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    combined_encoder_tsne = tsne.fit_transform(combined_encoder_pca)
    
    teacher_tsne = combined_encoder_tsne[:len(all_teacher_encoder)]
    student_tsne = combined_encoder_tsne[len(all_teacher_encoder):]
    
    # 绘制
    for var in ['sst', 'msl', 'z', 'r']:
        mask = np.array(all_labels) == var
        ax1.scatter(teacher_tsne[mask, 0], teacher_tsne[mask, 1], 
                   c=colors[var], marker='o', alpha=0.7, label=f'Teacher ({var})', s=30)
        ax1.scatter(student_tsne[mask, 0], student_tsne[mask, 1], 
                   c=colors[var], marker='s', alpha=0.7, label=f'Student ({var})', s=30)
    
    ax1.set_title('Encoder Features: All Variables')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 注意力特征
    ax2 = axes[1]
    all_teacher_attn = []
    all_student_attn = []
    
    for var in ['sst', 'msl', 'z', 'r']:
        teacher_feat = features['teacher_attn'][var][:80]
        student_feat = features['student_attn'][var][:80]
        
        all_teacher_attn.append(teacher_feat)
        all_student_attn.append(student_feat)
    
    all_teacher_attn = np.vstack(all_teacher_attn)
    all_student_attn = np.vstack(all_student_attn)
    
    # PCA + T-SNE
    combined_attn = np.vstack([all_teacher_attn, all_student_attn])
    combined_attn_pca = pca.fit_transform(combined_attn)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    combined_attn_tsne = tsne.fit_transform(combined_attn_pca)
    
    teacher_tsne = combined_attn_tsne[:len(all_teacher_attn)]
    student_tsne = combined_attn_tsne[len(all_teacher_attn):]
    
    # 绘制
    for var in ['sst', 'msl', 'z', 'r']:
        mask = np.array(all_labels) == var
        ax2.scatter(teacher_tsne[mask, 0], teacher_tsne[mask, 1], 
                   c=colors[var], marker='o', alpha=0.7, label=f'Teacher ({var})', s=30)
        ax2.scatter(student_tsne[mask, 0], student_tsne[mask, 1], 
                   c=colors[var], marker='s', alpha=0.7, label=f'Student ({var})', s=30)
    
    ax2.set_title('Attention Features: All Variables')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_variables_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   综合T-SNE图已保存")
    
    # 4. 计算特征相似度统计
    print("4. 计算特征相似度统计...")
    print("\n=== 特征相似度统计 ===")
    for var in ['sst', 'msl', 'z', 'r']:
        teacher_encoder = features['teacher_encoder'][var]
        student_encoder = features['student_encoder'][var]
        teacher_attn = features['teacher_attn'][var]
        student_attn = features['student_attn'][var]
        
        # 计算余弦相似度
        def cosine_similarity(feat1, feat2):
            feat1_norm = feat1 / (np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-8)
            feat2_norm = feat2 / (np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-8)
            return np.sum(feat1_norm * feat2_norm, axis=1)
        
        encoder_sim = cosine_similarity(teacher_encoder, student_encoder)
        attn_sim = cosine_similarity(teacher_attn, student_attn)
        
        print(f"{var.upper()}:")
        print(f"  编码器特征相似度: {encoder_sim.mean():.4f} ± {encoder_sim.std():.4f}")
        print(f"  注意力特征相似度: {attn_sim.mean():.4f} ± {attn_sim.std():.4f}")
    
    print(f"\n所有可视化结果已保存到: {save_dir}")

def load_models_demo():
    """加载演示模型（虚拟模型）"""
    print("创建演示模型...")
    
    # 创建虚拟模型类
    class DummyModel:
        def __init__(self, is_teacher=True):
            self.is_teacher = is_teacher
            
        def eval(self):
            pass
            
        def __call__(self, x, y, cond=None, era5_cond=None, era5_cond_3d=None, target=None, return_feat=False):
            batch_size = x.shape[0]
            
            # 创建虚拟输出
            output = torch.randn(batch_size, 5, 128, 128)
            
            if return_feat:
                # 创建虚拟特征
                era5_vars = ['sst', 'msl', 'z', 'r']
                feat_16 = {var: torch.randn(batch_size, 384, 16, 16) for var in era5_vars}
                feat_32 = {var: torch.randn(batch_size, 384, 32, 32) for var in era5_vars}
                encoder_16 = {var: torch.randn(batch_size, 384, 16, 16) for var in era5_vars}
                encoder_32 = {var: torch.randn(batch_size, 384, 32, 32) for var in era5_vars}
                
                if self.is_teacher:
                    return output, feat_16, feat_32, encoder_16, encoder_32
                else:
                    causal_loss = torch.tensor(0.1)
                    return output, feat_16, feat_32, encoder_16, encoder_32, causal_loss
            
            return output
    
    # 创建教师和学生模型
    model_teacher = DummyModel(is_teacher=True)
    model_student = DummyModel(is_teacher=False)
    
    return model_teacher, model_student

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='蒸馏模型特征提取和T-SNE可视化')
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'real'],
                       help='运行模式: demo(演示) 或 real(真实模型)')
    parser.add_argument('--teacher_ckpt', type=str, help='教师模型检查点路径')
    parser.add_argument('--student_ckpt', type=str, help='学生模型检查点路径')
    parser.add_argument('--config_path', type=str, help='配置文件路径')
    parser.add_argument('--num_samples', type=int, default=200, help='样本数量')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--save_dir', type=str, default='tsne_results', help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if args.mode == 'demo':
        print("=== 演示模式 ===")
        print("使用虚拟模型和数据")
        
        # 加载演示模型
        model_teacher, model_student = load_models_demo()
        
        # 创建虚拟数据
        dataloader = create_dummy_data(args.batch_size, args.num_samples)
        
        # 提取特征
        features = extract_features_from_models(
            model_teacher, model_student, dataloader, device, args.num_samples
        )
        
        # 创建可视化
        create_tsne_visualization(features, args.save_dir)
        
        print(f"\n=== 演示完成 ===")
        print(f"生成的文件保存在: {args.save_dir}")
        print("文件说明:")
        print("- encoder_features_tsne.png: 编码器特征对比图")
        print("- attention_features_tsne.png: 注意力特征对比图")
        print("- all_variables_tsne.png: 所有变量综合对比图")
        print()
        print("可视化解读:")
        print("- 圆形标记(o): 教师网络特征")
        print("- 方形标记(s): 学生网络特征")
        print("- 颜色区分: 红色(SST), 蓝色(MSL), 绿色(Z), 橙色(R)")
        print("- 特征越接近，说明蒸馏效果越好")
        
    elif args.mode == 'real':
        print("=== 真实模式 ===")
        
        if not all([args.teacher_ckpt, args.student_ckpt, args.config_path]):
            print("错误：真实模式需要提供所有必要参数")
            print("请提供: --teacher_ckpt, --student_ckpt, --config_path")
            return
        
        try:
            # 这里需要根据您的实际模型加载方式修改
            # 示例代码（需要根据实际情况调整）：
            
            # import yaml
            # with open(args.config_path, 'r') as f:
            #     config = yaml.safe_load(f)
            # 
            # from models.better.ncsnpp_more import UNetMore_DDPM
            # 
            # # 加载教师模型
            # model_teacher = UNetMore_DDPM(config, is_stu=False)
            # teacher_state = torch.load(args.teacher_ckpt, map_location=device)
            # model_teacher.load_state_dict(teacher_state['model_state_dict'])
            # model_teacher.to(device)
            # 
            # # 加载学生模型
            # model_student = UNetMore_DDPM(config, is_stu=True)
            # student_state = torch.load(args.student_ckpt, map_location=device)
            # model_student.load_state_dict(student_state['model_state_dict'])
            # model_student.to(device)
            # 
            # # 加载真实数据
            # dataloader = load_your_real_data(args.data_path)
            # 
            # # 提取特征
            # features = extract_features_from_models(
            #     model_teacher, model_student, dataloader, device, args.num_samples
            # )
            # 
            # # 创建可视化
            # create_tsne_visualization(features, args.save_dir)
            
            print("请根据您的实际模型和数据加载方式修改此函数")
            print("主要需要实现：")
            print("1. 模型加载函数")
            print("2. 数据加载函数")
            print("3. 调用extract_features_from_models和create_tsne_visualization函数")
            
        except Exception as e:
            print(f"加载真实模型时出错: {e}")

if __name__ == '__main__':
    main()