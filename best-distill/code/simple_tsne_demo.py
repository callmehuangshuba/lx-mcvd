#!/usr/bin/env python3
"""
简化的T-SNE可视化演示脚本
不依赖PyTorch，直接生成示例数据来展示可视化效果
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

def create_demo_features():
    """创建演示用的特征数据"""
    print("生成演示特征数据...")
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 参数设置
    num_samples_per_var = 150
    feature_dim = 384
    era5_vars = ['sst', 'msl', 'z', 'r']
    
    features = {
        'teacher_encoder': {},
        'student_encoder': {},
        'teacher_attn': {},
        'student_attn': {},
        'labels': []
    }
    
    # 为每个变量生成特征
    for i, var in enumerate(era5_vars):
        # 为每个变量创建不同的特征分布
        base_center = np.array([i * 2, i * 1.5])  # 每个变量有不同的中心
        
        # 教师网络编码器特征 - 更集中的分布
        teacher_encoder = np.random.multivariate_normal(
            mean=[0] * feature_dim, 
            cov=np.eye(feature_dim) * 0.5, 
            size=num_samples_per_var
        )
        
        # 学生网络编码器特征 - 稍微分散的分布（模拟蒸馏效果）
        student_encoder = teacher_encoder + np.random.normal(0, 0.3, teacher_encoder.shape)
        
        # 教师网络注意力特征
        teacher_attn = np.random.multivariate_normal(
            mean=[0] * feature_dim, 
            cov=np.eye(feature_dim) * 0.4, 
            size=num_samples_per_var
        )
        
        # 学生网络注意力特征 - 更接近教师（模拟更好的蒸馏）
        student_attn = teacher_attn + np.random.normal(0, 0.2, teacher_attn.shape)
        
        # 存储特征
        features['teacher_encoder'][var] = teacher_encoder
        features['student_encoder'][var] = student_encoder
        features['teacher_attn'][var] = teacher_attn
        features['student_attn'][var] = student_attn
        
        # 添加标签
        features['labels'].extend([var] * num_samples_per_var)
    
    features['labels'] = np.array(features['labels'])
    
    print(f"生成完成！总样本数: {len(features['labels'])}")
    print(f"每个变量样本数: {num_samples_per_var}")
    print(f"特征维度: {feature_dim}")
    
    return features

def create_tsne_visualization(features, save_dir='tsne_demo_results'):
    """创建T-SNE可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置颜色映射
    colors = {'sst': 'red', 'msl': 'blue', 'z': 'green', 'r': 'orange'}
    markers = {'teacher': 'o', 'student': 's'}
    
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
        pca = PCA(n_components=50)
        combined_feat = np.vstack([teacher_feat, student_feat])
        combined_feat_pca = pca.fit_transform(combined_feat)
        
        # 应用T-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
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
        pca = PCA(n_components=50)
        combined_feat = np.vstack([teacher_feat, student_feat])
        combined_feat_pca = pca.fit_transform(combined_feat)
        
        # 应用T-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
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
    
    # 3. 模态融合可视化 - 所有变量在同一图中
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
    pca = PCA(n_components=50)
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
    
    # 4. 计算特征距离统计
    print("4. 计算特征相似度统计...")
    print("\n=== 特征距离统计 ===")
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

def main():
    """主函数"""
    print("=== 蒸馏模型T-SNE可视化演示 ===")
    print("本演示展示了蒸馏模型的特征对齐效果")
    print("通过T-SNE降维可视化验证：")
    print("1. 学生网络特征是否成功靠近教师网络")
    print("2. 不同ERA5变量的特征是否保持结构化分布")
    print("3. 知识迁移效果")
    print()
    
    # 生成演示特征数据
    features = create_demo_features()
    
    # 创建T-SNE可视化
    save_dir = 'tsne_demo_results'
    create_tsne_visualization(features, save_dir)
    
    print(f"\n=== 演示完成 ===")
    print(f"生成的文件保存在: {save_dir}")
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

if __name__ == '__main__':
    main()