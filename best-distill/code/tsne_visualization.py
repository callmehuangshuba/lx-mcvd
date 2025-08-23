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

def extract_features(model_teacher, model_student, dataloader, device, num_samples=1000):
    """
    从教师模型和学生模型中提取特征
    """
    model_teacher.eval()
    model_student.eval()
    
    features = {
        'teacher_encoder': defaultdict(list),
        'student_encoder': defaultdict(list),
        'teacher_attn': defaultdict(list),
        'student_attn': defaultdict(list),
        'labels': []
    }
    
    era5_vars = ['sst', 'msl', 'z', 'r']
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if i * batch[0].shape[0] >= num_samples:
                break
                
            # 假设batch格式为 [x, y, cond, era5_cond, era5_cond_3d]
            x, y, cond, era5_cond, era5_cond_3d = [b.to(device) for b in batch]
            
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
                features['teacher_encoder'][var].append(
                    encoder_32_tea[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                )
                features['student_encoder'][var].append(
                    encoder_32_stu[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                )
                
                # 注意力特征 (32x32分辨率)
                features['teacher_attn'][var].append(
                    feat_32_tea[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                )
                features['student_attn'][var].append(
                    feat_32_stu[var].mean(dim=(2, 3)).cpu().numpy()  # [B, 384]
                )
            
            features['labels'].extend([var] * x.shape[0] for var in era5_vars)
    
    # 合并所有特征
    for key in ['teacher_encoder', 'student_encoder', 'teacher_attn', 'student_attn']:
        for var in era5_vars:
            features[key][var] = np.concatenate(features[key][var], axis=0)
    
    features['labels'] = np.concatenate(features['labels'], axis=0)
    
    return features

def create_tsne_visualization(features, save_dir='tsne_results'):
    """
    创建T-SNE可视化
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置颜色映射
    colors = {'sst': 'red', 'msl': 'blue', 'z': 'green', 'r': 'orange'}
    markers = {'teacher': 'o', 'student': 's'}
    
    # 1. 编码器特征对比可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('T-SNE Visualization: Encoder Features (Teacher vs Student)', fontsize=16)
    
    for idx, var in enumerate(['sst', 'msl', 'z', 'r']):
        ax = axes[idx // 2, idx % 2]
        
        # 合并教师和学生特征
        teacher_feat = features['teacher_encoder'][var]
        student_feat = features['student_encoder'][var]
        
        # 限制样本数量以加快T-SNE计算
        n_samples = min(500, len(teacher_feat))
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
    
    # 2. 注意力特征对比可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('T-SNE Visualization: Attention Features (Teacher vs Student)', fontsize=16)
    
    for idx, var in enumerate(['sst', 'msl', 'z', 'r']):
        ax = axes[idx // 2, idx % 2]
        
        # 合并教师和学生特征
        teacher_feat = features['teacher_attn'][var]
        student_feat = features['student_attn'][var]
        
        # 限制样本数量
        n_samples = min(500, len(teacher_feat))
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
    
    # 3. 模态融合可视化 - 所有变量在同一图中
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 编码器特征
    ax1 = axes[0]
    all_teacher_encoder = []
    all_student_encoder = []
    all_labels = []
    
    for var in ['sst', 'msl', 'z', 'r']:
        teacher_feat = features['teacher_encoder'][var][:200]  # 每个变量200个样本
        student_feat = features['student_encoder'][var][:200]
        
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
        teacher_feat = features['teacher_attn'][var][:200]
        student_feat = features['student_attn'][var][:200]
        
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
    
    # 4. 计算特征距离统计
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

def main():
    parser = argparse.ArgumentParser(description='T-SNE visualization for distillation model')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Path to teacher model checkpoint')
    parser.add_argument('--student_ckpt', type=str, required=True, help='Path to student model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='tsne_results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # 这里需要根据您的具体数据加载器和模型加载方式进行调整
    # 以下是示例代码，您需要根据实际情况修改
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型（需要根据您的模型加载方式调整）
    # model_teacher = load_teacher_model(args.teacher_ckpt)
    # model_student = load_student_model(args.student_ckpt)
    
    # 加载数据（需要根据您的数据加载方式调整）
    # dataloader = load_test_data(args.data_path)
    
    # 提取特征
    # features = extract_features(model_teacher, model_student, dataloader, device, args.num_samples)
    
    # 创建可视化
    # create_tsne_visualization(features, args.save_dir)
    
    print("请根据您的具体模型和数据加载方式修改此脚本")
    print("主要需要实现：")
    print("1. 模型加载函数")
    print("2. 数据加载函数")
    print("3. 调用extract_features和create_tsne_visualization函数")

if __name__ == '__main__':
    main()