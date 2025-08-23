# 蒸馏模型特征对齐T-SNE可视化使用指南

## 概述

本文档详细介绍了如何使用T-SNE可视化验证多模态蒸馏模型的特征对齐效果。该可视化工具可以帮助验证学生网络是否成功学习到了教师网络的特征表示。

## 模型架构要点

### 1. 核心创新技术

#### 1.1 多尺度特征蒸馏
- **16×16尺度**：高层抽象特征蒸馏，捕获全局语义信息
- **32×32尺度**：深层语义特征蒸馏，保留空间细节信息

#### 1.2 跨模态注意力机制
```python
# 标准跨模态注意力
CrossAttention(dim=384, heads=6, dim_head=64)

# 因果跨模态注意力（带掩码）
CrossAttention_Casual(dim=384, heads=6, dim_head=64, mask_ratio=0.7)
```

#### 1.3 条件互信息损失
```python
class ConditionalMutualInfoLoss:
    # 最大化因果特征与目标的相关性
    # 最小化非因果特征与目标的相关性
    # 通过margin loss确保特征区分度
```

#### 1.4 自适应权重融合
- ERA5变量（SST、MSL、Z、R）的动态权重调节
- 基于softmax的归一化权重分配

### 2. 损失函数设计

总损失函数包含四个主要组成部分：

```python
loss = loss_score + loss_distill + 10*loss_feat_encoder + lambda_adaptive*causal_loss
```

1. **基础生成损失 (loss_score)**：标准扩散模型去噪损失
2. **输出蒸馏损失 (loss_distill)**：教师-学生输出对齐
3. **多层特征蒸馏损失 (loss_feat_encoder)**：中间层特征对齐（权重×10）
4. **自适应因果损失 (causal_loss)**：动态调节的因果推理损失

## 安装依赖

```bash
# 安装必要的Python包
pip install torch torchvision
pip install numpy matplotlib seaborn
pip install scikit-learn
pip install tqdm pyyaml
pip install pandas
```

## 使用方法

### 1. 基础使用

```bash
cd best-distill/code
python tsne_visualization.py \
    --config configs/TCG.yml \
    --ckpt path/to/your/checkpoint.pth \
    --save_dir tsne_results \
    --max_samples 1000
```

### 2. 参数说明

- `--config`: 模型配置文件路径
- `--ckpt`: 预训练模型检查点路径
- `--data_path`: 数据集路径
- `--save_dir`: 结果保存目录
- `--max_samples`: 最大分析样本数
- `--perplexity`: T-SNE的困惑度参数（默认30）
- `--device`: 计算设备（cuda/cpu）

### 3. 高级配置

```bash
# 使用更多样本和更高困惑度
python tsne_visualization.py \
    --config configs/TCG.yml \
    --ckpt path/to/checkpoint.pth \
    --max_samples 2000 \
    --perplexity 50 \
    --device cuda

# 保存到自定义目录
python tsne_visualization.py \
    --config configs/TCG.yml \
    --ckpt path/to/checkpoint.pth \
    --save_dir /path/to/custom/results
```

## 输出结果说明

### 1. 可视化图片

生成的T-SNE可视化包含以下几类图片：

#### 1.1 特征尺度对比
- `tsne_16_features.png`: 16×16尺度特征分布
- `tsne_32_features.png`: 32×32尺度特征分布
- `tsne_encoder_16_features.png`: 编码器16×16特征
- `tsne_encoder_32_features.png`: 编码器32×32特征

#### 1.2 每张图包含四个子图
1. **教师vs学生特征分布**：整体对比教师和学生网络的特征聚类
2. **ERA5变量分布**：不同气象变量（SST、MSL、Z、R）的特征分布
3. **距离分析**：到各自聚类中心的距离分布直方图
4. **特征对齐度**：教师特征到最近学生特征的距离分布

### 2. 数值分析结果

#### 2.1 对齐指标文件
- `alignment_metrics_16.txt`: 16×16尺度的对齐指标
- `alignment_metrics_32.txt`: 32×32尺度的对齐指标
- `alignment_metrics_encoder_16.txt`: 编码器16×16指标
- `alignment_metrics_encoder_32.txt`: 编码器32×32指标

#### 2.2 综合分析报告
- `feature_alignment_analysis.txt`: 包含以下指标：
  - **余弦相似度**：特征方向的相似性（越接近1越好）
  - **欧氏距离**：特征空间的绝对距离（越小越好）
  - **特征形状**：确认特征维度匹配

### 3. 评估标准

#### 3.1 优秀的特征对齐应该满足：
1. **余弦相似度 > 0.8**：表示特征方向高度一致
2. **平均教师-学生距离 < 2.0**：表示特征空间接近
3. **聚类方差比 < 1.5**：学生和教师聚类紧密度相似
4. **特征重叠度高**：T-SNE图中教师和学生特征点混合分布

#### 3.2 问题诊断：
- **余弦相似度 < 0.5**：特征学习不充分，需要增加蒸馏权重
- **平均距离 > 5.0**：特征空间差异过大，检查损失函数权重
- **明显的分离聚类**：蒸馏效果差，需要调整训练策略

## 代码结构说明

### 1. 主要函数

#### 1.1 `load_model_and_config()`
- 加载YAML配置文件
- 初始化NCSNpp_stu模型
- 加载预训练权重

#### 1.2 `extract_features_batch()`
- 批量提取教师和学生网络特征
- 在16×16和32×32两个尺度提取
- 同时提取注意力特征和编码器特征

#### 1.3 `create_tsne_visualization()`
- 执行T-SNE降维
- 生成多种可视化图表
- 计算对齐度指标

#### 1.4 `create_detailed_alignment_analysis()`
- 计算余弦相似度和欧氏距离
- 生成详细的数值分析报告

### 2. 特征提取流程

```python
# 教师网络（包含ERA5条件）
h_teacher, feat_16_tea, feat_32_tea, encoder_16_tea, encoder_32_tea = model(
    perturbed_x, time_steps, cond=cond, 
    era5_cond=era5_cond, era5_cond_3d=era5_cond_3d,
    return_feat=True
)

# 学生网络（不包含ERA5条件）
h_student, feat_16_stu, feat_32_stu, encoder_16_stu, encoder_32_stu, _ = model(
    perturbed_x, time_steps, cond=cond, 
    era5_cond=None, era5_cond_3d=None,  # 关键：学生网络不使用ERA5
    return_feat=True, target=x
)
```

## 实验建议

### 1. 对比实验
- **实验1**：训练前vs训练后的特征对齐度
- **实验2**：不同蒸馏权重下的对齐效果
- **实验3**：有无因果损失的对比

### 2. 超参数调优
- **T-SNE困惑度**：30-100之间调整
- **样本数量**：500-2000样本进行分析
- **特征维度**：可以尝试PCA预降维

### 3. 定量分析
```python
# 建议的评估阈值
excellent_alignment = {
    'cosine_similarity': > 0.85,
    'mean_distance': < 1.5,
    'cluster_variance_ratio': < 1.3
}

good_alignment = {
    'cosine_similarity': > 0.7,
    'mean_distance': < 3.0,
    'cluster_variance_ratio': < 2.0
}
```

## 故障排除

### 1. 常见错误

#### 1.1 CUDA内存不足
```bash
# 减少批次大小或样本数
python tsne_visualization.py --max_samples 500
```

#### 1.2 模型加载失败
```bash
# 检查配置文件和检查点路径
python tsne_visualization.py --config configs/TCG.yml --ckpt None
```

#### 1.3 特征提取错误
- 确认模型支持return_feat=True参数
- 检查ERA5数据格式是否匹配

### 2. 性能优化

#### 2.1 加速建议
- 使用GPU加速（--device cuda）
- 减少样本数量进行快速测试
- 使用较小的T-SNE困惑度

#### 2.2 内存优化
- 批量处理大数据集
- 及时清理中间变量
- 使用mixed precision训练

## 结果解读指南

### 1. 理想的可视化结果
- 教师和学生特征在T-SNE空间中**高度重叠**
- 不同ERA5变量保持**各自的聚类结构**
- 学生特征**紧密围绕**教师特征分布

### 2. 需要改进的情况
- 教师和学生特征**明显分离**
- 学生特征聚类**过于松散**
- 不同变量间**边界模糊**

通过这个T-SNE可视化工具，您可以定量和定性地评估蒸馏模型的特征学习效果，为模型优化提供重要的指导信息。