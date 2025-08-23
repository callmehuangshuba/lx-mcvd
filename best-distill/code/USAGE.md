# 蒸馏模型T-SNE可视化使用指南

## 概述

本指南将帮助您运行T-SNE可视化脚本来观察蒸馏模型的特征对齐效果。通过这些可视化，您可以验证：

1. **特征对齐验证**: 学生网络特征是否成功靠近教师网络
2. **模态融合验证**: 不同ERA5变量的特征是否保持结构化分布
3. **知识迁移验证**: 从有ERA5条件的教师网络到无ERA5的学生网络的知识传递效果

## 文件说明

### 主要脚本文件

1. **`quick_visualization.py`** - 快速分析脚本（无需额外依赖）
   - 模拟蒸馏效果
   - 生成文本可视化
   - 提供分析报告

2. **`extract_and_visualize.py`** - 完整可视化脚本
   - 支持演示模式和真实模式
   - 生成T-SNE可视化图片
   - 计算特征相似度统计

3. **`run_real_visualization.py`** - 真实模型可视化脚本
   - 专门用于真实模型的可视化
   - 需要提供模型检查点和配置文件

## 使用方法

### 方法1: 快速分析（推荐新手）

```bash
cd best-distill/code
python3 quick_visualization.py
```

**输出结果:**
- 控制台显示蒸馏效果分析
- 生成 `distillation_analysis.txt` 分析报告

**特点:**
- 无需安装额外包
- 快速了解模型结构
- 模拟蒸馏效果

### 方法2: 演示模式可视化

```bash
# 安装依赖包
pip3 install numpy matplotlib scikit-learn tqdm

# 运行演示模式
python3 extract_and_visualize.py --mode demo
```

**输出结果:**
- `tsne_results/encoder_features_tsne.png` - 编码器特征对比图
- `tsne_results/attention_features_tsne.png` - 注意力特征对比图
- `tsne_results/all_variables_tsne.png` - 所有变量综合对比图

### 方法3: 真实模型可视化

```bash
python3 run_real_visualization.py \
    --teacher_ckpt /path/to/teacher/checkpoint.pth \
    --student_ckpt /path/to/student/checkpoint.pth \
    --config_path /path/to/config.yaml \
    --num_samples 100 \
    --batch_size 8 \
    --save_dir tsne_results \
    --device cuda
```

**参数说明:**
- `--teacher_ckpt`: 教师模型检查点路径
- `--student_ckpt`: 学生模型检查点路径
- `--config_path`: 配置文件路径
- `--num_samples`: 样本数量（默认100）
- `--batch_size`: 批次大小（默认8）
- `--save_dir`: 保存目录（默认tsne_results）
- `--device`: 设备（默认cuda）

## 可视化解读

### 图片说明

1. **编码器特征T-SNE图**
   - 显示教师和学生网络编码器特征的对齐情况
   - 每个变量（SST、MSL、Z、R）单独显示

2. **注意力特征T-SNE图**
   - 显示教师和学生网络注意力特征的对齐情况
   - 反映跨模态注意力机制的效果

3. **综合T-SNE图**
   - 所有变量在同一图中显示
   - 便于比较不同变量的对齐效果

### 标记说明

- **圆形标记(o)**: 教师网络特征
- **方形标记(s)**: 学生网络特征
- **颜色区分**: 
  - 红色: SST (海表温度)
  - 蓝色: MSL (平均海平面气压)
  - 绿色: Z (位势高度)
  - 橙色: R (相对湿度)

### 效果评估

**相似度解读:**
- **0.9-1.0**: 极好的蒸馏效果，特征几乎完全对齐
- **0.8-0.9**: 良好的蒸馏效果，特征高度相似
- **0.7-0.8**: 中等蒸馏效果，特征基本对齐
- **<0.7**: 蒸馏效果较差，需要进一步优化

**可视化效果:**
- **极好**: T-SNE图中教师(圆形)和学生(方形)特征点高度聚集
- **良好**: T-SNE图中教师和学生特征点形成清晰的聚类
- **中等**: T-SNE图中教师和学生特征点部分重叠
- **较差**: T-SNE图中教师和学生特征点分散分布

## 示例输出

### 快速分析输出示例

```
=== 蒸馏效果模拟结果 ===
变量            编码器特征      注意力特征      对齐效果
------------------------------------------------------------
SST             0.9139          0.9013          极好
MSL             0.8775          0.9112          良好
Z               0.9236          0.9338          极好
R               0.9392          0.9043          极好

=== 蒸馏效果总结报告 ===
平均编码器特征相似度: 0.9136
平均注意力特征相似度: 0.9126
整体蒸馏效果: 极好

结论:
✓ 学生网络特征成功对齐到教师网络
✓ 跨模态注意力机制工作良好
✓ 因果特征分解有效
✓ 知识迁移成功
```

### 特征相似度统计示例

```
=== 特征相似度统计 ===
SST:
  编码器特征相似度: 0.9139 ± 0.0456
  注意力特征相似度: 0.9013 ± 0.0523
MSL:
  编码器特征相似度: 0.8775 ± 0.0612
  注意力特征相似度: 0.9112 ± 0.0489
Z:
  编码器特征相似度: 0.9236 ± 0.0398
  注意力特征相似度: 0.9338 ± 0.0412
R:
  编码器特征相似度: 0.9392 ± 0.0367
  注意力特征相似度: 0.9043 ± 0.0491
```

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 解决方案：安装依赖包
   pip3 install numpy matplotlib scikit-learn tqdm
   ```

2. **模型加载失败**
   ```bash
   # 检查文件路径是否正确
   ls -la /path/to/your/checkpoint.pth
   ls -la /path/to/your/config.yaml
   ```

3. **内存不足**
   ```bash
   # 减少样本数量
   python3 run_real_visualization.py --num_samples 50 --batch_size 4
   ```

4. **CUDA内存不足**
   ```bash
   # 使用CPU或减少批次大小
   python3 run_real_visualization.py --device cpu --batch_size 2
   ```

### 性能优化

1. **减少样本数量**: 使用 `--num_samples 50` 而不是默认的100
2. **减少批次大小**: 使用 `--batch_size 4` 而不是默认的8
3. **使用CPU**: 如果GPU内存不足，使用 `--device cpu`

## 高级用法

### 自定义可视化

您可以修改脚本中的参数来自定义可视化：

```python
# 修改颜色映射
colors = {'sst': 'red', 'msl': 'blue', 'z': 'green', 'r': 'orange'}

# 修改图片大小
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 修改T-SNE参数
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
```

### 批量处理

如果您有多个模型需要比较，可以创建批处理脚本：

```bash
#!/bin/bash
for model in model1 model2 model3; do
    python3 run_real_visualization.py \
        --teacher_ckpt ${model}_teacher.pth \
        --student_ckpt ${model}_student.pth \
        --config_path config.yaml \
        --save_dir results_${model}
done
```

## 联系支持

如果您在使用过程中遇到问题，请：

1. 检查错误信息
2. 确认文件路径正确
3. 验证依赖包已安装
4. 查看本文档的故障排除部分

## 更新日志

- **v1.0**: 初始版本，支持基本的T-SNE可视化
- **v1.1**: 添加快速分析脚本
- **v1.2**: 优化性能和错误处理
- **v1.3**: 添加批量处理支持