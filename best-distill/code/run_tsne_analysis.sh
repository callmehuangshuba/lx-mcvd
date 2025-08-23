#!/bin/bash

# T-SNE特征对齐分析运行脚本
# 使用方法: ./run_tsne_analysis.sh [checkpoint_path] [config_path] [output_dir]

# 设置默认参数
CONFIG_PATH=${2:-"configs/TCG.yml"}
CHECKPOINT_PATH=${1:-"checkpoints/model_best.pth"}
OUTPUT_DIR=${3:-"tsne_analysis_results"}
MAX_SAMPLES=${4:-1000}
DEVICE=${5:-"cuda"}

echo "==================================="
echo "T-SNE特征对齐分析"
echo "==================================="
echo "配置文件: $CONFIG_PATH"
echo "模型检查点: $CHECKPOINT_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "最大样本数: $MAX_SAMPLES"
echo "计算设备: $DEVICE"
echo "==================================="

# 检查文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行T-SNE分析
python tsne_visualization.py \
    --config "$CONFIG_PATH" \
    --ckpt "$CHECKPOINT_PATH" \
    --save_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    --perplexity 30

# 检查是否成功完成
if [ $? -eq 0 ]; then
    echo "==================================="
    echo "分析完成！结果保存在: $OUTPUT_DIR"
    echo "==================================="
    echo "生成的文件:"
    ls -la "$OUTPUT_DIR"
    echo "==================================="
    
    # 显示对齐分析结果
    if [ -f "$OUTPUT_DIR/feature_alignment_analysis.txt" ]; then
        echo "特征对齐分析摘要:"
        cat "$OUTPUT_DIR/feature_alignment_analysis.txt"
    fi
else
    echo "分析失败，请检查错误信息"
    exit 1
fi