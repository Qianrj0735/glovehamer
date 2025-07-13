# HAMER LoRA Training

这个项目为HAMER（Hand Mesh Recovery）模型添加了LoRA（Low-Rank Adaptation）支持，使得可以高效地对模型进行微调。

## 概述

LoRA是一种参数高效的微调方法，通过在预训练模型的线性层中添加低秩矩阵来实现适应，而不需要更新所有模型参数。这种方法可以：

1. **显著减少可训练参数数量**（通常少于1%的原始参数）
2. **降低GPU内存需求**
3. **加快训练速度**
4. **保持与全量微调相当的性能**

## 项目结构

```
hamer/
├── models/
│   ├── components/
│   │   └── lora.py              # LoRA核心实现
│   ├── hamer_lora.py            # 带LoRA的HAMER模型
│   └── __init__.py              # 更新了导入
├── configs_hydra/
│   ├── train_lora.yaml          # LoRA训练配置
│   └── experiment/
│       └── hamer_lora.yaml      # LoRA实验配置
├── train_lora.py                # LoRA训练脚本
├── demo_lora.py                 # LoRA推理演示脚本
└── README_LORA.md               # 本文件
```

## LoRA实现细节

### 核心组件

1. **LoRALayer**: 基础LoRA层实现
2. **LoRALinear**: 针对Linear层的LoRA适配
3. **LoRAConv2d**: 针对Conv2d层的LoRA适配
4. **HAMERLoRA**: 集成LoRA的HAMER模型

### 目标模块

LoRA被应用到以下模块：

#### Backbone (ViT)
- `qkv`: 注意力机制的Query、Key、Value投影
- `proj`: 注意力输出投影
- `fc1`, `fc2`: MLP层

#### MANO Head (Transformer Decoder)
- `to_qkv`: 自注意力QKV投影
- `to_out`: 注意力输出投影
- `to_kv`, `to_q`: 交叉注意力投影
- `decpose`, `decshape`, `deccam`: 解码器头部

#### Discriminator
- `D_conv1`, `D_conv2`: 卷积层
- `betas_fc1`, `betas_fc2`: Beta处理层
- `D_alljoints_fc1`, `D_alljoints_fc2`: 关节处理层

## 使用方法

### 1. 安装依赖

确保已安装原始HAMER的所有依赖。

### 2. 训练LoRA模型

基础训练命令：

```bash
python train_lora.py
```

使用自定义配置：

```bash
python train_lora.py experiment=hamer_lora
```

指定LoRA参数：

```bash
python train_lora.py \
    lora.rank=8 \
    lora.alpha=16.0 \
    lora.dropout=0.1 \
    TRAIN.LR=5e-5
```

从预训练模型开始：

```bash
python train_lora.py \
    lora.pretrained_model_path=path/to/hamer_checkpoint.ckpt
```

### 3. 推理

使用训练好的LoRA权重进行推理：

```bash
python demo_lora.py \
    --lora_weights logs/lora_runs/YYYY-MM-DD_HH-MM-SS/final_lora_weights.pth \
    --img_folder example_data \
    --out_folder demo_out_lora \
    --save_mesh \
    --side_view
```

## 配置说明

### LoRA配置参数

```yaml
lora:
  rank: 4                    # LoRA秩，控制瓶颈大小
  alpha: 1.0                 # LoRA缩放因子
  dropout: 0.0               # LoRA dropout率
  target_modules:            # 目标模块配置
    backbone: [...]
    mano_head: [...]
    discriminator: [...]
  pretrained_model_path: null      # 预训练模型路径
  save_lora_separately: true       # 是否单独保存LoRA权重
  save_interval: 1000              # LoRA权重保存间隔
```

### 训练配置

```yaml
TRAIN:
  LR: 1e-4                   # 学习率（建议比全量训练低）
  WEIGHT_DECAY: 0.01         # 权重衰减
  GRAD_CLIP_VAL: 1.0         # 梯度裁剪

GENERAL:
  LOG_STEPS: 100             # 日志记录间隔
  CHECKPOINT_STEPS: 1000     # 检查点保存间隔
  CHECKPOINT_SAVE_TOP_K: 3   # 保存的检查点数量
```

## 输出文件

训练过程中会生成以下文件：

```
logs/lora_runs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/                    # 完整模型检查点
├── lora_weights/                   # LoRA权重
│   ├── lora_weights_step_001000.pth
│   ├── lora_weights_step_002000.pth
│   └── ...
├── final_lora_weights.pth          # 最终LoRA权重
├── tensorboard/                    # TensorBoard日志
├── model_config.yaml               # 模型配置
├── lora_config.yaml                # LoRA配置
└── dataset_config.yaml             # 数据集配置
```

## 性能对比

| 方法 | 可训练参数 | GPU内存 | 训练时间 | 性能 |
|------|------------|---------|----------|------|
| 全量微调 | 100% | 高 | 慢 | 基准 |
| LoRA (rank=4) | ~0.5% | 低 | 快 | 相当 |
| LoRA (rank=8) | ~1% | 低 | 快 | 更好 |

## 最佳实践

1. **选择合适的rank**：
   - rank=4: 内存友好，适合小数据集
   - rank=8: 平衡性能和效率
   - rank=16: 更高性能，但参数增加

2. **学习率设置**：
   - 使用比全量微调低的学习率（1e-4 到 5e-5）
   - LoRA参数对学习率较敏感

3. **alpha参数**：
   - 通常设置为rank的1-2倍
   - alpha/rank控制LoRA的影响强度

4. **目标模块选择**：
   - 注意力层通常是最重要的
   - 可以根据任务需求调整目标模块

## 故障排除

### 常见问题

1. **内存不足**：
   - 减少batch_size
   - 降低LoRA rank
   - 使用mixed precision (precision=16)

2. **训练不收敛**：
   - 检查学习率设置
   - 增加LoRA rank
   - 调整alpha参数

3. **性能下降**：
   - 尝试不同的target_modules组合
   - 增加训练步数
   - 调整LoRA超参数

### 调试技巧

```python
# 检查LoRA参数数量
model.print_lora_info()

# 验证只有LoRA参数可训练
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable}/{total} ({trainable/total*100:.2f}%)")
```

## 扩展

### 自定义LoRA配置

可以通过修改配置文件或代码来：

1. 添加新的目标模块
2. 调整不同模块的LoRA参数
3. 实现自定义的LoRA变体

### 与其他技术结合

LoRA可以与以下技术结合使用：

1. **Gradient Checkpointing**: 进一步减少内存使用
2. **Mixed Precision**: 加速训练
3. **DeepSpeed**: 大规模训练优化

## 引用

如果您使用了这个LoRA实现，请引用原始HAMER论文以及LoRA论文：

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}

@inproceedings{hu2022lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
    booktitle={ICLR},
    year={2022}
}
```
