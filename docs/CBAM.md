# CBAM Attention (Convolutional Block Attention Module)

## 🔍 Overview | 概述

**English**

CBAM (Convolutional Block Attention Module) applies attention in two sequential stages:
1) **Channel Attention**: learns *which feature channels* are important  
2) **Spatial Attention**: learns *which spatial locations* are important  

It improves feature representation by emphasizing informative features and suppressing irrelevant ones.

**中文**

CBAM（卷积块注意力模块）按顺序执行两种注意力：
1) **通道注意力**：学习“哪些通道更重要”  
2) **空间注意力**：学习“哪些位置更重要”  

通过增强关键信息、抑制无用信息来提升特征表达能力。

---

## 🎯 Motivation | 设计动机

**English**

CNN features contain redundant information. SE focuses only on channel attention, but ignores spatial importance.  
CBAM extends SE by additionally applying spatial attention, which is especially beneficial for detection/segmentation tasks.

**中文**

CNN 特征中包含冗余信息。SE 只关注通道注意力，忽略空间位置信息。  
CBAM 在通道注意力之外加入空间注意力，尤其适合检测/分割等对“位置”敏感的任务。

---

## ⚙️ How It Works | 工作原理

Given input feature map:

\[
X \in \mathbb{R}^{B \times C \times H \times W}
\]

CBAM performs:

### 1️⃣ Channel Attention | 通道注意力

**English**

CBAM uses both global average pooling and global max pooling to generate channel descriptors, then uses a shared MLP to produce channel weights.

**中文**

CBAM 同时使用全局平均池化与全局最大池化生成通道描述，再通过共享 MLP 输出通道权重。

Channel attention map:

\[
M_c \in \mathbb{R}^{B \times C \times 1 \times 1}
\]

Apply:

\[
X' = M_c \otimes X
\]

---

### 2️⃣ Spatial Attention | 空间注意力

**English**

CBAM compresses channel information using average and max operations along channels, concatenates them, then uses a convolution to generate spatial weights.

**中文**

CBAM 在通道维做平均与最大操作，拼接后通过卷积生成空间注意力图。

Spatial attention map:

\[
M_s \in \mathbb{R}^{B \times 1 \times H \times W}
\]

Apply:

\[
X'' = M_s \otimes X'
\]

---

## 🧾 Attention Tensors | 注意力张量在哪里？

- Channel attention weights:
  - **Shape:** `(B, C, 1, 1)`
  - Produced by: `ChannelAttention(x)`

- Spatial attention weights:
  - **Shape:** `(B, 1, H, W)`
  - Produced by: `SpatialAttention(x)`

---

## 🔑 Key Parameters | 关键参数

| Parameter | Description (English) | 中文说明 |
|----------|------------------------|----------|
| in_channels | Number of input channels (C) | 输入通道数 |
| reduction_ratio | Bottleneck ratio for channel MLP | 通道注意力压缩比 |
| kernel_size | Conv kernel size in spatial attention | 空间注意力卷积核大小 |

---

## ✅ Advantages | 优点

**English**

✔ Improves feature representation  
✔ Captures both channel and spatial importance  
✔ Works well for detection and segmentation  
✔ Lightweight overhead compared to many alternatives  

**中文**

✔ 提升特征表达能力  
✔ 同时关注通道与空间重要性  
✔ 对检测/分割任务效果好  
✔ 额外开销相对较小  

---

## ⚠️ Limitations | 局限

**English**

✖ Adds extra computation compared to SE  
✖ Spatial attention may be less beneficial for some purely channel-dominant tasks  

**中文**

✖ 相比 SE 增加一定计算量  
✖ 某些任务可能主要受通道影响，空间注意力增益有限  

---

## 🧩 When to Use | 适用场景

**English**

- Object detection (YOLO-style backbones)
- Semantic segmentation
- Medical imaging
- Any CNN backbone where spatial focus matters

**中文**

- 目标检测（如 YOLO 的 backbone）
- 语义分割
- 医学图像
- 任何“空间位置重要”的 CNN 特征增强场景

---

## 📚 Reference | 参考文献

Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018  
https://arxiv.org/abs/1807.06521

---

## 🧪 Minimal PyTorch Example | 最小PyTorch示例

```python
import torch
from attention.cbam import CBAM

x = torch.randn(1, 64, 32, 32)
cbam = CBAM(64)
y = cbam(x)
print(y.shape)  # (1, 64, 32, 32)