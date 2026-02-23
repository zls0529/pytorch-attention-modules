# Coordinate Attention (CA)

## 🔍 Overview | 概述

**English**

Coordinate Attention (CA) is an attention mechanism that embeds **positional (coordinate) information** into channel attention.  
Unlike SE, which compresses spatial dimensions into a single scalar per channel, CA performs **directional pooling** along height and width separately, allowing the network to capture long-range dependencies while preserving spatial location cues.

**中文**

Coordinate Attention（坐标注意力，简称 CA）是一种将**位置信息（坐标信息）**融入通道注意力的机制。  
与 SE 直接把空间维度压缩成每个通道一个标量不同，CA 会沿 **高度方向** 和 **宽度方向** 分别做池化，从而在捕获长距离依赖的同时保留位置信息。

---

## 🎯 Motivation | 设计动机

**English**

SE provides channel attention but loses positional information because it uses global pooling over both H and W.  
For tasks like detection and segmentation, location matters. CA is designed to keep coordinate-aware information while remaining lightweight.

**中文**

SE 虽然能学习通道重要性，但全局池化会丢失位置信息（H/W 都被压缩掉）。  
在检测、分割等任务中，位置很重要。CA 的目标是在保持轻量化的同时，保留坐标感知能力。

---

## ⚙️ How It Works | 工作原理

Given input feature map:

\[
X \in \mathbb{R}^{B \times C \times H \times W}
\]

CA computes two attention maps:

- Height attention: \(A_h \in \mathbb{R}^{B \times C \times H \times 1}\)
- Width attention:  \(A_w \in \mathbb{R}^{B \times C \times 1 \times W}\)

Final output:

\[
Y = X \otimes A_h \otimes A_w
\]

---

### 1️⃣ Directional Pooling | 方向池化

**English**

Pool along width to keep height context:

\[
X_h = AvgPool_{W}(X) \in \mathbb{R}^{B \times C \times H \times 1}
\]

Pool along height to keep width context:

\[
X_w = AvgPool_{H}(X) \in \mathbb{R}^{B \times C \times 1 \times W}
\]

**中文**

沿宽度池化（保留高度信息）：

\[
X_h = AvgPool_{W}(X) \in \mathbb{R}^{B \times C \times H \times 1}
\]

沿高度池化（保留宽度信息）：

\[
X_w = AvgPool_{H}(X) \in \mathbb{R}^{B \times C \times 1 \times W}
\]

---

### 2️⃣ Concatenation and Shared Transform | 拼接与共享变换

**English**

Concatenate along the spatial axis to form a combined descriptor:

\[
Y = Concat(X_h, X_w) \in \mathbb{R}^{B \times C \times (H+W) \times 1}
\]

Then apply a bottleneck transform (1×1 conv + BN + activation):

\[
Y' \in \mathbb{R}^{B \times d \times (H+W) \times 1}, \quad d=\max(8, C/r)
\]

**中文**

将高度与宽度描述在空间长度维拼接：

\[
Y = Concat(X_h, X_w) \in \mathbb{R}^{B \times C \times (H+W) \times 1}
\]

再通过瓶颈变换（1×1 卷积 + BN + 激活）：

\[
Y' \in \mathbb{R}^{B \times d \times (H+W) \times 1}, \quad d=\max(8, C/r)
\]

---

### 3️⃣ Split and Generate Attention Maps | 切分并生成注意力图

**English**

Split back into height and width parts:

- \(Y_h \in \mathbb{R}^{B \times d \times H \times 1}\)
- \(Y_w \in \mathbb{R}^{B \times d \times 1 \times W}\)

Then generate attention maps using two 1×1 conv layers:

\[
A_h = \sigma(Conv_h(Y_h)),\quad A_w = \sigma(Conv_w(Y_w))
\]

**中文**

把融合后的特征按高度与宽度切分回去：

- \(Y_h \in \mathbb{R}^{B \times d \times H \times 1}\)
- \(Y_w \in \mathbb{R}^{B \times d \times 1 \times W}\)

再用两个 1×1 卷积生成注意力图：

\[
A_h = \sigma(Conv_h(Y_h)),\quad A_w = \sigma(Conv_w(Y_w))
\]

---

### 4️⃣ Apply Attention | 应用注意力

**English**

Broadcast and multiply with the input:

\[
Out = X \otimes A_h \otimes A_w
\]

**中文**

通过广播机制将 \(A_h\) 和 \(A_w\) 乘回输入特征：

\[
Out = X \otimes A_h \otimes A_w
\]

---

## 🧾 Attention Tensors | 注意力张量在哪里？

- Height attention weights:
  - Shape: `(B, C, H, 1)`
  - In code: `a_h = conv_h(x_h).sigmoid()`

- Width attention weights:
  - Shape: `(B, C, 1, W)`
  - In code: `a_w = conv_w(x_w).sigmoid()`

These two tensors jointly reweight the input feature map.

---

## 🔑 Key Parameters | 关键参数

| Parameter | Description (English) | 中文说明 |
|----------|------------------------|----------|
| inp / oup | Input / output channels | 输入/输出通道数（通常相同） |
| reduction | Reduction ratio for bottleneck | 瓶颈降维比例 |
| mip | Bottleneck channels (`max(8, C//reduction)`) | 中间通道数（最少为 8） |

---

## ✅ Advantages | 优点

**English**

✔ Preserves positional information better than SE  
✔ Lightweight and easy to integrate  
✔ Effective for detection/segmentation where location matters

**中文**

✔ 比 SE 更能保留位置信息  
✔ 轻量、易集成  
✔ 对检测/分割等位置敏感任务更有效

---

## ⚠️ Limitations | 局限

**English**

✖ Slightly more computation than SE/ECA  
✖ Still not as global as full self-attention (Transformer)

**中文**

✖ 相比 SE/ECA 略增加计算  
✖ 不如 Transformer 那样全局建模强

---

## 🧩 When to Use | 适用场景

**English**

- Object detection backbones / necks
- Semantic segmentation
- Lightweight models needing coordinate awareness (e.g., MobileNet variants)

**中文**

- 目标检测的 backbone/neck
- 语义分割
- 需要位置信息但又要轻量的模型（如 MobileNet 系列）

---

## 📚 Reference | 参考文献

Hou et al., *Coordinate Attention for Efficient Mobile Network Design*, CVPR 2021  
https://arxiv.org/abs/2103.02907

---

## 🧪 Minimal PyTorch Example | 最小示例

```python
import torch
from attention.ca import CoordAtt

x = torch.randn(1, 64, 32, 32)
ca = CoordAtt(64, 64, reduction=32)
y = ca(x)
print(y.shape)  # (1, 64, 32, 32)