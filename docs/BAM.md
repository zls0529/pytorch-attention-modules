# BAM (Bottleneck Attention Module)

## 🔍 Overview | 概述

**English**

BAM (Bottleneck Attention Module) is a lightweight attention block designed to be inserted at the **bottleneck** stage of CNNs.  
It combines **Channel Attention** and **Spatial Attention** in a gating manner:

- Channel branch learns “which channels are important”
- Spatial branch learns “which spatial locations are important”

Then BAM merges them and applies a residual gating:
\[
Out = (1 + \sigma(CA + SA)) \otimes X
\]

**中文**

BAM（瓶颈注意力模块）是一种常用于 CNN **瓶颈层**的轻量注意力结构。  
它通过两条分支同时建模：

- 通道注意力：哪些通道重要
- 空间注意力：哪些位置重要

最后将两者融合，并采用残差门控：
\[
Out = (1 + \sigma(CA + SA)) \otimes X
\]

---

## ⚙️ Input / Output | 输入输出

- Input:  \(X \in \mathbb{R}^{B \times C \times H \times W}\)
- Output: \(Out \in \mathbb{R}^{B \times C \times H \times W}\)

---

## 🧠 How It Works | 工作流程

### 1️⃣ Channel Attention Branch | 通道注意力分支

**English**

1) Global Average Pooling:
\[
(B,C,H,W)\rightarrow(B,C,1,1)\rightarrow(B,C)
\]

2) MLP (multiple FC + BN + ReLU) to model channel dependencies:
\[
(B,C)\rightarrow(B,C)
\]

3) Broadcast back to spatial size:
\[
(B,C)\rightarrow(B,C,1,1)\rightarrow(B,C,H,W)
\]

**中文**

1) 全局平均池化压缩空间信息：
\[
(B,C,H,W)\rightarrow(B,C,1,1)\rightarrow(B,C)
\]

2) 多层 MLP（FC + BN + ReLU）学习通道间关系：
\[
(B,C)\rightarrow(B,C)
\]

3) 广播回原空间尺寸：
\[
(B,C)\rightarrow(B,C,1,1)\rightarrow(B,C,H,W)
\]

**Attention tensor shape | 注意力张量形状**

- Channel branch output: `(B, C, H, W)`（实际是 `(B, C, 1, 1)` 广播出来）

---

### 2️⃣ Spatial Attention Branch | 空间注意力分支

**English**

1) 1×1 conv reduces channels:
\[
(B,C,H,W)\rightarrow(B,C/r,H,W)
\]

2) Several 3×3 **dilated conv** blocks (BN + ReLU), capturing larger receptive fields.

3) Final 1×1 conv to 1 channel:
\[
(B,C/r,H,W)\rightarrow(B,1,H,W)
\]

4) Broadcast to `(B, C, H, W)`.

**中文**

1) 1×1 卷积降维：
\[
(B,C,H,W)\rightarrow(B,C/r,H,W)
\]

2) 多层 3×3 **空洞卷积**（BN + ReLU），扩大感受野捕获上下文。

3) 最后 1×1 卷积输出单通道空间图：
\[
(B,C/r,H,W)\rightarrow(B,1,H,W)
\]

4) 广播成 `(B, C, H, W)`。

**Attention tensor shape | 注意力张量形状**

- Spatial branch raw map: `(B, 1, H, W)`  
- Broadcasted map: `(B, C, H, W)`

---

### 3️⃣ Merge and Gate | 融合与门控

**English**

Merge two branches and apply sigmoid:

\[
W = \sigma(CA + SA)
\]

Then residual gating:

\[
Out = (1 + W)\otimes X
\]

Why `1 + W`?
- keeps identity path
- prevents over-suppression early in training

**中文**

两分支相加后 sigmoid 得到权重：

\[
W = \sigma(CA + SA)
\]

再用残差门控：

\[
Out = (1 + W)\otimes X
\]

为什么是 `1 + W`？
- 保留主干信息（identity）
- 避免训练初期把特征压得太狠

---

## 🔑 Key Parameters | 关键参数

| Parameter | Description (English) | 中文说明 |
|----------|------------------------|----------|
| channel | Number of channels C | 通道数 |
| reduction | Reduction ratio r | 降维比例 |
| num_layers | Depth of CA/SA subnets | 子网络层数 |
| dia_val | Dilation rate for SA | 空洞卷积扩张率 |

---

## ✅ Pros | 优点

**English**
- Channel + Spatial attention together
- Dilated conv helps capture wider context
- Residual gating stabilizes optimization

**中文**
- 同时建模通道与空间重要性
- 空洞卷积利于捕获更大上下文
- 残差门控训练更稳定

---

## ⚠️ Cons | 局限

**English**
- Heavier than SE/ECA (extra conv blocks)
- Spatial branch adds computation (especially with multiple layers)

**中文**
- 比 SE/ECA 更重（多卷积分支）
- 空间分支计算量更大（层数越多越明显）

---

## 🧩 When to Use | 适用场景

**English**
- Bottleneck stages of CNN backbones
- Detection / segmentation where spatial context matters
- When you can afford slightly more compute than SE/ECA

**中文**
- CNN backbone 的瓶颈层
- 对空间上下文敏感的检测/分割任务
- 计算量允许略增加时

---

## 📚 Reference | 参考

Park et al., *BAM: Bottleneck Attention Module*, BMVC 2018  
https://arxiv.org/abs/1807.06514

---

## 🧪 Minimal Example | 最小示例

```python
import torch
from attention.bam import BAMBlock

x = torch.randn(2, 256, 32, 32)
bam = BAMBlock(channel=256, reduction=16, dia_val=2)
y = bam(x)
print(y.shape)  # (2, 256, 32, 32)