# ELA Attention

## 🔍 Overview | 概述

**English**

ELA is a lightweight attention mechanism that generates two **1D attention maps**:
- one along the height direction \((H)\)
- one along the width direction \((W)\)

It uses **mean pooling** to squeeze one spatial dimension, then applies a shared **grouped 1D convolution** to model local dependencies along that axis.  
Finally, the two attention maps are broadcast-multiplied back to the original feature map.

**中文**

ELA 是一种轻量注意力机制，它生成两条 **一维注意力图**：
- 高度方向注意力 \((H)\)
- 宽度方向注意力 \((W)\)

它先用均值池化压缩一个空间维度，再用共享的 **分组一维卷积（Conv1D）** 在该方向上建模局部依赖，最后将两条注意力图广播回原特征图并逐元素相乘。

---

## ⚙️ Input / Output | 输入输出

- Input:  \(X \in \mathbb{R}^{B \times C \times H \times W}\)
- Output: \(Y \in \mathbb{R}^{B \times C \times H \times W}\)

---

## 🧠 How It Works | 工作流程

### 1️⃣ Directional Mean Pooling | 方向均值池化

**English**

Pool along width (keep height information):

\[
X_h = mean_W(X) \in \mathbb{R}^{B \times C \times H}
\]

Pool along height (keep width information):

\[
X_w = mean_H(X) \in \mathbb{R}^{B \times C \times W}
\]

**中文**

沿宽度做均值池化（保留高度信息）：

\[
X_h = mean_W(X) \in \mathbb{R}^{B \times C \times H}
\]

沿高度做均值池化（保留宽度信息）：

\[
X_w = mean_H(X) \in \mathbb{R}^{B \times C \times W}
\]

---

### 2️⃣ Shared Grouped Conv1D | 共享分组一维卷积

**English**

Apply a shared Conv1D on both sequences:

\[
\tilde{X}_h = Conv1D(X_h), \quad \tilde{X}_w = Conv1D(X_w)
\]

This captures local context along H and W with kernel size \(k\).

**中文**

对两条序列共享一套 Conv1D：

\[
\tilde{X}_h = Conv1D(X_h), \quad \tilde{X}_w = Conv1D(X_w)
\]

卷积核大小 \(k\) 控制沿 H/W 的局部感受野。

---

### 3️⃣ Normalize + Sigmoid | 归一化 + Sigmoid

**English**

GroupNorm stabilizes the signal, sigmoid converts it to attention weights:

\[
A_h = \sigma(GN(\tilde{X}_h)) \in \mathbb{R}^{B \times C \times H \times 1}
\]
\[
A_w = \sigma(GN(\tilde{X}_w)) \in \mathbb{R}^{B \times C \times 1 \times W}
\]

**中文**

GroupNorm 用于稳定训练，Sigmoid 输出权重：

\[
A_h = \sigma(GN(\tilde{X}_h)) \in \mathbb{R}^{B \times C \times H \times 1}
\]
\[
A_w = \sigma(GN(\tilde{X}_w)) \in \mathbb{R}^{B \times C \times 1 \times W}
\]

---

### 4️⃣ Apply Attention | 应用注意力

**English**

Broadcast and multiply:

\[
Y = X \otimes A_h \otimes A_w
\]

**中文**

通过广播逐元素相乘：

\[
Y = X \otimes A_h \otimes A_w
\]

---

## 🧾 Attention Tensors | 注意力张量在哪里？

- Height attention:
  - shape: `(B, C, H, 1)`
  - in code: `x_h = sigmoid(GN(conv1(x_h))).view(B, C, H, 1)`

- Width attention:
  - shape: `(B, C, 1, W)`
  - in code: `x_w = sigmoid(GN(conv1(x_w))).view(B, C, 1, W)`

---

## 🔑 Key Parameters | 关键参数（phi 的含义）

ELA uses `phi` to control kernel size and grouping strategy:

| phi | kernel_size | groups | GN groups | Notes (English) | 中文说明 |
|-----|------------:|-------:|----------:|------------------|---------|
| T | 5 | C | 32 | very lightweight (depthwise-like) | 超轻量（接近深度卷积） |
| B | 7 | C | 16 | larger kernel for more context | 更大卷积核，上下文更强 |
| S | 5 | C/8 | 16 | more channel mixing than T/B | 通道混合更强 |
| L | 7 | C/8 | 16 | strongest context, for large nets | 更强上下文，适合大网络 |

---

## ✅ Pros | 优点

**English**
- Very lightweight (especially T/B variants)
- Encodes directional context along H and W
- GroupNorm works well with small batch sizes

**中文**
- 很轻量（尤其 T/B）
- 能沿 H/W 捕获方向上下文
- GroupNorm 对小 batch 更友好

---

## ⚠️ Cons | 局限

**English**
- Does not build full 2D spatial attention map (only separable H and W)
- Choice of `phi` impacts performance and cost

**中文**
- 不是完整的 2D 空间注意力（只做 H/W 可分离）
- `phi` 的选择会影响效果与开销

---

## 🧩 When to Use | 适用场景

**English**
- Lightweight CNN backbones
- Detection/segmentation where directional cues help
- When you want a CA-like effect but even simpler/cheaper

**中文**
- 轻量 CNN backbone
- 检测/分割中需要方向性信息
- 想要类似 CA 的效果但更便宜的方案

---

## 🧪 Minimal Example | 最小示例

```python
import torch
from attention.ela import ELA

x = torch.randn(1, 64, 128, 128)
ela = ELA(in_channels=64, phi='T')
y = ela(x)
print(y.shape)  # (1, 64, 128, 128)