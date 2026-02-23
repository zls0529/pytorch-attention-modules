# SK Attention (Selective Kernel Attention)

## 🔍 Overview | 概述

**English**

Selective Kernel (SK) Attention enables a CNN to dynamically adapt its receptive field by selecting among multiple convolution branches with different kernel sizes.  
Instead of using a fixed kernel (e.g., 3×3), SK learns **branch-wise attention weights** and produces an adaptive weighted combination of multi-scale features.

**中文**

SK（Selective Kernel）注意力机制使 CNN 能够根据输入内容动态调整感受野：  
通过多个不同卷积核分支提取多尺度特征，并学习 **分支选择权重**，最终对各尺度特征进行加权融合，从而实现自适应卷积核选择。

---

## 🎯 Motivation | 设计动机

**English**

Objects in images vary in scale. A fixed kernel size may be suboptimal:
- Small objects benefit from small receptive fields (e.g., 3×3)
- Large objects benefit from larger receptive fields (e.g., 5×5 / 7×7)

SK attention allows the model to **choose the most suitable receptive field** for each input.

**中文**

图像目标存在明显尺度变化，固定卷积核可能不够灵活：
- 小目标更需要小感受野（如 3×3）
- 大目标更需要大感受野（如 5×5 / 7×7）

SK 通过注意力机制让模型自动选择合适的感受野。

---

## ⚙️ How It Works | 工作原理

Given input feature map:

\[
X \in \mathbb{R}^{B \times C \times H \times W}
\]

SK performs:

### 1️⃣ Multi-branch Convolution | 多分支多尺度卷积

**English**

Apply K convolution branches with different kernel sizes:

\[
U_i = f_i(X), \quad i=1..K
\]

Each branch output:

\[
U_i \in \mathbb{R}^{B \times C \times H \times W}
\]

**中文**

使用 K 个不同卷积核分支提取多尺度特征：

\[
U_i = f_i(X), \quad i=1..K
\]

每个分支输出：

\[
U_i \in \mathbb{R}^{B \times C \times H \times W}
\]

---

### 2️⃣ Fuse Features | 特征融合

**English**

Aggregate multi-scale features by summation:

\[
U = \sum_{i=1}^{K} U_i
\]

**中文**

将多尺度特征相加融合：

\[
U = \sum_{i=1}^{K} U_i
\]

---

### 3️⃣ Squeeze (Global Pooling) | 压缩（全局池化）

**English**

Global average pooling produces a channel descriptor:

\[
S = GAP(U) \in \mathbb{R}^{B \times C}
\]

**中文**

对融合特征做全局平均池化，得到通道描述：

\[
S = GAP(U) \in \mathbb{R}^{B \times C}
\]

---

### 4️⃣ Gating (Dim Reduction) | 门控（降维）

**English**

Reduce channels via a shared FC:

\[
Z = W S \in \mathbb{R}^{B \times d}
\]

where:

\[
d = \max(L, C / reduction)
\]

**中文**

通过共享全连接层进行降维：

\[
Z = W S \in \mathbb{R}^{B \times d}
\]

其中：

\[
d = \max(L, C / reduction)
\]

---

### 5️⃣ Branch-wise Soft Selection | 分支 Softmax 选择权重

**English**

For each branch i, generate a channel-wise weight vector:

\[
a_i = W_i Z \in \mathbb{R}^{B \times C}
\]

Normalize across branches using Softmax:

\[
\alpha_i = \frac{\exp(a_i)}{\sum_{j=1}^{K} \exp(a_j)}
\]

So that:

\[
\sum_{i=1}^{K}\alpha_i = 1
\]

**中文**

为每个分支生成通道权重：

\[
a_i = W_i Z \in \mathbb{R}^{B \times C}
\]

通过 Softmax 在分支维度归一化：

\[
\alpha_i = \frac{\exp(a_i)}{\sum_{j=1}^{K} \exp(a_j)}
\]

满足：

\[
\sum_{i=1}^{K}\alpha_i = 1
\]

---

### 6️⃣ Weighted Feature Aggregation | 加权融合输出

**English**

Final output is the weighted sum of branch features:

\[
V = \sum_{i=1}^{K} \alpha_i \otimes U_i
\]

**中文**

最终输出为分支特征的加权和：

\[
V = \sum_{i=1}^{K} \alpha_i \otimes U_i
\]

---

## 🧾 Attention Tensor | 注意力张量在哪里？

**English**

- Branch attention weights tensor shape: `(K, B, C, 1, 1)`
- Softmax is applied on the **K dimension** (across branches)
- These weights determine how much each kernel branch contributes.

**中文**

- 分支注意力权重张量形状：`(K, B, C, 1, 1)`
- Softmax 在 **K 维度** 上进行（跨分支归一化）
- 权重用于决定每个卷积核分支的贡献比例。

---

## 🔑 Key Parameters | 关键参数

| Parameter | Description (English) | 中文说明 |
|----------|------------------------|----------|
| channel | Number of channels (C) | 输入/输出通道数 |
| kernels | Kernel sizes for branches | 分支卷积核大小列表 |
| reduction | Reduction ratio for gating | 降维比例 |
| group | Groups in Conv2d | 分组卷积组数（可设为 C 做深度卷积） |
| L | Minimum reduced dimension | 降维后的最小通道数下限 |

---

## ✅ Advantages | 优点

**English**

✔ Adaptive receptive field selection  
✔ Stronger multi-scale representation  
✔ Works well when object scales vary significantly  

**中文**

✔ 自适应选择感受野  
✔ 多尺度表征能力强  
✔ 对尺度变化大的任务更有效  

---

## ⚠️ Limitations | 局限

**English**

✖ Multiple branches increase computation and memory  
✖ More complex than SE/ECA for lightweight deployment  

**中文**

✖ 多分支带来更高计算量和显存开销  
✖ 相比 SE/ECA 更重，不一定适合极轻量场景  

---

## 🧩 When to Use | 适用场景

**English**

- Image classification with scale variation
- Detection/segmentation backbones needing adaptive receptive fields
- Complex visual tasks with multi-scale patterns

**中文**

- 目标尺度变化明显的分类任务
- 需要自适应感受野的检测/分割 backbone
- 多尺度模式丰富的视觉任务

---

## 📚 Reference | 参考文献

Li et al., *Selective Kernel Networks*, CVPR 2019  
https://arxiv.org/abs/1903.06586

---

## 🧪 Minimal PyTorch Example | 最小示例

```python
import torch
from attention.skattention import SKAttention

x = torch.randn(1, 64, 64, 64)
sk = SKAttention(channel=64, kernels=[1, 3, 5, 7], reduction=8)
y = sk(x)
print(y.shape)  # (1, 64, 64, 64)