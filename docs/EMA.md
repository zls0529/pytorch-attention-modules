# EMA Attention

## 🔍 Overview | 概述

**English**

EMA is a **group-wise efficient attention** module.  
It splits channels into **G groups** to reduce computation, then for each group:

1) extracts **direction-aware signals** along height and width (via pooling + 1×1 conv)
2) builds two feature branches:
   - a gated + normalized branch
   - a local 3×3 conv branch
3) computes a **per-group spatial weight map** using two cross interactions (matmul)
4) reweights group features and reshapes back to the original tensor

**中文**

EMA 是一种 **按组（group-wise）高效注意力**模块。  
它把通道分成 **G 组**降低开销，然后在每一组内部：

1) 通过 H/W 方向池化 + 1×1 卷积提取方向信息  
2) 构建两条分支：
   - 门控 + 归一化分支
   - 3×3 局部分支  
3) 通过两项交互（矩阵乘法）计算**每组的空间权重图**  
4) 用权重重标定组内特征，再恢复回原形状

---

## ⚙️ Input / Output | 输入输出

- Input:  \(X \in \mathbb{R}^{B \times C \times H \times W}\)
- Output: \(Y \in \mathbb{R}^{B \times C \times H \times W}\)

---

## 🧠 Key Shapes | 关键形状

Let:
- \(G\) = number of groups (`factor`)
- \(C_g = C/G\)

Reshape:

\[
X \rightarrow X_g \in \mathbb{R}^{(B\cdot G) \times C_g \times H \times W}
\]

EMA computes weights per group:

\[
W_g \in \mathbb{R}^{(B\cdot G)\times 1 \times H \times W}
\]

Then:

\[
Y_g = X_g \otimes \sigma(W_g)
\Rightarrow
Y \in \mathbb{R}^{B \times C \times H \times W}
\]

---

## 🔄 How It Works | 工作流程

### 1️⃣ Grouping | 分组

**English**
Split channels into G groups:

\[
X_g = reshape(X) \in \mathbb{R}^{(B\cdot G)\times C_g \times H \times W}
\]

**中文**
把通道分成 G 组：

\[
X_g = reshape(X) \in \mathbb{R}^{(B\cdot G)\times C_g \times H \times W}
\]

---

### 2️⃣ Direction-aware gating | 方向门控（H/W）

**English**

Pool along width and height:

- \(X_h = pool_W(X_g) \in \mathbb{R}^{(B\cdot G)\times C_g \times H \times 1}\)
- \(X_w = pool_H(X_g) \in \mathbb{R}^{(B\cdot G)\times C_g \times 1 \times W}\)

Concatenate and fuse with 1×1 conv, then split back to obtain \(X_h\) and \(X_w\).

Then gate:

\[
X_1 = GN(X_g \otimes \sigma(X_h) \otimes \sigma(X_w))
\]

**中文**

沿宽度/高度池化：

- \(X_h = pool_W(X_g)\) 保留高度信息
- \(X_w = pool_H(X_g)\) 保留宽度信息

拼接后用 1×1 卷积融合，再切分回 \(X_h, X_w\)。

得到门控分支：

\[
X_1 = GN(X_g \otimes \sigma(X_h) \otimes \sigma(X_w))
\]

---

### 3️⃣ Local branch | 局部分支

**English**
Apply 3×3 conv:

\[
X_2 = Conv_{3\times3}(X_g)
\]

**中文**
3×3 卷积提取局部特征：

\[
X_2 = Conv_{3\times3}(X_g)
\]

---

### 4️⃣ Cross interactions to form weights | 交互生成权重

EMA builds weights using two terms:

- Term A: global descriptor from \(X_1\) interacts with spatial tokens of \(X_2\)
- Term B: global descriptor from \(X_2\) interacts with spatial tokens of \(X_1\)

Each term uses:

\[
softmax(AGP(\cdot)) \in \mathbb{R}^{(B\cdot G)\times 1 \times C_g}
\]

and flatten spatial tokens:

\[
(\cdot) \rightarrow \mathbb{R}^{(B\cdot G)\times C_g \times (H\cdot W)}
\]

Then:

\[
W_g = (Q_1 \cdot T_2 + Q_2 \cdot T_1) \rightarrow \mathbb{R}^{(B\cdot G)\times 1 \times H \times W}
\]

Finally apply sigmoid:

\[
Y_g = X_g \otimes \sigma(W_g)
\]

---

## 🧾 Where is the Attention? | 注意力在哪里？

- **Group weight map**:
  - shape: `((B*G), 1, H, W)`
  - in code: `weights = (...)reshape(B*G, 1, H, W)` then `weights.sigmoid()`

This is the actual attention map applied to each group.

---

## 🔑 Key Parameters | 关键参数

| Parameter | Description (English) | 中文说明 |
|----------|------------------------|----------|
| channels | Input channels C | 输入通道数 |
| factor | Number of groups G | 分组数（越大越省算力，但每组通道更少） |

**Practical tip | 实用建议**

- Ensure `channels % factor == 0` to avoid mismatch.
- Typical values: `factor=8/16/32` depending on C.

---

## ✅ Pros | 优点

**English**
- Efficient due to grouping
- Direction-aware gating helps encode coordinate-like cues
- Produces spatial weight maps without expensive NxN attention

**中文**
- 分组带来较高效率
- 方向门控保留一定坐标信息
- 不需要 NxN 注意力矩阵也能生成空间权重图

---

## ⚠️ Cons | 局限

**English**
- Grouping choice is sensitive: too many groups may reduce capacity
- More complex than SE/ECA/CA in implementation

**中文**
- 分组数选得不对会影响表达能力（组太多每组通道太少）
- 实现逻辑比 SE/ECA/CA 更复杂

---

## 🧩 When to Use | 适用场景

**English**
- Detection / segmentation backbones where spatial weighting helps
- When you want stronger attention than CA/ELA but still efficient
- Models with medium/large channel sizes (C>=64)

**中文**
- 需要空间加权的检测/分割 backbone
- 想比 CA/ELA 更强但仍希望高效
- 通道数较大的网络（C>=64）更适合

---

## 🧪 Minimal Example | 最小示例

```python
import torch
from attention.ema import EMA

x = torch.randn(1, 128, 64, 64)
ema = EMA(channels=128, factor=32)
y = ema(x)
print(y.shape)  # (1, 128, 64, 64)