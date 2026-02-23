# GAM Attention (Global Attention Mechanism)

## 🔍 Overview | 概述

**English**

GAM is an attention block that combines **channel attention** and **spatial attention**.
In this implementation:

1) **Channel attention is computed per spatial location** (token-wise):
   - treat each (h,w) position as a token of length C
   - apply an MLP on the channel dimension to produce a gating map

2) **Spatial attention is computed by large-kernel convolutions (7×7)**:
   - captures broader spatial context
   - produces another gating map

Final output:
\[
Out = X \otimes \sigma(A_c) \otimes \sigma(A_s)
\]

**中文**

GAM 是一种将 **通道注意力 + 空间注意力**结合的注意力模块。
在你这份实现中：

1) **通道注意力是“按像素/按token”计算的**：
   - 把每个 (h,w) 当成一个 token，长度为 C
   - 对通道维做 MLP，得到门控权重

2) **空间注意力通过 7×7 大卷积核实现**：
   - 捕获更大范围的空间上下文
   - 输出另一张门控权重图

最终输出：
\[
Out = X \otimes \sigma(A_c) \otimes \sigma(A_s)
\]

---

## ⚙️ Input / Output | 输入输出

- Input:  \(X \in \mathbb{R}^{B \times C \times H \times W}\)
- Output: \(Out \in \mathbb{R}^{B \times C \times H \times W}\)

---

## 🧠 Step-by-step | 工作流程

### 1️⃣ Token-wise Channel Attention | 按位置的通道注意力

**English**

Rearrange:

\[
(B,C,H,W)\rightarrow(B,H\cdot W,C)
\]

Apply MLP:

\[
MLP: C \rightarrow C/r \rightarrow C
\]

Reshape back:

\[
(B,H\cdot W,C)\rightarrow(B,C,H,W)
\]

Then sigmoid:

\[
A_c = \sigma(\cdot) \in \mathbb{R}^{B\times C\times H\times W}
\]

**中文**

变形为 token：

\[
(B,C,H,W)\rightarrow(B,H\cdot W,C)
\]

对每个 token 的通道维做 MLP：

\[
MLP: C \rightarrow C/r \rightarrow C
\]

再变回原形状并 sigmoid 得到权重：

\[
A_c \in \mathbb{R}^{B\times C\times H\times W}
\]

✅ 注意：这里的“通道注意力”不是 SE 那种 **全局共享的 (B,C,1,1)**，而是 **每个位置都有自己的通道权重 (B,C,H,W)**。

---

### 2️⃣ Apply Channel Gate | 应用通道门控

\[
X' = X \otimes A_c
\]

---

### 3️⃣ Spatial Attention (Large Kernel Conv) | 空间注意力（大卷积核）

**English**

Use 7×7 convs to produce a gating map:

\[
X' \rightarrow Conv7\times7 \rightarrow BN/ReLU \rightarrow Conv7\times7 \rightarrow BN
\]

Then:

\[
A_s = \sigma(\cdot) \in \mathbb{R}^{B\times C\times H\times W}
\]

**中文**

用 7×7 卷积产生空间门控图：

\[
X' \rightarrow Conv7\times7 \rightarrow BN/ReLU \rightarrow Conv7\times7 \rightarrow BN
\]

然后 sigmoid：

\[
A_s \in \mathbb{R}^{B\times C\times H\times W}
\]

---

### 4️⃣ Final Output | 最终输出

\[
Out = X' \otimes A_s
\]

---

## 🧾 Where is the Attention? | 注意力到底在哪？

- Channel attention map:
  - shape: `(B, C, H, W)`
  - in code: `x_channel_att = sigmoid(MLP(tokens)).reshape(...).permute(...)`

- Spatial attention map:
  - shape: `(B, C, H, W)`
  - in code: `x_spatial_att = sigmoid(spatial_attention(x))`

Both are **gating masks** multiplied element-wise with the feature map.

---

## 🔑 Key Parameters | 关键参数

| Parameter | Description (English) | 中文说明 |
|----------|------------------------|----------|
| in_channels | input channels C | 输入通道数 |
| rate | reduction ratio r | MLP/Conv 的缩减比例 |

---

## ✅ Pros | 优点

**English**
- Stronger than SE-style channel attention because it is **position-aware**
- 7×7 conv captures wider spatial context

**中文**
- 比 SE 更“细”：每个位置都有自己的通道权重（位置敏感）
- 7×7 大卷积核能捕获更大范围空间信息

---

## ⚠️ Cons | 局限

**English**
- More compute/memory than SE/ECA because channel attention runs on all H×W tokens
- Spatial attention outputs C channels (heavier than typical 1-channel spatial maps)

**中文**
- 比 SE/ECA 更耗算力/显存：MLP 作用于所有 H×W 位置
- 空间注意力输出的是 C 通道门控图（通常空间注意力只输出 1 通道）

---

## 🧩 When to Use | 适用场景

**English**
- When feature maps are small (e.g., bottleneck stages), token-wise MLP becomes affordable
- When you want attention to be both channel-selective and position-aware

**中文**
- 特征图较小的层（比如 backbone 后段），token-wise MLP 开销可接受
- 需要“通道选择 + 位置敏感”的注意力时

---

## 🧪 Minimal Example | 最小示例

```python
import torch
from attention.gam import GAM_Attention

x = torch.randn(2, 64, 14, 14)
gam = GAM_Attention(in_channels=64, rate=4)
y = gam(x)
print(y.shape)  # (2, 64, 14, 14)