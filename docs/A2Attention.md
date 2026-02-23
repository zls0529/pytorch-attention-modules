# A2 Attention (Double Attention)

## 🔍 Overview | 概述

**English**

A2 (Double Attention) is an attention mechanism that performs attention twice:
1) **Gather**: aggregate spatial information into a set of **global descriptors**
2) **Distribute**: redistribute the global descriptors back to every spatial position

This design captures long-range dependencies efficiently without explicitly building an NxN attention matrix like Non-local.

**中文**

A2（Double Attention，双重注意力）会执行两次注意力：
1) **Gather（汇聚）**：把全图信息汇聚成一组 **全局描述符**
2) **Distribute（分配）**：再把全局描述符分配回每个空间位置

它能高效建模长距离依赖，并且不像 Non-local 那样显式构造 NxN 的注意力矩阵。

---

## ⚙️ Input / Output | 输入输出

- Input:  \(X \in \mathbb{R}^{B \times C \times H \times W}\)
- Output:
  - If `reconstruct=True`: \(Out \in \mathbb{R}^{B \times C \times H \times W}\)
  - Else: \(Out \in \mathbb{R}^{B \times c_m \times H \times W}\)

---

## 🧠 Key Components | 核心张量

A2 uses three 1×1 conv projections:

- \(A = Conv_A(X)\)  → `(B, c_m, H, W)`
- \(B = Conv_B(X)\)  → `(B, c_n, H, W)`  (attention maps for gathering)
- \(V = Conv_V(X)\)  → `(B, c_n, H, W)`  (attention vectors for distributing)

Flatten spatial dimension \(N=H\cdot W\):

- \(A \rightarrow (B, c_m, N)\)
- \(B \rightarrow (B, c_n, N)\)
- \(V \rightarrow (B, c_n, N)\)

Softmax is applied along spatial dimension `N`:

\[
\hat{B} = softmax(B), \quad \hat{V}=softmax(V)
\]

---

## 🔄 How It Works | 工作流程

### 1️⃣ Gather (Feature Gating) | 汇聚（特征门控）

**English**

Compute global descriptors by aggregating A using attention maps:

\[
G = A \cdot \hat{B}^{T}
\]

Shapes:

- \(A: (B, c_m, N)\)
- \(\hat{B}^{T}: (B, N, c_n)\)
- \(G: (B, c_m, c_n)\)

**中文**

使用注意力图把 A 汇聚成全局描述符：

\[
G = A \cdot \hat{B}^{T}
\]

形状：

- \(A: (B, c_m, N)\)
- \(\hat{B}^{T}: (B, N, c_n)\)
- \(G: (B, c_m, c_n)\)

直观理解：  
`c_n` 可以看作 “全局描述符的数量/组数”，每组都从全图汇聚信息。

---

### 2️⃣ Distribute (Feature Distribution) | 分配（特征分布）

**English**

Distribute global descriptors back to spatial positions using attention vectors:

\[
Z = G \cdot \hat{V}
\]

Shapes:

- \(G: (B, c_m, c_n)\)
- \(\hat{V}: (B, c_n, N)\)
- \(Z: (B, c_m, N)\rightarrow(B, c_m, H, W)\)

**中文**

用注意力向量把全局描述符分配回每个空间位置：

\[
Z = G \cdot \hat{V}
\]

形状：

- \(G: (B, c_m, c_n)\)
- \(\hat{V}: (B, c_n, N)\)
- \(Z: (B, c_m, N)\rightarrow(B, c_m, H, W)\)

直观理解：  
每个位置会从若干组全局描述符中按权重“取信息”。

---

### 3️⃣ Optional Reconstruction | 可选通道重建

If `reconstruct=True`, apply a 1×1 conv to map:

\[
(B, c_m, H, W)\rightarrow(B, C, H, W)
\]

---

## 🧾 Where is the Attention? | 注意力在哪里？

**English**

A2 does NOT build an NxN attention matrix. Instead:
- `attention_maps` (B_hat): `(B, c_n, N)` controls **how to gather** global descriptors
- `attention_vectors` (V_hat): `(B, c_n, N)` controls **how to distribute** them back

**中文**

A2 不像 Non-local 那样构造 `(B, N, N)` 的注意力矩阵。  
它的注意力体现在两个张量上：

- `attention_maps`：`(B, c_n, N)` 决定如何从全图汇聚信息（Gather）
- `attention_vectors`：`(B, c_n, N)` 决定如何把信息分配回每个位置（Distribute）

---

## 🔑 Key Parameters | 关键参数

| Parameter | Description (English) | 中文说明 |
|----------|------------------------|----------|
| in_channels | Input channels C | 输入通道数 |
| c_m | Channels of A / output before reconstruct | A 的通道数 / 重建前输出通道 |
| c_n | Channels used for attention maps/vectors | 注意力映射/向量的通道数（可理解为“全局组数”） |
| reconstruct | Project back to C or not | 是否用 1×1 卷积重建回原通道 |

---

## ✅ Pros | 优点

**English**
- Captures long-range context efficiently
- Avoids explicit NxN attention matrix (lower memory than Non-local)
- Works well in detection/segmentation backbones

**中文**
- 高效建模长距离依赖
- 不显式构造 NxN 注意力矩阵（比 Non-local 更省显存）
- 适合检测/分割 backbone 等需要全局上下文的任务

---

## ⚠️ Cons | 局限

**English**
- More compute than SE/ECA
- Hyperparameters `c_m` / `c_n` need tuning for best trade-off

**中文**
- 比 SE/ECA 更重
- `c_m` / `c_n` 需要调参以平衡性能与开销

---

## 🧩 When to Use | 适用场景

**English**
- Dense prediction tasks (segmentation, detection)
- When global context is important but Non-local is too expensive

**中文**
- 密集预测任务（分割、检测）
- 需要全局上下文但 Non-local 太贵时

---

## 📚 Reference | 参考

Chen et al., *A²-Nets: Double Attention Networks*, NeurIPS 2018  
https://arxiv.org/abs/1810.11579

---

## 🧪 Minimal Example | 最小示例

```python
import torch
from attention.a2attention import DoubleAttention

x = torch.randn(2, 64, 32, 32)
a2 = DoubleAttention(in_channels=64, c_m=128, c_n=128, reconstruct=True)
y = a2(x)
print(y.shape)  # (2, 64, 32, 32)