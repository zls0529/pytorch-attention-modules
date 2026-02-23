mobilevit_attention.md
# MobileViT Attention Block

## Overview | 概述

**EN**  
MobileViT Attention combines convolutional local feature extraction with Transformer-based global context modeling.  
It first captures local patterns using CNN layers, then models long-range dependencies through patch-based self-attention, and finally fuses both representations.

**中文**  
MobileViT Attention 将卷积网络的局部特征提取能力与 Transformer 的全局建模能力结合。  
它先用 CNN 提取局部特征，再通过 patch 级自注意力建模全局关系，最后融合两种信息。

---

## Data Flow | 数据流

Input:
x ∈ R^{B×C×H×W}

---

### 1️⃣ Local Representation (CNN)
局部特征提取
y = Conv3×3(x)
y = Conv1×1(y)

Output:
(B, dim, H, W)

**EN**: captures local texture & spatial structure  
**中文**：提取局部纹理与空间结构信息

---

### 2️⃣ Patch Unfolding
Patch 划分

Feature map is split into non-overlapping patches:
(B, dim, H, W)
→
(B, P², N, dim)

Where:

- P = patch size  
- N = number of patches  

**EN**: each patch becomes a token  
**中文**：每个 patch 被视为一个 token

---

### 3️⃣ Global Representation (Transformer)
全局关系建模
y = Transformer(y)

Transformer learns:

- long-range dependencies  
- cross-patch relationships  
- global semantic structure  

**中文**

Transformer 学习：

- 长距离依赖关系  
- patch 之间的联系  
- 全局语义结构  

---

### 4️⃣ Reshape Back to Spatial Map
恢复空间结构
(B, P², N, dim)
→
(B, dim, H, W)

Now the feature map contains global context.

---

### 5️⃣ Feature Fusion
特征融合

Restore channels:
dim → C

Concatenate with input:
cat(x, y) → (B, 2C, H, W)

Fusion convolution:
Conv → (B, C, H, W)

**EN**: combines local detail + global context  
**中文**：融合局部细节与全局语义信息

---

## Output
out ∈ R^{B×C×H×W}

---

## Where is the “attention”? | 注意力在哪里？

**EN**

- Transformer self-attention models relationships between patches.
- Enables global context understanding beyond CNN receptive fields.

**中文**

- Transformer 自注意力建模 patch 之间的关系。
- 实现 CNN 无法获得的全局上下文理解。

---

## Key Design Insights | 设计核心思想

### CNN provides

✔ local spatial precision  
✔ efficient computation  

### Transformer provides

✔ global dependency modeling  
✔ contextual reasoning  

### MobileViT combines both

✔ lightweight  
✔ expressive  
✔ mobile-friendly  

---

## Key Hyperparameters | 关键参数

- **patch_size**: controls token granularity  
- **dim**: Transformer embedding dimension  
- **depth**: number of Transformer layers  
- **heads**: number of attention heads  

---

## Practical Notes | 实践注意

### Patch size constraint

Input spatial size must be divisible by patch size.

输入尺寸必须能被 patch_size 整除。

---

### When to use MobileViT

Best suited for:

- lightweight vision networks  
- mobile & edge deployment  
- classification / detection / segmentation  

适用于：

- 轻量视觉网络  
- 移动端与边缘计算  
- 分类、检测、分割任务  

---

## Intuition | 直观理解

**EN**

1. CNN extracts local patterns  
2. feature map is split into patches  
3. Transformer learns relationships between patches  
4. spatial structure is reconstructed  
5. local & global features are fused  

**中文**

1. CNN 提取局部模式  
2. 特征图划分为 patch  
3. Transformer 学习 patch 关系  
4. 恢复空间结构  
5. 融合局部与全局信息  

---

## Comparison with CNN Attention Blocks | 与CNN注意力模块对比

| Module | Global Context | Lightweight |
|--------|--------|--------|
| SE | ✗ | ✓ |
| CBAM | ✗ | ✓ |
| LSK | ✗ | ✓ |
| MLCA | ✗ | ✓ |
| **MobileViT** | ✓ | ✓ |

---

## One-line Summary | 一句话总结

MobileViT brings Transformer global reasoning into lightweight CNN pipelines.

MobileViT 将 Transformer 的全局建模能力引入轻量 CNN 网络。