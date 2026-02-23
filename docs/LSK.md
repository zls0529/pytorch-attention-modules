# LSKNet / LSK Block (Large Selective Kernel)

## Overview | 概述
**EN**:  
LSK block builds a *large receptive field* attention-like reweighting using two depthwise convolution branches (local vs large-kernel via dilation).  
It learns spatially-adaptive gating weights to fuse the two branches and produces a recalibration map `attn` to scale the input: `out = x * attn`.

**中文**：  
LSK block 通过两条深度可分离卷积分支（局部 5×5 vs 膨胀 7×7）提取不同感受野的特征，并学习一个空间自适应的 gating 权重来融合两条分支，得到重标定图 `attn`，最终输出 `out = x * attn`。

---

## Data Flow | 数据流
Input: `x ∈ R^{B×C×H×W}`

1) Local branch (depthwise 5×5):  
`attn1 = DWConv5x5(x)` → `(B, C, H, W)`

2) Large spatial branch (depthwise 7×7, dilation=3):  
`attn2 = DWConv7x7_d3(attn1)` → `(B, C, H, W)`  
Effective kernel size ≈ `7 + (7-1)*(3-1) = 19`.

3) Channel reduction (1×1):  
`attn1, attn2: C → C/2`

4) Pooling-based gating:  
Concatenate: `cat(attn1, attn2)` → `(B, C, H, W)`  
Compute spatial stats: `avg` and `max` over channels → `(B, 2, H, W)`  
Generate two gating maps: `sig = sigmoid(Conv7x7(agg))` → `(B, 2, H, W)`

5) Spatially adaptive fusion:  
`attn = attn1 * sig[:,0] + attn2 * sig[:,1]` → `(B, C/2, H, W)`

6) Restore channels and recalibrate:  
`attn = Conv1x1(C/2→C)` → `(B, C, H, W)`  
`out = x * attn`

---

## What is the “attention” here? | 注意力在哪里？
**EN**:  
- `sig ∈ R^{B×2×H×W}` provides spatial gating weights for selecting between two receptive fields.  
- `attn ∈ R^{B×C×H×W}` is the final recalibration map applied to the input.

**中文**：  
- `sig(B,2,H,W)` 是两条分支的空间权重图（选择注意力）。  
- `attn(B,C,H,W)` 是最终对输入做重标定的权重图。

---

## Key Hyperparameters | 关键参数
- `groups=dim`: depthwise conv (lightweight)
- `dilation=3`: expands receptive field without increasing parameters too much
- `conv_squeeze(7×7)`: produces spatial gates for branch fusion

---

## Practical Notes | 实践注意
This implementation multiplies `x * attn` without constraining `attn` to [0,1].  
If training becomes unstable, consider adding `attn = sigmoid(attn)` or using a residual form `out = x * (1 + attn)`.

本实现的 `attn` 未做 sigmoid 约束，可能出现负值或>1。若训练不稳定，可考虑 `attn = sigmoid(attn)` 或 `out = x * (1 + attn)`。