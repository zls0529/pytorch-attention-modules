# MLCA (Multi-Scale Local Context Attention) | 多尺度局部上下文注意力

## Overview | 概述
**EN**  
MLCA combines **global channel attention** (ECA-style) and **local context attention** computed on a coarse spatial grid (e.g., 5×5).
It generates a final attention map `att_all ∈ R^{B×C×H×W}` and reweights the input feature map by element-wise multiplication: `out = x * att_all`.

**中文**  
MLCA 将 **全局通道注意力（ECA 风格）** 与 **局部网格上下文注意力（如 5×5）** 融合，
最终得到注意力权重 `att_all (B,C,H,W)`，并对输入特征做逐元素加权：`out = x * att_all`。

---

## Input / Output | 输入输出
- Input: `x ∈ R^{B×C×H×W}`
- Output: `out ∈ R^{B×C×H×W}`

---

## Data Flow | 工作流程（含 shape）
Let `local_size = L`.

1) **Local pooling**  
`local = AdaptiveAvgPool2d(L)(x)`  
Shape: `(B, C, L, L)`

2) **Global pooling (from local)**  
`global = AdaptiveAvgPool2d(1)(local)`  
Shape: `(B, C, 1, 1)`

3) **Prepare for 1D conv**  
- Local sequence: reshape to `(B, 1, L²*C)`  
- Global sequence: reshape to `(B, 1, C)`

4) **ECA-style 1D conv**
- `y_local = Conv1d(y_local_input)` → `(B, 1, L²*C)`
- `y_global = Conv1d(y_global_input)` → `(B, 1, C)`

5) **Recover attention maps**
- `att_local = sigmoid(reshape(y_local))` → `(B, C, L, L)`
- `att_global = sigmoid(y_global)` then expand to `(B, C, L, L)`

6) **Fuse local & global**
`att_mix = (1-α)*att_global + α*att_local`, where `α = local_weight`  
Shape: `(B, C, L, L)`

7) **Upsample to original resolution**
`att_all = AdaptiveAvgPool2d((H,W))(att_mix)`  
Shape: `(B, C, H, W)`

8) **Reweight input**
`out = x * att_all`

---

## Where is the attention? | 注意力在哪里？
- `att_local (B,C,L,L)`: channel attention varying across coarse local grid cells  
- `att_global (B,C,L,L)`: global channel attention broadcast to local grid  
- `att_all (B,C,H,W)`: final attention map applied to input

---

## Key Hyperparameters | 关键超参数
- `local_size (L)`: size of the local pooling grid (e.g., 5). Larger L means finer local context but higher cost.
- `local_weight (α)`: balance between local and global attention (0 → global only, 1 → local only).
- `gamma, b`: control ECA kernel size `k` as a function of channel number `C`:
  `k = odd( |log2(C)+b| / gamma )`.

---

## Notes | 注意事项
This implementation uses 1D convolution for attention (ECA-style), which is lightweight and avoids heavy fully-connected layers.
Local attention is computed on a coarse grid and then resized back to `(H, W)` to match the input feature resolution.