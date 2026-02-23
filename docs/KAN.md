# KAN (Kolmogorov–Arnold Networks) — PyTorch Implementation

## 🔍 What is KAN? | KAN 是什么？

**English**

KAN replaces a standard MLP linear layer with a **learnable function** built from:
- a **base linear branch** with an activation (e.g., SiLU)
- a **B-spline expansion branch** that approximates a flexible 1D function per input feature

A KAN layer output is:

\[
y = W_{base}\,\phi(x) + W_{spline}\,B(x)
\]

where:
- \(\phi(\cdot)\) is a base activation
- \(B(x)\) are B-spline basis values computed on a grid

**中文**

KAN 的核心思想是：把传统 MLP 的“线性层”替换为更灵活的函数形式：
- 基础分支：\(\phi(x)\) + 线性
- 样条分支：对输入做 **B-样条基函数展开**，再线性组合

一层 KAN 近似为：

\[
y = W_{base}\,\phi(x) + W_{spline}\,B(x)
\]

---

## ⚙️ KANLinear Parameters | KANLinear 参数说明

| Param | Meaning (EN) | 中文 |
|------|--------------|------|
| in_features | input dim | 输入维度 |
| out_features | output dim | 输出维度 |
| grid_size | number of grid intervals | 网格区间数 |
| spline_order | spline order (degree-related) | 样条阶数 |
| scale_noise | init noise for spline curve | 样条初始化噪声 |
| scale_base | scale for base weights | base 权重缩放 |
| scale_spline | scale for spline weights | spline 权重缩放 |
| enable_standalone_scale_spline | per-(out,in) learnable scaler | 是否启用独立缩放参数 |
| base_activation | base branch activation | base 分支激活函数 |
| grid_eps | blend uniform/adaptive grid | 网格混合系数 |
| grid_range | initial grid range | 初始网格范围 |

---

## 🧠 Key Tensors & Shapes | 关键张量与形状

Let:
- \(B\)=batch size, \(I\)=in_features, \(O\)=out_features
- \(K = grid\_size + spline\_order\)  (number of spline basis functions)

### Parameters

- `base_weight`: \((O, I)\)
- `spline_weight`: \((O, I, K)\)
- optional `spline_scaler`: \((O, I)\)

### Forward shapes

Input:
- `x`: \((B, I)\)

Base branch:
- `base_output = Linear(phi(x), base_weight)` → \((B, O)\)

Spline branch:
- `bases = b_splines(x)` → \((B, I, K)\)
- flatten: \((B, I*K)\)
- flatten weights: \((O, I*K)\)
- `spline_output = Linear(flat_bases, flat_weights)` → \((B, O)\)

Final:
\[
y = base\_output + spline\_output
\]

---

## 🔄 How B-splines are computed | B-样条基函数怎么来的？

**English**

`b_splines(x)` builds basis values using the **Cox–de Boor recursion**.
The grid is stored as a buffer:

- `grid`: \((I, grid_size + 2*spline_order + 1)\)

For each input feature, it computes \(K\) basis functions (local support).

**中文**

`b_splines(x)` 用 Cox–de Boor 递推公式计算 B-样条基函数。  
网格 `grid` 是 buffer，按每个输入维度分别维护。  
B-样条具有“局部支撑”，因此表达灵活且稳定。

---

## 🧩 Where does “learning” happen? | 学习发生在哪里？

KAN 的可学习参数主要在两处：

1) `base_weight`：base 分支线性层权重  
2) `spline_weight`（以及可选的 `spline_scaler`）：样条分支的系数（basis 的线性组合权重）

训练时反向传播会对这些权重更新，从而学习到：
- base 分支的线性映射
- spline 分支对每个输入维度的非线性形状（通过基函数组合实现）

---

## 🧱 `update_grid`: Adaptive grid | 自适应网格更新

**English**

`update_grid(x)` adjusts grid points based on the data distribution:
- sort each input dimension
- pick quantiles to form an adaptive grid (`grid_adaptive`)
- also build a uniform grid (`grid_uniform`)
- blend them with `grid_eps`

Then it **re-fits spline coefficients** so the function remains consistent under the new grid.

**中文**

`update_grid(x)` 根据数据分布动态更新网格：
- 对每个输入维度排序，取分位点构造自适应网格
- 同时构造均匀网格以保持稳定
- 用 `grid_eps` 混合二者
- 网格变化后会重新拟合 spline 系数，尽量保持函数不“跳变”

---

## 🧾 Regularization | 正则化

This implementation uses a proxy regularization:
- L1 on spline weights (mean abs)
- entropy-like term on normalized L1 distribution

Purpose:
- encourage sparsity / smoothness
- avoid overfitting

---

## ✅ When to Use KAN? | 什么时候用 KAN？

**English**
- function fitting / regression tasks
- tabular data (often works well)
- when you want a flexible 1D nonlinearity per input feature

**中文**
- 回归/函数拟合
- 表格数据
- 希望每个输入维度都有更灵活的非线性表达时

---

## 🧪 Minimal Example | 最小示例

```python
import torch
from kan import KAN

model = KAN([10, 64, 1], grid_size=5, spline_order=3)
x = torch.randn(32, 10)
y = model(x)  # (32, 1)
loss = y.mean() + 1e-4 * model.regularization_loss()
loss.backward()