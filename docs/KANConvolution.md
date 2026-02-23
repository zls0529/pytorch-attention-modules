# KAN Convolution (KAN-Conv) — PyTorch Implementation

## What is KAN-Conv? | 什么是 KAN-Conv？

**English**

KAN-Conv replaces the standard convolution kernel (a fixed k×k weight matrix) with a **KANLinear** module.
Instead of computing:

\[
y = \sum_{i=1}^{k^2} w_i \cdot x_i
\]

KAN-Conv computes:

\[
y = \text{KANLinear}( \text{flatten}(patch) )
\]

So each local patch (size k×k) is mapped to a scalar by a learnable nonlinear function.

**中文**

KAN-Conv 的核心是：把传统卷积核（k×k 的固定权重）替换为一个 **KANLinear 小网络**。  
传统卷积是 patch 的线性加权求和；KAN-Conv 是 patch → 标量的可学习非线性映射。

---

## Data Flow | 工作流程

Input:
- `x`: (B, C, H, W)

Step 1 — Extract patches with `Unfold`:
- `unfold(x)` → (B, k*k, L), where L = H_out * W_out

Step 2 — Rearrange:
- transpose → (B, L, k*k)

Step 3 — Apply KAN kernel:
- KANLinear: (L, k*k) → (L, 1)

Step 4 — Reshape back:
- (L, 1) → (H_out, W_out)

Output:
- single kernel: (B, C, H_out, W_out)
- multiple kernels: (B, C*n_convs, H_out, W_out)

---

## Key Modules | 关键模块

### `KAN_Convolution`
- wraps one `KANLinear(k*k -> 1)`
- acts like one convolution kernel

### `KAN_Convolutional_Layer`
- holds `n_convs` kernels in a `ModuleList`
- supports multiple kernels per input channel

---

## Pros & Cons | 优缺点

**Pros**
- More expressive than linear convolution kernels
- Potentially better function approximation in local neighborhoods

**Cons**
- Much slower than Conv2d if implemented with Python loops
- Needs careful vectorization and GPU-friendly batching

---

## Notes / Known Issues | 注意事项

- Avoid nested loops over batch and channels (very slow).
- `regularization_loss()` in the provided code must reference the correct parameters (e.g., `self.conv`), not undefined `self.layers`.
- Prefer using the input tensor device instead of hard-coded `device="cuda"`.

---

## Minimal Example | 最小示例

```python
x = torch.randn(4, 3, 32, 32).cuda()
layer = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3,3), padding=(1,1), device="cuda")
y = layer(x)
print(y.shape)  # (4, 3*2, 32, 32)