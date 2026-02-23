import math
import torch
from torch import nn
import torch.nn.functional as F


class MLCA(nn.Module):
    """
    Multi-Scale Local Context Attention (MLCA)

    This module combines:
    1. Global channel attention (ECA-style)
    2. Local context attention computed on a coarse spatial grid

    The final attention map is resized to the original feature size
    and applied via element-wise multiplication.

    Input shape:
        (B, C, H, W)

    Output shape:
        (B, C, H, W)
    """

    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()

        # ===== PARAMETERS =====
        # in_size: number of input channels
        # local_size: spatial grid size for local context pooling
        # gamma & b: used to compute adaptive kernel size (ECA-style)
        # local_weight: balance between local and global attention

        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        self.local_weight = local_weight

        # ----- Compute kernel size (ECA-style) -----
        # kernel size grows with channel dimension
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1  # ensure odd kernel for symmetric padding

        # 1D convolution for global channel attention
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            padding=(k - 1) // 2,
            bias=False
        )

        # 1D convolution for local context attention
        self.conv_local = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            padding=(k - 1) // 2,
            bias=False
        )

        # Adaptive pooling to extract local grid features
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)

        # Adaptive pooling to extract global features
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: input tensor (B, C, H, W)

        Returns:
            attention-weighted tensor (B, C, H, W)
        """

        B, C, H, W = x.shape

        # ===== 1. LOCAL CONTEXT POOLING =====
        # Compress spatial info into an L×L grid
        local_arv = self.local_arv_pool(x)      # (B, C, L, L)

        # Extract global summary from local grid
        global_arv = self.global_arv_pool(local_arv)  # (B, C, 1, 1)

        # ===== 2. PREPARE LOCAL FEATURES FOR 1D CONV =====
        # reshape → (B, 1, L²*C)
        temp_local = local_arv.view(B, C, -1) \
                               .transpose(-1, -2) \
                               .reshape(B, 1, -1)

        # ===== 3. PREPARE GLOBAL FEATURES FOR 1D CONV =====
        # reshape → (B, 1, C)
        temp_global = global_arv.view(B, C, -1) \
                                .transpose(-1, -2)

        # ===== 4. COMPUTE ATTENTION WEIGHTS =====
        y_local = self.conv_local(temp_local)   # (B, 1, L²*C)
        y_global = self.conv(temp_global)       # (B, 1, C)

        # ===== 5. RESTORE LOCAL ATTENTION MAP =====
        # reshape back to (B, C, L, L)
        y_local = y_local.reshape(B, self.local_size * self.local_size, C) \
                         .transpose(-1, -2) \
                         .view(B, C, self.local_size, self.local_size)

        att_local = y_local.sigmoid()

        # ===== 6. GLOBAL ATTENTION MAP =====
        # expand to match local grid size
        y_global = y_global.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        att_global = F.adaptive_avg_pool2d(
            y_global.sigmoid(),
            [self.local_size, self.local_size]
        )

        # ===== 7. FUSE LOCAL & GLOBAL ATTENTION =====
        att_mix = (
            att_global * (1 - self.local_weight) +
            att_local * self.local_weight
        )

        # resize attention to original spatial size
        att_all = F.adaptive_avg_pool2d(att_mix, [H, W])

        # ===== 8. APPLY ATTENTION =====
        out = x * att_all

        return out


# ===== TEST =====
if __name__ == '__main__':
    attention = MLCA(in_size=256)
    inputs = torch.randn((2, 256, 16, 16))
    result = attention(inputs)

    print(result.size())  # expected: (2, 256, 16, 16)