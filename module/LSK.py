import torch
import torch.nn as nn

class LSKNet(nn.Module):
    def __init__(self, dim: int):
        """
        LSK block (Large Selective Kernel) implementation.

        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        # Depthwise 5x5 conv: local receptive field
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        # Depthwise 7x7 conv with dilation=3: large receptive field (effective ~19x19)
        self.conv_spatial = nn.Conv2d(
            dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3
        )

        # 1x1 conv to reduce channels: dim -> dim//2 (branch 1)
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=1)

        # 1x1 conv to reduce channels: dim -> dim//2 (branch 2)
        self.conv2 = nn.Conv2d(dim, dim // 2, kernel_size=1)

        # Squeeze conv to generate 2 spatial gating maps for two branches
        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3)

        # 1x1 conv to restore channels: dim//2 -> dim
        self.conv = nn.Conv2d(dim // 2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Branch 1: local features
        attn1 = self.conv0(x)                 # (B, C, H, W)

        # Branch 2: large-kernel spatial features (via dilation)
        attn2 = self.conv_spatial(attn1)      # (B, C, H, W)

        # Channel reduction for both branches
        attn1 = self.conv1(attn1)             # (B, C//2, H, W)
        attn2 = self.conv2(attn2)             # (B, C//2, H, W)

        # Concatenate branch features for pooling-based gating
        attn = torch.cat([attn1, attn2], dim=1)  # (B, C, H, W)

        # Channel-wise statistics (spatial maps)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)       # (B, 1, H, W)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)     # (B, 1, H, W)
        agg = torch.cat([avg_attn, max_attn], dim=1)           # (B, 2, H, W)

        # Two spatial gating maps for two branches
        sig = self.conv_squeeze(agg).sigmoid()                 # (B, 2, H, W)

        # Weighted sum of two branches (spatially adaptive)
        attn = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]  # (B, C//2, H, W)

        # Restore channels
        attn = self.conv(attn)                                  # (B, C, H, W)

        # Recalibrate input features
        return x * attn


if __name__ == '__main__':
    block = LSKNet(64).cuda()
    x = torch.rand(1, 64, 64, 64).cuda()
    y = block(x)
    print(x.size(), y.size())