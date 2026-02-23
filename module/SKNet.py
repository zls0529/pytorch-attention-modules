import numpy as np
import torch
from torch import nn
from collections import OrderedDict

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        """
        Selective Kernel (SK) Attention module.

        Args:
            channel (int): Number of input/output channels (C).
            kernels (list[int]): List of kernel sizes for multi-branch convolutions (multi-scale).
            reduction (int): Reduction ratio used in the gating (channel squeeze) stage.
            group (int): Number of groups for grouped convolution (can be set to C for depthwise conv).
            L (int): Minimum dimension after reduction (lower bound for reduced channels).
        """
        super().__init__()

        # Reduced channel dimension d = max(L, C / reduction)
        self.d = max(L, channel // reduction)

        # Multi-branch convolutions with different receptive fields (kernel sizes)
        # Each branch keeps channel dimension unchanged: (B, C, H, W) -> (B, C, H, W)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    # Grouped convolution (set group=channel for depthwise conv)
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group, bias=False)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU(inplace=True))
                ]))
            )

        # Shared FC to squeeze channel descriptor from C -> d
        self.fc = nn.Linear(channel, self.d)

        # Branch-specific FC layers to expand from d -> C (one per kernel branch)
        self.fcs = nn.ModuleList([])
        for _ in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))

        # Softmax over branches (k dimension) to obtain normalized selection weights
        # Note: attention_weights is stacked as (K, B, C, 1, 1), so dim=0 means across branches
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Output feature map of shape (B, C, H, W).
        """
        bs, c, _, _ = x.size()
        conv_outs = []

        # 1) Multi-scale feature extraction (K branches)
        for conv in self.convs:
            conv_outs.append(conv(x))  # each: (B, C, H, W)

        # Stack branch outputs: feats shape = (K, B, C, H, W)
        feats = torch.stack(conv_outs, dim=0)

        # 2) Fuse features by summation: U shape = (B, C, H, W)
        U = sum(conv_outs)

        # 3) Squeeze (global pooling over spatial dims): S shape = (B, C)
        S = U.mean(-1).mean(-1)

        # 4) Reduce dimension: Z shape = (B, d)
        Z = self.fc(S)

        # 5) Compute branch-wise attention weights
        weights = []
        for fc in self.fcs:
            # weight: (B, C)
            weight = fc(Z)
            # reshape for broadcasting with feature maps: (B, C, 1, 1)
            weights.append(weight.view(bs, c, 1, 1))

        # Stack weights across branches: (K, B, C, 1, 1)
        attention_weights = torch.stack(weights, dim=0)

        # Normalize weights across branches (soft selection): (K, B, C, 1, 1)
        attention_weights = self.softmax(attention_weights)

        # 6) Weighted sum of multi-scale features: (B, C, H, W)
        V = (attention_weights * feats).sum(dim=0)
        return V


if __name__ == '__main__':
    # Simple sanity check
    x = torch.randn(1, 64, 64, 64)  # (B, C, H, W)
    sk_attention = SKAttention(channel=64, reduction=8)
    y = sk_attention(x)
    print(y.shape)  # Expected: torch.Size([1, 64, 64, 64])