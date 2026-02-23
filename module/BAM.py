import numpy as np
import torch
from torch import nn
from torch.nn import init

def autopad(k, p=None, d=1):
    """
    Compute padding size to keep output spatial size "same" as input (for stride=1).

    Args:
        k (int or list): kernel size
        p (int or list or None): padding. If None, auto-compute.
        d (int): dilation

    Returns:
        padding value(s)
    """
    if d > 1:
        # Effective kernel size under dilation: k_eff = d*(k-1)+1
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Flatten(nn.Module):
    """Flatten a tensor from (B, C, 1, 1) to (B, C)."""
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):
    """
    Channel Attention branch in BAM.

    Pipeline:
        x -> GAP -> (B, C, 1, 1)
          -> flatten -> (B, C)
          -> MLP (with BN + ReLU) -> (B, C)
          -> reshape/broadcast -> (B, C, H, W)

    Args:
        channel (int): number of channels C
        reduction (int): reduction ratio for hidden channels
        num_layers (int): number of intermediate FC layers (depth of MLP)
    """
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Build a channel list for the MLP: C -> C/r -> ... -> C/r -> C
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())

        # Add (FC + BN + ReLU) blocks
        for i in range(len(gate_channels) - 2):
            self.ca.add_module(f'fc{i}', nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module(f'bn{i}', nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module(f'relu{i}', nn.ReLU(inplace=True))

        # Final FC maps back to C
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avgpool(x)          # (B, C, 1, 1)
        y = self.ca(y)               # (B, C)
        y = y.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # (B, C, H, W)
        return y


class SpatialAttention(nn.Module):
    """
    Spatial Attention branch in BAM.

    Pipeline:
        x (B, C, H, W)
          -> 1x1 conv reduce channels: C -> C/r
          -> several 3x3 dilated conv blocks (BN + ReLU)
          -> 1x1 conv to 1 channel: (B, 1, H, W)
          -> broadcast to (B, C, H, W)

    Args:
        channel (int): number of channels C
        reduction (int): reduction ratio for intermediate channels
        num_layers (int): number of dilated conv blocks
        dia_val (int): dilation value for 3x3 conv blocks
    """
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.sa = nn.Sequential()

        # Reduce channels by 1x1 conv: C -> C/r
        self.sa.add_module(
            'conv_reduce1',
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False)
        )
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU(inplace=True))

        # Dilated 3x3 conv blocks (keep channels = C/r)
        for i in range(num_layers):
            self.sa.add_module(
                f'conv_{i}',
                nn.Conv2d(
                    in_channels=channel // reduction,
                    out_channels=channel // reduction,
                    kernel_size=3,
                    padding=autopad(3, None, dia_val),
                    dilation=dia_val,
                    bias=False
                )
            )
            self.sa.add_module(f'bn_{i}', nn.BatchNorm2d(channel // reduction))
            self.sa.add_module(f'relu_{i}', nn.ReLU(inplace=True))

        # Project to 1-channel spatial map: (B, 1, H, W)
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.sa(x)          # (B, 1, H, W)
        y = y.expand_as(x)      # (B, C, H, W) broadcast to match x
        return y


class BAMBlock(nn.Module):
    """
    BAM (Bottleneck Attention Module) block.

    It computes:
        - Channel attention map:  (B, C, H, W)
        - Spatial attention map:  (B, C, H, W)
      Then merges them:
        weight = sigmoid(CA + SA)  -> (B, C, H, W)
        out = (1 + weight) * x     -> residual gating

    Args:
        channel (int): number of channels C
        reduction (int): reduction ratio used in CA/SA
        dia_val (int): dilation used in SA branch
    """
    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        """Optional: custom weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, C, H, W)
        sa_out = self.sa(x)               # (B, C, H, W) broadcasted spatial map
        ca_out = self.ca(x)               # (B, C, H, W) broadcasted channel map

        # Merge gates and squash to (0,1)
        weight = self.sigmoid(sa_out + ca_out)  # (B, C, H, W)

        # Residual gating: keep identity path (1 + weight)
        out = (1 + weight) * x
        return out


if __name__ == '__main__':
    x = torch.randn(32, 512, 7, 7)
    bam = BAMBlock(channel=512, reduction=16, dia_val=2)
    y = bam(x)
    print(y.shape)  # Expected: torch.Size([32, 512, 7, 7])