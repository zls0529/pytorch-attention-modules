import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class DoubleAttention(nn.Module):
    """
    Double Attention (A2) Module.

    Key idea:
    - First, gather global descriptors (a compact set of global context vectors)
      by aggregating spatial features with an attention map.
    - Then, distribute these global descriptors back to all spatial positions
      using another attention vector.

    Args:
        in_channels (int): Number of input channels C.
        c_m (int): Channel size of feature A (descriptor features).
        c_n (int): Channel size used for attention maps/vectors.
        reconstruct (bool): If True, project output from c_m back to in_channels.
    """
    def __init__(self, in_channels, c_m=128, c_n=128, reconstruct=True):
        super().__init__()

        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n

        # 1x1 convs to generate three tensors:
        # A: features for building global descriptors (B, c_m, H, W)
        # B: attention maps for gathering (B, c_n, H, W)
        # V: attention vectors for distributing (B, c_n, H, W)
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size=1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size=1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size=1)

        # Optional: project c_m back to in_channels
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        """Custom weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            Tensor: Output feature map of shape (B, C, H, W) if reconstruct=True,
                    else (B, c_m, H, W)
        """
        b, c, h, w = x.shape
        assert c == self.in_channels

        # Compute A, B, V
        A = self.convA(x)  # (B, c_m, H, W)
        B_map = self.convB(x)  # (B, c_n, H, W)
        V_vec = self.convV(x)  # (B, c_n, H, W)

        # Flatten spatial dims: N = H*W
        # tmpA: (B, c_m, N)
        tmpA = A.view(b, self.c_m, -1)

        # attention_maps: softmax over spatial positions (dim=-1)
        # (B, c_n, N)
        attention_maps = F.softmax(B_map.view(b, self.c_n, -1), dim=-1)

        # attention_vectors: softmax over spatial positions (dim=-1)
        # (B, c_n, N)
        attention_vectors = F.softmax(V_vec.view(b, self.c_n, -1), dim=-1)

        # Step 1) GATHER: build global descriptors
        # tmpA: (B, c_m, N)
        # attention_maps.permute(0,2,1): (B, N, c_n)
        # global_descriptors: (B, c_m, c_n)
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))

        # Step 2) DISTRIBUTE: broadcast global descriptors back to spatial positions
        # attention_vectors: (B, c_n, N)
        # tmpZ: (B, c_m, N)
        tmpZ = global_descriptors.matmul(attention_vectors)

        # Reshape to feature map: (B, c_m, H, W)
        tmpZ = tmpZ.view(b, self.c_m, h, w)

        # Optional reconstruction to original channel dimension: (B, C, H, W)
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


if __name__ == '__main__':
    x = torch.randn(64, 32, 7, 7)
    a2 = DoubleAttention(in_channels=32)
    y = a2(x)
    print(y.shape)  # Expected: torch.Size([64, 32, 7, 7])