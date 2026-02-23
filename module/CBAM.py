import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Channel Attention module in CBAM.

        Args:
            in_channels (int): Number of input channels (C).
            reduction_ratio (int): Reduction ratio for the bottleneck MLP.
                                   Default is 16.
        """
        super(ChannelAttention, self).__init__()
        # Global average pooling and global max pooling over spatial dimensions (H, W)
        # Output shape for both: (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP implemented by 1x1 convolutions (equivalent to FC on channel dimension)
        # C -> C//r -> C
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        # Sigmoid produces channel attention weights in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Channel attention map of shape (B, C, 1, 1).
        """
        # Pool along spatial dimensions to get channel descriptors
        # avg_pool(x): (B, C, 1, 1)
        # max_pool(x): (B, C, 1, 1)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # Combine both descriptors (avg + max) and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Spatial Attention module in CBAM.

        Args:
            kernel_size (int): Kernel size for the spatial attention convolution.
                               Common choice is 7.
        """
        super(SpatialAttention, self).__init__()
        
        # Convolution over the concatenated (avg, max) maps: (B, 2, H, W) -> (B, 1, H, W)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
        # Sigmoid produces spatial attention weights in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Spatial attention map of shape (B, 1, H, W).
        """
        # Compute channel-wise average and max to produce two spatial descriptors
        # avg_out: (B, 1, H, W)
        # max_out: (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension: (B, 2, H, W)
        x = torch.cat([avg_out, max_out], dim=1)
        # Convolution + sigmoid to generate spatial attention map: (B, 1, H, W)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        """
        CBAM: Convolutional Block Attention Module (Channel + Spatial Attention).

        Args:
            in_channels (int): Number of input channels (C).
            reduction_ratio (int): Reduction ratio for channel attention MLP.
            kernel_size (int): Kernel size for spatial attention convolution.
        """
        super(CBAM, self).__init__()
        
        # Channel attention and spatial attention modules
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Output feature map with CBAM applied. Shape is (B, C, H, W).
        """
        # 1) Apply channel attention: (B, C, 1, 1) * (B, C, H, W) -> (B, C, H, W)
        out = self.channel_attention(x) * x
        
        # 2) Apply spatial attention: (B, 1, H, W) * (B, C, H, W) -> (B, C, H, W)
        out = self.spatial_attention(out) * out
        
        return out

# Simple sanity check
if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)  # (B, C, H, W)
    cbam = CBAM(64)
    out = cbam(x)
    print(out.shape)  # Expected: torch.Size([4, 64, 32, 32])
