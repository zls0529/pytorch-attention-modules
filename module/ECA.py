import torch
from torch import nn

class ECA_layer(nn.Module):
    """
    Efficient Channel Attention (ECA) module.

    Args:
        channel (int): Number of input feature channels (C).
        k_size (int): Kernel size of the 1D convolution used for local cross-channel interaction.
                      Default is 3.
    """
    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()

        # Global Average Pooling compresses spatial information (H, W) into a single value per channel
        # Output shape: (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1D convolution to capture local cross-channel interactions (no dimensionality reduction)
        # Input/Output channels are both 1 because we convolve along the "channel-length" dimension.
        # Padding keeps the length unchanged so output remains aligned with input channels.
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        # Sigmoid maps attention weights into (0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of ECA.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Output feature map with channel attention applied. Shape is (B, C, H, W).
        """
        # 1) Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)

        # 2) Reshape for Conv1d:
        #    squeeze last dim: (B, C, 1, 1) -> (B, C, 1)
        #    transpose to make channel dimension the "length": (B, C, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)

        # 3) 1D convolution along the channel dimension: (B, 1, C) -> (B, 1, C)
        y = self.conv(y)

        # 4) Restore shape back to (B, C, 1, 1):
        #    (B, 1, C) -> (B, C, 1) -> (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # 5) Sigmoid to obtain attention weights in (0, 1): (B, C, 1, 1)
        y = self.sigmoid(y)

        # 6) Apply channel attention (broadcast along H, W): (B, C, H, W)
        return x * y.expand_as(x)


if __name__ == "__main__":
    # Simple sanity check
    # Simulated input feature map: batch=4, channels=64, spatial size=32x32
    x = torch.randn(4, 64, 32, 32)

    # Create an ECA module for 64 channels
    eca = ECA_layer(channel=64)

    # Forward pass
    y = eca(x)

    # Output shape should match input shape
    print(y.shape)  # Expected: torch.Size([4, 64, 32, 32])