import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Initialize the Squeeze-and-Excitation (SE) Layer.

        Args:
            channel (int): Number of input feature channels.
            reduction (int, optional): Channel reduction ratio used to decrease
                dimensionality in the bottleneck fully connected layers.
                Default is 16.
        """
        super(SELayer, self).__init__()
        # Adaptive average pooling reduces each channel's spatial dimension (H, W)
        # to a single value (1×1), producing a global channel descriptor.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected bottleneck used to learn channel-wise attention weights.
        self.fc = nn.Sequential(
            # Reduce channel dimensionality: C → C // reduction
            # nn.Linear(in_features, out_features) # batch 维度始终保留，用于并行计算
            nn.Linear(channel, channel // reduction, bias=False),
            # Non-linear activation to model complex channel dependencies
            nn.ReLU(inplace=True),
            # Restore dimensionality: C // reduction → C
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid activation generates channel attention weights in [0, 1]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass of the SE layer.

        Args:
            x (Tensor): Input feature map of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output feature map with channel-wise attention applied.
                    Shape is identical to the input.
        """
        # Get input dimensions
        b, c, h, w = x.size()
        # Squeeze: aggregate spatial information into channel descriptors
        # Result shape: (B, C, 1, 1) → reshape to (B, C)
        y = self.avg_pool(x).view(b, c)
        # Excitation: learn channel attention weights via bottleneck MLP
        # Output shape: (B, C) → reshape to (B, C, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: reweight input feature map using learned channel attention
        return x * y.expand_as(x)

        
# how to use this module
if __name__ == "__main__":
    # Create a random tensor to simulate input features:
    # batch size = 4, channels = 64, spatial size = 32×32
    x = torch.randn(4, 64, 32, 32)
    # Instantiate the SE layer with 64 input channels
    se_layer = SELayer(channel=64)
    # Apply channel attention to the input tensor
    # y is the attetion we learn here
    y = se_layer(x)
    # Print output shape (should match input shape)
    print(y.shape)  # Expected: torch.Size([4, 64, 32, 32])
