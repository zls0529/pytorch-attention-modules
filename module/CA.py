import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    """
    Hard-Sigmoid activation:
        h_sigmoid(x) = ReLU6(x + 3) / 6
    This is a piecewise-linear approximation of sigmoid, commonly used in MobileNetV3.
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """
    Hard-Swish activation:
        h_swish(x) = x * h_sigmoid(x)
    Also widely used in MobileNetV3 for efficiency.
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention (CA) module.

    Key idea:
    - Encode long-range dependencies along one spatial direction at a time (height / width).
    - Generate two attention maps: one for height and one for width.
    - Apply them to the input feature map to preserve positional information better than SE.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels (usually same as inp).
        reduction (int): Reduction ratio for the intermediate bottleneck channels.
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()

        # Pool along width -> keep height information: output (B, C, H, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))

        # Pool along height -> keep width information: output (B, C, 1, W)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Bottleneck channels: mip = max(8, C / reduction)
        mip = max(8, inp // reduction)

        # 1x1 conv to reduce channels after concatenation of H and W descriptors
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # Two separate 1x1 convs to generate attention weights for H and W directions
        # a_h: (B, oup, H, 1), a_w: (B, oup, 1, W)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            Tensor: Output feature map of shape (B, C, H, W)
        """
        identity = x
        b, c, h, w = x.size()

        # 1) Directional pooling
        # x_h: (B, C, H, 1) encodes height-wise context
        x_h = self.pool_h(x)

        # x_w: (B, C, 1, W) -> permute to (B, C, W, 1) so that we can concat on the "spatial length" axis
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # 2) Concatenate along the spatial dimension (height-length + width-length)
        # y: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)

        # 3) Shared transform (channel reduction + BN + activation)
        # y: (B, mip, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 4) Split back into height and width tensors
        # x_h: (B, mip, H, 1)
        # x_w: (B, mip, W, 1)
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Restore x_w to shape (B, mip, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)

        # 5) Produce attention maps and apply sigmoid
        # a_h: (B, oup, H, 1)
        # a_w: (B, oup, 1, W)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 6) Apply attention (broadcast along missing dimensions)
        # identity: (B, C, H, W)
        # a_h broadcasts across W; a_w broadcasts across H
        out = identity * a_h * a_w
        return out


if __name__ == '__main__':
    # Sanity check: input/output shapes should match
    block = CoordAtt(64, 64)
    x = torch.rand(1, 64, 64, 64)
    y = block(x)
    print(x.size(), y.size())  # Expected: torch.Size([1, 64, 64, 64]) torch.Size([1, 64, 64, 64])