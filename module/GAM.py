import torch
import torch.nn as nn

class GAM_Attention(nn.Module):
    """
    GAM (Global Attention Mechanism) - a channel + spatial attention block.

    This implementation applies:
    1) Channel attention computed per spatial position (token-wise MLP on channel dimension)
    2) Spatial attention computed by 7x7 convs on the channel-attended feature map

    Args:
        in_channels (int): input channels C
        rate (int): reduction ratio r (default 4)
    """
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()

        hidden = int(in_channels / rate)

        # Channel attention (MLP): C -> C/r -> C
        # NOTE: In this code, MLP is applied to each spatial location independently.
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels)
        )

        # Spatial attention: (B, C, H, W) -> (B, C/r, H, W) -> (B, C, H, W)
        # Using 7x7 conv to capture wider spatial context.
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, H, W)

        Returns:
            Tensor: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ---- 1) Channel attention (token-wise) ----
        # Rearrange to tokens: (B, C, H, W) -> (B, H*W, C)
        x_tokens = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # Apply MLP on channel dimension for each token:
        # (B, H*W, C) -> (B, H*W, C)
        x_tokens_att = self.channel_attention(x_tokens)

        # Reshape back: (B, H*W, C) -> (B, H, W, C) -> (B, C, H, W)
        x_channel_att = x_tokens_att.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # Sigmoid to get channel attention weights in (0,1)
        x_channel_att = x_channel_att.sigmoid()  # (B, C, H, W)

        # Apply channel attention
        x = x * x_channel_att  # (B, C, H, W)

        # ---- 2) Spatial attention ----
        # Spatial attention produces another (B, C, H, W) gating map
        x_spatial_att = self.spatial_attention(x).sigmoid()  # (B, C, H, W)

        # Apply spatial attention
        out = x * x_spatial_att
        return out


if __name__ == '__main__':
    x = torch.randn(1, 64, 7, 7)
    net = GAM_Attention(in_channels=64, rate=4)
    y = net(x)
    print(y.size())  # Expected: torch.Size([1, 64, 7, 7])