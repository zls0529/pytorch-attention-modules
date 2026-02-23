import torch
import torch.nn as nn

class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        """
        ELA (Efficient/Edge-aware Lightweight Attention) module.

        Notes (as provided in your comments):
        - ELA-T and ELA-B are lightweight variants, suitable for shallow or lightweight CNNs.
        - ELA-B and ELA-S tend to work better in deeper networks.
        - ELA-L is designed for large networks.

        Args:
            in_channels (int): Number of input channels C.
            phi (str): Variant selector in {'T', 'B', 'S', 'L'} controlling kernel size / groups / GN groups.
        """
        super(ELA, self).__init__()

        # Kernel size for 1D convolution (controls receptive field along H or W)
        kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]

        # Group count for grouped Conv1d
        # - 'T'/'B': depthwise-like (groups=C) -> very lightweight
        # - 'S'/'L': groups=C/8 -> less grouped, more cross-channel mixing
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]

        # GroupNorm groups (controls normalization granularity)
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]

        # "Same" padding for Conv1d
        pad = kernel_size // 2

        # Shared 1D convolution applied on both height-pooled and width-pooled signals
        # Input/output: (B, C, L) -> (B, C, L)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=pad,
            groups=groups,
            bias=False
        )

        # GroupNorm normalizes over channels for each sample: works well for small batch sizes
        self.gn = nn.GroupNorm(num_groups, in_channels)

        # Sigmoid produces attention weights in (0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Output feature map of shape (B, C, H, W) with ELA applied.
        """
        b, c, h, w = x.size()

        # 1) Pool along width to get a height-wise descriptor
        # mean over W: (B, C, H, W) -> (B, C, H, 1) -> reshape -> (B, C, H)
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)

        # 2) Pool along height to get a width-wise descriptor
        # mean over H: (B, C, H, W) -> (B, C, 1, W) -> reshape -> (B, C, W)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        # 3) Apply the shared Conv1d to model local dependencies along H and W
        # (B, C, H) -> (B, C, H), (B, C, W) -> (B, C, W)
        x_h = self.conv1(x_h)
        x_w = self.conv1(x_w)

        # 4) Normalize + sigmoid to produce two 1D attention maps
        # Note: GroupNorm can accept (N, C, L) and normalizes across C dimension.
        # x_h: (B, C, H) -> GN+sigmoid -> (B, C, H) -> reshape -> (B, C, H, 1)
        x_h = self.sigmoid(self.gn(x_h)).view(b, c, h, 1)

        # x_w: (B, C, W) -> GN+sigmoid -> (B, C, W) -> reshape -> (B, C, 1, W)
        x_w = self.sigmoid(self.gn(x_w)).view(b, c, 1, w)

        # 5) Apply attention (broadcast multiply)
        # x_h broadcasts across W; x_w broadcasts across H
        return x_h * x_w * x


if __name__ == "__main__":
    # Sanity check
    x = torch.randn(1, 32, 256, 256)
    ela = ELA(in_channels=32, phi='T')
    y = ela(x)
    print(y.size())  # Expected: torch.Size([1, 32, 256, 256])