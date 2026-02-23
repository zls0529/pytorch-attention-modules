import torch
from torch import nn

class EMA(nn.Module):
    """
    EMA Attention module (group-wise efficient attention).

    High-level idea:
    - Split channels into G groups to reduce computation.
    - Inside each group, generate direction-aware gating (H/W) similar in spirit to coordinate-like pooling.
    - Build an attention weight map (per group) using two cross terms:
        (channel-to-spatial) and (spatial-to-channel) interactions via matmul.
    - Apply sigmoid weights to group features and reshape back.

    Args:
        channels (int): Number of input channels C.
        c2: unused (kept for compatibility).
        factor (int): Number of groups G (default 32). Must satisfy C // G > 0.
    """
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0, "channels must be >= groups"

        self.softmax = nn.Softmax(dim=-1)

        # Global average pool to (1,1) for group-wise channel descriptors
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        # Directional pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # keep H
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # keep W

        # NOTE:
        # Here GroupNorm is applied on each grouped tensor with shape (B*G, Cg, H, W).
        # num_groups = channels_per_group, i.e. instance-norm-like within each group.
        channels_per_group = channels // self.groups
        self.gn = nn.GroupNorm(num_groups=channels_per_group, num_channels=channels_per_group)

        # 1x1 conv and 3x3 conv operate within each group (channels_per_group -> channels_per_group)
        self.conv1x1 = nn.Conv2d(channels_per_group, channels_per_group, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels_per_group, channels_per_group, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, H, W)

        Returns:
            Tensor: (B, C, H, W)
        """
        b, c, h, w = x.size()
        g = self.groups
        cg = c // g

        # 1) Reshape to grouped tensor:
        # group_x: (B*G, Cg, H, W)
        group_x = x.reshape(b * g, cg, h, w)

        # 2) Directional pooling within each group
        # x_h: (B*G, Cg, H, 1)
        x_h = self.pool_h(group_x)

        # x_w: (B*G, Cg, 1, W) -> permute to (B*G, Cg, W, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 3) Concatenate along "length" axis and fuse with 1x1 conv
        # cat: (B*G, Cg, H+W, 1)
        # hw:  (B*G, Cg, H+W, 1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))

        # Split back
        # x_h: (B*G, Cg, H, 1)
        # x_w: (B*G, Cg, W, 1)
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # Restore x_w to (B*G, Cg, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)

        # 4) Build two feature branches
        # x1: gated + normalized branch
        # x2: local conv branch
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.sigmoid())      # (B*G, Cg, H, W)
        x2 = self.conv3x3(group_x)                                 # (B*G, Cg, H, W)

        # 5) Build attention weights via two cross terms
        # Term A: from x1 global channel descriptor -> weights over channels, then dot with x2 spatial tokens
        # agp(x1): (B*G, Cg, 1, 1) -> reshape -> (B*G, Cg, 1)
        # permute -> (B*G, 1, Cg) then softmax over Cg
        x11 = self.softmax(self.agp(x1).reshape(b * g, cg, 1).permute(0, 2, 1))  # (B*G, 1, Cg)

        # x12: flatten x2 to spatial tokens: (B*G, Cg, H*W)
        x12 = x2.reshape(b * g, cg, -1)                                           # (B*G, Cg, HW)

        # Term B: from x2 global channel descriptor -> weights, then dot with x1 spatial tokens
        x21 = self.softmax(self.agp(x2).reshape(b * g, cg, 1).permute(0, 2, 1))  # (B*G, 1, Cg)
        x22 = x1.reshape(b * g, cg, -1)                                          # (B*G, Cg, HW)

        # Matmul:
        # (B*G, 1, Cg) @ (B*G, Cg, HW) -> (B*G, 1, HW) -> reshape -> (B*G, 1, H, W)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * g, 1, h, w)

        # 6) Apply sigmoid weights and reshape back to (B, C, H, W)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return out


if __name__ == '__main__':
    # CUDA test (optional)
    block = EMA(64).cuda()
    x = torch.rand(1, 64, 64, 64).cuda()
    y = block(x)
    print(x.size(), y.size())  # Expected: torch.Size([1, 64, 64, 64]) torch.Size([1, 64, 64, 64])