import torch
from torch import nn

class ContextBlock(nn.Module):
    """
    GCNet ContextBlock (Global Context Block).

    Core idea:
    1) Use spatial pooling to obtain a global context vector of shape (B, C, 1, 1)
       - pooling_type='avg' : global average pooling
       - pooling_type='att' : attention pooling (softmax over H*W)
    2) Transform this context with a small bottleneck network (1x1 conv -> norm -> ReLU -> 1x1 conv)
    3) Fuse the transformed context back to the input feature by:
       - channel_add: residual add
       - channel_mul: channel-wise gating (sigmoid) then multiply

    Args:
        inplanes (int): input channels C
        ratio (float): bottleneck ratio, planes = int(C * ratio)
        pooling_type (str): 'avg' or 'att'
        fusion_types (tuple/list): any of ('channel_add', 'channel_mul')
    """

    def __init__(self, inplanes, ratio=0.25, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()

        valid_fusion_types = ['channel_add', 'channel_mul']
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        # ---- 1) Pooling to get context (B, C, 1, 1) ----
        if pooling_type == 'att':
            # Produce a 1-channel mask then softmax over spatial positions
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)  # softmax over HW dimension after flatten
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ---- 2) Transform networks for fusion ----
        # The same bottleneck structure is used for add and mul branches (if enabled).
        def make_transform():
            return nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )

        self.channel_add_conv = make_transform() if 'channel_add' in fusion_types else None
        self.channel_mul_conv = make_transform() if 'channel_mul' in fusion_types else None

    def spatial_pool(self, x):
        """
        Pool spatial information into a context vector.

        Args:
            x: (B, C, H, W)

        Returns:
            context: (B, C, 1, 1)
        """
        batch, channel, height, width = x.size()

        if self.pooling_type == 'att':
            # x -> (B, C, HW)
            input_x = x.view(batch, channel, height * width)

            # mask: (B, 1, H, W) -> (B, 1, HW)
            context_mask = self.conv_mask(x).view(batch, 1, height * width)

            # softmax over HW so weights sum to 1
            context_mask = self.softmax(context_mask)  # (B, 1, HW)

            # (B, C, HW) @ (B, HW, 1) -> (B, C, 1)
            context = torch.matmul(input_x, context_mask.permute(0, 2, 1))

            # -> (B, C, 1, 1)
            context = context.view(batch, channel, 1, 1)
        else:
            # global average pooling -> (B, C, 1, 1)
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        context = self.spatial_pool(x)  # (B, C, 1, 1)
        out = x

        # channel-wise multiplication fusion (gating)
        if self.channel_mul_conv is not None:
            gate = torch.sigmoid(self.channel_mul_conv(context))  # (B, C, 1, 1)
            out = out * gate

        # channel-wise addition fusion (residual bias)
        if self.channel_add_conv is not None:
            bias = self.channel_add_conv(context)  # (B, C, 1, 1)
            out = out + bias

        return out


if __name__ == "__main__":
    x = torch.ones((1, 64, 128, 128))
    cb = ContextBlock(inplanes=64, ratio=0.25, pooling_type='att', fusion_types=('channel_add',))
    y = cb(x)
    print(x.shape)  # torch.Size([1, 64, 128, 128])
    print(y.shape)  # torch.Size([1, 64, 128, 128])