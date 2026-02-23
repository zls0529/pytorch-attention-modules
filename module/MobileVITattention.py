from torch import nn
import torch
from einops import rearrange


# -------------------------------------------------
# PreNorm: LayerNorm applied before a given function
# -------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn  # function (attention or feed-forward)

    def forward(self, x, **kwargs):
        # Apply LayerNorm before function (Pre-Norm Transformer)
        return self.fn(self.ln(x), **kwargs)


# -------------------------------------------------
# FeedForward (MLP block used in Transformer)
# -------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Multi-head Self Attention
# -------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()

        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            if project_out else nn.Identity()
        )

    def forward(self, x):
        # compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # reshape into multi-head format
        q, k, v = map(
            lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads),
            qkv
        )

        # scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        # weighted sum
        out = torch.matmul(attn, v)

        # merge heads
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


# -------------------------------------------------
# Transformer Encoder Block
# -------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)   # residual attention
            out = out + ffn(out)   # residual MLP
        return out


# -------------------------------------------------
# MobileViT Attention Block
# -------------------------------------------------
class MobileViTAttention(nn.Module):
    """
    MobileViT-style attention block combining:

    1. Local representation via convolution
    2. Global representation via Transformer
    3. Feature fusion for enhanced spatial awareness

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H, W)
    """

    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()

        self.ph, self.pw = patch_size, patch_size

        # Local feature extraction
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        # Global representation via Transformer
        self.trans = Transformer(
            dim=dim,
            depth=3,
            heads=8,
            head_dim=64,
            mlp_dim=1024
        )

        # Restore channels
        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)

        # Fusion layer
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size, padding=kernel_size // 2)

    def forward(self, x):

        # Preserve input for fusion
        y = x.clone()

        # -------- Local Representation --------
        y = self.conv2(self.conv1(x))

        # -------- Global Representation --------
        _, _, h, w = y.shape

        # split into patches
        y = rearrange(
            y,
            'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim',
            ph=self.ph,
            pw=self.pw
        )

        y = self.trans(y)

        # reconstruct spatial map
        y = rearrange(
            y,
            'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)',
            ph=self.ph,
            pw=self.pw,
            nh=h // self.ph,
            nw=w // self.pw
        )

        # -------- Fusion --------
        y = self.conv3(y)
        y = torch.cat([x, y], dim=1)
        y = self.conv4(y)

        return y


# ---------------- TEST ----------------
if __name__ == '__main__':
    m = MobileViTAttention(in_channel=512)
    input = torch.randn(1, 512, 49, 49)
    output = m(input)
    print(output.shape)