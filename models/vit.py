import math
import torch
import torch.nn as nn


def _trunc_normal_(tensor, std=0.02):
    if hasattr(torch.nn.init, "trunc_normal_"):
        return torch.nn.init.trunc_normal_(tensor, std=std)
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std)


class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and flatten."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = in_chans * patch_size * patch_size

    def forward(self, x):
        b, c, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"Expected input size {(self.img_size, self.img_size)}, got {(h, w)}.")
        patches = (
            x.unfold(2, self.patch_size, self.patch_size)
             .unfold(3, self.patch_size, self.patch_size)
             .permute(0, 2, 3, 1, 4, 5)
             .reshape(b, self.num_patches, self.patch_dim)
        )
        return patches


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = embed_dim ** 0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, num_tokens, _ = x.shape
        q = self.q_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, num_tokens, self.embed_dim)
        attn_output = self.dropout(self.out_proj(attn_output))
        return attn_output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=384,
        depth=7,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        input_normalization=True,
        use_cls_token=True,
    ):
        super().__init__()
        self.input_normalization = input_normalization
        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans)
        self.embed = nn.Linear(self.patch_embed.patch_dim, embed_dim)

        token_count = self.patch_embed.num_patches + (1 if use_cls_token else 0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.randn(1, token_count, embed_dim))

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        _trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.embed.weight, std=0.02)
        if self.embed.bias is not None:
            nn.init.zeros_(self.embed.bias)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in {self.embed}:
                _trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def per_image_standardization(self, x):
        flat_dim = x.shape[1] * x.shape[2] * x.shape[3]
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        adjusted_stddev = torch.clamp(std, min=1.0 / math.sqrt(flat_dim))
        return (x - mean) / adjusted_stddev

    def forward(self, x):
        if self.input_normalization:
            x = self.per_image_standardization(x)

        tokens = self.embed(self.patch_embed(x))
        if self.use_cls_token:
            cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)

        if self.use_cls_token:
            tokens = tokens[:, 0]
        else:
            tokens = tokens.mean(dim=1)
        logits = self.head(tokens)
        return logits


def vit_cifar(
    num_classes,
    img_size=32,
    patch_size=4,
    embed_dim=384,
    depth=7,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.0,
    input_normalization=True,
    use_cls_token=True,
):
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        input_normalization=input_normalization,
        use_cls_token=use_cls_token,
    )


__all__ = [
    "VisionTransformer",
    "vit_cifar",
]
