"""Vision Transformer for CIFAR-10, derived from torchvision's implementation, with kernel attention.

License: includes code from torchvision (BSD 3-Clause).
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import MLP

from .attention import attention
from .kernels import TAU


class KernelMultiheadAttention(nn.Module):
    """Self-attention with a pluggable kernel; param-compatible with nn.MultiheadAttention."""

    def __init__(self, embed_dim, num_heads, kernel="softmax", tau=None, impl="efficient"):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel = kernel
        self.tau = TAU[kernel] if tau is None else tau
        self.impl = impl

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)      # (B, S, 3E)
        q, k, v = proj.view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (B, H, S, D)
        out = attention(q, k, v, self.kernel, self.tau, impl=self.impl)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, E))


class MLPBlock(MLP):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, kernel, tau, impl, norm_layer):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = KernelMultiheadAttention(hidden_dim, num_heads, kernel, tau, impl)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, kernel, tau, impl, norm_layer):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, kernel, tau, impl, norm_layer)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer (https://arxiv.org/abs/2010.11929)."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        num_classes: int = 10,
        kernel: str = "softmax",
        tau: Optional[float] = None,
        impl: str = "efficient",
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        seq_length = (image_size // patch_size) ** 2

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, kernel, tau, impl, norm_layer)
        self.seq_length = seq_length
        self.heads = nn.Sequential(OrderedDict([("head", nn.Linear(hidden_dim, num_classes))]))

        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w) -> (n, n_h * n_w, hidden_dim)
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, (h // p) * (w // p))
        return x.permute(0, 2, 1)

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        return self.heads(x)
