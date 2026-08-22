import math
from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from torch.nn import init

# Replace with model path later
from ..shared_utils.lr_rope_temporary import RotaryEmbedding, apply_rotary_emb
from ..shared_utils.normalization import RMSGroupNorm
from ..shared_utils.position_biases import (
    RelativePositionBias,
)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FullAttention(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        mlp_dim=None,
        num_heads=12,
        drop_path=0,
        layer_scale_init_value=1e-6,
        bias_type="rel",
        max_d=3,
        weight_tied_axes=True,
        gradient_checkpointing=False,
        norm_layer: Callable = RMSGroupNorm,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim or hidden_dim * 4
        if self.mlp_dim % 2 != 0:
            raise ValueError(
                f"mlp_dim must be divisible by 2, got {self.mlp_dim} instead."
            )
        self.num_heads = num_heads
        self.max_d = max_d
        self.weight_tied_axes = weight_tied_axes
        self.norm1 = norm_layer(num_heads, hidden_dim, affine=True)
        self.fused_dims = (
            self.mlp_dim,
            hidden_dim,
            hidden_dim,
            hidden_dim,
        )  # FF, Q, K, V
        self.fused_ff_qkv = nn.Linear(hidden_dim, sum(self.fused_dims))

        self.activation = SwiGLU()
        self.ff_out = nn.Linear(int(self.mlp_dim // 2), hidden_dim)
        # Initialize ff_out weight and bias to include gamma_att
        init.kaiming_uniform_(
            self.ff_out.weight, a=math.sqrt(5) / layer_scale_init_value
        )
        if self.ff_out.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.ff_out.weight)
            bound = 1 / math.sqrt(fan_in) * layer_scale_init_value
            init.uniform_(self.ff_out.bias, -bound, bound)

        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Initialize attn weight and bias to include gamma_att
        init.kaiming_uniform_(
            self.attn_out.weight, a=math.sqrt(5) / layer_scale_init_value
        )
        if self.attn_out.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.attn_out.weight)
            bound = 1 / math.sqrt(fan_in) * layer_scale_init_value
            init.uniform_(self.attn_out.bias, -bound, bound)

        self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
        self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        if False and bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif True or bias_type == "rotary":
            # self.pos_emb = None
            self.rotary_emb = RotaryEmbedding(
                hidden_dim // num_heads // 4, freqs_for="pixel", max_freq=256
            )  # Do divide by dimension
            # self.register_buffer("pos_emb", None, persistent=False)
        else:
            self.rel_pos_biases = nn.ModuleList(
                [RelativePositionBias(n_heads=num_heads) for _ in range(3)]
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def make_rope_learnable(self, per_axis=False):
        """
        Make the RoPE learnable.
        """
        if hasattr(self, "rotary_emb"):
            self.rotary_emb.make_learnable(per_axis)

    def get_rotary_embedding(self, n, device):
        # if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
        #     return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        # self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, bcs, return_att=False):
        # input is t x b x c x h x w
        B, C, H, W, D = x.shape

        input = x.clone()
        x = self.norm1(x)

        fused_ff_qkv = rearrange(x, "b c h w d -> b h w d c").contiguous()
        ff, q, k, v = self.fused_ff_qkv(fused_ff_qkv).split(self.fused_dims, dim=-1)

        # Split into heads and process q, k
        q, k, v = map(
            lambda t: rearrange(t, "b h w d (he c) -> b he h w d c", he=self.num_heads),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        pos_emb = self.rotary_emb.get_axial_freqs(H, W, D)
        q, k = map(lambda t: apply_rotary_emb(pos_emb, t), (q, k))
        q, k, v = map(
            lambda t: rearrange(t, "b he h w d c -> b he (h w d) c"), (q, k, v)
        )
        att = F.scaled_dot_product_attention(q, k, v)
        att = rearrange(att, "b he (h w d) c -> b h w d (he c)", h=H, w=W)
        att_out = self.attn_out(att)
        x = self.drop_path(att_out + self.ff_out(self.activation(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + input

        return x, []
