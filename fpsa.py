import torch
import torch.nn as nn
from fix import FixedPointIteration


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, layer_norm=True, residual=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.residual = residual

        # Single matrix for Q, K, V projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Optional layer norm
        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B, N, C = x.shape

        # Efficient combined Q,K,V projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        # Optional residual connection
        if self.residual:
            out = out + x

        # Optional layer normalization
        if self.layer_norm is not None:
            out = self.layer_norm(out)

        return out


class FixedPointSelfAttentionStep(nn.Module):
    def __init__(
        self, embed_dim, num_heads=1, normalize=True, causal=False, block_size=64
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # mating sure all heads get initialized with different temperature
        temperature_init = torch.ones(self.num_heads)
        # for h in range(self.num_heads):
        #     temperature_init[h] += (h + 1) * 0.1
        self.temperature = nn.Parameter(temperature_init)
        self.normalize = nn.Tanh() if normalize else None
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.causal = causal
        self._init_weights()
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(1, 1, block_size, block_size), diagonal=1).bool(),
                persistent=False,
            )

    def _init_weights(self):
        nn.init.orthogonal_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)

    def forward(self, z_k, x):
        B, N, C = z_k.shape
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv(z_k).reshape(B, -1, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]

        scale = D**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn / self.temperature.view(1, H, 1, 1)
        if self.causal:
            mask = self.causal_mask[..., :N, :N]
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)

        v = x.reshape(B, -1, H, D).transpose(1, 2)
        z_next = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.normalize is not None:
            z_next = self.normalize(z_next)
        return z_next


class FixedPointSelfAttentionStepFlash(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=1,
        normalize=True,
        causal=False,
        dropout=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # mating sure all heads get initialized with different temperature
        temperature_init = torch.ones(self.num_heads)
        for h in range(self.num_heads):
            temperature_init[h] += (h + 1) * 0.1
        self.temperature = nn.Parameter(temperature_init)
        self.normalize = nn.Tanh() if normalize else None
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.causal = causal
        self.dropout = 0
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)

    def forward(self, z_k, x):
        B, N, C = z_k.shape
        H = self.num_heads
        D = self.head_dim

        # Compute q, k from z_k
        qkv = self.qkv(z_k).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # Compute v from x
        v = x.reshape(B, N, H, D).transpose(1, 2)  # (B, H, N, D)

        # Apply scaling by sqrt(temperature) and sqrt(head_dim)
        scale = 1.0 / self.temperature.sqrt().view(1, H, 1, 1)
        q = q * scale
        k = k * scale

        # PyTorch 2.0 native efficient attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=self.causal, dropout_p=self.dropout
        )  # (B, H, N, D)

        # Final reshape to (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)

        # Optional normalization
        if self.normalize is not None:
            out = self.normalize(out)

        return out


class FixedPointSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=1,
        max_iter=10,
        eps=1e-6,
        layer_norm=True,  # calling it this to preserve vanilla SA interface
        residual=True,
        causal=False,
        flash=False,
        dropout=0,
        block_size=64,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.eps = eps
        self.residual = residual
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.normalize = nn.Tanh() if layer_norm else None
        if flash and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            self.attention_step = FixedPointSelfAttentionStepFlash(
                embed_dim, num_heads, layer_norm, causal=False, dropout=dropout
            )
            self.attention_step_final = FixedPointSelfAttentionStepFlash(
                embed_dim, num_heads, layer_norm, causal=causal, dropout=dropout
            )
        else:
            self.attention_step = FixedPointSelfAttentionStep(
                embed_dim,
                num_heads,
                layer_norm,
                causal=causal,
                block_size=block_size,
            )

    def forward(self, x):
        z_init = x

        # Use the custom fixed-point autograd function
        z_star = FixedPointIteration.apply(
            lambda inp, z: self.attention_step(z, inp),
            x,
            z_init,
            self.max_iter,
            self.eps,
        )

        out = self.out_proj(z_star)

        if self.residual:
            out = out + x
        if self.normalize is not None:
            out = self.normalize(out)

        return out
