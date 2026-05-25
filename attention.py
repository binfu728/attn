"""Deep Transformer Architecture with Decoupled Attention Modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ==========================================
# 1. 核心 Attention 算子 (之前写好的)
# ==========================================

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    """标准的 Feed Forward Network (MLP)"""
    def __init__(self, in_features: int, hidden_features: int, act_layer: nn.Module = nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

# ==========================================
# 2. Block 封装 (添加残差、Norm 和 MLP)
# ==========================================

class SelfAttentionBlock(nn.Module):
    """单层 Self-Attention Transformer Block (Pre-LN)"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_norm: bool = False, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, qk_norm, norm_layer)
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """单层 Cross-Attention Transformer Block (Pre-LN)"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_norm: bool = False, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        # Cross-Attention 的特点是：目标序列 x 参与残差和后续的 MLP 计算
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, qk_norm, norm_layer)
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x 是被更新的主干流，context 仅提供外部信息
        x = x + self.cross_attn(self.norm1(x), context=context, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 3. 多层 Encoder 网络
# ==========================================

class SelfEncoder(nn.Module):
    """多层 Self-Attention 网络"""
    def __init__(self, depth: int, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_norm: bool = False, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_norm, norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(dim) # 最终的归一化层

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return self.norm(x)


class CrossEncoder(nn.Module):
    """多层 Cross-Attention 网络"""
    def __init__(self, depth: int, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_norm: bool = False, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_norm, norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(dim) # 最终的归一化层

    def forward(self, x: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            # context 贯穿所有层，作为提供外部知识的锚点
            x = blk(x, context=context, attn_mask=attn_mask)
        return self.norm(x)