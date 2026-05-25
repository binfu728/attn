import torch
from attention import SelfAttention, CrossAttention

# --- 1. 全局超参数设定 ---
batch_size = 2
dim = 768            # 统一的特征维度
num_heads = 8

N = 196      # 目标序列长度 (例如: 图像 Patch 数量)
context_N = 50     # 上下文序列长度 (例如: 另一模态的 Token 数量)

print("--- 1. 初始化张量 ---")
x = torch.randn(batch_size, N, dim)
context = torch.randn(batch_size, context_N, dim)
print(f"Target 流 (x):       {x.shape}")
print(f"Context 流 (context): {context.shape}\n")

# --- 2. 测试 Self-Attention ---
print("--- 2. 测试 Self-Attention ---")
self_attn = SelfAttention(dim=dim, num_heads=num_heads)

# 前向传播
out_self = self_attn(x)
print(f"Self-Attention 输出: {out_self.shape} \n(预期: 与 x 形状完全一致)\n")

# --- 3. 测试 Cross-Attention ---
print("--- 3. 测试 Cross-Attention ---")
cross_attn = CrossAttention(dim=dim, num_heads=num_heads)

# 创建掩码：假设我们要屏蔽 context 序列的后 10 个 Token
attn_mask = torch.ones(batch_size, context_N, dtype=torch.bool)
attn_mask[:, -10:] = False 

# 前向传播 (x 作为 Query，context 作为 Key/Value)
out_cross = cross_attn(x=x, context=context, attn_mask=attn_mask)
print(f"Cross-Attention 输出: {out_cross.shape} \n(预期: 吸收了 context 信息，但最终形状依然与 x 绝对一致)")
