import torch
# 导入支持深度堆叠的 Encoder 类
from attention import SelfEncoder, CrossEncoder

def main():
    # --- 1. 全局超参数设定 ---
    batch_size = 2
    dim = 768            # 统一的特征维度
    num_heads = 8
    depth = 6            # <--- 新增：网络的深度（层数）

    N = 196              # 目标序列长度 (例如: 图像 Patch 数量)
    context_N = 50       # 上下文序列长度 (例如: 另一模态的 Token 数量)

    print("--- 1. 初始化张量 ---")
    x = torch.randn(batch_size, N, dim)
    context = torch.randn(batch_size, context_N, dim)
    print(f"Target 流 (x):       {x.shape}")
    print(f"Context 流 (context): {context.shape}\n")

    # --- 2. 测试多层 Self-Encoder ---
    print(f"--- 2. 测试 {depth} 层 Self-Encoder ---")
    # 实例化 SelfEncoder，传入 depth
    self_encoder = SelfEncoder(depth=depth, dim=dim, num_heads=num_heads)

    # 前向传播
    out_self = self_encoder(x)
    print(f"Self-Encoder 输出: {out_self.shape} \n(预期: 与 x 形状完全一致)\n")

    # --- 3. 测试多层 Cross-Encoder ---
    print(f"--- 3. 测试 {depth} 层 Cross-Encoder ---")
    # 实例化 CrossEncoder，传入 depth
    cross_encoder = CrossEncoder(depth=depth, dim=dim, num_heads=num_heads)

    # 创建掩码：假设我们要屏蔽 context 序列的后 10 个 Token
    attn_mask = torch.ones(batch_size, context_N, dtype=torch.bool)
    attn_mask[:, -10:] = False 

    # 前向传播 (x 作为 Query，context 作为 Key/Value 贯穿所有层)
    # 注意：在深层 CrossEncoder 中，残差是加在 x 上的
    out_cross = cross_encoder(x=x, context=context, attn_mask=attn_mask)
    print(f"Cross-Encoder 输出: {out_cross.shape} \n(预期: 吸收了 context 信息，但最终形状依然与 x 绝对一致)")

if __name__ == "__main__":
    main()