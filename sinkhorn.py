# sinkhorn.py
import torch
import torch.nn as nn

class SinkhornLayer(nn.Module):
    def __init__(self, n_iter=20, tau=1.0, epsilon=1e-8):
        super().__init__()
        self.n_iter = n_iter
        self.tau = tau
        self.epsilon = epsilon

    def forward(self, z):
        """
        输入 z: (Batch_Size, N, N) 或 (N, N)
        """
        # 兼容非 Batch 输入
        if z.dim() == 2:
            z = z.unsqueeze(0)
        
        # === 数值稳定性修复 ===
        # 1. 防止除以过小的 tau 导致数值爆炸
        # 实际 tau 不应低于 1e-3
        effective_tau = max(self.tau, 1e-3)
        
        # 2. Log-Sum-Exp 技巧的前置步骤：减去最大值防止 exp 溢出
        # z_max: (B, N, 1)
        z_max = torch.max(z, dim=-1, keepdim=True)[0]
        z_stable = z - z_max
        
        # 3. 指数映射
        A = torch.exp(z_stable / effective_tau)
        
        # B, N, _ = A.shape
        
        # 4. 初始化缩放向量
        r = torch.ones((A.shape[0], A.shape[1], 1), device=A.device, dtype=A.dtype)
        c = torch.ones((A.shape[0], 1, A.shape[2]), device=A.device, dtype=A.dtype)

        # 5. 交替归一化
        for _ in range(self.n_iter):
            # c = 1 / (r^T @ A)
            c = 1.0 / (torch.matmul(r.transpose(1, 2), A) + self.epsilon)
            
            # r = 1 / (A @ c^T)
            r = 1.0 / (torch.matmul(A, c.transpose(1, 2)) + self.epsilon)

        # 6. P = r * A * c
        P = r * A * c
        
        # 7. 再次进行数值裁剪，防止极个别 NaN 逃逸
        # 将 NaN 替换为 0，将极小负数截断（虽理论上不应有负数）
        P = torch.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
        
        return P.squeeze()

# sinkhorn.py
# import torch
# import torch.nn as nn

# class SinkhornLayer(nn.Module):
#     """
#     Sinkhorn-Knopp Layer
#     作用：将任意实数矩阵投影到 Birkhoff 多面体 (满足行和=1, 列和=1)
#     """
#     def __init__(self, n_iter=20, tau=1.0, epsilon=1e-8):
#         super().__init__()
#         self.n_iter = n_iter
#         self.tau = tau  # 温度参数：越小越接近硬分配(Hard Assignment)
#         self.epsilon = epsilon

#     def forward(self, z):
#         """
#         Args:
#             z: [n, n] 任意实数得分矩阵 (Logits)
#         Returns:
#             P: [n, n] 双随机矩阵
#         """
#         # 1. 指数映射 (Score -> Positive Matrix)
#         A = torch.exp(z / self.tau)
        
#         # 2. 初始化缩放向量
#         r = torch.ones(A.shape[0], 1, device=A.device, dtype=A.dtype)
#         c = torch.ones(1, A.shape[1], device=A.device, dtype=A.dtype)

#         # 3. 交替归一化 (Alternating Normalization)
#         for _ in range(self.n_iter):
#             # Update c (Col norm): c = 1 / (r^T * A)
#             c = 1.0 / (torch.matmul(r.T, A) + self.epsilon)
#             # Update r (Row norm): r = 1 / (A * c^T)
#             r = 1.0 / (torch.matmul(A, c.T) + self.epsilon)

#         # 4. 计算 P = D(r) * A * D(c)
#         P = r * A * c
#         return P