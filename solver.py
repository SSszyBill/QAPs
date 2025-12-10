# solver.py
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
from sinkhorn import SinkhornLayer

class QAPSolver:
    def __init__(self, n, A_np, B_np, device='cpu', tau=1.0, batch_size=100):
        self.n = n
        self.device = device
        self.batch_size = batch_size  # 新增：并行数量
        
        # 1. 数据扩展到 Batch 维度
        # A, B: (N, N) -> (1, N, N) 以便广播
        self.A = torch.tensor(A_np, dtype=torch.float32, device=device).unsqueeze(0)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 2. 初始化 Primal Z (Batch, N, N)
        # 使用不同的随机种子初始化，探索不同区域
        self.Z = torch.randn((batch_size, n, n), device=device, requires_grad=True)
        
        # 3. 初始化 Dual Y (Batch, N, N)
        self.Y = torch.ones((batch_size, n, n), device=device) * 5.0
        
        # 4. Sinkhorn
        self.init_tau = tau
        self.sk_layer = SinkhornLayer(n_iter=20, tau=tau)
        
        self.optimizer = torch.optim.Adam([self.Z], lr=0.1)

    def solve(self, max_iter=2000, lr_dual=0.01):
        # 记录全局最优
        global_best_score = float('inf')
        global_best_perm = None
        
        for t in range(max_iter):
            # === 温度衰减策略 (Annealing) ===
            # 从 init_tau 慢慢降到 0.1，让结构先模糊后清晰
            progress = t / max_iter
            current_tau = max(0.05, self.init_tau * (1 - progress))
            self.sk_layer.tau = current_tau

            # === Step 1: Forward ===
            self.optimizer.zero_grad()
            
            # P: (Batch, N, N)
            P = self.sk_layer(self.Z)
            
            # === Loss 计算 (Batch Wise) ===
            # QAP Objective: tr(A P B P^T)
            # 矩阵乘法: (B, N, N) @ (B, N, N) -> (B, N, N)
            term1 = torch.matmul(self.A, P)       # A 自动广播
            term2 = torch.matmul(self.B, P.transpose(1, 2)) # B 自动广播
            
            # Trace 的 Batch 版本: 对角线求和
            # term1 @ term2.T 的对角线
            # 更高效的方法: sum(elementwise_product)
            # tr(X Y) = sum(X * Y^T)
            prod = torch.matmul(term1, term2.transpose(1, 2))
            loss_qap = torch.diagonal(prod, dim1=1, dim2=2).sum(-1) # (Batch,)
            
            # PDBO Penalty: sum(Y * (P^2 - P))
            g_val = P**2 - P
            loss_pdbo = torch.sum(self.Y * g_val, dim=(1, 2)) # (Batch,)
            
            # 总 Loss (对 Batch 求和或平均供 optimizer 使用)
            loss_total = (loss_qap + loss_pdbo).mean()
            
            # === Step 2: Backward ===
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([self.Z], max_norm=1.0)
            self.optimizer.step()
            
            # === Step 3: Dual Update ===
            with torch.no_grad():
                # 每个 Batch 独立更新自己的 Y
                self.Y += lr_dual * g_val
                
                # 扰动机制
                if t % 100 == 0:
                     self.Z.data += torch.randn_like(self.Z) * 0.01 * (1-progress)

            # === Step 4: 评估 (定期只取 Batch 中最好的打印) ===
            if t % 500 == 0 or t == max_iter - 1:
                # 找到当前 Batch 中 Loss 最小的那个索引
                min_loss_idx = torch.argmin(loss_qap + loss_pdbo)
                
                # 离散化评估这个最好的
                best_in_batch_P = P[min_loss_idx]
                cost, perm = self.evaluate_single(best_in_batch_P)
                
                if cost < global_best_score:
                    global_best_score = cost
                    global_best_perm = perm
                
                # 打印日志
                int_gap = torch.abs(g_val).mean().item()
                # print(f"Iter {t} | Tau: {current_tau:.3f} | BatchBestCost: {cost:.2f} | GlobalBest: {global_best_score:.2f} | Gap: {int_gap:.4f}")

        return global_best_score, global_best_perm

    def evaluate_single(self, P_soft):
        """评估单个矩阵 (非 Batch)"""
        with torch.no_grad():
            P_np = P_soft.detach().cpu().numpy()
            if not np.isfinite(P_np).all():
                # print("[Warning] NaN/Inf detected in P, sanitizing...")
                P_np = np.nan_to_num(P_np, nan=0.0, posinf=0.0, neginf=0.0)

            row_ind, col_ind = linear_sum_assignment(-P_np)
            
            P_bin = torch.zeros((self.n, self.n), device=self.device)
            P_bin[row_ind, col_ind] = 1.0

            A_curr = self.A.squeeze() if self.A.dim() == 3 else self.A
            B_curr = self.B.squeeze() if self.B.dim() == 3 else self.B
            
            # 注意 A 和 B 在 self 里是 (1, N, N)，计算时要 squeeze
            cost = torch.trace(self.A.squeeze() @ P_bin @ self.B.squeeze() @ P_bin.t())
            return cost.item(), P_bin