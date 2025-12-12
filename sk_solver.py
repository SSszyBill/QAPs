# solver.py
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
from sinkhorn import SinkhornLayer

class SK_QAP_Solver:
    def __init__(self, n, A_np, B_np, device='cpu', tau=1.0, batch_size=100):
        self.n = n
        self.device = device
        self.batch_size = batch_size
        
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
        
        # === 新增：用于绘图的 Cost 历史记录 ===
        cost_history = []
        
        for t in range(max_iter):
            # === 温度衰减策略 (Annealing) ===
            progress = t / max_iter
            current_tau = max(0.05, self.init_tau * (1 - progress))
            self.sk_layer.tau = current_tau

            # === Step 1: Forward ===
            self.optimizer.zero_grad()
            
            # P: (Batch, N, N)
            P = self.sk_layer(self.Z)
            
            # === Loss 计算 (Batch Wise) ===
            # QAP Objective: tr(A P B P^T)
            term1 = torch.matmul(self.A, P)       
            term2 = torch.matmul(self.B, P.transpose(1, 2)) 
            
            # Trace 的 Batch 版本: 对角线求和
            prod = torch.matmul(term1, term2.transpose(1, 2))
            loss_qap = torch.diagonal(prod, dim1=1, dim2=2).sum(-1) # (Batch,)
            
            # PDBO Penalty: sum(Y * (P^2 - P))
            g_val = P**2 - P
            loss_pdbo = torch.sum(self.Y * g_val, dim=(1, 2)) # (Batch,)
            
            # 总 Loss
            loss_total = (loss_qap + loss_pdbo).mean()
            
            # === 新增：记录当前 Batch 中最好的连续 Cost (用于画图) ===
            # 这里记录 loss_qap 的最小值，代表当前优化方向的“能量”
            current_best_continuous = loss_qap.min().item()
            cost_history.append(current_best_continuous)
            
            # === Step 2: Backward ===
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([self.Z], max_norm=1.0)
            self.optimizer.step()
            
            # === Step 3: Dual Update ===
            with torch.no_grad():
                self.Y += lr_dual * g_val
                
                # 扰动机制
                if t % 100 == 0:
                     self.Z.data += torch.randn_like(self.Z) * 0.01 * (1-progress)

            # === Step 4: 评估 ===
            # 提高频率到每 100 次检查一次，或者最后一次必定检查
            if t % 100 == 0 or t == max_iter - 1:
                # 找到当前 Batch 中 Loss (QAP+Penalty) 最小的索引
                # 注意：这里我们用 loss_qap + loss_pdbo 来选最“合法”且 Cost 低的
                min_loss_idx = torch.argmin(loss_qap + loss_pdbo)
                
                # 离散化评估这个最好的
                best_in_batch_P = P[min_loss_idx]
                cost, perm = self.evaluate_single(best_in_batch_P)
                
                if cost < global_best_score:
                    global_best_score = cost
                    global_best_perm = perm
                
                # 打印日志 (可选)
                # int_gap = torch.abs(g_val).mean().item()
                # print(f"Iter {t} | Cost: {cost:.2f} | Best: {global_best_score:.2f}")

        # === 修改返回值：增加 cost_history ===
        return global_best_score, global_best_perm, cost_history

    def evaluate_single(self, P_soft):
        """评估单个矩阵 (非 Batch)"""
        with torch.no_grad():
            P_np = P_soft.detach().cpu().numpy()
            
            # 处理 NaN/Inf 情况
            if not np.isfinite(P_np).all():
                P_np = np.nan_to_num(P_np, nan=0.0, posinf=0.0, neginf=0.0)

            # 匈牙利算法离散化
            row_ind, col_ind = linear_sum_assignment(-P_np)
            
            P_bin = torch.zeros((self.n, self.n), device=self.device)
            P_bin[row_ind, col_ind] = 1.0

            # 计算离散 Cost
            # squeeze() 确保维度匹配: (N, N) @ (N, N) ...
            A_sq = self.A.squeeze(0)
            B_sq = self.B.squeeze(0)
            
            cost = torch.trace(A_sq @ P_bin @ B_sq @ P_bin.t())
            return cost.item(), P_bin