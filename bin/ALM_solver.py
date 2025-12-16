# alm_solver.py
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np

class ALM_QAP_Solver:
    def __init__(self, n, A_np, B_np, device='cpu', rho=10.0, batch_size=100):
        self.n = n
        self.device = device
        self.batch_size = batch_size
        self.rho = rho  # ALM 的惩罚系数 (Penalty Parameter)
        
        # 1. 数据扩展 (Batch, N, N)
        self.A = torch.tensor(A_np, dtype=torch.float32, device=device).unsqueeze(0)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 2. Primal 变量 Z (Logits)
        # 我们用 Sigmoid(Z) 来得到 P，所以 Z 初始化在 0 附近即可
        self.Z = torch.randn((batch_size, n, n), device=device, requires_grad=True)
        
        # 3. Dual 变量 Y (PDBO - 整数约束)
        self.Y = torch.ones((batch_size, n, n), device=device) * 5.0
        
        # 4. Dual 变量 Lambda (ALM - 结构约束)
        # lambda_row: (Batch, N) 对应每一行的和
        # lambda_col: (Batch, N) 对应每一列的和
        self.lam_row = torch.ones((batch_size, n), device=device)
        self.lam_col = torch.ones((batch_size, n), device=device)
        
        self.optimizer = torch.optim.Adam([self.Z], lr=0.01) # 学习率可以适当调小

    def solve(self, max_iter=2000, lr_dual=0.01):
        global_best_score = float('inf')
        global_best_perm = None
        cost_history = []
        
        # 辅助向量：全1向量，用于计算行和列和
        ones_n = torch.ones(self.n, device=self.device)

        for t in range(max_iter):
            self.optimizer.zero_grad()
            
            # === Step 1: Forward (计算 P) ===
            # 不再用 Sinkhorn，而是简单的 Sigmoid 约束在 [0,1]
            P = torch.sigmoid(self.Z)
            
            # === Step 2: Loss 计算 ===
            
            # (A) QAP Objective: tr(A P B P^T)
            term1 = torch.matmul(self.A, P)
            term2 = torch.matmul(self.B, P.transpose(1, 2))
            prod = torch.matmul(term1, term2.transpose(1, 2))
            loss_qap = torch.diagonal(prod, dim1=1, dim2=2).sum(-1) # (Batch,)
            
            # (B) PDBO Penalty: sum Y * (P^2 - P)
            g_int = P**2 - P
            loss_pdbo = torch.sum(self.Y * g_int, dim=(1, 2)) # (Batch,)
            
            # (C) ALM Feasibility Terms (结构约束)
            # Row Error: sum_j P_ij - 1
            row_sum = P.sum(dim=2) # (Batch, N)
            err_row = row_sum - 1
            
            # Col Error: sum_i P_ij - 1
            col_sum = P.sum(dim=1) # (Batch, N)
            err_col = col_sum - 1
            
            # ALM Lagrangian: lambda * err + (rho/2) * ||err||^2
            # 这里的 dot product 对 Batch 中的每个样本单独算
            term_row = (self.lam_row * err_row).sum(1) + 0.5 * self.rho * (err_row**2).sum(1)
            term_col = (self.lam_col * err_col).sum(1) + 0.5 * self.rho * (err_col**2).sum(1)
            loss_alm = term_row + term_col
            
            # 总 Loss
            loss_total = (loss_qap + loss_pdbo + loss_alm).mean()
            
            # 记录 QAP Cost (便于画图)
            cost_history.append(loss_total.min().item())
            
            # === Step 3: Backward (Primal Update) ===
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([self.Z], max_norm=1.0)
            self.optimizer.step()
            
            # === Step 4: Dual Updates (Ascent) ===
            with torch.no_grad():
                # 4.1 Update PDBO Y (PDBO Rule)
                self.Y += lr_dual * g_int
                
                # 4.2 Update ALM Lambdas (Standard ALM Rule)
                self.lam_row += self.rho * err_row * 0.1 # 0.1 是 ALM 的 dual 学习率
                self.lam_col += self.rho * err_col * 0.1
                
                # (可选) 动态调整 rho (Penalty Scheduling)
                # 如果误差很难下降，可以逐渐增大 rho，但初期不宜过大
                # if t % 500 == 0: self.rho *= 1.1

            # === Step 5: 评估 ===
            if t % 100 == 0 or t == max_iter - 1:
                # 选取原则：综合考虑 Cost 和 可行性误差
                total_err = (err_row**2).sum(1) + (err_col**2).sum(1)
                selection_metric = loss_qap + 1000 * total_err # 优先选满足约束的
                
                min_idx = torch.argmin(selection_metric)
                best_in_batch_P = P[min_idx]
                
                cost, perm = self.evaluate_single(best_in_batch_P)
                
                if cost < global_best_score:
                    global_best_score = cost
                    global_best_perm = perm
                
                # 打印日志
                # mean_err = total_err.mean().item()
                # print(f"Iter {t} | Cost: {cost:.2f} | FeasErr: {mean_err:.4f}")

        return global_best_score, global_best_perm, cost_history

    def evaluate_single(self, P_soft):
        """评估单个矩阵 (非 Batch)"""
        with torch.no_grad():
            P_np = P_soft.detach().cpu().numpy()
            
            if not np.isfinite(P_np).all():
                P_np = np.nan_to_num(P_np, nan=0.0, posinf=0.0, neginf=0.0)

            row_ind, col_ind = linear_sum_assignment(-P_np)
            
            P_bin = torch.zeros((self.n, self.n), device=self.device)
            P_bin[row_ind, col_ind] = 1.0

            A_sq = self.A.squeeze(0)
            B_sq = self.B.squeeze(0)
            
            cost = torch.trace(A_sq @ P_bin @ B_sq @ P_bin.t())
            return cost.item(), P_bin