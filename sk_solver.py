import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class SK_QAP_Solver(nn.Module):
    def __init__(self, n, A_np, B_np, device='cpu', batch_size=200, **kwargs):
        super().__init__()
        self.n = n
        self.device = device
        self.batch_size = batch_size
        
        # 1. 流量与距离矩阵
        self.A = torch.tensor(A_np, dtype=torch.float32, device=device).unsqueeze(0)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 2. 决策变量 Z (连续，可正可负)
        self.Z = nn.Parameter(torch.randn(batch_size, n, n, device=device))
        
        # 3. 对偶变量 Y
        self.Y = torch.ones(batch_size, n, n, device=device) * 5.0
        
        # 优化器
        self.optimizer = torch.optim.Adam([self.Z], lr=0.1)

    def knight_sinkhorn_step(self, A, n_iter=20):
        """Knight 算法：把非负矩阵 A 修理成双随机矩阵 P"""
        r = torch.ones((A.shape[0], A.shape[1], 1), device=A.device)
        c = torch.ones((A.shape[0], A.shape[1], 1), device=A.device)
        
        for _ in range(n_iter):
            # c = 1 / (A^T * r)
            val_c = torch.matmul(A.transpose(1, 2), r)
            c = 1.0 / (val_c + 1e-12)
            
            # r = 1 / (A * c)
            val_r = torch.matmul(A, c)
            r = 1.0 / (val_r + 1e-12)
            
        # P = D A E
        P = r * A * c.transpose(1, 2)
        return P

    def solve(self, max_iter=2000, lr_dual=0.01, log_interval=100, log_path=None):
        global_best_score = float('inf')
        global_best_perm = None
        
        soft_cost_history = [] # 记录每一步的连续 Loss
        real_cost_history = [] # 记录每 log_interval 步的真实离散 Cost (iter, cost)
        
        # === 1. 初始化日志头 (固定宽度对齐) ===
        # 格式说明: 
        # :<8  左对齐，占8格
        # :<12 左对齐，占12格
        header = f"{'Iter':<8} | {'SoftCost':<12} | {'RealCost':<12} | {'Vio(Bin)':<12} | {'Grad|Z|':<12}"
        
        if log_path:
            with open(log_path, 'w') as f:
                f.write(header + "\n")
        
        # 终端打印
        print("-" * 70)
        print(header)
        print("-" * 70)

        for t in range(max_iter):
            self.optimizer.zero_grad()
            
            # === 2. 核心逻辑 ===
            A_in = torch.sigmoid(self.Z)               
            P = self.knight_sinkhorn_step(A_in)        
            
            # === 3. 计算 Loss ===
            term1 = torch.matmul(self.A, P)       
            term2 = torch.matmul(self.B, P.transpose(1, 2)) 
            prod = torch.matmul(term1, term2.transpose(1, 2))
            loss_qap_batch = torch.diagonal(prod, dim1=1, dim2=2).sum(-1)
            
            g_val = P**2 - P
            loss_pdbo_batch = torch.sum(self.Y * g_val, dim=(1, 2))
            
            loss_total = (loss_qap_batch + loss_pdbo_batch).mean()
            
            # === 4. 梯度更新 ===
            loss_total.backward()
            grad_norm = self.Z.grad.norm().item() 
            torch.nn.utils.clip_grad_norm_([self.Z], max_norm=1.0)
            self.optimizer.step()
            
            # === 5. 对偶变量更新 & 记录 Soft Cost ===
            with torch.no_grad():
                self.Y += lr_dual * g_val
                
                current_soft_min = loss_qap_batch.min().item()
                soft_cost_history.append(current_soft_min)
                
                if t % 100 == 0:
                     self.Z.data += torch.randn_like(self.Z) * 0.01

            # === 6. 评估与打印 ===
            if t % log_interval == 0 or t == max_iter - 1:
                
                # 找到 Soft Loss 最小的解
                min_idx = torch.argmin(loss_qap_batch + loss_pdbo_batch)
                best_soft_P = P[min_idx]
                
                # 计算真实离散 Cost
                real_cost, perm = self.evaluate_single(best_soft_P)
                
                # 记录 Real Cost (带上时间戳 t，方便画图)
                real_cost_history.append((t, real_cost))
                
                if real_cost < global_best_score:
                    global_best_score = real_cost
                    global_best_perm = perm
                
                vio = g_val.abs().mean().item()
                
                # 格式化打印 (注意宽度要和 header 一致)
                log_str = f"{t:<8} | {current_soft_min:<12.2f} | {real_cost:<12.2f} | {vio:<12.4f} | {grad_norm:<12.4f}"
                print(log_str)
                
                if log_path:
                    with open(log_path, 'a') as f:
                        f.write(log_str + "\n")
                if t == max_iter - 1:
                    print(t,"\n")
                    print(best_soft_P,"\n")
                    print(self.evaluate_single(best_soft_P),"\n")
        # 返回两个 history
        return global_best_score, global_best_perm, soft_cost_history, real_cost_history

    def evaluate_single(self, P_soft):
        """匈牙利算法离散化"""
        with torch.no_grad():
            P_np = P_soft.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-P_np)
            P_bin = torch.zeros((self.n, self.n), device=self.device)
            P_bin[row_ind, col_ind] = 1.0
            A_sq = self.A.squeeze(0)
            B_sq = self.B.squeeze(0)
            cost = torch.trace(A_sq @ P_bin @ B_sq @ P_bin.t())
            return cost.item(), P_bin