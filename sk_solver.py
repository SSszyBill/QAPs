import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class SK_QAP_Solver(nn.Module):
    def __init__(self, n, A_np, B_np, device = 'cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        super().__init__()
        self.n = n
        self.device = device
        # 1. 流量与距离矩阵
        self.A = torch.tensor(A_np, dtype=torch.float32, device=device)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device)
        # 2. 决策变量 P (连续，可正可负)
        self.P = torch.rand(size=(n, n), device=device) + 1e-8
        # 3. 对偶变量 Y
        self.Y = torch.ones(n, n, device=device) * 5.0

    def knight_sinkhorn_step(self, A, n_iter=500):
        """Knight 算法：把非负矩阵 A 修理成双随机矩阵 P"""
        n = A.shape[0]
        r = torch.ones((n, 1), device=A.device)
        c = torch.ones((n, 1), device=A.device)

        for _ in range(n_iter):
            val_c = torch.matmul(A.t(), r)
            c = 1.0 / (val_c + 1e-12)
            
            val_r = torch.matmul(A, c)
            r = 1.0 / (val_r + 1e-12)
        P = r * A * c.t()
        return P
    def check_sinkhorn(self, P):
        """
        检查矩阵 P 是否接近双随机矩阵
        功能增强：
        - 计算每行/每列的和
        - 若某行/列和与1的差距>判定阈值(0.1)，打印提示并标注行号/列号
        """
        # 统一阈值：和判定不合格的阈值对齐（可根据需要调整）
        ERROR_THRESHOLD = 1e-2  
        has_large_error = False

        P = torch.clamp(P, min=1e-12)  # 确保非负
        if P.dim() == 2:
            # 2维情况：P shape [n, n]
            n = P.shape[0]
            row_sum = P.sum(dim=1)  # 每行求和 → [n]
            col_sum = P.sum(dim=0)  # 每列求和 → [n]
            # 检查行和
            for row_idx in range(n):
                row_diff = abs(row_sum[row_idx].item() - 1.0)
                if row_diff > ERROR_THRESHOLD:
                    has_large_error = True
                    print(f"⚠️ P (2维) 行{row_idx} 和为 {row_sum[row_idx].item():.6f},与1的差距 {row_diff:.6f} > {ERROR_THRESHOLD} → 不满足双随机")
            # 检查列和
            for col_idx in range(n):
                col_diff = abs(col_sum[col_idx].item() - 1.0)
                if col_diff > ERROR_THRESHOLD:
                    has_large_error = True
                    print(f"⚠️ P (2维) 列{col_idx} 和为 {col_sum[col_idx].item():.6f},与1的差距 {col_diff:.6f} > {ERROR_THRESHOLD} → 不满足双随机")
        else:
            raise ValueError(f"不支持的P维度:{P.dim()},仅支持2维")
        
        # 计算全局最大误差（原逻辑）
        row_error = torch.max(torch.abs(row_sum - 1.0)).item()
        col_error = torch.max(torch.abs(col_sum - 1.0)).item()
        max_error = max(row_error, col_error)
        is_valid = max_error < ERROR_THRESHOLD # 判定不合格的核心阈值
        
        # 若存在超标情况，补充全局提示
        if has_large_error:
            print(f"❌ 全局最大误差: {max_error:.6f},P 不是双随机矩阵(判定阈值:{ERROR_THRESHOLD})")
        return is_valid, max_error
    
    def solve(self, max_iter, lr_primal=0.1, lr_dual=0.5, log_interval=100, log_path=None):
        global_best_score = float('inf')
        global_best_perm = None
        
        soft_cost_history = [] # 记录每一步的连续 Loss
        real_cost_history = [] # 记录每 log_interval 步的真实离散 Cost (iter, cost)
        
        # === 1. 初始化日志头 (固定宽度对齐) ===
        # 格式说明: 
        # :<8  左对齐，占8格
        # :<12 左对齐，占12格
        header = f"{'Iter':<8} | {'SoftCost':<12} | {'RealCost':<12} | {'Vio(Bin)':<12} |\
             {'Grad|Z|':<12} | {'Loss_pdbo':<12} | {'y':<12} | {'y_min':<12}" 
        
        if log_path:
            with open(log_path, 'w') as f:
                f.write(header + "\n")
        
        # 终端打印
        print("-" * 70)
        print(header)
        print("-" * 70)

        for t in range(max_iter):
            # === 2. 核心逻辑 ===            
            self.P = self.knight_sinkhorn_step(self.P)
            is_valid, max_error = self.check_sinkhorn(self.P)
            if not is_valid:
                print(f"Iteration {t}: Sinkhorn failed to produce double stochastic matrix (max error: {max_error:.6f}). Stopping early.")  
                break
            if torch.any(self.P < 0.0):
                print(f"Iteration {t}: After Sinkhorn, P has negative values. Stopping early.")
                break
            # === 3. 计算 Loss ===
            # P = torch.clamp(P, 1e-8, 1.0)
            term1 = torch.matmul(self.A, self.P)       
            term2 = torch.matmul(self.B, self.P.t()) 
            prod = torch.matmul(term1, term2.t())
            loss_qap = torch.trace(prod)
            
            g_val = self.P**2 - self.P
            loss_pdbo = torch.sum(self.Y * g_val)
            loss_total = loss_qap + loss_pdbo
            # === 4. 梯度更新 ===
            grad_qap = self.A @ self.P @ self.B.t() + self.A.t() @ self.P @ self.B
            grad_pdbo = 2 * self.Y * self.P - self.Y
            grad_total = grad_qap + grad_pdbo
            grad_norm = torch.norm(grad_total).item()
            self.P = self.P - lr_primal * grad_total
            self.P = torch.clamp(self.P, min=1e-12)
            if torch.any(self.P > 1):
                print(f"Iteration {t}: After gradient update, P has values >1. Stopping early.")
            if torch.any(self.P < 0):
                print(f"Iteration {t}: After gradient update, P has negative values. Stopping early.")
                
            
            # === 5. 对偶变量更新 & 记录 Soft Cost ===
            with torch.no_grad():
                self.Y += lr_dual * g_val
                
                current_soft_cost = loss_qap.item()
                soft_cost_history.append(current_soft_cost)
                
                # 每100步添加小噪音
                if t % 100 == 0:
                     self.P.data += torch.randn_like(self.P) * 0.05

            # === 6. 评估与打印 ===
            if t % log_interval == 0 or t == max_iter - 1:
                
                # 计算真实离散 Cost
                real_cost, perm = self.evaluate_single(self.P)
                
                # 记录 Real Cost (带上时间戳 t，方便画图)
                real_cost_history.append((t, real_cost))
                
                if real_cost < global_best_score:
                    global_best_score = real_cost
                    global_best_perm = perm
                
                vio = g_val.abs().mean().item()
                
                # 格式化打印 (注意宽度要和 header 一致)
                log_str = f"{t:<8} | {current_soft_cost:<12.2f} | {real_cost:<12.2f} | {vio:<12.4f} |\
                {grad_norm:<12.4f} | {loss_pdbo.item():<12.2f} | {self.Y.mean().item():<12.4f} | {self.Y.min().item():<12.4f}"
                print(log_str)
                
                if log_path:
                    with open(log_path, 'a') as f:
                        f.write(log_str + "\n")
                if t == max_iter - 1:
                    print(t,"\n")
                    row_sum = self.P.sum(dim=1) 
                    col_sum = self.P.sum(dim=0)
                    row_error = torch.max(torch.abs(row_sum - 1.0)).item()
                    col_error = torch.max(torch.abs(col_sum - 1.0)).item()
                    print(f"Final Check - Max Row Error: {row_error:.6f}, Max Col Error: {col_error:.6f}")

        # 返回两个 history
        return global_best_score, global_best_perm, soft_cost_history, real_cost_history

    def evaluate_single(self, P_soft):
        """匈牙利算法离散化"""
        with torch.no_grad():
            P_np = P_soft.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-P_np)
            P_bin = torch.zeros((self.n, self.n), device=self.device)
            P_bin[row_ind, col_ind] = 1.0
            cost = torch.trace(self.A @ P_bin @ self.B @ P_bin.t())
            return cost.item(), P_bin