from sk_solver import SK_QAP_Solver
import torch
import numpy as np

class TestSK:
    def __init__(self, n, A, B, z, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n = n
        self.z = z
        self.device = device
    def test_sk (self):
        solver = SK_QAP_Solver(self.n, device=self.device)
        z_sk = solver.knight_sinkhorn_step(self.z, n_iter=100)
        return z_sk
    def check_sk(self, P):
        """
        检查矩阵 P 是否接近双随机矩阵
        功能增强：
        - 计算每行/每列的和
        - 若某行/列和与1的差距>判定阈值(0.1)，打印提示并标注行号/列号
        """
        # 统一阈值：和判定不合格的阈值对齐（可根据需要调整）
        ERROR_THRESHOLD = 1e-6 
        has_large_error = False
        
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

        # 计算全局最大误差（原逻辑）
        row_error = torch.max(torch.abs(row_sum - 1.0)).item()
        col_error = torch.max(torch.abs(col_sum - 1.0)).item()
        max_error = max(row_error, col_error)
        is_valid = max_error < ERROR_THRESHOLD  # 判定不合格的核心阈值
        
        # 若存在超标情况，补充全局提示
        if has_large_error:
            print(f"❌ 全局最大误差: {max_error:.6f},P 不是双随机矩阵(判定阈值:{ERROR_THRESHOLD})")
        return is_valid, max_error

if __name__ == "__main__":
    n = 26
    z = torch.rand(size=(n, n)) + 1e-8
    print(z)
    if torch.any(z < 0):
        print("Initial z has negative values!")
    if torch.any(z > 1):
        print("Initial z has values greater than 1!")
    tester = TestSK(n, z)
    P_sk = tester.test_sk()
    is_valid, max_error = tester.check_sk(P_sk)
    if is_valid:
        print(f"✅ P 是双随机矩阵，最大误差: {max_error:.6f}")
    else:
        print(f"❌ P 不是双随机矩阵，最大误差: {max_error:.6f}")