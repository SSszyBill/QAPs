from sk_solver import SK_QAP_Solver
import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linear_sum_assignment
class TestSK:
    def __init__(self, n, A, B, z, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n = n
        self.A = torch.tensor(A, dtype=torch.float32, device=device)
        self.B = torch.tensor(B, dtype=torch.float32, device=device)
        self.z = z
        self.device = device
    def test_sk (self):
        solver = SK_QAP_Solver(self.n, A, B, device=self.device)
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

def solve_birkhoff_projection(A_tensor, n):
    """
    使用 Gurobi 将矩阵 A 投影到双随机矩阵集合 (Birkhoff Polytope) 上。
    min ||X - A||_F^2
    s.t. X >= 0, X*1 = 1, X.T*1 = 1
    """
    
    # 1. 数据准备：将 PyTorch Tensor 转为 Numpy 方便 Gurobi 处理
    A = A_tensor.detach().cpu().numpy()
    
    # 2. 建立模型
    model = gp.Model("Birkhoff_Projection")
    
    #以此关闭 Gurobi 的控制台输出（如果想看求解过程可设为 1）
    model.setParam('OutputFlag', 0) 
    
    # 3. 定义变量
    # X 是 n x n 的矩阵，lb=0.0 满足非负性约束 (X >= 0)
    # ub=1.0 是隐式的（因为和为1），但显式加上有助于求解器预处理
    x = model.addVars(n, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    
    # 4. 设置目标函数：最小化 Frobenius 范数的平方
    # min sum((x_ij - A_ij)^2)
    # 展开后等价于 min sum(x_ij^2 - 2*A_ij*x_ij)，常数项 A_ij^2 不影响最优解，
    # 但为了保持目标函数值为真实的距离平方，我们保留完整形式。
    obj_expr = gp.quicksum((x[i, j] - A[i, j]) * (x[i, j] - A[i, j]) 
                           for i in range(n) for j in range(n))
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # 5. 添加约束
    
    # 约束 1: 行和为 1 (X * 1 = 1)
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1.0, name=f"row_{i}")
        
    # 约束 2: 列和为 1 (X.T * 1 = 1)
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1.0, name=f"col_{j}")
        
    # 6. 开始求解
    model.optimize()
    
    # 7. 获取结果并转回 PyTorch Tensor
    X_sol = torch.zeros((n, n))
    
    if model.status == GRB.OPTIMAL:
        # 提取解
        solution = model.getAttr('X', x)
        for i in range(n):
            for j in range(n):
                X_sol[i, j] = solution[i, j]
        return X_sol
    else:
        print("Optimization failed or was infeasible.")
        return None
    
def evaluate_single(P_soft):
    """匈牙利算法离散化"""
    with torch.no_grad():
        P_np = P_soft.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(-P_np)
        P_bin = torch.zeros((n, n), device=P_soft.device)
        P_bin[row_ind, col_ind] = 1.0
        return P_bin

if __name__ == "__main__":
    n = 10
    z = torch.randn(size=(n, n)) 
    A = torch.ones(size=(n, n))
    B = torch.ones(size=(n, n))
    print(z)

    if torch.any(z < 0):
        print("Initial z has negative values!")
    if torch.any(z > 1):
        print("Initial z has values greater than 1!")
    #sinkhorn
    z_clamp = torch.clamp(z, 0.0)
    tester = TestSK(n, A.numpy(), B.numpy(), z_clamp)
    P_sk = tester.test_sk()
    is_valid, max_error = tester.check_sk(P_sk)
    if is_valid:
        print(f"✅ P 是双随机矩阵，最大误差: {max_error:.6f}")
    else:
        print(f"❌ P 不是双随机矩阵，最大误差: {max_error:.6f}")

    #gurobi birkhoff projection
    X_projected = solve_birkhoff_projection(z, n)
    if X_projected is not None:
        print("\nOptimization Finished!")
    
    # 检查行和与列和
    row_sums = torch.sum(X_projected, dim=1)
    col_sums = torch.sum(X_projected, dim=0)
    print(f"X_projected row sums: {row_sums}")
    print(f"X_projected col sums: {col_sums}")
    print(f"Max deviation in Row Sums (should be 0): {torch.max(torch.abs(row_sums - 1.0)).item():.2e}")
    print(f"Max deviation in Col Sums (should be 0): {torch.max(torch.abs(col_sums - 1.0)).item():.2e}")
    print(f"Min value in X (should be >= 0): {torch.min(X_projected).item():.2e}")
    error = X_projected - P_sk
    print(error)

    P_bin = evaluate_single(P_sk)
    X_bin = evaluate_single(X_projected)
    if torch.allclose(X_bin, P_bin, atol=1e-3):
        print("✅ X_bin 与 P_bin 相等")
    else:
        print("❌ X_bin 与 P_bin 不相等\n")
        print(f"max error: {torch.max(torch.abs(error)).item():.2e}")
