import torch
import argparse
import numpy as np
import time
from typing import Tuple, Optional

@torch.jit.script
def rmsprop_batch_jit(X: torch.Tensor, 
                      grad: torch.Tensor, 
                      v: torch.Tensor, 
                      buffer: torch.Tensor, 
                      gamma: float, 
                      alpha: float, 
                      lam: float, 
                      mu: float, 
                      eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. Weight Decay
    # grad: (bs, n, n), X: (bs, n, n)
    if lam != 0.0:
        grad = grad + lam * X
        
    # 2. Update Moving Average (v)
    v = alpha * v + (1.0 - alpha) * (grad * grad)
    
    # 3. Denominator
    denom = torch.sqrt(v) + eps
    
    # 4. Update Step
    if mu > 0.0:
        buffer = mu * buffer + grad / denom
        X = X - gamma * buffer
    else:
        X = X - gamma * grad / denom
        
    return X, v, buffer

@torch.jit.script
def dykstra_proj_batch_jit(M: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
    # X, P, Q: (bs, n, n)
    X = M.clone()
    p = torch.zeros_like(M)
    q = torch.zeros_like(M)
    
    n = M.size(1)
    
    for _ in range(max_iter):
        # --- Project onto affine subspace (Row/Col Sums = 1) ---
        Y = X + p
        
        # mannually unroll 2 iterations for jit acceleration
        Y = Y - (Y.sum(dim=2, keepdim=True) - 1.0) / n
        Y = Y - (Y.sum(dim=1, keepdim=True) - 1.0) / n

        Y = Y - (Y.sum(dim=2, keepdim=True) - 1.0) / n
        Y = Y - (Y.sum(dim=1, keepdim=True) - 1.0) / n
        
        p = X + p - Y
        X = Y
        
        # --- Project onto non-negative orthant (X >= 0) ---
        Y = X + q
        X = torch.clamp(Y, min=0.0)
        q = Y - X
        
    return X

def greedy_round_batch(M):
    """
    M: (bs, n, n)
    """
    bs, n, _ = M.shape
    assignment = torch.zeros_like(M)
    M_temp = M.clone()
    
    batch_indices = torch.arange(bs, device=M.device)
    
    for _ in range(n):
        # 1. 找到每个 batch 中当前最大的元素
        # view(bs, -1) 将矩阵展平为向量
        flat_M = M_temp.view(bs, -1)
        _, idx = flat_M.max(dim=1) # (bs,)
        
        # 2. 还原为行列索引
        rows = idx.div(n, rounding_mode='floor')
        cols = idx % n
        
        # 3. 填充结果
        assignment[batch_indices, rows, cols] = 1.0
        
        # 4. Mask (设为极小值)
        M_temp[batch_indices, rows, :] = -1e10
        M_temp[batch_indices, :, cols] = -1e10
        
    return assignment

def grad_L_X_batch(F, D, X, Y):
    """
    F X D^T + F^T X D + Y * (2X - 1)
    F, D: (n, n)
    X, Y: (bs, n, n)
    """
    # Term 1: F @ X @ D.T
    FX = torch.matmul(F, X) # (bs, n, n)
    FXDt = torch.matmul(FX, D.t()) # (bs, n, n)
    
    # Term 2: F.T @ X @ D
    FtX = torch.matmul(F.t(), X)
    FtXD = torch.matmul(FtX, D)
    
    # Term 3: Penalty
    penalty = Y * (2.0 * X - 1.0)
    
    return FXDt + FtXD + penalty

def grad_L_Y_batch(X):
    return X * X - X

def obj_fn_batch(F, D, X):
    """
    trace(F * X * D.T * X.T)
    """
    # X @ D.T @ X.T
    tmp = torch.matmul(torch.matmul(X, D.t()), X.transpose(1, 2))
    # trace(F @ tmp) = sum(F * tmp.T) element-wise
    # trace(A B) = sum(A * B^T)
    F_batch = F.unsqueeze(0) # (1, n, n)
    prod = torch.matmul(F_batch, tmp) # (bs, n, n)
    
    obj = prod.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) # (bs,)
    return obj

# ==========================================
# 3. 主求解流程
# ==========================================

def solve_torch_batch(instance, batch_size=100, dual_init=10, 
                      gamma=0.01, beta=0.01, num_iters=1000, 
                      device='cuda'):
    
    # 1. 读取数据 (CPU)
    n, F_np, D_np, obj_label, x_label = read_instance(instance)
    
    x_label = torch.tensor(x_label[np.newaxis, :, :], dtype=torch.float64, device=device)
    
    # 2. 转为 GPU Tensor (Float64)
    dtype = torch.float64 
    
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)

    
    # 3. 初始化
    # 随机初始化 X
    X = torch.rand((batch_size, n, n), device=device, dtype=dtype)
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    
    # RMSProp 状态
    v = torch.zeros_like(X)
    buffer = torch.zeros_like(X)
    
    # incumbent
    incumbent_obj = float('inf')
    incumbent_X = None
    
    print(f"Start solving {instance} with Batch Size={batch_size}, Device={device}, Dtype={dtype}")
    
    for it in range(num_iters):
        # --- Gradient & Update ---
        grad_X = grad_L_X_batch(F, D, X, Y) # (bs, n, n)
        grad_Y = grad_L_Y_batch(X)
        
        X, v, buffer = rmsprop_batch_jit(X, grad_X, v, buffer, gamma=gamma, 
                                         alpha=0.99, lam=0.98, mu=0.91, eps=1e-8)
        
        Y = Y + beta * grad_Y
        
        # --- Projection ---
        X = dykstra_proj_batch_jit(X, max_iter=20) 
        
        # --- Rounding & Evaluation ---
        X_int = greedy_round_batch(X)
        obj_vals = obj_fn_batch(F, D, X_int) # (bs,)
        
        # find the best in the batch
        min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
        
        # update incumbent        
        if min_obj_batch < incumbent_obj:
            incumbent_obj = min_obj_batch.item()
            incumbent_X = X_int[min_idx].clone()
            
            # row_sum = incumbent_X.sum(dim=1).mean().item()
            # print(f"Iter {it}: New Best {incumbent_obj:.4f}")

        if it % 100 == 0:
            penalty = (X * X - X).sum(dim=(1,2)).mean().item()
            print(f"Iter {it}, Best: {incumbent_obj:.4f}, BatchMeanObj: {obj_vals.mean().item():.2f}, Pen: {penalty:.2e}")

    return incumbent_X.cpu().numpy(), incumbent_obj, obj_label, x_label


def read_instance(instance):
    # 请确保路径正确
    problem_file = f"./qaplibs/qapdata/{instance}.dat"
    solution_file = f"./qaplibs/qapsoln/{instance}.sln"
    
    with open(problem_file, "r") as f:
        data = f.read().split()        
    data_iter = iter(map(int, data))
    n = next(data_iter)
    F_flat = [next(data_iter) for _ in range(n * n)]
    F_np = np.array(F_flat).reshape(n, n)
    D_flat = [next(data_iter) for _ in range(n * n)]
    D_np = np.array(D_flat).reshape(n, n)
    
    with open(solution_file, "r") as f:
        sol_data = f.read().split()
    sol_data_iter = iter(map(int, sol_data))
    n_sol = next(sol_data_iter)
    obj_label = next(sol_data_iter)
    x_label = [next(sol_data_iter) - 1 for _ in range(n)]
    x_label_np = np.zeros((n, n))
    for i in range(n):
        x_label_np[i, x_label[i]] = 1
        
    return n, F_np, D_np, obj_label, x_label_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default="nug12")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # solve
    start_event.record()
    X_best, obj_best, obj_label, x_label = solve_torch_batch(
        args.instance, 
        batch_size=args.batch_size, 
        dual_init=10, 
        gamma=0.02, 
        beta=0.02, 
        num_iters=args.iters,
        device=device
    )
    end_event.record()
    torch.cuda.synchronize()
    
    solve_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
    
    print("-" * 30)
    print(f"Time: {solve_time:.2f}s")
    print(f"My Best Obj: {obj_best}")
    print(f"Sol File Obj: {obj_label}")
    
    # 验证最终解是否合法
    row_sums = X_best.sum(axis=1)
    col_sums = X_best.sum(axis=0)
    if row_sums.min() < 0.99 or row_sums.max() > 1.01:
        print("Warning: Solution might not be a valid permutation.")
    else:
        print(row_sums)
        print(col_sums)
        print("Solution is a valid permutation matrix.")
        
    # check solution
    # n, F_np, D_np, _, _ = read_instance(args.instance)
    # final_obj = 0
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             for l in range(n):
    #                 final_obj += F_np[i, j] * D_np[k, l] * X_best[i, k] * X_best[j, l]
    # print(f"Final Obj Check: {final_obj}")
    
    
    # write the result to a file
    with open(f"result.txt", "a") as f:
        f.write(f"{args.instance} {solve_time:.2f} {obj_best} {obj_label}\n")