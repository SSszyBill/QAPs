import torch
import argparse
import numpy as np
import time
from typing import Tuple, Optional
import triton
import triton.language as tl

@torch.jit.script
def rmsprop_batch_jit(X: torch.Tensor, 
                      grad: torch.Tensor, 
                      v: torch.Tensor, 
                      m: torch.Tensor, 
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
        m = mu * m + grad / denom
        X = X - gamma * m
    else:
        X = X - gamma * grad / denom
        
    return X, v, m


@torch.jit.script
def adam_torch(X: torch.Tensor, 
               grad: torch.Tensor, 
               v: Optional[torch.Tensor] = None, 
               m: Optional[torch.Tensor] = None, 
               t: int = 1, 
               gamma: float = 0.001, 
               beta1: float = 0.9, 
               beta2: float = 0.999, 
               lam: float = 0.0, 
               eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if v is None:
        v = torch.zeros_like(X)
    if m is None:
        m = torch.zeros_like(X)
        
    if lam != 0.0:
        grad = grad + lam * X
        
    m = beta1 * m + (1.0 - beta1) * grad
    
    v = beta2 * v + (1.0 - beta2) * (grad * grad)
    
    bias_correction1 = 1.0 - (beta1 ** t)
    bias_correction2 = 1.0 - (beta2 ** t)
    
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    
    denom = torch.sqrt(v_hat) + eps
    step = gamma * (m_hat / denom)
    
    X = X - step
    
    return X, v, m

@torch.jit.script
def dykstra_proj_batch_jit(M: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
    # X, P, Q: (bs, n, n)
    X = M.clone()
    p = torch.zeros_like(M)
    q = torch.zeros_like(M)
    
    n = M.size(1)
    n_float = float(n)
    
    for _ in range(max_iter):
        Y = X + p
        
        row_diff = Y.sum(dim=2, keepdim=True) - 1.0
        col_diff = Y.sum(dim=1, keepdim=True) - 1.0
        grand_diff = row_diff.sum(dim=1, keepdim=True) 
        Y = Y - (row_diff / n_float) - (col_diff / n_float) + (grand_diff / (n_float * n_float))
        
        p = X + p - Y
        X = Y
        
        # Project onto non-negative orthant
        Y = X + q
        X = torch.clamp(Y, min=0.0)
        q = Y - X
        
    return X

def greedy_round_large_batch(M):
    """
    M: (bs, n, n)
    """
    bs, n, _ = M.shape
    assignment = torch.zeros_like(M)
    M_temp = M.clone()
    
    batch_indices = torch.arange(bs, device=M.device)
    
    for _ in range(n):
        flat_M = M_temp.view(bs, -1)
        _, idx = flat_M.max(dim=1) # (bs,)
        
        rows = idx.div(n, rounding_mode='floor')
        cols = idx % n
        
        assignment[batch_indices, rows, cols] = 1.0
        
        M_temp[batch_indices, rows, :] = -1e10
        M_temp[batch_indices, :, cols] = -1e10
        
    return assignment


@triton.jit
def greedy_kernel(
    M_ptr,          # 输入矩阵指针 (BS, N, N)
    Out_ptr,        # 输出矩阵指针 (BS, N, N)
    stride_b, stride_h, stride_w,  # M 的 strides
    N: tl.constexpr, # 矩阵大小
    BLOCK_SIZE: tl.constexpr # 设置为大于等于 N*N 的最小 2^k
):
    # 1. 获取当前程序的 batch ID
    pid = tl.program_id(0)
    
    # 2. 计算当前 Batch 在内存中的起始偏移量
    # M 和 Out 布局相同
    m_start_ptr = M_ptr + pid * stride_b
    out_start_ptr = Out_ptr + pid * stride_b
    
    # 3. 加载整个 N*N 矩阵到寄存器/SRAM
    # Triton 会尝试将其放在最快的存储器中
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < (N * N)
    
    # 加载 M，对于越界部分填充 -inf
    val = tl.load(m_start_ptr + offs, mask=mask, other=-float('inf'))
    
    # 记录被占用的行和列 (位掩码会更高效，但数组更容易实现)
    # 在寄存器中维护状态：如果 val 被选中，将其设为 -inf
    
    # 循环 N 次寻找最大值
    for _ in range(N):
        # a. 找到当前剩余元素中的最大值索引
        # argmax 在 Triton 中稍微复杂，我们先找 max val，再找 index
        max_val = tl.max(val, axis=0)
        
        # b. 找到该最大值对应的 index (第一个匹配的)
        # 比较 val == max_val，得到布尔向量
        is_max = (val == max_val)
        
        # 将布尔转为索引，取第一个 (argmax)
        # 这是一个标准的 argmax trick
        idx = tl.argmax(is_max.to(tl.int32), axis=0) 
        # 注意：triton 的 argmax 返回的是 index，如果存在多个最大值，它返回第一个
        
        # c. 计算 Row 和 Col
        r = idx // N
        c = idx % N
        
        # d. 写入结果 1.0
        tl.store(out_start_ptr + idx, 1.0)
        
        # e. Masking (核心加速点)
        # 我们不需要重新写回 Global Memory，直接在寄存器变量 `val` 上修改
        # 将第 r 行的所有元素设为 -inf
        # 将第 c 列的所有元素设为 -inf
        
        # 构建行掩码和列掩码
        # offs // N == r  -> 这一行的所有元素
        # offs % N == c   -> 这一列的所有元素
        
        row_mask = (offs // N) == r
        col_mask = (offs % N) == c
        combined_mask = row_mask | col_mask
        
        # 更新 val: 凡是 mask 命中的地方，更新为 -inf
        val = tl.where(combined_mask, -float('inf'), val)

def greedy_round_batch(M):
    bs, n, _ = M.shape
    
    if n < 60:
        assignment = torch.zeros_like(M)
        
        # 算出需要的 Block Size (必须是 2 的幂)
        # 例如 N=12 -> 144 -> Block=256
        # N=20 -> 400 -> Block=512
        # N=50 -> 2500 -> Block=4096 (Triton 处理 4K 元素非常轻松)
        block_size = triton.next_power_of_2(n * n)
        
        # 启动 Grid，每个 Batch 一个 Kernel 实例
        grid = (bs,)
        
        greedy_kernel[grid](
            M, 
            assignment,
            M.stride(0), M.stride(1), M.stride(2),
            N=n,
            BLOCK_SIZE=block_size
        )
    else:
        assignment = greedy_round_large_batch(M)
    
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
    return torch.einsum('ij,bjk,lk,bil->b', F, X, D, X)

def solve_torch_batch(instance, optimizer="adam",
                      batch_size=100, dual_init=10, 
                      gamma=0.01, beta=0.01, num_iters=1000, 
                      device='cuda', dtype=torch.float64):
    
    n, F_np, D_np, obj_label, x_label = read_instance(instance)
    
    x_label = torch.tensor(x_label[np.newaxis, :, :], dtype=dtype, device=device)
    
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    
    X = torch.rand((batch_size, n, n), device=device, dtype=dtype)
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    
    # optimizer 状态
    v = torch.zeros_like(X)
    m = torch.zeros_like(X)
    
    # incumbent
    incumbent_obj = float('inf')
    incumbent_X = None
    
    print(f"Start solving {instance} with Batch Size={batch_size}, Device={device}, Dtype={dtype}")
    
    for it in range(num_iters):
        # gradient steps
        grad_X = grad_L_X_batch(F, D, X, Y) # (bs, n, n)
        grad_Y = grad_L_Y_batch(X)
        
        if optimizer == "rmsprop":
            X, v, m = rmsprop_batch_jit(X, grad_X, v, m, gamma=gamma, 
                                             alpha=0.99, lam=0.98, mu=0.91, eps=1e-8)
        elif optimizer == "adam":
            X, v, m = adam_torch(X, grad_X, v, m, t=it+1, gamma=gamma)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        Y -= beta * grad_Y
        
        # projection
        X = dykstra_proj_batch_jit(X, max_iter=20) 
        
        
        # rounding and evaluation
        X_int = greedy_round_batch(X)
        obj_vals = obj_fn_batch(F, D, X_int) # (bs,)
        
        # update incumbent        
        min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
        if min_obj_batch < incumbent_obj:
            incumbent_obj = min_obj_batch.item()
            incumbent_X = X_int[min_idx].clone()

        if (it+1) % 1000 == 0:
            penalty = (X * X - X).mean(dim=(1,2)).mean().item()
            print(f"Iter {it+1}, Best: {incumbent_obj:.4f}, BatchMeanObj: {obj_vals.mean().item():.2f}, Pen: {penalty:.2e}")
            if np.abs(penalty) < 1e-2:
                print("Early stopping due to low penalty.")
                break

    return incumbent_X.cpu().numpy(), incumbent_obj, obj_label, x_label, it


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
    parser.add_argument('--optimizer', type=str, default="rmsprop", choices=["rmsprop", "adam"])
    
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
    X_best, obj_best, obj_label, x_label, it = solve_torch_batch(
        args.instance, 
        args.optimizer,
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
    
    gap = (obj_best - obj_label) / obj_label
    
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
        f.write(f"{args.instance} {solve_time:.2f} {obj_best} {obj_label} {gap:.4f}\n")