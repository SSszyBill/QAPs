import torch
import torch.optim as optim
import triton
import triton.language as tl
import numpy as np
import argparse
import time
import os
import scipy
from scipy.optimize import linear_sum_assignment

@triton.jit
def greedy_kernel_optimized(
    M_ptr,          # 输入 (BS, N, N)
    Out_ptr,        # 输出 (BS, N, N)
    stride_b, stride_h, stride_w,
    N: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    m_ptr = M_ptr + pid * stride_b
    out_ptr = Out_ptr + pid * stride_b
    offs = tl.arange(0, BLOCK_SIZE)
    mask_load = offs < (N * N)
    val = tl.load(m_ptr + offs, mask=mask_load, other=-1.0e10)
    for _ in range(N):
        max_val = tl.max(val, axis=0)
        is_max = (val == max_val)
        idx = tl.argmax(is_max.to(tl.int32), axis=0)
        r = idx // N
        c = idx % N
        tl.store(out_ptr + idx, 1.0)
        row_mask = (offs // N) == r
        col_mask = (offs % N) == c
        combined_mask = row_mask | col_mask
        val = tl.where(combined_mask, -1.0e10, val)

def run_greedy_triton(M):
    bs, n, _ = M.shape
    block_size = triton.next_power_of_2(n * n)
    assignment = torch.zeros_like(M)
    if n <= 64:
        grid = (bs,)
        greedy_kernel_optimized[grid](
            M, assignment,
            M.stride(0), M.stride(1), M.stride(2),
            N=n,
            BLOCK_SIZE=block_size
        )
    else:
        assignment = greedy_round_large_batch_torch(M)
        
    return assignment

def greedy_round_large_batch_torch(M):
    bs, n, _ = M.shape
    assignment = torch.zeros_like(M)
    M_temp = M.clone()
    batch_indices = torch.arange(bs, device=M.device)
    
    for _ in range(n):
        flat_M = M_temp.view(bs, -1)
        _, idx = flat_M.max(dim=1) 
        rows = idx.div(n, rounding_mode='floor')
        cols = idx % n
        assignment[batch_indices, rows, cols] = 1.0
        M_temp[batch_indices, rows, :] = -1e10
        M_temp[batch_indices, :, cols] = -1e10
    return assignment

@torch.compile
def sinkhorn_step(log_alpha, num_iters: int = 10):
    log_S = log_alpha
    for _ in range(num_iters):
        log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
    return torch.exp(log_S)

@torch.compile
def compute_loss_and_grad(X, Y, F, D_T):
    """
    将 Loss 计算和梯度相关的操作融合，减少中间变量显存占用。
    """
    # 1. Sinkhorn Forward
    n = X.shape[-1]
    if n > 500:
        num_iters = 25
    else:
        num_iters = 30
    
    S = sinkhorn_step(X, num_iters=num_iters)
    
    # 2. QAP Objective: Trace(F S D^T S^T)
    # 利用矩阵乘法结合律减少计算量
    # 路径: (F @ S) -> M1; (M1 @ D_T) -> M2; Sum(M2 * S)
    M1 = torch.matmul(F, S) 
    M2 = torch.matmul(M1, D_T)
    # term1 = torch.sum(M2 * S)
    term1 = torch.sum(M2 * S, dim=(1, 2)).mean()
    
    # row_pen: for the i-th row, sum_j S_ij^2 should be 1
    S_sq = S * S
    
    # the row penalty term is Y_i * (sum_j S_ij^2 - 1)
    row_pen = torch.sum(S_sq, dim=-1) - 1.0
    term2 = torch.sum(Y[0] * row_pen, dim=1).mean()
    
    col_pen = torch.sum(S_sq, dim=-2) - 1.0
    term2 += torch.sum(Y[0] * col_pen, dim=1).mean()
        
    loss = term1 + term2
    
    return loss, S, [row_pen, col_pen], term1, term2

def read_instance(instance):
    # 请确保路径正确
    problem_file = f"./qaplibs/{instance}.dat"
    solution_file = f"./qaplibs/{instance}.sln"
    
    with open(problem_file, "r") as f:
        data = f.read().split()        
    data_iter = iter(map(int, data))
    n = next(data_iter)
    F_flat = [next(data_iter) for _ in range(n * n)]
    F_np = np.array(F_flat).reshape(n, n)
    D_flat = [next(data_iter) for _ in range(n * n)]
    D_np = np.array(D_flat).reshape(n, n)
    
    if not os.path.exists(solution_file):
        return n, F_np, D_np, 1.0, None
    
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

def latin_hypercube_matrices(m, n):
    """使用拉丁超立方体设计生成多样矩阵"""
    # 将矩阵向量化后的参数空间采样
    d = m * m  # 参数维度
    
    # 生成拉丁超立方体设计
    samples = np.zeros((n, d))
    for j in range(d):
        perm = np.random.permutation(n)
        samples[:, j] = (perm + np.random.rand(n)) / n
    
    # 转换为矩阵
    matrices = []
    for i in range(n):
        # 映射到不同分布
        vec = samples[i, :]
        
        # 可选：应用不同变换获得不同模式
        transform_type = i % 4
        if transform_type == 0:
            vec = np.tanh(vec * 4 - 2)  # 压缩到[-1,1]
        elif transform_type == 1:
            vec = np.sin(vec * 2 * np.pi)  # 周期性
        elif transform_type == 2:
            vec = np.exp(vec * 3 - 1.5)  # 正数
        else:
            vec = vec * 4 - 2  # 线性
        
        mat = vec.reshape(m, m)
        matrices.append(mat)
    
    return matrices


def spectral_initialization_qap(F, D, num=None):
    n = F.shape[0]
    if D.shape[0] != n:
        raise ValueError("Matrices F and D must have the same dimension.")

    P_list = []
    
    if num is None:
        k_list = np.arange(n) 
    else:
        # k_list = np.arange(min(num, n)) + 1
        k_list = np.arange(n-min(num, n), n) + 1
        print(k_list)
    
    val_F, vec_F = scipy.linalg.eigh(F)
    val_D, vec_D = scipy.linalg.eigh(D)
    idx_F = np.argsort(val_F)[::-1]
    idx_D = np.argsort(val_D)[::-1]
    
    U_F = vec_F[:, idx_F]
    U_D = vec_D[:, idx_D]
    for k in k_list:
        U_F_k = U_F[:, :k]
        U_D_k = U_D[:, :k]

        X = np.abs(U_F_k) @ np.abs(U_D_k).T

        row_ind, col_ind = linear_sum_assignment(-X)
        
        P = np.zeros((n, n))
        P[row_ind, col_ind] = 1
        P_list.append(P)

    return np.array(P_list)


def run_optimization(F_np, D_np, dual_init, batch_size, num_steps=100, lr=0.01, optimizer_type='adam', x_label_np=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dtype = torch.float32 
    
    # Warmup (对于 Triton 和 torch.compile 很重要)
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    X = torch.rand((2, F_np.shape[0], F_np.shape[0]), device=device, dtype=dtype, requires_grad=True)
    Y1 = torch.full((2, F_np.shape[0]), dual_init, device=device, dtype=dtype)
    Y2 = torch.full((2, F_np.shape[0]), dual_init, device=device, dtype=dtype)
    Y = [Y1, Y2]
    _ = compute_loss_and_grad(X, Y, F, D.transpose(-1, -2).contiguous())
    
    t0 = time.time()
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    D_T = D.transpose(-1, -2).contiguous() 
    n = F_np.shape[0]
    
    X_spectral = spectral_initialization_qap(F_np, D_np, batch_size)
    # # add random noise to the spectral initializations to generate the full batch
    # if X_spectral.shape[0] < batch_size:
    #     num_random = batch_size - X_spectral.shape[0]
    #     # generate random noise matrices
    #     noise_matrices = np.random.rand(num_random, n, n) * 0.1
    #     X_spectral = np.concatenate([X_spectral, noise_matrices], axis=0)
    np.random.shuffle(X_spectral)
    X = torch.tensor(X_spectral, device=device, dtype=dtype, requires_grad=True)
    print(X.shape)
    
    # X = torch.tensor(X_spectral, device=device, dtype=dtype)
    # # add random samples to fill the batch
    # if X.shape[0] < batch_size:
    #     num_random = batch_size - X.shape[0]
    #     # X_random = torch.rand((num_random, n, n), device=device, dtype=dtype)
    #     X_rand = np.array(latin_hypercube_matrices(n, num_random))
    #     X_random = torch.tensor(X_rand, device=device, dtype=dtype)
    #     X = torch.cat([X, X_random], dim=0)
    # X = X[:batch_size]
    # X.requires_grad_(True)
    print(X.shape[0])
    
    # X = torch.tensor(latin_hypercube_matrices(n, batch_size), device=device, dtype=dtype, requires_grad=True)
    
    
    # X = torch.rand((batch_size, n, n), device=device, dtype=dtype, requires_grad=True)
    # Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    Y1 = torch.full((batch_size, n), dual_init, device=device, dtype=dtype)
    Y2 = torch.full((batch_size, n), dual_init, device=device, dtype=dtype)
    Y = [Y1, Y2]
    
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam([X], lr=lr)
    else:
        optimizer = optim.RMSprop([X], lr=lr)
    
    incumbent_obj = float('inf')
    
    print(f"Starting Optimization [N={n}, Batch={batch_size}, Device={device}]")
    
    incumbents = []
    incum_time = []
    
    for it in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        
        loss, P, P_sq_minus_P, term1, term2 = compute_loss_and_grad(X, Y, F, D_T)
        
        loss.backward()
        optimizer.step()
        
        # 2. Dual Update & Evaluation
        with torch.no_grad():
            # In-place update
            # Y.add_(P_sq_minus_P, alpha=lr)
            Y[0].add_(P_sq_minus_P[0], alpha=lr)
            Y[1].add_(P_sq_minus_P[1], alpha=lr)
            X_int = run_greedy_triton(P)
            
            val = torch.matmul(F, X_int)
            val = torch.matmul(val, D_T)
            obj_vals = torch.sum(val * X_int, dim=(1, 2))
            
            min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
            if min_obj_batch < incumbent_obj:
                incumbent_obj = min_obj_batch.item()
                incumbent_X = X_int[min_idx].clone() 
                incumbents.append(incumbent_obj)
                time_now = time.time() - t0
                incum_time.append(time_now)
                
            if (it+1) % 100 == 0:
                print(f"Iter {it+1}: Best Obj {incumbent_obj:.4f}, Loss {loss.item():.4f}, Term1 {term1.item():.4f}, Term2 {term2.item():.4f}")
                
    total_time = time.time() - t0
    print(f"Total Time: {total_time:.2f}s, FPS: {num_steps/total_time:.1f}")
    return incumbent_X, incumbent_obj, total_time, incumbents, incum_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default="nug12")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default="rmsprop", choices=["rmsprop", "adam"])
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    n, F_np, D_np, obj_label, x_label_np = read_instance(args.instance)
    print(np.sum(np.diag(F_np)), np.sum(np.diag(D_np)))
    
    batch_size = args.batch_size
    num_steps = args.iters
    
    if n < 200:
        # batch_size = 2000
        # num_steps = 1000
        batch_size = 14
        num_steps = 5000
    elif n < 500:
        batch_size = 1000
        num_steps = 1200
    else:
        batch_size = 200
        num_steps = 1200
    
    lr = 0.01
    dual_init = 100
    dtype = torch.float32
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    X_best, obj_best, solve_time_raw, incumbents, incum_time = run_optimization(F_np, D_np, dual_init, batch_size, num_steps, lr, optimizer_type=args.optimizer, x_label_np=x_label_np)
    end_event.record()
    torch.cuda.synchronize()
    
    row_sums = X_best.sum(axis=1)
    col_sums = X_best.sum(axis=0)
    if row_sums.min() < 0.99 or row_sums.max() > 1.01:
        print("Warning: Solution might not be a valid permutation.")
    else:
        print(row_sums)
        print(col_sums)
        print("Solution is a valid permutation matrix.")
        
    # # check solution
    n, F_np, D_np, _, _ = read_instance(args.instance)

    # Compute final objective value
    X_best = X_best.cpu().numpy()
    tmp = X_best @ D_np.T @ X_best.T
    obj_best = np.trace(F_np @ tmp)
    
    solve_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
    
    gap = (obj_best - obj_label) / obj_label
    
    instance_name = args.instance.split('/')[-1]
    
    with open(f"result.txt", "a") as f:
        f.write(f"{instance_name} {solve_time:.2f} {solve_time_raw:.2f} {obj_best}\n")
        
    # write incumbents over time to file
    with open(f"/home/xjx/A-xjx/QAPs/results/pdbo_square/{instance_name}.txt", "w") as f:
        for t, val in zip(incum_time, incumbents):
            f.write(f"{t:.4f} {val}\n")