import torch
import torch.optim as optim
import triton
import triton.language as tl
import numpy as np
import argparse
import time
import os


def run_greedy_triton(M):
    assignment = torch.zeros_like(M)
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

def sinkhorn_step(log_alpha, num_iters: int = 10):
    log_S = log_alpha / 0.5
    for _ in range(num_iters):
        log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
    return torch.exp(log_S)

def compute_loss_and_grad(X, Y, F, D_T):
    """
    将 Loss 计算和梯度相关的操作融合，减少中间变量显存占用。
    """
    S = sinkhorn_step(X, num_iters=20)
    
    M1 = torch.matmul(F, S) 
    M2 = torch.matmul(M1, D_T)
    # term1 = torch.sum(M2 * S)
    term1 = torch.einsum('bij,bij->b', M2, S).mean()
    
    # 3. Penalty Term: sum(Y * (S^2 - S))
    # 提前计算 S^2 - S，既用于 Loss 也用于后续 Dual 更新
    S_sq_minus_S = S * (S - 1.0)
    # term2 = torch.sum(Y * S_sq_minus_S)
    term2 = torch.einsum('bij,bij->b', Y, S_sq_minus_S).mean()
    
    loss = term1 + term2
    
    return loss, S, S_sq_minus_S

# -----------------------------------------------------------------------------
# 3. 主逻辑
# -----------------------------------------------------------------------------

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

def run_optimization(F_np, D_np, dual_init, batch_size, num_steps=100, lr=0.01, optimizer_type='adam', args=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dtype = torch.float32 
    
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    
    D_T = D.transpose(-1, -2).contiguous() 
    n = F_np.shape[0]
    
    
    # _, F_np_orig, D_np_orig, _, _ = read_instance(args.instance)
    
    # F_orig = torch.tensor(F_np_orig, device=device, dtype=dtype)
    # D_orig = torch.tensor(D_np_orig, device=device, dtype=dtype)
    # D_T_orig = D_orig.transpose(-1, -2).contiguous()
    
    # 初始化变量
    X_rand = np.array(latin_hypercube_matrices(n, batch_size))
    X = torch.tensor(X_rand, device=device, dtype=dtype, requires_grad=True)
    
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam([X], lr=lr)
    else:
        optimizer = optim.RMSprop([X], lr=0.003)
    
    incumbent_obj = float('inf')
    
    print(f"Starting Optimization [N={n}, Batch={batch_size}, Device={device}]")
    
    # Warmup (对于 Triton 和 torch.compile 很重要)
    _ = compute_loss_and_grad(X[:2], Y[:2], F, D_T)
    
    t0 = time.time()
    
    for it in range(num_steps):
        # set_to_none=True 比 zero_grad() 稍微快一点
        optimizer.zero_grad()
        
        # 1. 计算 Loss 和所需的中间变量
        # 得益于 torch.compile，这里会融合成极少的 Kernel
        loss, P, P_sq_minus_P = compute_loss_and_grad(X, Y, F, D_T)
        
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        with torch.no_grad():
            Y.add_(P_sq_minus_P, alpha=lr)
            
            X_int = run_greedy_triton(P)
            val = torch.matmul(F, X_int)
            val = torch.matmul(val, D_T)
            obj_vals = torch.sum(val * X_int, dim=(1, 2))
        
            min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
            if min_obj_batch < incumbent_obj:
                incumbent_obj = min_obj_batch.item()
                incumbent_X = X_int[min_idx].clone() 
            if (it+1) % 100 == 0:
                print(f"Iter {it+1}: Best Obj {incumbent_obj:.4f}, Loss {loss.item():.4f}")
                
    total_time = time.time() - t0
    print(f"Total Time: {total_time:.2f}s, FPS: {num_steps/total_time:.1f}")
    return incumbent_X, incumbent_obj, total_time

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

    # print(np.sum(np.diag(F_np)), np.sum(np.diag(D_np)))
    
    # F_np = F_np / np.linalg.norm(F_np, ord='fro')
    # D_np = D_np / np.linalg.norm(D_np, ord='fro')
    
    # # kronecker product of F and D 
    # FD = np.kron(F_np, D_np)
    # eigvals = np.linalg.eigvalsh(FD)
    # # print the 10 smallest eigenvalues
    # print("10 smallest eigenvalues of Kronecker(F,D):", eigvals[:10])
    # print("10 smallest eigenvalues of Kronecker(F,D):", eigvals[-10:])
    
    batch_size = args.batch_size
    num_steps = args.iters
    
    if n < 300:
        batch_size = 1
        num_steps = 20000
    elif n < 500:
        batch_size = 2000
        num_steps = 2000
    else:
        batch_size = 500
        num_steps = 5000
    
    lr = 0.003
    dual_init = 1
    dtype = torch.float32
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    X_best, obj_best, solve_time_raw = run_optimization(F_np, D_np, dual_init, batch_size, num_steps, lr, optimizer_type=args.optimizer, args=args)
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
    
    # # print the result:
    # res = []
    # for i in range(n):
    #     for j in range(n):
    #         if X_best[i, j] > 0.5:
    #             res.append(j+1)
    #             break
    # print(res)
    
    # final_obj = 0
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             for l in range(n):
    #                 final_obj += F_np[i, j] * D_np[k, l] * X_best[i, k] * X_best[j, l]
    # print(f"Final Obj Check: {final_obj}")
    
    solve_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
    
    gap = (obj_best - obj_label) / obj_label
    
    instance_name = args.instance.split('/')[-1]
    
    with open(f"result.txt", "a") as f:
        f.write(f"{instance_name} {solve_time:.2f} {solve_time_raw:.2f} {obj_best} {obj_label} {gap:.4f}\n")