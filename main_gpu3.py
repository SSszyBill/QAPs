import torch
import torch.optim as optim
import triton
import triton.language as tl
import numpy as np
import argparse
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast

# torch.set_float32_matmul_precision('high')
# -----------------------------------------------------------------------------
# 1. 优化后的 Triton Greedy Kernel
# -----------------------------------------------------------------------------

@triton.jit
def greedy_kernel_optimized(
    M_ptr,          # 输入 (BS, N, N)
    Out_ptr,        # 输出 (BS, N, N)
    stride_b, stride_h, stride_w,
    N: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    """
    优化的 Greedy Rounding Kernel。
    优化点：
    1. 将整个 N*N 矩阵加载到寄存器/SRAM (适用于 N <= 64 或适当调整 Block)。
    2. 在寄存器中维护 Mask，避免昂贵的 Global Memory 写回操作。
    3. 只在确定 1 的位置写回 Global Memory，减少显存带宽占用。
    """
    pid = tl.program_id(0)
    
    # 指针偏移
    m_ptr = M_ptr + pid * stride_b
    out_ptr = Out_ptr + pid * stride_b
    
    # 1. 加载数据到寄存器
    offs = tl.arange(0, BLOCK_SIZE)
    mask_load = offs < (N * N)
    
    # 加载 M，越界部分设为极小值
    val = tl.load(m_ptr + offs, mask=mask_load, other=-1.0e10)
    
    # 循环 N 次进行匹配
    for _ in range(N):
        # a. 寻找当前最大值 (Reduction)
        max_val = tl.max(val, axis=0)
        
        # b. 找到最大值的索引 (Argmax)
        # 注意: 如果有多个相同最大值，Argmax 返回第一个
        is_max = (val == max_val)
        idx = tl.argmax(is_max.to(tl.int32), axis=0)
        
        # c. 计算行列坐标
        r = idx // N
        c = idx % N
        
        # d. 写入结果
        # 我们假设 Out_ptr 外部已初始化为 0，这里只写 1
        tl.store(out_ptr + idx, 1.0)
        
        # e. In-Register Masking (关键加速点)
        # 不写回 Global Memory，直接在寄存器变量 val 上操作
        row_mask = (offs // N) == r
        col_mask = (offs % N) == c
        combined_mask = row_mask | col_mask
        
        # 将被选中的行和列的值设为极小值，使其不再被选中
        val = tl.where(combined_mask, -1.0e10, val)

def run_greedy_triton(M):
    bs, n, _ = M.shape
    # Block size 必须是 2 的幂
    block_size = triton.next_power_of_2(n * n)
    
    # 准备输出容器
    assignment = torch.zeros_like(M)
    
    # Triton Kernel 限制: 如果 N 很大 (例如 > 64)，N*N 超过 4096，
    # 单个线程块寄存器压力会很大。
    # 对于 NUG12-NUG30 这类问题，N <= 64，此 Kernel 极快。
    if n <= 64:
        grid = (bs,)
        greedy_kernel_optimized[grid](
            M, assignment,
            M.stride(0), M.stride(1), M.stride(2),
            N=n,
            BLOCK_SIZE=block_size
        )
    else:
        # 对于超大 N，回退到 PyTorch 批处理实现
        assignment = greedy_round_large_batch_torch(M)
        
    return assignment

def greedy_round_large_batch_torch(M):
    """针对大 N 的 PyTorch 批处理实现"""
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
        # 这种 masking 在 PyTorch 中比较慢，因为涉及大量内存拷贝
        M_temp[batch_indices, rows, :] = -1e10
        M_temp[batch_indices, :, cols] = -1e10
    return assignment

# -----------------------------------------------------------------------------
# 2. 算子融合与计算图优化 (PyTorch 2.0)
# -----------------------------------------------------------------------------

# 使用 torch.compile 替代 torch.jit.script
# mode="reduce-overhead" 专门针对这种小尺寸矩阵、多次迭代的场景优化 CUDA Graph 启动
@torch.compile
def sinkhorn_step(log_alpha, num_iters: int = 10):
    log_S = log_alpha
    for _ in range(num_iters):
        # Row Normalization
        log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
        # Column Normalization
        log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
    return torch.exp(log_S)

@torch.compile
def compute_loss_and_grad(X, Y, F, D_T):
    """
    将 Loss 计算和梯度相关的操作融合，减少中间变量显存占用。
    """
    # 1. Sinkhorn Forward
    S = sinkhorn_step(X, num_iters=20)
    
    # 2. QAP Objective: Trace(F S D^T S^T)
    # 利用矩阵乘法结合律减少计算量
    # 路径: (F @ S) -> M1; (M1 @ D_T) -> M2; Sum(M2 * S)
    M1 = torch.matmul(F, S) 
    M2 = torch.matmul(M1, D_T)
    # term1 = torch.sum(M2 * S)
    # term1 = torch.einsum('bij,bij->b', M2, S).mean()
    term1 = torch.sum(M2.float() * S, dim=(1, 2)).mean()
    
    # 3. Penalty Term: sum(Y * (S^2 - S))
    # 提前计算 S^2 - S，既用于 Loss 也用于后续 Dual 更新
    S_sq_minus_S = S * (S - 1.0)
    # term2 = torch.sum(Y * S_sq_minus_S)
    # term2 = torch.einsum('bij,bij->b', Y, S_sq_minus_S).mean()
    term2 = torch.sum(Y * S_sq_minus_S, dim=(1, 2)).mean()
    
    loss = term1 + term2
    
    # print(term1.item(), term2.item())
    
    return loss, S, S_sq_minus_S

# -----------------------------------------------------------------------------
# 3. 主逻辑
# -----------------------------------------------------------------------------

def read_instance(instance):
    # 请确保路径正确
    problem_file = f"./qaplibs/{instance}.dat"
    solution_file = f"./qaplibs/{instance}.sln"
    
    with open(problem_file, "r") as f:
        line = f.readline()
        while not line.strip(): # 跳过文件开头的空行(如果有)
            line = f.readline()
            
        n = int(line.split()[0])
        
        rest_data = f.read().split()

    # 3. 生成迭代器 (注意：这里不再包含 n 了)
    data_iter = iter(map(int, rest_data))
    
    # 4. 直接开始读取矩阵 (不需要再 next(data_iter) 读取 n)
    F_flat = [next(data_iter) for _ in range(n * n)]
    F_np = np.array(F_flat).reshape(n, n)
    
    D_flat = [next(data_iter) for _ in range(n * n)]
    D_np = np.array(D_flat).reshape(n, n)
    
    if not os.path.exists(solution_file):
        # 如果没有解文件，返回默认值
        return n, F_np, D_np, 0.0, None # obj_label 改为 0.0 防止报错
    
    
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

def run_optimization(F_np, D_np, dual_init, batch_size, num_steps=100, lr=0.01, optimizer_type='adam'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    n = F_np.shape[0]
    
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)

    D_T = D.transpose(-1, -2).contiguous() 
    
    # 初始化变量
    X_rand = np.array(latin_hypercube_matrices(n, batch_size))
    X = torch.tensor(X_rand, device=device, dtype=dtype, requires_grad=True)
    
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam([X], lr=lr)
    else:
        # optimizer = optim.RMSprop([X], lr=lr)
        optimizer = optim.AdamW([X], lr=lr)
    
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=lr*0.01)
    incumbent_obj = float('inf')
    
    print(f"Starting Optimization [N={n}, Batch={batch_size}, Device={device}]")
    
    # Warmup (对于 Triton 和 torch.compile 很重要)
    _ = compute_loss_and_grad(X[:2], Y[:2], F, D_T)
    
    t0 = time.time()
    
    for it in range(num_steps):
        # set_to_none=True 比 zero_grad() 稍微快一点
        for _ in range(1):
            optimizer.zero_grad(set_to_none=True)

            # 1. 计算 Loss 和所需的中间变量
            # 得益于 torch.compile，这里会融合成极少的 Kernel
            loss, P, P_sq_minus_P = compute_loss_and_grad(X, Y, F, D_T)
        
            loss.backward()
            optimizer.step()
            # scheduler.step()
        
        # 2. Dual Update & Evaluation
        with torch.no_grad():
            # In-place update
            Y.add_(P_sq_minus_P, alpha=lr)
            
            # 使用优化的 Triton Kernel 进行 Rounding
            X_int = run_greedy_triton(P)
            
            # 评估目标函数值
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
    
    batch_size = args.batch_size
    num_steps = args.iters
    
    if n < 100:
        batch_size = 20000
        num_steps = 500
    elif n < 500:
        batch_size = 5000
        num_steps = 500
    else:
        batch_size = 500
        num_steps = 2000
    
    lr = 0.02
    dual_init = 0
    dtype = torch.float32
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    X_best, obj_best, solve_time_raw = run_optimization(F_np, D_np, dual_init, batch_size, num_steps, lr, optimizer_type=args.optimizer)
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
    X_best = X_best.cpu().numpy()
    tmp = X_best @ D_np.T @ X_best.T
    obj_best = np.trace(F_np @ tmp)
    
    
    if obj_best < obj_label or obj_label == 1:
        # print the result:
        res = []
        for i in range(n):
            for j in range(n):
                if X_best[i, j] > 0.5:
                    res.append(j+1)
                    break
        print(res)
        
        # write results to file
        with open(f"./qaplibs/{args.instance}.sln", "w") as f:
            f.write(f"{n} {int(obj_best)}\n")
            f.write(' '.join(map(str, res)) + '\n')
    
    
    
    # final_obj = 0
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             for l in range(n):
    #                 final_obj += F_np[i, j] * D_np[k, l] * X_best[i, k] * X_best[j, l]
    # print(f"Final Obj Check: {final_obj}")
    
    solve_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
    
    gap = (obj_best - obj_label) / obj_label
    
    with open(f"result.txt", "a") as f:
        f.write(f"{args.instance.split('/')[-1]} {solve_time:.2f} {solve_time_raw:.2f} {obj_best:.6f} {obj_label} {gap:.6f}\n")