import torch
import torch.optim as optim
import triton
import triton.language as tl
import numpy as np
import argparse

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

def obj_fn_batch(F, D, X):
    """
    trace(F * X * D.T * X.T)
    """
    return torch.einsum('ij,bjk,lk,bil->b', F, X, D, X)

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

def log_sinkhorn(log_alpha, num_iters=20):
    """
    Performs Sinkhorn normalization in the log-domain for numerical stability.
    
    Args:
        log_alpha: Input tensor of shape (Batch_Size, N, N). 
                   Can contain any real values (positive or negative).
        num_iters: Number of normalization iterations.
        
    Returns:
        S: Doubly stochastic matrix in linear space (Batch_Size, N, N).
    """
    # Initialize log_S as the input logits (clone to avoid modifying input)
    log_S = log_alpha.clone()

    for _ in range(num_iters):
        # 1. Row Normalization in log space
        # equivalent to: S = S / sum(S)
        # becomes: log_S = log_S - logsumexp(log_S)
        lse_row = torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - lse_row

        # 2. Column Normalization in log space
        lse_col = torch.logsumexp(log_S, dim=-2, keepdim=True)
        log_S = log_S - lse_col

    # Exponentiate at the very end to get the transport matrix S
    return torch.exp(log_S)

def compute_loss(X, Y, F, D):
    """
    Calculates L(X, Y) = trace(F * S * D' * S') + sum(Y * (X^2 - X))
    using the stable log_sinkhorn function.
    """
    # Get the doubly stochastic matrix S using log-space iterations
    S = log_sinkhorn(X)
    
    # --- Term 1: Trace(F * S * D^T * S^T) ---
    D_T = D.transpose(-1, -2)
    
    # M = F * S * D^T
    M = torch.matmul(torch.matmul(F, S), D_T)
    
    # Trace(M * S^T) = sum(M * S) element-wise
    term1 = torch.sum(M * S)
    
    # --- Term 2: sum(Y * (X^2 - X)) ---
    # Note: X is used directly here, not S
    term2 = torch.sum(Y * (S**2 - S))
    
    return term1 + term2

def run_optimization(F_np, D_np, dual_init, batch_size, num_steps=100, lr=0.01, optimizer_type='adam'):
    # Initialize X with random values (can include negatives)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = F_np.shape[0]
    # X = torch.randn(batch_size, N, N, device=Y_batch.device, requires_grad=True)
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    
    X = torch.rand((batch_size, n, n), device=device, dtype=dtype, requires_grad=True)
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam([X], lr=lr)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop([X], lr=lr)
        
    incumbent_obj = float('inf')
    incumbent_X = None
    
    for it in range(num_steps):
        # update X
        optimizer.zero_grad()
        loss = compute_loss(X, Y, F, D)
        loss.backward()
        optimizer.step()
        
        # update Y
        with torch.no_grad():
            P = log_sinkhorn(X)
            Y += lr * (P * P - P)
    
            # rounding and evaluation
            X_int = greedy_round_batch(P)
            obj_vals = obj_fn_batch(F, D, X_int) # (bs,)
            
            # update incumbent        
            min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
            if min_obj_batch < incumbent_obj:
                incumbent_obj = min_obj_batch.item()
                incumbent_X = X_int[min_idx].clone()

            if (it+1) % 1000 == 0:
                penalty = (P * P - P).mean(dim=(1,2)).mean().item()
                print(f"Iter {it+1}, Best: {incumbent_obj:.4f}, BatchMeanObj: {obj_vals.mean().item():.2f}, Pen: {penalty:.2e}")
                if np.abs(penalty) < 1e-2:
                    print("Early stopping due to low penalty.")
                    break
        
    return incumbent_X, incumbent_obj

# --- Example Usage ---
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
    lr = 0.02
    dual_init = 10.0
    dtype = torch.float64
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    optimized_X, obj_best = run_optimization(F_np, D_np, dual_init, batch_size, num_steps, lr, optimizer_type=args.optimizer)
    end_event.record()
    torch.cuda.synchronize()
    
    solve_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
    
    gap = (obj_best - obj_label) / obj_label
    
    with open(f"result.txt", "a") as f:
        f.write(f"{args.instance} {solve_time:.2f} {obj_best} {obj_label} {gap:.4f}\n")