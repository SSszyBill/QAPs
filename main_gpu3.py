import torch
import torch.optim as optim
import triton
import triton.language as tl
import numpy as np
import argparse
import wandb

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
    Optimized implementation of trace(F * X * D.T * X.T) using matmul instead of einsum.
    F: (N, N)
    D: (N, N)
    X: (BS, N, N)
    """
    # Einsum path: torch.einsum('ij,bjk,lk,bil->b', F, X, D, X)
    # Optimized path:
    # 1. M1 = F @ X  -> PyTorch broadcasts F (N,N) against X (B,N,N) efficiently
    # 2. M2 = M1 @ D.T
    # 3. Trace(M2 @ X.T) = sum(M2 * X)
    
    # Pre-transpose D here or ensure inputs are correct. 
    # Note: To match einsum 'lk', we need D.T if input is D
    
    # F (N,N) @ X (B,N,N) -> (B,N,N)
    # We rely on broadcasting behavior of matmul
    val = torch.matmul(F, X)
    
    # (B,N,N) @ D.T (N,N) -> (B,N,N)
    val = torch.matmul(val, D.t())
    
    # Element-wise multiply and sum dimensions 1 and 2
    return torch.sum(val * X, dim=(1, 2))

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
    m_start_ptr = M_ptr + pid * stride_b
    out_start_ptr = Out_ptr + pid * stride_b
    
    # 3. 加载整个 N*N 矩阵到寄存器/SRAM
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < (N * N)
    
    # 加载 M，对于越界部分填充 -inf
    val = tl.load(m_start_ptr + offs, mask=mask, other=-float('inf'))
    
    # 循环 N 次寻找最大值
    for _ in range(N):
        # a. 找到当前剩余元素中的最大值索引
        max_val = tl.max(val, axis=0)
        
        # b. 找到该最大值对应的 index
        is_max = (val == max_val)
        idx = tl.argmax(is_max.to(tl.int32), axis=0) 
        
        # c. 计算 Row 和 Col
        r = idx // N
        c = idx % N
        
        # d. 写入结果 1.0
        tl.store(out_start_ptr + idx, 1.0)
        
        # e. Masking (核心加速点)
        row_mask = (offs // N) == r
        col_mask = (offs % N) == c
        combined_mask = row_mask | col_mask
        
        # 更新 val: 凡是 mask 命中的地方，更新为 -inf
        val = tl.where(combined_mask, -float('inf'), val)

def greedy_round_batch(M):
    bs, n, _ = M.shape
    
    # 计算需要的 Block Size
    block_size = triton.next_power_of_2(n * n)
    
    # 检查 Block Size 是否在合理的 Shared Memory 范围内
    # Float64 (8 bytes) * 8192 = 64KB (标准 consumer GPU 上限)
    # 如果 n > 90, block_size 变为 16384 (128KB)，可能需要 A100/H100 或 L2 cache spill
    # 这里的阈值设为 85，涵盖大多数 QAP benchmark 且安全
    if n <= 60: 
        assignment = torch.zeros_like(M)
        grid = (bs,)
        greedy_kernel[grid](
            M, 
            assignment,
            M.stride(0), M.stride(1), M.stride(2),
            N=n,
            BLOCK_SIZE=block_size
        )
    else:
        # Fallback for very large N
        assignment = greedy_round_large_batch(M)
    
    return assignment

# 使用 TorchScript JIT 编译 Sinkhorn，减少 Python 循环开销并融合算子
@torch.jit.script
def log_sinkhorn(log_alpha, num_iters: int = 10):
    """
    Performs Sinkhorn normalization in the log-domain for numerical stability.
    JIT compiled for loop fusion and speed.
    """
    # Initialize log_S as the input logits
    log_S = log_alpha.clone()

    for _ in range(num_iters):
        # 1. Row Normalization
        lse_row = torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - lse_row

        # 2. Column Normalization
        lse_col = torch.logsumexp(log_S, dim=-2, keepdim=True)
        log_S = log_S - lse_col

    return torch.exp(log_S)

def compute_loss(X, Y, F, D_T):
    """
    Calculates L(X, Y) = trace(F * S * D' * S') + sum(Y * (X^2 - X))
    Note: D passed here should be already transposed (D_T) for efficiency.
    """
    S = log_sinkhorn(X)
    
    # --- Term 1: Trace(F * S * D^T * S^T) ---
    # Optimized using matmul: (F @ S @ D_T) * S
    # F: (N,N), S: (B,N,N), D_T: (N,N)
    
    # Broadcast F matmul:
    M = torch.matmul(F, S) # (B,N,N)
    # Broadcast D_T matmul:
    M = torch.matmul(M, D_T) # (B,N,N)
    
    # Trace logic: sum element-wise product
    term1 = torch.sum(M * S)
    
    # Fusing the squaring and subtraction
    term2 = torch.sum(Y * (S.square() - S))
    
    return term1 + term2

def run_optimization(F_np, D_np, dual_init, batch_size, num_steps=100, lr=0.01, optimizer_type='adam'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = F_np.shape[0]
    
    # 使用 float64，如原代码要求。如果在非科研场景，建议改为 float32 以获得 2x 速度。
    dtype = torch.float64 
    
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    
    # Pre-calculate D transpose outside the loop
    D_T = D.transpose(-1, -2).contiguous()
    
    X = torch.rand((batch_size, n, n), device=device, dtype=dtype, requires_grad=True)
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam([X], lr=lr)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop([X], lr=lr)
        
    incumbent_obj = float('inf')
    incumbent_X = None
    
    wandb = False
    
    if wandb:
        import wandb
        wandb.init(project="QAP_solver", name=f"nug30_np_{optimizer_type}")
        wandb.config.update({
            "instance": "nug30",
            "optimizer": optimizer_type,
            "dual_init": dual_init,
            "gamma": lr,
            "beta": lr,
            "num_iters": num_steps,
        })
    
    for it in range(num_steps):
        optimizer.zero_grad()
        
        # Pass D_T instead of D
        loss = compute_loss(X, Y, F, D_T)
        loss.backward()
        optimizer.step()
        
        # update Y
        with torch.no_grad():
            # Re-compute P inside no_grad (fast due to JIT)
            P = log_sinkhorn(X)
            
            # Fused update
            Y.add_(P * P - P, alpha=lr)
    
            # rounding and evaluation
            X_int = greedy_round_batch(P)
            
            # 使用优化后的 obj_fn_batch
            obj_vals = obj_fn_batch(F, D, X_int) 
            
            # update incumbent        
            min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
            if min_obj_batch < incumbent_obj:
                incumbent_obj = min_obj_batch.item()
                incumbent_X = X_int[min_idx].clone()

            if (it+1) % 1000 == 0:
                # Optimized penalty calc
                penalty = (P.square() - P).mean().item()
                print(f"Iter {it+1}, Best: {incumbent_obj:.4f}, BatchMeanObj: {obj_vals.mean().item():.2f}, Pen: {penalty:.2e}")
                if np.abs(penalty) < 1e-6:
                    print("Early stopping due to low penalty.")
                    break
            
            if wandb:
                wandb.log({
                    # "iteration": it,
                    "objective": obj_fn_batch(F, D, P).item(),
                    "incumbent_objective": incumbent_obj,
                    "integrality_penalty": torch.abs((P*P-P)).mean().item(),
                    "lagrangian": (obj_fn_batch(F, D, P) + torch.sum(Y * (P * P - P))).mean().item(),
                    "dual_mean": torch.mean(Y),
                })
            
        
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