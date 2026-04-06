import torch
import argparse
import numpy as np
import time
from typing import Tuple, Optional
import triton
import triton.language as tl
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import torch.nn as nn
import torch.optim as optim

def read_instance(instance):
    # 请确保路径正确
    base_path = "/home/szy/QAPszy/qaplibs/"
    problem_file = f"{base_path}qapdata/{instance}.dat"
    solution_file = f"{base_path}qapsoln/{instance}.sln"
    
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

@triton.jit
def greedy_kernal_optimized(
        M_ptr, Out_ptr, stride_b, stride_h, stride_w, 
        N: tl.constexpr, 
        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    m_ptr = M_ptr + pid * stride_b
    out_ptr = Out_ptr + pid * stride_b
    offs = tl.arange(0, BLOCK_SIZE)
    mask_load = offs < (N * N)
    val = tl.load(m_ptr + offs, mask=mask_load, other=-1e10)
    for _ in range(N):
        max_val = tl.max(val, axis=0)
        is_max = (val == max_val)
        idx = tl.argmax(is_max.to(tl.int32), axis=0)
        r = idx // N
        c = idx % N
        tl.store(out_ptr + idx, 1.0)

        row_mask = (offs // N == r)
        col_mask = (offs % N == c)
        combined_mask = row_mask | col_mask
        val = tl.where(combined_mask, -1e10, val)

def greedy_round_large_batch(X):
    """
    Input: (B, N, N) continuous matrix
    Output: (B, N, N) binary permutation matrix
    TODO: Insert your Triton kernel or Torch implementation here.
    """
    bs, n, _ = X.shape
    assignment = torch.zeros_like(X)
    X_temp = X.clone()
    batch_indices = torch.arange(bs, device=X.device)

    for _ in range(n):
        flot_X = X_temp.view(bs, -1)
        _, idx = flot_X.max(dim=1)

        rows = idx.div(n, rounding_mode='floor')
        cols = idx % n
        
        assignment[batch_indices, rows, cols] = 1.0
        
        X_temp[batch_indices, rows, :] = -1e10
        X_temp[batch_indices, :, cols] = -1e10
        
    return assignment

def greedy_round(X):
    bs, n, _ = X.shape
    if n < 60:
        assignment = torch.zeros_like(X)
        block_size = triton.next_power_of_2(n * n)
        grid = (bs,)
        greedy_kernal_optimized[grid](
            X, 
            assignment,
            X.stride(0), X.stride(1), X.stride(2), 
            N=n, BLOCK_SIZE=block_size)
    else:
        assignment = greedy_round_large_batch(X)    

    return assignment

def checker(X):
    row_sums = X.sum(dim=1)
    col_sums = X.sum(dim=0)
    if row_sums.min() < 0.99 or row_sums.max() > 1.01:
        print("Warning: Solution might not be a valid permutation.")
        return False
    else:
        print("Solution is a valid permutation matrix.")
        return True

def analyze_history(history,bks):
    """
    'iter': it + 1,
    'obj': min_val,
    'time': current_time
    分析优化历史记录，返回最优解、收敛速度等指标
    """
    targets = [0.01, 0.02, 0.05, 0.1]  # 目标gap百分比
    metrics = {f"T_to_{int(t*100)}%": None for t in targets}

    final_obj = history[-1]['obj']
    time_to_best = None

    for record in history:
        t = record['time']
        obj = record['obj']
        gap = (obj - bks) / bks

        for target in targets:
            k = f"T_to_{int(target*100)}%"
            if gap <= target + 1e-6 and metrics[k] is None:
                metrics[k] = t

        if abs(obj - final_obj) < 1e-6 and time_to_best is None:
            time_to_best = t

    return metrics, time_to_best

# def latin_hypercube_matrices(m, n):
#         """使用拉丁超立方体设计生成多样矩阵"""
#         # 将矩阵向量化后的参数空间采样
#         d = m * m  # 参数维度
        
#         # 生成拉丁超立方体设计
#         samples = np.zeros((n, d))
#         for j in range(d):
#             perm = np.random.permutation(n)
#             samples[:, j] = (perm + np.random.rand(n)) / n
        
#         # 转换为矩阵
#         matrices = []
#         for i in range(n):
#             # 映射到不同分布
#             vec = samples[i, :]
            
#             # 可选：应用不同变换获得不同模式
#             transform_type = i % 4
#             if transform_type == 0:
#                 vec = np.tanh(vec * 4 - 2)  # 压缩到[-1,1]
#             elif transform_type == 1:
#                 vec = np.sin(vec * 2 * np.pi)  # 周期性
#             elif transform_type == 2:
#                 vec = np.exp(vec * 3 - 1.5)  # 正数
#             else:
#                 vec = vec * 4 - 2  # 线性
            
#             mat = vec.reshape(m, m)
#             matrices.append(mat)
        
#         return matrices

def get_spectral_init(F, D, n, batch_size, device='cuda'):
    """
    生成谱初始化矩阵 X (Continuous)。
    基于 Umeyama 算法或谱松弛:X ~ |U_F| @ |U_D|^T
    
    返回:
    - X_init (torch.Tensor): 形状为 (batch_size, n, n)，在 [0, 1] 之间或是原始相似度值。
    """
    # 1. 确保输入是 Tensor
    if isinstance(F, np.ndarray):
        F = torch.tensor(F, device=device, dtype=torch.float32)
    if isinstance(D, np.ndarray):
        D = torch.tensor(D, device=device, dtype=torch.float32)

    # 2. 特征值分解 (Eigendecomposition)
    # 既然是实对称矩阵，使用 eigh 更快更稳
    L_F, U_F = torch.linalg.eigh(F) 
    L_D, U_D = torch.linalg.eigh(D)
    
    # 3. 排序 (eigh 默认是从小到大，我们需要统一顺序，通常从大到小或者从小到大一致即可)
    # 为了对齐结构，我们把特征值从大到小排列
    idx_F = torch.argsort(L_F, descending=True)
    idx_D = torch.argsort(L_D, descending=True)
    
    U_F = U_F[:, idx_F]
    U_D = U_D[:, idx_D]
    
    # 4. 构造关联矩阵 X
    # 核心思想：结构相似的节点在特征空间中应该靠得近
    # 常见做法取绝对值以避免符号模糊性 (Sign Ambiguity)
    # X = |U_F| @ |U_D|^T
    U_F_abs = torch.abs(U_F)
    U_D_abs = torch.abs(U_D)
    
    X_spectral = torch.matmul(U_F_abs, U_D_abs.t())
    
    # 5. (可选) 归一化或微调
    # 这里建议加一点小的随机噪声，防止完全对称导致的卡死（虽然谱初始化通常已经破坏了对称性）
    # X_spectral = X_spectral + 1e-4 * torch.rand_like(X_spectral)

    # 6. 扩展到 Batch 维度
    # (n, n) -> (batch_size, n, n)
    X_init = X_spectral.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    return X_init.to(device)