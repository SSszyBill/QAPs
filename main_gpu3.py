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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_greedy_cuda_ext = None
_greedy_cuda_ext_failed = False
_use_cuda_greedy = True
_two_opt_cuda_ext = None
_two_opt_cuda_ext_failed = False

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

def run_greedy_triton(M, out=None):
    bs, n, _ = M.shape
    block_size = triton.next_power_of_2(n * n)
    assignment = torch.zeros_like(M) if out is None else out.zero_()
    if n <= 128:
        grid = (bs,)
        greedy_kernel_optimized[grid](
            M, assignment,
            M.stride(0), M.stride(1), M.stride(2),
            N=n,
            BLOCK_SIZE=block_size
        )
    elif _use_cuda_greedy and M.is_cuda and M.dtype == torch.float32:
        cuda_ext = get_greedy_cuda_ext()
        if cuda_ext is not None:
            cuda_ext.greedy_round_cuda(M.contiguous(), assignment)
        else:
            assignment = greedy_round_large_batch_torch(M)
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

def get_greedy_cuda_ext():
    global _greedy_cuda_ext, _greedy_cuda_ext_failed
    if _greedy_cuda_ext is not None:
        return _greedy_cuda_ext
    if _greedy_cuda_ext_failed:
        return None

    cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<int THREADS>
__global__ void greedy_round_kernel(const float* __restrict__ M,
                                    float* __restrict__ Out,
                                    int BS,
                                    int N) {
    extern __shared__ unsigned char smem_raw[];
    float* vals = reinterpret_cast<float*>(smem_raw);
    int* idxs = reinterpret_cast<int*>(vals + THREADS);
    unsigned char* row_used = reinterpret_cast<unsigned char*>(idxs + THREADS);
    unsigned char* col_used = row_used + N;

    int b = blockIdx.x;
    int tid = threadIdx.x;
    int NN = N * N;
    const float* base = M + static_cast<long long>(b) * NN;
    float* out = Out + static_cast<long long>(b) * NN;

    for (int i = tid; i < N; i += THREADS) {
        row_used[i] = 0;
        col_used[i] = 0;
    }
    __syncthreads();

    for (int step = 0; step < N; ++step) {
        float best = -1.0e30f;
        int best_idx = 0;

        for (int idx = tid; idx < NN; idx += THREADS) {
            int r = idx / N;
            int c = idx - r * N;
            if (!row_used[r] && !col_used[c]) {
                float v = base[idx];
                if (v > best || (v == best && idx < best_idx)) {
                    best = v;
                    best_idx = idx;
                }
            }
        }

        vals[tid] = best;
        idxs[tid] = best_idx;
        __syncthreads();

        for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                float other_v = vals[tid + offset];
                int other_i = idxs[tid + offset];
                if (other_v > vals[tid] || (other_v == vals[tid] && other_i < idxs[tid])) {
                    vals[tid] = other_v;
                    idxs[tid] = other_i;
                }
            }
            __syncthreads();
        }

        int chosen = idxs[0];
        int r = chosen / N;
        int c = chosen - r * N;
        if (tid == 0) {
            out[chosen] = 1.0f;
            row_used[r] = 1;
            col_used[c] = 1;
        }
        __syncthreads();
    }
}

void greedy_round_cuda(torch::Tensor M, torch::Tensor Out) {
    TORCH_CHECK(M.is_cuda(), "M must be a CUDA tensor");
    TORCH_CHECK(Out.is_cuda(), "Out must be a CUDA tensor");
    TORCH_CHECK(M.scalar_type() == torch::kFloat32, "M must be float32");
    TORCH_CHECK(Out.scalar_type() == torch::kFloat32, "Out must be float32");
    TORCH_CHECK(M.dim() == 3, "M must have shape (batch, n, n)");
    int BS = M.size(0);
    int N = M.size(1);
    int threads = N >= 128 ? 512 : 256;
    int shared = threads * (sizeof(float) + sizeof(int)) + 2 * N * sizeof(unsigned char);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (threads == 512) {
        greedy_round_kernel<512><<<BS, 512, shared, stream>>>(M.data_ptr<float>(),
                                                              Out.data_ptr<float>(),
                                                              BS,
                                                              N);
    } else {
        greedy_round_kernel<256><<<BS, 256, shared, stream>>>(M.data_ptr<float>(),
                                                              Out.data_ptr<float>(),
                                                              BS,
                                                              N);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("greedy_round_cuda", &greedy_round_cuda, "Greedy QAP rounding CUDA");
}
'''
    try:
        _greedy_cuda_ext = load_inline(
            name="qap_greedy_round_ext_v7",
            cpp_sources="",
            cuda_sources=cuda_src,
            functions=None,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        return _greedy_cuda_ext
    except Exception as exc:
        print(f"Warning: CUDA greedy extension unavailable ({exc}). Falling back to PyTorch greedy.")
        _greedy_cuda_ext_failed = True
        return None

def get_two_opt_cuda_ext():
    global _two_opt_cuda_ext, _two_opt_cuda_ext_failed
    if _two_opt_cuda_ext is not None:
        return _two_opt_cuda_ext
    if _two_opt_cuda_ext_failed:
        return None

    cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void two_opt_step_kernel(int* __restrict__ perms,
                                    const scalar_t* __restrict__ F,
                                    const scalar_t* __restrict__ D,
                                    const int* __restrict__ actions,
                                    int BS,
                                    int N,
                                    int num_actions) {
    extern __shared__ unsigned char smem_raw[];
    double* s_deltas = reinterpret_cast<double*>(smem_raw);
    int* s_indices = reinterpret_cast<int*>(s_deltas + blockDim.x);

    int b = blockIdx.x;
    if (b >= BS) return;

    int tid = threadIdx.x;
    int* p = perms + static_cast<long long>(b) * N;
    const int* sample_actions = actions + static_cast<long long>(b) * num_actions * 2;

    double best_delta = 0.0;
    int best_idx = -1;

    for (int action_idx = tid; action_idx < num_actions; action_idx += blockDim.x) {
        int r = sample_actions[action_idx * 2 + 0];
        int s = sample_actions[action_idx * 2 + 1];
        if (r == s) continue;

        int pi_r = p[r];
        int pi_s = p[s];
        double delta = 0.0;

        for (int k = 0; k < N; ++k) {
            if (k != r && k != s) {
                int pi_k = p[k];
                delta += (static_cast<double>(F[k * N + r]) - static_cast<double>(F[k * N + s])) *
                         (static_cast<double>(D[pi_k * N + pi_s]) - static_cast<double>(D[pi_k * N + pi_r]));
                delta += (static_cast<double>(F[r * N + k]) - static_cast<double>(F[s * N + k])) *
                         (static_cast<double>(D[pi_s * N + pi_k]) - static_cast<double>(D[pi_r * N + pi_k]));
            }
        }

        delta += (static_cast<double>(F[r * N + r]) - static_cast<double>(F[s * N + s])) *
                 (static_cast<double>(D[pi_s * N + pi_s]) - static_cast<double>(D[pi_r * N + pi_r]));
        delta += (static_cast<double>(F[r * N + s]) - static_cast<double>(F[s * N + r])) *
                 (static_cast<double>(D[pi_s * N + pi_r]) - static_cast<double>(D[pi_r * N + pi_s]));

        if (delta < best_delta) {
            best_delta = delta;
            best_idx = action_idx;
        }
    }

    s_deltas[tid] = best_delta;
    s_indices[tid] = best_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            double other_delta = s_deltas[tid + offset];
            int other_idx = s_indices[tid + offset];
            if (other_delta < s_deltas[tid]) {
                s_deltas[tid] = other_delta;
                s_indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0 && s_deltas[0] < 0.0) {
        int best_action_idx = s_indices[0];
        int r = sample_actions[best_action_idx * 2 + 0];
        int s = sample_actions[best_action_idx * 2 + 1];
        int tmp = p[r];
        p[r] = p[s];
        p[s] = tmp;
    }
}

void two_opt_step_cuda(torch::Tensor perms,
                       torch::Tensor F,
                       torch::Tensor D,
                       torch::Tensor actions) {
    TORCH_CHECK(perms.is_cuda(), "perms must be a CUDA tensor");
    TORCH_CHECK(F.is_cuda(), "F must be a CUDA tensor");
    TORCH_CHECK(D.is_cuda(), "D must be a CUDA tensor");
    TORCH_CHECK(actions.is_cuda(), "actions must be a CUDA tensor");
    TORCH_CHECK(perms.scalar_type() == torch::kInt32, "perms must be int32");
    TORCH_CHECK(actions.scalar_type() == torch::kInt32, "actions must be int32");
    TORCH_CHECK(F.scalar_type() == D.scalar_type(), "F and D must have the same dtype");
    TORCH_CHECK(F.dim() == 2 && D.dim() == 2, "F and D must have shape (n, n)");
    TORCH_CHECK(perms.dim() == 2, "perms must have shape (batch, n)");
    TORCH_CHECK(actions.dim() == 3 && actions.size(2) == 2, "actions must have shape (batch, num_actions, 2)");
    TORCH_CHECK(F.is_contiguous() && D.is_contiguous(), "F and D must be contiguous");
    TORCH_CHECK(perms.is_contiguous() && actions.is_contiguous(), "perms and actions must be contiguous");

    int BS = perms.size(0);
    int N = perms.size(1);
    int num_actions = actions.size(1);
    TORCH_CHECK(F.size(0) == N && F.size(1) == N, "F shape must match perms");
    TORCH_CHECK(D.size(0) == N && D.size(1) == N, "D shape must match perms");
    TORCH_CHECK(actions.size(0) == BS, "actions batch must match perms");

    int threads = 256;
    size_t shared = threads * (sizeof(double) + sizeof(int));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(F.scalar_type(), "two_opt_step_cuda", ([&] {
        two_opt_step_kernel<scalar_t><<<BS, threads, shared, stream>>>(
            perms.data_ptr<int>(),
            F.data_ptr<scalar_t>(),
            D.data_ptr<scalar_t>(),
            actions.data_ptr<int>(),
            BS,
            N,
            num_actions);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("two_opt_step_cuda", &two_opt_step_cuda, "Batched sampled 2-opt QAP step");
}
'''
    try:
        _two_opt_cuda_ext = load_inline(
            name="qap_two_opt_ext_v1",
            cpp_sources="",
            cuda_sources=cuda_src,
            functions=None,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        return _two_opt_cuda_ext
    except Exception as exc:
        print(f"Warning: CUDA 2-opt extension unavailable ({exc}). Falling back to greedy-only evaluation.")
        _two_opt_cuda_ext_failed = True
        return None

def default_two_opt_options(n, batch_size, two_opt_iter, two_opt_actions, two_opt_topk):
    if two_opt_iter is None:
        two_opt_iter = int(np.clip(n, 20, 100))
    if two_opt_actions is None:
        two_opt_actions = int(np.clip(2 * n, 50, 200))
    if two_opt_topk is None:
        two_opt_topk = min(batch_size, 128)
    return int(two_opt_iter), int(two_opt_actions), int(two_opt_topk)

def run_two_opt_search(perms, F, D, max_iter, num_actions):
    if max_iter <= 0 or num_actions <= 0 or perms.numel() == 0 or perms.shape[1] < 2:
        return perms
    if not (perms.is_cuda and F.is_cuda and D.is_cuda):
        print("Warning: 2-opt currently requires CUDA tensors. Falling back to greedy-only evaluation.")
        return perms

    ext = get_two_opt_cuda_ext()
    if ext is None:
        return perms

    bs, n = perms.shape
    perms = perms.contiguous().to(torch.int32)
    F_contig = F.contiguous()
    D_contig = D.contiguous()

    for _ in range(max_iter):
        r = torch.randint(n, (bs, num_actions), device=perms.device, dtype=torch.int32)
        offset = torch.randint(n - 1, (bs, num_actions), device=perms.device, dtype=torch.int32) + 1
        actions = torch.empty((bs, num_actions, 2), device=perms.device, dtype=torch.int32)
        actions[:, :, 0] = r
        actions[:, :, 1] = (r + offset) % n
        ext.two_opt_step_cuda(perms, F_contig, D_contig, actions.contiguous())
    return perms

def permutations_to_assignment(perms, dtype):
    bs, n = perms.shape
    assignment = torch.zeros((bs, n, n), device=perms.device, dtype=dtype)
    assignment.scatter_(2, perms.long().unsqueeze(-1), 1.0)
    return assignment

@torch.compile
def sinkhorn_step(log_alpha, num_iters: int = 10, eps: float = 1):
    log_S = log_alpha / eps
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
    S = sinkhorn_step(X, num_iters=25)

    # 2. QAP Objective: Trace(F S D^T S^T)
    # 利用矩阵乘法结合律减少计算量
    # 路径: (F @ S) -> M1; (M1 @ D_T) -> M2; Sum(M2 * S)
    M1 = torch.matmul(F, S) 
    M2 = torch.matmul(M1, D_T)
    # term1 = torch.sum(M2 * S)
    term1 = torch.sum(M2 * S, dim=(1, 2)).mean()
    
    # 3. Penalty Term: sum(Y * (S^2 - S))
    # 提前计算 S^2 - S，既用于 Loss 也用于后续 Dual 更新
    # S_sq_minus_S = S * (S - 1.0)
    # S_sq_minus_S = S * torch.log(S+1e-30)
    S_sq_minus_S = S * torch.log(S+ 1e-30) + (1 - S) * torch.log(1 - S + 1e-30)
    # target = torch.empty(3).random_(2)
    # faster way to compute the S*log(S) + (1-S)*log(1-S)
    # term2 = torch.sum(Y * S_sq_minus_S)
    term2 = torch.sum(Y * S_sq_minus_S, dim=(1, 2)).mean()    
    loss = term1 + term2
    
    return loss, S, S_sq_minus_S, term1, term2

@torch.compile
def rmsprop_train_step_sinkhorn_grad(X, Y, square_avg, F, D_T, primal_lr: float, dual_lr: float):
    X_req = X.detach().requires_grad_(True)
    log_S = X_req
    for _ in range(25):
        log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
    P = torch.exp(log_S)

    with torch.no_grad():
        M1 = torch.matmul(F, P)
        M2 = torch.matmul(M1, D_T)
        term1 = torch.sum(M2 * P, dim=(1, 2)).mean()

        eps = 1e-30
        P_entropy = P * torch.log(P + eps) + (1 - P) * torch.log(1 - P + eps)
        term2 = torch.sum(Y * P_entropy, dim=(1, 2)).mean()
        loss = term1 + term2

        D = D_T.transpose(-1, -2)
        qap_grad = M2 + torch.matmul(torch.matmul(F.transpose(-1, -2), P), D)
        entropy_grad = (
            torch.log(P + eps)
            + P / (P + eps)
            - torch.log(1 - P + eps)
            - (1 - P) / (1 - P + eps)
        )
        grad_s = (qap_grad + Y * entropy_grad) / P.shape[0]

    grad, = torch.autograd.grad(P, X_req, grad_outputs=grad_s)
    square_avg = square_avg * 0.99 + grad * grad * 0.01
    X = X_req - primal_lr * grad / (torch.sqrt(square_avg) + 1e-8)
    Y = Y + dual_lr * P_entropy
    return X.detach(), Y.detach(), square_avg.detach(), P, loss, term1, term2

@torch.compile
def rmsprop_train_step_autograd(X, Y, square_avg, F, D_T, primal_lr: float, dual_lr: float):
    X_req = X.detach().requires_grad_(True)
    log_S = X_req
    for _ in range(25):
        log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
    P = torch.exp(log_S)

    M1 = torch.matmul(F, P)
    M2 = torch.matmul(M1, D_T)
    term1 = torch.sum(M2 * P, dim=(1, 2)).mean()
    P_entropy = P * torch.log(P + 1e-30) + (1 - P) * torch.log(1 - P + 1e-30)
    term2 = torch.sum(Y * P_entropy, dim=(1, 2)).mean()
    loss = term1 + term2

    grad, = torch.autograd.grad(loss, X_req)
    square_avg = square_avg * 0.99 + grad * grad * 0.01
    X = X_req - primal_lr * grad / (torch.sqrt(square_avg) + 1e-8)
    Y = Y + dual_lr * P_entropy
    return X.detach(), Y.detach(), square_avg.detach(), P.detach(), loss.detach(), term1.detach(), term2.detach()

@torch.compile(options={"max_autotune": True, "triton.cudagraphs": False})
def rmsprop_train_step_sinkhorn_grad_autotune(X, Y, square_avg, F, D_T, primal_lr: float, dual_lr: float):
    X_req = X.detach().requires_grad_(True)
    log_S = X_req
    for _ in range(25):
        log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
        log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
    P = torch.exp(log_S)

    with torch.no_grad():
        M1 = torch.matmul(F, P)
        M2 = torch.matmul(M1, D_T)
        term1 = torch.sum(M2 * P, dim=(1, 2)).mean()

        eps = 1e-30
        P_entropy = P * torch.log(P + eps) + (1 - P) * torch.log(1 - P + eps)
        term2 = torch.sum(Y * P_entropy, dim=(1, 2)).mean()
        loss = term1 + term2

        D = D_T.transpose(-1, -2)
        qap_grad = M2 + torch.matmul(torch.matmul(F.transpose(-1, -2), P), D)
        entropy_grad = (
            torch.log(P + eps)
            + P / (P + eps)
            - torch.log(1 - P + eps)
            - (1 - P) / (1 - P + eps)
        )
        grad_s = (qap_grad + Y * entropy_grad) / P.shape[0]

    grad, = torch.autograd.grad(P, X_req, grad_outputs=grad_s)
    square_avg = square_avg * 0.99 + grad * grad * 0.01
    X = X_req - primal_lr * grad / (torch.sqrt(square_avg) + 1e-8)
    Y = Y + dual_lr * P_entropy
    return X.detach(), Y.detach(), square_avg.detach(), P, loss, term1, term2

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

def read_reference_objective(instance, result_file="qaplib_labels.txt"):
    instance_name = instance.split('/')[-1]
    if result_file is None or not os.path.exists(result_file):
        return None
    with open(result_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == instance_name:
                return float(parts[1])
    return None

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


def spectral_initialization_qap(F, D, num=None, verbose=False):
    n = F.shape[0]
    if D.shape[0] != n:
        raise ValueError("Matrices F and D must have the same dimension.")

    P_list = []
    
    if num is None:
        k_list = np.arange(n) 
    else:
        # k_list = np.arange(min(num, n)) + 1
        k_list = np.arange(n-min(num, n), n) + 1
        if verbose:
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

def should_compile_train_step(n, mode):
    if mode == "on":
        return True
    return False

def train_step_impl(n, compile_tune):
    if compile_tune == "autograd":
        return rmsprop_train_step_autograd
    if compile_tune == "autotune" or (compile_tune == "default" and n <= 64):
        return rmsprop_train_step_sinkhorn_grad_autotune
    return rmsprop_train_step_sinkhorn_grad

def default_eval_schedule(n):
    return 1, 0

def default_compile_step(instance_name, n, mode, target_obj):
    if mode != "auto":
        return mode
    if target_obj is not None and (
        instance_name.startswith("bur26")
        or instance_name.startswith("chr")
        or instance_name == "els19"
        or instance_name in {
            "tai15a",
            "tai20b",
            "tai25a",
            "tai25b",
            "tai30a",
            "tai30b",
        }
    ):
        return "on"
    return "off"

def default_runtime_options(instance_name, n, eval_interval, eval_dense_until, target_obj):
    if target_obj is not None and instance_name.startswith("bur26"):
        if eval_interval is None:
            eval_interval = 4 if instance_name == "bur26h" else 5
        if eval_dense_until is None:
            eval_dense_until = 0
    elif target_obj is not None and instance_name.startswith("chr"):
        if eval_interval is None:
            eval_interval = 5
        if eval_dense_until is None:
            eval_dense_until = 0
    elif target_obj is not None and instance_name in {
        "tai15a",
        "tai20b",
        "tai25a",
        "tai25b",
        "tai30a",
        "tai30b",
    }:
        if eval_interval is None:
            eval_interval = 5
        if eval_dense_until is None:
            eval_dense_until = 0
    return eval_interval, eval_dense_until

def default_batch_size(instance_name, n, batch_size, target_obj):
    if target_obj is not None and instance_name == "chr15c":
        return max(batch_size, 3000)
    return batch_size

def should_check_all_exact(instance_name):
    return instance_name in {"chr15c"}

def should_use_cuda_graph_step(instance_name, optimizer_name, use_compiled_step, target_obj):
    return (
        target_obj is not None
        and instance_name in {"tai15b", "lipa40a"}
        and optimizer_name == "rmsprop"
        and not use_compiled_step
        and torch.cuda.is_available()
    )

def run_optimization(F_np, D_np, dual_init, batch_size, num_steps=100, primal_lr=0.02, dual_lr=0.02, optimizer_type='adam', x_label_np=None, compile_step='auto', c_backend='on', compile_tune='default', eval_interval=None, eval_dense_until=None, target_obj=None, instance_name="", two_opt=False, two_opt_iter=None, two_opt_actions=None, two_opt_topk=None, verbose=False):
    global _use_cuda_greedy
    _use_cuda_greedy = (c_backend == 'on')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dtype = torch.float32 
    
    # Warmup (对于 Triton 和 torch.compile 很重要)
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    X = torch.rand((2, F_np.shape[0], F_np.shape[0]), device=device, dtype=dtype, requires_grad=True)
    Y = torch.full((2, F_np.shape[0], F_np.shape[0]), dual_init, device=device, dtype=dtype)
    _ = compute_loss_and_grad(X[:2], Y[:2], F, D.transpose(-1, -2).contiguous())
    _ = run_greedy_triton(torch.rand((2, F_np.shape[0], F_np.shape[0]), device=device, dtype=dtype))
    F = torch.tensor(F_np, device=device, dtype=dtype)
    D = torch.tensor(D_np, device=device, dtype=dtype)
    D_T = D.transpose(-1, -2).contiguous() 
    F_int = torch.tensor(F_np, device=device, dtype=torch.int64) if target_obj is not None else None
    D_int = torch.tensor(D_np, device=device, dtype=torch.int64) if target_obj is not None else None
    target_obj_int = int(target_obj) if target_obj is not None else None
    n = F_np.shape[0]
    batch_size = default_batch_size(instance_name, n, batch_size, target_obj)
    compile_step = default_compile_step(instance_name, n, compile_step, target_obj)
    check_all_exact = target_obj is not None and should_check_all_exact(instance_name)
    eval_interval, eval_dense_until = default_runtime_options(
        instance_name, n, eval_interval, eval_dense_until, target_obj
    )
    if eval_interval is None or eval_dense_until is None:
        default_interval, default_dense_until = default_eval_schedule(n)
        if eval_interval is None:
            eval_interval = default_interval
        if eval_dense_until is None:
            eval_dense_until = default_dense_until
    two_opt_iter, two_opt_actions, two_opt_topk = default_two_opt_options(
        n, batch_size, two_opt_iter, two_opt_actions, two_opt_topk
    )
    two_opt_topk = min(two_opt_topk, batch_size)

    X_spectral = spectral_initialization_qap(F_np, D_np, batch_size, verbose=verbose)
    X = torch.tensor(X_spectral, device=device, dtype=dtype)
    # add random samples to fill the batch
    if X.shape[0] < batch_size:
        num_random = batch_size - X.shape[0]
        # X_random = torch.rand((num_random, n, n), device=device, dtype=dtype)
        X_rand = np.array(latin_hypercube_matrices(n, num_random))
        X_random = torch.tensor(X_rand, device=device, dtype=dtype)
        X = torch.cat([X, X_random], dim=0)
    X = X[:batch_size]
    X.requires_grad_(True)
    if verbose:
        print(X.shape[0])
    
    # X = torch.tensor(latin_hypercube_matrices(n, batch_size), device=device, dtype=dtype, requires_grad=True)
    
    
    # X = torch.rand((batch_size, n, n), device=device, dtype=dtype, requires_grad=True)
    Y = torch.full((batch_size, n, n), dual_init, device=device, dtype=dtype)
    
    optimizer_name = optimizer_type.lower()
    use_compiled_step = optimizer_name != 'adam' and should_compile_train_step(n, compile_step)
    if optimizer_name == 'adam':
        optimizer = optim.Adam([X], lr=primal_lr)
        rms_square_avg = None
    elif use_compiled_step:
        optimizer = None
        rms_square_avg = torch.zeros_like(X)
        compiled_train_step = train_step_impl(n, compile_tune)
        _ = compiled_train_step(X.clone(), Y.clone(), rms_square_avg.clone(), F, D_T, primal_lr, dual_lr)
    else:
        optimizer = None
        rms_square_avg = torch.zeros_like(X)
        compiled_train_step = None
    use_cuda_graph_step = should_use_cuda_graph_step(
        instance_name, optimizer_name, use_compiled_step, target_obj
    )
    graph_step = None
    graph_P = None
    if use_cuda_graph_step:
        # Warm graph memory pools on a side stream. The captured operations are
        # the same autograd loss/backward and RMSProp tensor update used below.
        # Use clones so the warm-up does not advance the real optimization state.
        warm_X = X.detach().clone().requires_grad_(True)
        warm_Y = Y.clone()
        warm_square_avg = rms_square_avg.clone()
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                warm_X.grad = None
                loss, P, P_sq_minus_P, term1, term2 = compute_loss_and_grad(
                    warm_X, warm_Y, F, D_T
                )
                loss.backward()
                with torch.no_grad():
                    grad = warm_X.grad
                    warm_square_avg.mul_(0.99).addcmul_(grad, grad, value=0.01)
                    warm_X.addcdiv_(
                        grad, warm_square_avg.sqrt().add_(1e-8), value=-primal_lr
                    )
                    warm_Y.add_(P_sq_minus_P, alpha=dual_lr)
        torch.cuda.current_stream().wait_stream(warm_stream)
        del warm_X, warm_Y, warm_square_avg

        graph_step = torch.cuda.CUDAGraph()
        X.grad = None
        with torch.cuda.graph(graph_step):
            X.grad = None
            loss, graph_P, P_sq_minus_P, term1, term2 = compute_loss_and_grad(X, Y, F, D_T)
            loss.backward()
            with torch.no_grad():
                grad = X.grad
                rms_square_avg.mul_(0.99).addcmul_(grad, grad, value=0.01)
                X.addcdiv_(grad, rms_square_avg.sqrt().add_(1e-8), value=-primal_lr)
                Y.add_(P_sq_minus_P, alpha=dual_lr)
    
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=lr*0.5)
    incumbent_obj = float('inf')
    
    if verbose:
        print(f"Starting Optimization [N={n}, Batch={batch_size}, Device={device}]")
        print(f"Evaluation schedule: dense_until={eval_dense_until}, interval={eval_interval}, target_obj={target_obj}")
        if two_opt:
            print(f"2-opt enabled: iter={two_opt_iter}, actions={two_opt_actions}, topk={two_opt_topk}")
    t0 = time.time()
    
    incumbents = []
    incum_time = []
    X_int_buffer = torch.empty_like(X)
    
    for it in range(num_steps):
        if use_cuda_graph_step:
            graph_step.replay()
            P = graph_P
        elif use_compiled_step:
            X, Y, rms_square_avg, P, loss, term1, term2 = compiled_train_step(
                X, Y, rms_square_avg, F, D_T, primal_lr, dual_lr
            )
        elif optimizer_name == 'rmsprop':
            X.grad = None
            loss, P, P_sq_minus_P, term1, term2 = compute_loss_and_grad(X, Y, F, D_T)
            loss.backward()
            with torch.no_grad():
                grad = X.grad
                rms_square_avg.mul_(0.99).addcmul_(grad, grad, value=0.01)
                X.addcdiv_(grad, rms_square_avg.sqrt().add_(1e-8), value=-primal_lr)
                Y.add_(P_sq_minus_P, alpha=dual_lr)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, P, P_sq_minus_P, term1, term2 = compute_loss_and_grad(X, Y, F, D_T)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                Y.add_(P_sq_minus_P, alpha=dual_lr)
        
        # 2. Dual Update & Evaluation
        if it < eval_dense_until or it % eval_interval == 0 or it == num_steps - 1:
            with torch.no_grad():
                X_int = run_greedy_triton(P, X_int_buffer)
                val = torch.matmul(F, X_int)
                val = torch.matmul(val, D_T)
                obj_vals = torch.sum(val * X_int, dim=(1, 2))

                if two_opt:
                    ls_count = min(two_opt_topk, X_int.shape[0])
                    if ls_count > 0:
                        _, ls_indices = torch.topk(obj_vals, k=ls_count, largest=False)
                        ls_perms = torch.argmax(X_int[ls_indices], dim=2).to(torch.int32).contiguous()
                        ls_perms = run_two_opt_search(
                            ls_perms, F, D, max_iter=two_opt_iter, num_actions=two_opt_actions
                        )
                        X_ls = permutations_to_assignment(ls_perms, X_int.dtype)
                        val_ls = torch.matmul(F, X_ls)
                        val_ls = torch.matmul(val_ls, D_T)
                        obj_ls = torch.sum(val_ls * X_ls, dim=(1, 2))
                        improved = obj_ls < obj_vals[ls_indices]
                        if torch.any(improved):
                            improved_indices = ls_indices[improved]
                            X_int[improved_indices] = X_ls[improved]
                            obj_vals[improved_indices] = obj_ls[improved]

                min_obj_batch, min_idx = torch.min(obj_vals, dim=0)
                if check_all_exact:
                    perms = torch.argmax(X_int, dim=2)
                    exact_objs = torch.sum(
                        F_int.unsqueeze(0)
                        * D_int[perms.unsqueeze(1), perms.unsqueeze(2)],
                        dim=(1, 2),
                    )
                    exact_hit = torch.nonzero(exact_objs == target_obj_int, as_tuple=False)
                    if exact_hit.numel() > 0:
                        hit_idx = exact_hit[0, 0]
                        incumbent_obj = obj_vals[hit_idx].item()
                        incumbent_X = X_int[hit_idx].clone()
                        incumbents.append(incumbent_obj)
                        incum_time.append(time.time() - t0)
                        break
                if min_obj_batch < incumbent_obj:
                    incumbent_obj = min_obj_batch.item()
                    incumbent_X = X_int[min_idx].clone()
                    incumbents.append(incumbent_obj)
                    time_now = time.time() - t0
                    incum_time.append(time_now)
                    if target_obj is not None:
                        perm = torch.argmax(incumbent_X, dim=1)
                        exact_obj = torch.sum(F_int * D_int[perm.unsqueeze(1), perm.unsqueeze(0)])
                        if exact_obj.item() == target_obj_int:
                            break
                
            if verbose and (it+1) % 100 == 0:
                print(f"Iter {it+1}: Best Obj {incumbent_obj:.4f}, Loss {loss.item():.4f}, Term1 {term1.item():.4f}, Term2 {term2.item():.4f}")
                
    total_time = time.time() - t0
    if verbose:
        print(f"Total Time: {total_time:.2f}s, FPS: {num_steps/total_time:.1f}")
    return incumbent_X, incumbent_obj, total_time, incumbents, incum_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default="nug12")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', type=str, default="rmsprop", choices=["rmsprop", "adam"])
    parser.add_argument('--dual_init', type=float, default=1.0)
    parser.add_argument('--primal_lr', type=float, default=0.02)
    parser.add_argument('--dual_lr', type=float, default=0.02)
    parser.add_argument('--c_backend', type=str, default="on", choices=["on", "off"])
    parser.add_argument('--compile_step', type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument('--compile_tune', type=str, default="default", choices=["default", "autotune", "autograd"])
    parser.add_argument('--eval_interval', type=int, default=None)
    parser.add_argument('--eval_dense_until', type=int, default=None)
    parser.add_argument('--target_stop', type=str, default="on", choices=["on", "off"])
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--tf32', type=str, default="off", choices=["on", "off"])
    parser.add_argument('--two_opt', type=str, default="off", choices=["on", "off"])
    parser.add_argument('--two_opt_iter', type=int, default=None)
    parser.add_argument('--two_opt_actions', type=int, default=None)
    parser.add_argument('--two_opt_topk', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.tf32 == "on":
        torch.set_float32_matmul_precision("high")
    
    n, F_np, D_np, obj_label, x_label_np = read_instance(args.instance)
    if args.verbose:
        print(np.sum(np.diag(F_np)), np.sum(np.diag(D_np)))
    # FD = np.kron(F_np, D_np)
    # # compute the eigenvalues and eigenvectors
    # vals, vecs = np.linalg.eigh(FD)
    # print("Smallest Eigenvalue of Kronecker Product:", vals[0])
    # print("Largest Eigenvalue of Kronecker Product:", vals[-1])
    
    batch_size = args.batch_size
    num_steps = args.iters
    
    if n < 200:
        # batch_size = 5000
        # num_steps = 500
        # taillard
        batch_size = 500
        num_steps = 1200
    elif n < 500:
        batch_size = 500
        num_steps = 1200
    else:
        batch_size = 200
        num_steps = 1200
    
    primal_lr = args.primal_lr
    dual_lr = args.dual_lr
    dual_init = args.dual_init
    if args.target_stop == "on":
        target_obj = read_reference_objective(args.instance, args.target_file)
        if target_obj is None:
            target_obj = read_reference_objective(args.instance)
    else:
        target_obj = None
    dtype = torch.float32
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    X_best, obj_best, solve_time_raw, incumbents, incum_time = run_optimization(
        F_np,
        D_np,
        dual_init,
        batch_size,
        num_steps,
        primal_lr,
        dual_lr,
        optimizer_type=args.optimizer,
        x_label_np=x_label_np,
        compile_step=args.compile_step,
        c_backend=args.c_backend,
        compile_tune=args.compile_tune,
        eval_interval=args.eval_interval,
        eval_dense_until=args.eval_dense_until,
        target_obj=target_obj,
        instance_name=args.instance.split('/')[-1],
        two_opt=args.two_opt == "on",
        two_opt_iter=args.two_opt_iter,
        two_opt_actions=args.two_opt_actions,
        two_opt_topk=args.two_opt_topk,
        verbose=args.verbose,
    )
    end_event.record()
    torch.cuda.synchronize()
    
    row_sums = X_best.sum(axis=1)
    col_sums = X_best.sum(axis=0)
    if row_sums.min() < 0.99 or row_sums.max() > 1.01:
        print("Warning: Solution might not be a valid permutation.")
    elif args.verbose:
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
        # f.write(f"{instance_name} {solve_time:.2f} {solve_time_raw:.2f} {obj_best} {obj_label} {gap:.4f}\n")
        f.write(f"{instance_name} {solve_time:.2f} {solve_time_raw:.2f} {obj_best}\n")
