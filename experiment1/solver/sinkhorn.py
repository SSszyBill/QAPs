import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from .common import read_instance
from .common import greedy_kernal_optimized
from .common import greedy_round_large_batch
from .common import greedy_round
from .common import get_spectral_init
import scipy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time

class SinkhornSolver:
    def __init__(self, F, D, device='cuda', seed=0):
        """
        初始化函数：
        负责将外部传进来的 Numpy 矩阵转换为 GPU 上的 Tensor,
        并完成所有只需要做一次的预处理工作。
        
        参数:
        F (np.ndarray 或 torch.Tensor): 流量矩阵 (Flow)
        D (np.ndarray 或 torch.Tensor): 距离矩阵 (Distance)
        device (str): 'cuda' 或 'cpu'
        seed (int): 随机种子
        """
        self.device = device
        self.seed = seed
        self.n = F.shape[0]

        # 1. 统一数据格式：Numpy -> Tensor (GPU)
        # 这样无论外面传进来的是什么，内部统一用 float32 的 Tensor
        if isinstance(F, np.ndarray):
            self.F = torch.tensor(F, device=device, dtype=torch.float32)
        else:
            self.F = F.to(device).float()
            
        if isinstance(D, np.ndarray):
            self.D = torch.tensor(D, device=device, dtype=torch.float32)
        else:
            self.D = D.to(device).float()
            
        # 2. 预计算常量 (Pre-computation)
        # 你的 Loss 计算中频繁用到 D 的转置，初始化时算一次存下来，比在循环里每次都算快
        self.D_T = self.D.transpose(-1, -2).contiguous()

    def sinkhorn_step(self, log_alpha, num_iters: int = 10):
        log_S = log_alpha
        for _ in range(num_iters):
            log_S = log_S - torch.logsumexp(log_S, dim=-1, keepdim=True)
            log_S = log_S - torch.logsumexp(log_S, dim=-2, keepdim=True)
        return torch.exp(log_S)
    
    def compute_loss_and_grad(self, X, Y):
        """
        将 Loss 计算和梯度相关的操作融合，减少中间变量显存占用。
        """
        # 1. Sinkhorn Forward
        torch.cuda.nvtx.range_push("Core_Sinkhorn_Loop") 
        S = self.sinkhorn_step(X, num_iters=50)
        torch.cuda.nvtx.range_pop()
        
        # 2. QAP Objective: Trace(F S D^T S^T)
        # 利用矩阵乘法结合律减少计算量
        # 路径: (F @ S) -> M1; (M1 @ D_T) -> M2; Sum(M2 * S)
        torch.cuda.nvtx.range_push("Calc_Trace_Objective")
        M1 = torch.matmul(self.F, S) 
        M2 = torch.matmul(M1, self.D_T)
        term1 = torch.sum(M2 * S, dim=(1, 2)).mean()
        torch.cuda.nvtx.range_pop()
        
        # 3. Penalty Term: sum(Y * (S^2 - S))
        torch.cuda.nvtx.range_push("Calc_Penalty")
        S_sq_minus_S = S * torch.log(S+1e-30) # 注意：你变量名写的是 sq 但逻辑是 log，这是熵正则？
        term2 = torch.sum(Y * S_sq_minus_S, dim=(1, 2)).mean()    
        loss = term1 + term2
        torch.cuda.nvtx.range_pop()
        
        return loss, S, S_sq_minus_S, term1, term2
    
    def solve(self, batch_size=100, max_iter=100, params=None, X_init=None, log_interval=20):
        if params is None: params = {}
        # 从 params 字典中获取超参数，如果没有则使用默认值
        lr = params.get('lr', 0.02)
        dual_init = params.get('dual_init', 1.0)
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.time()

        if X_init is not None:
            X = X_init.clone().to(self.device)
        else:
            X_init = get_spectral_init(self.F, self.D, self.n, batch_size, self.device)
            
        X = X_init.detach().clone()
        X.requires_grad_(True)
        Y = torch.full((batch_size, self.n, self.n), dual_init, device=self.device)
        
        optimizer = optim.Adam([X], lr=lr)
        
        incumbent_obj = float('inf')
        history = []
        
        # # Warmup
        # if self.n < 300:
        #     _ = self.compute_loss_and_grad(X[:2], Y[:2])
        
        # === 注意：这里的循环变量要是 max_iter ===
        best_X = None
        for it in range(max_iter):

            torch.cuda.nvtx.range_push(f"Iter_{it}")
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.nvtx.range_push("Forward_Pass")
            loss, S, S_sq_minus_S, _, _ = self.compute_loss_and_grad(X, Y)
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("Backward_Pass")
            loss.backward()
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("Update_Dual_Variables")
            optimizer.step()
            torch.cuda.nvtx.range_pop()
            
            with torch.no_grad():
                Y.add_(S_sq_minus_S, alpha=lr)
                
                if it == 0 or (it + 1) % log_interval == 0 or it == max_iter - 1:
                    torch.cuda.nvtx.range_push("Logging_and_Rounding")
                    X_int = greedy_round(S) 
                    
                    val = torch.matmul(self.F, X_int)
                    val = torch.matmul(val, self.D_T)
                    obj_vals = torch.sum(val * X_int, dim=(1, 2))
                    
                    min_val = obj_vals.min().item()
                    if min_val < incumbent_obj:
                        incumbent_obj = min_val
                        best_X = X_int.clone().to('cpu')
                
                    current_time = time.time() - t0
                    history.append({
                        'iter': it + 1,
                        'obj': min_val,
                        'time': current_time
                    })
                    torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()  # End of Iter
                        
        total_time = time.time() - t0
        return incumbent_obj, total_time, best_X, history