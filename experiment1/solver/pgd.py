import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from .common import read_instance
from .common import greedy_kernal_optimized
from .common import greedy_round_large_batch
from .common import greedy_round
import time

class PGDSolver:
    def __init__(self, F, D, device='cuda', seed=0):
        # 1. 统一转为 Float Tensor
        if isinstance(F, np.ndarray):
            self.F = torch.tensor(F, device=device, dtype=torch.float32)
        else:
            self.F = F.to(device).float()
            
        if isinstance(D, np.ndarray):
            self.D = torch.tensor(D, device=device, dtype=torch.float32)
        else:
            self.D = D.to(device).float()
            
        self.n = self.F.shape[0]
        self.device = device
        self.seed = seed
        
        # === 核心修复：必须在这里定义 F_nodiag 等属性 ===
        # 提取对角线
        self.diag_F = torch.diagonal(self.F)
        self.diag_D = torch.diagonal(self.D)
        
        # 构造去对角线矩阵 (Off-diagonal matrices)
        self.F_nodiag = self.F - torch.diag(self.diag_F)
        self.D_nodiag = self.D - torch.diag(self.diag_D)

    def grad_trace(self,F, D, X):
        """
        F X D^T + F^T X D
        F, D: (n, n)
        X: (n, n)
        """
        # Term 1: F @ X @ D.T
        FX = torch.matmul(F, X) # (n, n)
        FXDt = torch.matmul(FX, D.t()) # (n, n)
        
        # Term 2: F.T @ X @ D
        FtX = torch.matmul(F.t(), X)
        FtXD = torch.matmul(FtX, D)
        
        return FXDt + FtXD

    def grad_L_X_batch(self,F, D, X, Y, diag_F=None, diag_D=None):
        """
        F X D^T + F^T X D + Y * (2X - 1)
        F, D: (n, n)
        X, Y: (bs, n, n)
        """
        # # Term 1: F @ X @ D.T
        # FX = torch.matmul(F, X) # (bs, n, n)
        # FXDt = torch.matmul(FX, D.t()) # (bs, n, n)
        
        # # Term 2: F.T @ X @ D
        # FtX = torch.matmul(F.t(), X)
        # FtXD = torch.matmul(FtX, D)
        obj_grad = self.grad_trace(F, D, X)  # (bs, n, n)
        
        # Term 3: Penalty
        penalty = Y * (2.0 * X - 1.0)
        
        if diag_F is None and diag_D is None:
            return obj_grad + penalty
        else:
            diag_term = torch.einsum('i,k->ik', diag_F, diag_D)  # (n, n)
            return obj_grad + penalty + diag_term


    def grad_L_Y_batch(self, X):
        return X * X - X

    def obj_fn_batch(self, F, D, X, diag_F=None, diag_D=None):
        """
        trace(F * X * D.T * X.T)
        F, D: (n, n)
        """
        if diag_F is None and diag_D is None:
            return torch.einsum('ij,bjk,lk,bil->b', F, X, D, X) 
        else:
            # return trace(F * X * D.T * X.T) + trace(C.T * X)
            return torch.einsum('ij,bjk,lk,bil->b', F, X, D, X) \
                + torch.einsum('i,k,bik->b', diag_F, diag_D, X)

    def dykstra_project(self, M: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
        """The Dykstra or Sinkhorn projection step"""
        X = M.clone()
        p = torch.zeros_like(X)
        q = torch.zeros_like(X)

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
    def adam_torch(self, X: torch.Tensor, 
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

    def solve(self, batch_size=100, max_iter=1000, params=None, X_init=None, log_interval=20):
        if params is None: params = {}
        gamma = params.get('gamma', 0.05)
        beta = params.get('beta', 0.05)

        torch.manual_seed(self.seed)
        if X_init is not None:
            X = X_init.clone().to(self.device)

        else:
            X = torch.rand((batch_size, self.n, self.n), device=self.device)

        Y = torch.full((batch_size, self.n, self.n), 10.0, device=self.device)
        v = torch.zeros_like(X)
        m = torch.zeros_like(X)
        
        best_obj = float('inf')
        best_X = None

        t0 = time.time()
        history = []

        for it in range(max_iter):
            # 1. 梯度计算 (使用 self.F_nodiag 等)
            FX = torch.matmul(self.F_nodiag, X)
            grad_obj = torch.matmul(FX, self.D_nodiag.t()) + torch.matmul(self.F_nodiag.t(), X).matmul(self.D_nodiag)
            
            penalty = Y * (2.0 * X - 1.0)
            diag_term = torch.einsum('i,k->ik', self.diag_F, self.diag_D)
            grad_X = grad_obj + penalty + diag_term
            
            grad_Y = X * X - X
            
            # 2. 更新
            m = 0.9 * m + 0.1 * grad_X
            v = 0.999 * v + 0.001 * (grad_X * grad_X)
            m_hat = m / (1 - 0.9**(it+1))
            v_hat = v / (1 - 0.999**(it+1))
            X = X - gamma * m_hat / (torch.sqrt(v_hat) + 1e-8)
            
            Y = Y - beta * grad_Y
            
            # 3. 投影
            X = torch.clamp(X, min=1e-10)
            for _ in range(10):
                 X = X / X.sum(dim=-1, keepdim=True)
                 X = X / X.sum(dim=-2, keepdim=True)

            # 4. 评估
            if it == 0 or (it + 1) % log_interval == 0 or it == max_iter - 1:
                X_int = greedy_round(X)
                obj_vals = self.obj_fn_batch(self.F, self.D, X_int)
                current_min = obj_vals.min().item()

                if current_min < best_obj:
                    best_obj = current_min
                    best_X = X_int.clone().to('cpu')
                
                current_time = time.time() - t0
                history.append({
                    'iter': it + 1,
                    'obj': current_min,
                    'time': current_time
                })

        total_time = time.time() - t0
        return best_obj, total_time, best_X, history

