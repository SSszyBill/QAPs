import torch
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

def read_instance(instance):
    problem_file = f"/home/xjx/A-xjx/QAP/qapdata/{instance}.dat"
    solution_file = f"/home/xjx/A-xjx/QAP/qapsoln/{instance}.sln"
    
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
    assert n_sol == n, "Solution size does not match instance size"
    obj_label = next(sol_data_iter)
    x_label = [next(sol_data_iter) - 1 for _ in range(n)]
    x_label_np = np.zeros((n, n))
    for i in range(n):
        x_label_np[i, x_label[i]] = 1
        
    return n, F_np, D_np, obj_label, x_label_np

def obj_fn_np(F, D, X):
    tmp = X @ D.T @ X.T
    obj = np.trace(F @ tmp)
    return obj

def grad_L_X_np(F, D, X, Y):
    return F @ X @ D.T + F.T @ X @ D + Y * (2 * X - np.ones_like(X))

def grad_L_Y_np(X):
    return X * X - X

def rmsprop_np(X, grad, v=None, buffer=None, 
                      gamma=0.01, alpha=0.99, lam=0.01, mu=0.6, eps=1e-8):
    if v is None:
        v = np.zeros_like(X)
    if buffer is None:
        buffer = np.zeros_like(X)
        
    v = alpha * v + (1.0 - alpha) * (grad ** 2)
    
    denom = np.sqrt(v) + eps
    scaled_grad = grad / denom
    
    if lam != 0.0:
        scaled_grad = scaled_grad + lam * X
        
    if mu > 0.0:
        buffer = mu * buffer + scaled_grad
        step = gamma * buffer
    else:
        step = gamma * scaled_grad
        
    X = X - step
    
    return X, v, buffer


def adam_np(X, grad, v=None, m=None, t=1, 
            gamma=0.001, beta1=0.9, beta2=0.999, lam=0.0, eps=1e-8):
    if v is None:
        v = np.zeros_like(X)
    if m is None:
        m = np.zeros_like(X)
    
    if lam != 0:
        grad = grad + lam * X
        
    m = beta1 * m + (1 - beta1) * grad
    
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    X -= gamma * m_hat / (np.sqrt(v_hat) + eps)
    
    return X, v, m

def dykstra_proj_np(M, max_iter=100, tol=1e-5):
    '''
    dykstra algorithm for projecting onto doubly stochastic matrices
    X: (n, n) numpy array
    '''
    X = M.copy()
    # correction terms
    p = np.zeros_like(M)
    q = np.zeros_like(M)
    
    n = float(M.shape[0])
    
    for k in range(max_iter):
        X_prev = X.copy()

        Y = X + p

        row_diff = (np.sum(Y, axis=1, keepdims=True) - 1.0) / n
        col_diff = (np.sum(Y, axis=0, keepdims=True) - 1.0) / n
        grand_diff = np.sum(row_diff) / n
        Y = Y - row_diff - col_diff + grand_diff
        
        p = X + p - Y
        X = Y
        
        # Project onto non-negative orthant
        Y = X + q
        X = np.maximum(Y, 0.0)
        q = Y - X
        
        # Check convergence
        if np.max(np.abs(X - X_prev)) < tol:
            break
    return X

def dykstra_proj_gurobi(A):
    m, n = A.shape
    model = gp.Model("ProjectionToDoublyStochastic")
    x = model.addVars(m, n, lb=0.0, name="x")
    obj = 0
    for i in range(m):
        for j in range(n):
            obj += 0.5 * x[i,j] * x[i,j] - A[i,j] * x[i,j]
    model.setObjective(obj, GRB.MINIMIZE)
    for i in range(m):
        model.addConstr(gp.quicksum(x[i,j] for j in range(n)) == 1.0, 
                       f"row_sum_{i}")
    for j in range(n):
        model.addConstr(gp.quicksum(x[i,j] for i in range(m)) == 1.0,
                       f"col_sum_{j}")
    model.setParam('OutputFlag', 0)  # 关闭输出日志
    model.optimize()
    if model.status == GRB.OPTIMAL:
        X_sol = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                X_sol[i, j] = x[i, j].X
        return X_sol
    else:
        print(f"求解失败，状态码: {model.status}")
        return None


def greedy_round_np(M):
    '''
    Greedy rounding to get a permutation matrix from a doubly stochastic matrix
    M: (n, n) numpy array
    '''
    N = M.shape[0]
    assignment = np.zeros((N, N), dtype=int)
    
    M_temp = M.copy().astype(float)
    
    for _ in range(N):
        # 1. 找到当前剩余矩阵中的最大值索引
        idx = np.argmax(M_temp)
        r, c = np.unravel_index(idx, (N, N))
        
        # 2. 填充结果
        assignment[r, c] = 1
        
        # 3. Mask 掉该行和该列 (设为极小值，确保下次不会被选中)
        M_temp[r, :] = -np.inf
        M_temp[:, c] = -np.inf
        
    return assignment


def solve_np(instance, optimizer="adam", dual_init=10.0, gamma=0.01, beta=0.01, num_iters=1000, wandb=False):
    n, F_np, D_np, obj_label, x_label_np = read_instance(instance)
    
    X = np.random.rand(n, n)
    Y = np.full((n, n), dual_init, dtype=float)
    v = None
    buffer = None
    
    incumbent_obj = float('inf')
    incumbent_X = None
    
    integrity_penalty = []
    
    if wandb:
        import wandb
        wandb.init(project="QAP_solver", name=f"{instance}_np_{optimizer}")
        wandb.config.update({
            "instance": instance,
            "optimizer": optimizer,
            "dual_init": dual_init,
            "gamma": gamma,
            "beta": beta,
            "num_iters": num_iters
        })
        
    for it in range(num_iters):
        grad_X = grad_L_X_np(F_np, D_np, X, Y)
        grad_Y = grad_L_Y_np(X)
        
        if optimizer == "adam":
            X, v, buffer = adam_np(X, grad_X, v, buffer, t=it+1, gamma=gamma)
        elif optimizer == "rmsprop":
            X, v, buffer = rmsprop_np(X, grad_X, v, buffer, gamma=gamma)
        elif optimizer == "gd":
            np.add(X, -gamma * grad_X, out=X)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        np.add(Y, grad_Y * beta, out=Y)
        
        X = dykstra_proj_np(X)
        integrality_penalty = np.abs(np.sum(X * X - X))
        
        X_int = greedy_round_np(X)
        obj_int = obj_fn_np(F_np, D_np, X_int)
        if obj_int < incumbent_obj:
            incumbent_obj = obj_int
            incumbent_X = X_int.copy()
            
            # check the row and column sums
            row_sums = incumbent_X.sum(axis=1)
            col_sums = incumbent_X.sum(axis=0)
            
            if row_sums.min() < 0.999 or row_sums.max() > 1.001 or col_sums.min() < 0.999 or col_sums.max() > 1.001:
                print("Warning: Obtained solution is not a valid permutation matrix.")
    
        if it % 100 == 0:
            print(f"Iter {it}, Incumbent obj: {incumbent_obj}, obj: {obj_fn_np(F_np, D_np, X)}, Integrality penalty: {integrality_penalty}, Y mean: {Y.min()}")

        # log to wandb
        if wandb:
            wandb.log({
                "iteration": it,
                "objective": obj_fn_np(F_np, D_np, X),
                "incumbent_objective": incumbent_obj,
                "integrality_penalty": integrality_penalty,
            })
        
        if np.abs(integrality_penalty) < 1e-8:
            print(f"Converged with integrality penalty {integrality_penalty} at iteration {it}")
            break
        
    return incumbent_X, incumbent_obj, obj_label



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default="nug12")
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "rmsprop", "gd"])
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    
    start_time = time.time()
    X, incumbent_obj, obj_label = solve_np(args.instance, args.optimizer, dual_init=5.0, gamma=0.02, beta=0.02, num_iters=100000000, wandb=args.wandb)
    end_time = time.time()
    print(f"Solve time: {end_time - start_time} seconds")
    
    print(f"Final incumbent objective: {incumbent_obj}, Solution file objective: {obj_label}")