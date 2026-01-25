import torch
import numpy as np
import os
import argparse
import functools
import time

import pygmtools as pygm
pygm.set_backend('pytorch')


# from ..main_GI import generate_isomorphic


def read_instance(instance):
    problem_file = f"/home/xjx/A-xjx/QAPs/qaplibs/{instance}.dat"
    solution_file = f"/home/xjx/A-xjx/QAPs/qaplibs/{instance}.sln"
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default="nug12")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--solver', type=str, default="ipfp", choices=["rrwm", "sm", "ipfp", "ngm", "astar"])
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    solver = None
    if args.solver == "rrwm":
        solver = pygm.rrwm
    elif args.solver == "sm":
        solver = pygm.sm
    elif args.solver == "ipfp":
        solver = pygm.ipfp
    elif args.solver == "ngm":
        solver = pygm.ngm
    elif args.solver == "astar":
        solver = pygm.astar        
    
        
    if args.instance.startswith("GI"):
        n, F_np, D_np, obj_label, x_label_np = generate_isomorphic(args.instance)
    else:
        n, F_np, D_np, obj_label, x_label_np = read_instance(args.instance)
    
    
    try:
        # normalize F and D
        # F = F_np / np.linalg.norm(F_np, ord='fro')
        # D = D_np / np.linalg.norm(D_np, ord='fro')
        
        start_time = time.time()
        K = torch.tensor(np.kron(D_np, F_np), dtype=torch.float32)
        
        n1 = n2 = torch.tensor([n])
        X = solver(-K, n1, n2)
        X = pygm.hungarian(X)
        
        solve_time = time.time() - start_time
        
        X_np = X.detach().numpy()
        obj = np.sum(F_np * (X_np @ D_np @ X_np.T))
    except Exception as e:
        print(f"Solver {args.solver} failed on instance {args.instance} with error: {e}")
        solve_time = time.time() - start_time
        obj = -1
    print(obj)
    
    instance_name = args.instance.split('/')[-1]
    with open(f"/home/xjx/A-xjx/QAPs/results/pygm/{args.solver}.txt", "a") as f:
        f.write(f"{instance_name} {solve_time:.2f} {obj}\n")