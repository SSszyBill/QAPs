import argparse
import torch
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parse import parse_qap_dat, parse_qap_solution
from sk_solver import SK_QAP_Solver

# === 配置类 ===
class Config:
    data_dir = 'qaplibs/qapdata'
    soln_dir = 'qaplibs/qapsoln'
    base_result_dir = '/home/szy/QAP/results'
    output_dir = os.path.join(base_result_dir, 'logs')    
    plot_dir = os.path.join(base_result_dir, 'plots')     
    
    rho = 100
    limit = 50  
    iter = 5000           
    lr_dual = 0.05       

def check_constraints_gpu(perm_matrix):
    """
    [GPU 加速版] 检查排列矩阵是否满足约束
    输入: perm_matrix (torch.Tensor, likely on GPU)
    输出: is_valid (bool), max_error (float)
    """
    # 确保是 Tensor
    if not isinstance(perm_matrix, torch.Tensor):
        P = torch.tensor(perm_matrix)
    else:
        P = perm_matrix # 保持在原设备(GPU)上
    
    # 1. 二值约束: 元素应该接近 0 或 1
    # 计算 min(|P|, |P-1|)
    dist_to_binary = torch.min(torch.abs(P), torch.abs(P - 1.0))
    bin_error = torch.max(dist_to_binary).item() # .item() 会同步传回 CPU
    
    # 2. 行和约束: sum(row) == 1
    row_sum = torch.sum(P, dim=1)
    row_error = torch.max(torch.abs(row_sum - 1.0)).item()
    
    # 3. 列和约束: sum(col) == 1
    col_sum = torch.sum(P, dim=0)
    col_error = torch.max(torch.abs(col_sum - 1.0)).item()

    max_error = max(bin_error, row_error, col_error)
    is_valid = max_error < 1e-4
    
    return is_valid, max_error

def ensure_dirs(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

def plot_convergence(soft_history, real_history, instance_name, save_dir):
    """
    绘图函数必须在 CPU 上运行 (Matplotlib 不支持 GPU Tensor)
    """
    plt.figure(figsize=(10, 6))
    
    # 1. 画 Soft Cost
    plt.plot(soft_history, label='Soft Cost (Relaxed)', color='blue', alpha=0.6, linewidth=1)
    
    # 2. 画 Real Cost
    if real_history:
        # 解压
        iters, vals = zip(*real_history)
        plt.plot(iters, vals, label='Real Cost (Hungarian)', color='red', marker='o', markersize=3, linestyle='--', linewidth=0.8)
    
    plt.title(f"Convergence: {instance_name} (Soft vs Real)")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    save_path = os.path.join(save_dir, f"{instance_name}.png")
    plt.savefig(save_path)
    plt.close()

def run_single_instance(dat_path, soln_dir, args, device):
    filename = os.path.basename(dat_path)
    instance_name = os.path.splitext(filename)[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    
    # 1. 读取数据 (IO 仍在 CPU)
    try:
        n, A, B = parse_qap_dat(dat_path)
    except Exception as e:
        print(f"[Error] Failed to parse data: {e}")
        return None

    # 读取已知最优解
    sin_path = os.path.join(soln_dir, f"{instance_name}.sln")
    opt_val, opt_perm_vec = parse_qap_solution(sin_path)

    log_file_path = os.path.join(args.output_dir, f"{instance_name}_log.txt")
    print(f"[System] Logging details to: {log_file_path}")
    
    # 2. 初始化 Solver (数据自动上 GPU)
    solver = SK_QAP_Solver(n, A, B, device=device)
    start_t = time.time()
    
    # 3. 求解 (全 GPU 计算)
    best_cost, best_perm_matrix, soft_hist, real_hist = solver.solve(
        max_iter=args.iter, 
        lr_dual=args.lr_dual, 
        log_interval=100,      
        log_path=log_file_path 
    )
    end_t = time.time()
    elapsed = end_t - start_t

    # 4. 约束检查 (GPU 加速版)
    is_feasible, constr_error = check_constraints_gpu(best_perm_matrix)
    if not is_feasible:
        print(f"[WARNING] Solution violates constraints! Max Error: {constr_error:.6f}")
    
    # 5. 绘图 (数据需传回 CPU)
    if soft_hist:
        plot_convergence(soft_hist, real_hist, instance_name, args.plot_dir)
        print(f"-> Plot saved to {args.plot_dir}/{instance_name}.png")

    # 6. 结果统计与比对
    gap = 0.0
    perm_match = "N/A"
    
    # 处理 Cost (best_cost 可能是 tensor 或 float)
    my_cost_val = best_cost if isinstance(best_cost, (float, int)) else best_cost.item()

    if opt_val is not None:
        if abs(opt_val) < 1e-9: 
            if abs(my_cost_val - opt_val) < 1e-4: gap = 0.0
            else: gap = float('inf') 
        else:
            gap = (my_cost_val - opt_val) / abs(opt_val) * 100

        # 排列比对
        if opt_perm_vec is not None and len(opt_perm_vec) == n:
            # [GPU] 使用 torch.argmax 快速得到排列向量
            my_perm_vec_tensor = torch.argmax(best_perm_matrix, dim=1) 
            # 转回 CPU numpy 进行比对 (因为 opt_perm_vec 是 numpy/list)
            my_perm_vec = my_perm_vec_tensor.cpu().numpy()
            
            if np.array_equal(my_perm_vec, opt_perm_vec): perm_match = "YES"
            else: perm_match = "NO"
    
    feas_str = "OK" if is_feasible else "FAIL"
    print(f"-> Found: {my_cost_val:.2f} | Opt: {opt_val if opt_val else 'N/A'} | Gap: {gap:.2f}% | Feas: {feas_str}")
    
    return {
        "Instance": instance_name,
        "Size": n,
        "MyCost": float(my_cost_val),
        "OptCost": float(opt_val) if opt_val else None,
        "Gap(%)": float(gap) if opt_val else None,
        "PermMatch": perm_match,
        "Feasible": is_feasible,
        "MaxVio": float(constr_error),
        "Time(s)": float(elapsed)
    }

def main():
    args = Config()
    ensure_dirs(args)
    
    # 自动检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[System] Using device: {device}")
    print(f"[System] Results will be saved to: {args.base_result_dir}")

    if not os.path.exists(args.data_dir):
        print(f"[Error] Data directory '{args.data_dir}' not found.")
        return

    all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.dat')]
    all_files.sort()
    
    # target_filename = 'nug12.dat'  # 要筛选的文件名
    # if target_filename in all_files:
    #     target_files = [target_filename]  # 只保留这个文件
    
    target_files = all_files[6:20] 
    
    results = []

    for idx, filename in enumerate(target_files):
        dat_path = os.path.join(args.data_dir, filename)
        res = run_single_instance(dat_path, args.soln_dir, args, device)
        if res:
            results.append(res)

    print("\n" + "="*85)
    print("FINAL BENCHMARK SUMMARY")
    print("="*85)
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        if 'Gap(%)' in df.columns and df['Gap(%)'].notna().any():
            avg_gap = df['Gap(%)'].replace([np.inf, -np.inf], np.nan).dropna().abs().mean()
            print(f"Average Optimality Gap: {avg_gap :.2f}%")
            print("-" * 85)

        cols = ['Instance', 'Size', 'MyCost', 'OptCost', 'Gap(%)', 'MaxVio', 'PermMatch', 'Time(s)']
        final_cols = [c for c in cols if c in df.columns]
        
        # Pandas 展示需要 CPU 数据，我们在 run_single_instance 中已经转好 float 了
        print(df[final_cols].to_string(index=False, float_format="%.2f"))

        # excel_path = os.path.join(args.output_dir, 'SK_benchmark_summary.xlsx')
        # try:
        #     df.to_excel(excel_path, index=False, sheet_name='Benchmark')
        #     print(f"\n[Success] Excel report saved to: {excel_path}")
        # except Exception as e:
        #     print(f"\n[Error] Could not save Excel file: {e}")
    else:
        print("No results to show.")

if __name__ == "__main__":
    main()

# === 配置类 ===
# class Config:
#     data_dir = 'qaplibs/qapdata'
#     soln_dir = 'qaplibs/qapsoln'
#     base_result_dir = '/home/szy/QAP/results'
#     output_dir = os.path.join(base_result_dir, 'logs')   
#     plot_dir = os.path.join(base_result_dir, 'plots')    
    
#     rho = 100
#     limit = 50  
#     iter = 3000          
#     lr_dual = 0.05       

# def check_constraints(perm_matrix):
#     """检查排列矩阵是否满足约束"""
#     if isinstance(perm_matrix, torch.Tensor):
#         P = perm_matrix.cpu().detach().numpy()
#     else:
#         P = perm_matrix
    
#     # 1. 二值约束
#     dist_to_binary = np.minimum(np.abs(P), np.abs(P - 1))
#     bin_error = np.max(dist_to_binary)
    
#     # 2. 行和约束
#     row_sum = np.sum(P, axis=1)
#     row_error = np.max(np.abs(row_sum - 1))
    
#     # 3. 列和约束
#     col_sum = np.sum(P, axis=0)
#     col_error = np.max(np.abs(col_sum - 1))

#     max_error = max(bin_error, row_error, col_error)
#     is_valid = max_error < 1e-4
    
#     return is_valid, max_error

# def ensure_dirs(args):
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     if not os.path.exists(args.plot_dir):
#         os.makedirs(args.plot_dir)

# def plot_convergence(soft_history, real_history, instance_name, save_dir):
#     """
#     绘制 Soft Cost 和 Real Cost 对比图
#     soft_history: list of float (每一步)
#     real_history: list of (iter, val) (每 log_interval 步)
#     """
#     plt.figure(figsize=(10, 6))
    
#     # 1. 画 Soft Cost (连续曲线)
#     plt.plot(soft_history, label='Soft Cost (Relaxed)', color='blue', alpha=0.6, linewidth=1)
    
#     # 2. 画 Real Cost (离散点)
#     if real_history:
#         # 解压 (iteration, value)
#         iters, vals = zip(*real_history)
#         plt.plot(iters, vals, label='Real Cost (Hungarian)', color='red', marker='o', markersize=3, linestyle='--', linewidth=0.8)
    
#     plt.title(f"Convergence: {instance_name} (Soft vs Real)")
#     plt.xlabel("Iteration")
#     plt.ylabel("Cost")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend()
    
#     save_path = os.path.join(save_dir, f"{instance_name}.png")
#     plt.savefig(save_path)
#     plt.close()

# def run_single_instance(dat_path, soln_dir, args, device):
#     filename = os.path.basename(dat_path)
#     instance_name = os.path.splitext(filename)[0]
    
#     print(f"\n{'='*60}")
#     print(f"Processing: {filename}")
    
#     try:
#         n, A, B = parse_qap_dat(dat_path)
#     except Exception as e:
#         print(f"[Error] Failed to parse data: {e}")
#         return None

#     sin_path = os.path.join(soln_dir, f"{instance_name}.sln")
#     opt_val, opt_perm_vec = parse_qap_solution(sin_path)

#     log_file_path = os.path.join(args.output_dir, f"{instance_name}_log.txt")
#     print(f"[System] Logging details to: {log_file_path}")
#     solver = SK_QAP_Solver(n, A, B, device=device)
#     start_t = time.time()
    
#     # === 核心修改：接收 4 个返回值 ===
#     best_cost, best_perm_matrix, soft_hist, real_hist = solver.solve(
#         max_iter=args.iter, 
#         lr_dual=args.lr_dual, 
#         log_interval=100,      
#         log_path=log_file_path 
#     )
#     end_t = time.time()
#     elapsed = end_t - start_t

#     is_feasible, constr_error = check_constraints(best_perm_matrix)
#     if not is_feasible:
#         print(f"[WARNING] Solution violates constraints! Max Error: {constr_error:.6f}")
    
#     # === 修改：传入两个 history 进行绘图 ===
#     if soft_hist:
#         plot_convergence(soft_hist, real_hist, instance_name, args.plot_dir)
#         print(f"-> Plot saved to {args.plot_dir}/{instance_name}.png")

#     # 结果统计
#     gap = 0.0
#     is_optimal_cost = False
#     perm_match = "N/A"
    
#     if opt_val is not None:
#         if abs(opt_val) < 1e-9: 
#             if abs(best_cost - opt_val) < 1e-4: gap = 0.0
#             else: gap = float('inf') 
#         else:
#             gap = (best_cost - opt_val) / abs(opt_val) * 100

#         if abs(best_cost - opt_val) < 1e-4: is_optimal_cost = True
            
#         if opt_perm_vec is not None and len(opt_perm_vec) == n:
#             my_perm_mat = best_perm_matrix.cpu().numpy()
#             my_perm_vec = np.argmax(my_perm_mat, axis=1)
#             if np.array_equal(my_perm_vec, opt_perm_vec): perm_match = "YES"
#             else: perm_match = "NO"
    
#     feas_str = "OK" if is_feasible else "FAIL"
#     print(f"-> Found: {best_cost:.2f} | Opt: {opt_val if opt_val else 'N/A'} | Feas: {feas_str}")
    
#     return {
#         "Instance": instance_name,
#         "Size": n,
#         "MyCost": float(best_cost),
#         "OptCost": float(opt_val) if opt_val else None,
#         "Gap(%)": float(gap) if opt_val else None,
#         "PermMatch": perm_match,
#         "Feasible": is_feasible,
#         "MaxVio": float(constr_error),
#         "Time(s)": float(elapsed)
#     }

# def main():
#     args = Config()
#     ensure_dirs(args)
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"[System] Using device: {device}")
#     print(f"[System] Results will be saved to: {args.base_result_dir}")

#     if not os.path.exists(args.data_dir):
#         print(f"[Error] Data directory '{args.data_dir}' not found.")
#         return

#     all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.dat')]
#     all_files.sort()
#     target_files = all_files[:1]
    
#     results = []

#     for idx, filename in enumerate(target_files):
#         dat_path = os.path.join(args.data_dir, filename)
#         res = run_single_instance(dat_path, args.soln_dir, args, device)
#         if res:
#             results.append(res)

#     print("\n" + "="*85)
#     print("FINAL BENCHMARK SUMMARY")
#     print("="*85)
    
#     df = pd.DataFrame(results)
    
#     if not df.empty:
#         if 'Gap(%)' in df.columns and df['Gap(%)'].notna().any():
#             avg_gap = df['Gap(%)'].replace([np.inf, -np.inf], np.nan).dropna().abs().mean()
#             print(f"Average Optimality Gap: {avg_gap :.2f}%")
#             print("-" * 85)

#         cols = ['Instance', 'Size', 'MyCost', 'OptCost', 'Gap(%)', 'MaxVio', 'PermMatch', 'Time(s)']
#         final_cols = [c for c in cols if c in df.columns]
        
#         print(df[final_cols].to_string(index=False, float_format="%.2f"))

#         excel_path = os.path.join(args.output_dir, 'SK_benchmark_summary.xlsx')
#         try:
#             df.to_excel(excel_path, index=False, sheet_name='Benchmark')
#             print(f"\n[Success] Excel report saved to: {excel_path}")
#         except Exception as e:
#             print(f"\n[Error] Could not save Excel file: {e}")
#     else:
#         print("No results to show.")

# if __name__ == "__main__":
#     main()