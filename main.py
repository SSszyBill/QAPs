import argparse
import torch
import os
import time
import numpy as np
import pandas as pd              # 新增：用于处理表格和导出Excel
import matplotlib.pyplot as plt  # 新增：用于画图
from parse import parse_qap_dat, parse_qap_solution
from sk_solver import SK_QAP_Solver
from ALM_solver import *

# === 配置类 ===
class Config:
    data_dir = 'qaplibs/qapdata'
    soln_dir = 'qaplibs/qapsoln'
    output_dir = 'results'       # 新增：结果输出总目录
    plot_dir = 'results/plots_ALM'   # 新增：折线图存放目录
    rho = 100
    limit = 20           
    batch_size = 200     
    iter = 3000          
    lr_dual = 0.05       
    tau = 2   

def check_constraints(perm_matrix):
    """
    检查排列矩阵是否满足约束
    返回: (是否通过, 最大违背误差)
    """
    # 转为 numpy 或保持 tensor
    if isinstance(perm_matrix, torch.Tensor):
        P = perm_matrix.cpu().detach().numpy()
    else:
        P = perm_matrix

    n = P.shape[0]
    
    # 1. 检查二值约束 (是否只有 0 和 1)
    # 计算元素距离 0 或 1 的最近距离
    dist_to_binary = np.minimum(np.abs(P), np.abs(P - 1))
    bin_error = np.max(dist_to_binary)
    
    # 2. 检查行和约束 (是否为 1)
    row_sum = np.sum(P, axis=1)
    row_error = np.max(np.abs(row_sum - 1))
    
    # 3. 检查列和约束 (是否为 1)
    col_sum = np.sum(P, axis=0)
    col_error = np.max(np.abs(col_sum - 1))
    
    # 总误差
    max_error = max(bin_error, row_error, col_error)
    
    # 允许 1e-4 的数值误差
    is_valid = max_error < 1e-4
    
    return is_valid, max_error

def ensure_dirs(args):
    """确保输出目录存在"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

def plot_convergence(cost_history, instance_name, save_dir):
    """绘制并保存收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label='Best Cost per Iteration')
    plt.title(f"Convergence Curve: {instance_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图片
    save_path = os.path.join(save_dir, f"{instance_name}.png")
    plt.savefig(save_path)
    plt.close() # 关闭画布，防止内存泄漏

def run_single_instance(dat_path, soln_dir, args, device):
    filename = os.path.basename(dat_path)
    instance_name = os.path.splitext(filename)[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    
    # 1. 解析数据
    try:
        n, A, B = parse_qap_dat(dat_path)
    except Exception as e:
        print(f"[Error] Failed to parse data: {e}")
        return None

    # 2. 解析解文件
    sin_path = os.path.join(soln_dir, f"{instance_name}.sln")
    opt_val, opt_perm_vec = parse_qap_solution(sin_path)

    # 3. 运行求解
    # solver = SK_QAP_Solver(n, A, B, device=device, tau=args.tau, batch_size=args.batch_size)
    solver = ALM_QAP_Solver(n, A, B, device=device, rho=10.0, batch_size=args.batch_size) # 注意参数变化
    start_t = time.time()
    best_cost, best_perm_matrix, cost_history = solver.solve(max_iter=args.iter, lr_dual=args.lr_dual)
    end_t = time.time()
    elapsed = end_t - start_t

    is_feasible, constr_error = check_constraints(best_perm_matrix)
    if not is_feasible:
        print(f"[WARNING] Solution violates constraints! Max Error: {constr_error:.6f}")
    
    # === 新增：绘制折线图 ===
    if cost_history is not None and len(cost_history) > 0:
        plot_convergence(cost_history, instance_name, args.plot_dir)
        print(f"-> Plot saved to {args.plot_dir}/{instance_name}.png")

    # 4. 结果比对逻辑
    gap = 0.0
    is_optimal_cost = False
    perm_match = "N/A"
    
    if opt_val is not None:
        gap = (best_cost - opt_val) / abs(opt_val) * 100
        if abs(best_cost - opt_val) < 1e-4:
            is_optimal_cost = True
        if opt_perm_vec is not None and len(opt_perm_vec) == n:
            my_perm_mat = best_perm_matrix.cpu().numpy()
            my_perm_vec = np.argmax(my_perm_mat, axis=1)
            if np.array_equal(my_perm_vec, opt_perm_vec):
                perm_match = "YES"
            else:
                perm_match = "NO" 
    
    status_str = "OPTIMAL" if is_optimal_cost else f"Gap {gap:.2f}%"
    feas_str = "OK" if is_feasible else "FAIL"
    print(f"-> Found: {best_cost:.2f} | Opt: {opt_val if opt_val else 'N/A'} | Feas: {feas_str}")
    
    return {
        "Instance": instance_name,
        "Size": n,
        "MyCost": float(best_cost),
        "OptCost": float(opt_val) if opt_val else None,
        "Gap(%)": float(gap) if opt_val else None,
        "PermMatch": perm_match,
        "Feasible": is_feasible,      # 新增字段
        "MaxVio": float(constr_error), # 新增字段：最大违背量
        "Time(s)": float(elapsed)
    }

def main():
    args = Config()
    ensure_dirs(args) # 创建文件夹
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[System] Using device: {device}")
    print(f"[System] Results will be saved to: {os.path.abspath(args.output_dir)}")

    if not os.path.exists(args.data_dir):
        print(f"[Error] Data directory '{args.data_dir}' not found.")
        return

    all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.dat')]
    all_files.sort()
    target_files = all_files[:args.limit]
    
    results = []

    for idx, filename in enumerate(target_files):
        dat_path = os.path.join(args.data_dir, filename)
        # 传入 args 以获取 plot_dir
        res = run_single_instance(dat_path, args.soln_dir, args, device)
        if res:
            results.append(res)

    # 5. 输出汇总表格 (终端打印)
    print("\n" + "="*85)
    print("FINAL BENCHMARK SUMMARY")
    print("="*85)
    
    # 使用 Pandas 打印漂亮的表格
    df = pd.DataFrame(results)
    
    # 处理一下显示格式（将 None 替换为 NaN 或其他）
    if not df.empty:
        # 打印到终端
        print(df.to_string(index=False, float_format="%.2f"))
        
        # 计算平均 Gap
        if 'Gap(%)' in df.columns and df['Gap(%)'].notna().any():
            avg_gap = df['Gap(%)'].abs().mean()
            print("-" * 85)
            print(f"Average Optimality Gap: {avg_gap :.2f}%")

        # === 新增：导出到 Excel ===
        excel_path = os.path.join(args.output_dir, 'ALM_benchmark_summary.xlsx')
        try:
            df.to_excel(excel_path, index=False, sheet_name='Benchmark')
            print(f"\n[Success] Excel report saved to: {excel_path}")
        except Exception as e:
            print(f"\n[Error] Could not save Excel file: {e}")
    else:
        print("No results to show.")

    if not df.empty:
        # 调整列顺序，把 Feasible 加上
        cols = ['Instance', 'Size', 'MyCost', 'OptCost', 'Gap(%)', 'Feasible', 'PermMatch', 'Time(s)']
        # 仅保留存在的列
        final_cols = [c for c in cols if c in df.columns]
        print(df[final_cols].to_string(index=False, float_format="%.2f"))

if __name__ == "__main__":
    main()