import os
import pandas as pd
import torch
import numpy as np
import time
from datetime import datetime

from solver.common import read_instance 
from solver.common import checker
from solver.common import analyze_history
from solver.common import get_spectral_init
from solver.pgd import PGDSolver
from solver.sinkhorn import SinkhornSolver
import config

def main():
    # 1. 环境初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = './result/'
    output_summary_csv = os.path.join(save_dir, f"benchmark_result_time_analysis_{timestamp}.csv")
    output_trace_csv = os.path.join(save_dir, f"result_trace_{timestamp}.csv")
    # output_summary_csv = f"benchmark_result_time_analysis_{timestamp}.csv"
    # output_trace_csv = f"result_trace_{timestamp}.csv"
    
    print(f"=== QAP Time-to-Converge Benchmark ===")
    print(f"Device: {device}")
    print(f"Output: {output_summary_csv}\n")
    
    summary_data = []
    trace_data = []
    
    # 2. 遍历算例
    for inst_name in config.TARGET_INSTANCES:
        bks = config.BKS_DICT.get(inst_name, None)
        if bks is None:
            print(f"Skipping {inst_name}: No BKS.")
            continue
            
        print(f"--- Processing {inst_name} (BKS={int(bks)}) ---")
        
        # 读取数据
        try:
            n, F_np, D_np, _, _ = read_instance(inst_name)
        except Exception as e:
            print(f"Error loading {inst_name}: {e}")
            continue

        settings = config.get_adaptive_settings(n)
        bs = settings['batch_size']
        iters = settings['max_iter']
        print(f"    Size N={n} | Batch={bs} | Iters={iters}")

        # 3. 遍历种子
        for seed in config.SEEDS:
            
            init_X = get_spectral_init(F_np, D_np, n, bs, device)
            
            def run_solver(SolverClass, name, params, start_X):
                try:
                    torch.cuda.empty_cache()
                    solver = SolverClass(F_np, D_np, device=device, seed=seed)
                    
                    # 运行求解 (注意：现在接收 4 个返回值)
                    # 这里的 _ 是 total_time, 我们稍后用 history 里的数据更准
                    obj, _, sol, history = solver.solve(
                        batch_size=bs, 
                        max_iter=iters, 
                        params=params, 
                        X_init=start_X, 
                        log_interval=20)
                    
                    # 基础指标
                    gap = (obj - bks) / bks * 100
                    feasible = checker(sol)
                    total_time = history[-1]['time'] if history else 0
                    
                    # 高级指标：收敛时间分析
                    time_metrics, t_best = analyze_history(history, bks)
                    # 如果没返回 history (异常情况)
                    if t_best is None: t_best = total_time

                    summary_row = {
                        "Instance": inst_name, "Size": n, "Seed": seed, "BKS": bks,
                        f"{name}_Obj": obj,
                        f"{name}_Gap": gap,
                        f"{name}_Time": total_time,
                        f"{name}_Feasible": feasible
                    }
                    # 解包 T_to_X%
                    for k, v in time_metrics.items(): 
                        summary_row[f"{name}_{k}"] = v
                    
                    # B. 准备 Trace 数据 (Long Format)
                    # 格式: Instance, Seed, Method, Iter, Time, Gap
                    for h in history:
                        trace_row = {
                            "Instance": inst_name,
                            "Seed": seed,
                            "Method": name, # PGD 或 SK
                            "Iter": h['iter'],
                            "Time": h['time'],
                            "Obj": h['obj'],
                            "Gap": (h['obj'] - bks) / bks * 100 if bks!=0 else 0
                        }
                        trace_data.append(trace_row)
                        
                    return summary_row

                except Exception as e:
                    print(f"    [{name} Error] Seed {seed}: {e}")
                    return {f"{name}_Gap": float('inf'), f"{name}_TotalTime": 0}

            # --- 运行 PGD ---
            pgd_data = run_solver(PGDSolver, "PGD", config.PGD_PARAMS, start_X=init_X)
            
            # --- 运行 Sinkhorn ---
            sk_data = run_solver(SinkhornSolver, "SK", config.SINKHORN_PARAMS, start_X=init_X)
            
            if pgd_data and sk_data:
                merged_summary = {**pgd_data, **sk_data} # 合并字典
                # 修正重复键 (Instance, Seed 等)
                summary_data.append(merged_summary)
                print(f"    Seed {seed}: PGD={merged_summary.get('PGD_Gap'):.2f}% | SK={merged_summary.get('SK_Gap'):.2f}%")

            # ==========================
            # 实时保存 (防止程序中途崩溃)
            # ==========================
            # 保存 Summary
            pd.DataFrame(summary_data).to_csv(output_summary_csv, index=False)
            # 保存 Trace
            pd.DataFrame(trace_data).to_csv(output_trace_csv, index=False)
           
if __name__ == "__main__":
    main()

