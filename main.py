# main.py
import argparse
import torch
import os
import time
import numpy as np
from parse import parse_qap_dat, parse_qap_solution  # 记得导入新函数
from solver import QAPSolver

# === 配置类 ===
class Config:
    # 你的文件夹结构:
    # qaplibs/
    #   ├── qapdata/  (放 .dat)
    #   └── qapsoln/  (放 .sin)
    data_dir = 'qaplibs/qapdata'
    soln_dir = 'qaplibs/qapsoln'
    
    limit = 20           # 测试文件数量
    batch_size = 200     # 并行求解数量
    iter = 3000          # 迭代次数
    lr_dual = 0.05       # 学习率
    tau = 2            # Sinkhorn 温度

def run_single_instance(dat_path, soln_dir, args, device):
    filename = os.path.basename(dat_path)
    instance_name = os.path.splitext(filename)[0] # 去掉 .dat 后缀
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    
    # 1. 解析数据
    try:
        n, A, B = parse_qap_dat(dat_path)
    except Exception as e:
        print(f"[Error] Failed to parse data: {e}")
        return None

    # 2. 尝试解析对应的解文件 (.sin)
    # 假设文件名对应关系是: nug12.dat -> nug12.sin
    sin_path = os.path.join(soln_dir, f"{instance_name}.sln")
    opt_val, opt_perm_vec = parse_qap_solution(sin_path)

    # 3. 运行求解
    solver = QAPSolver(n, A, B, device=device, tau=args.tau, batch_size=args.batch_size)
    start_t = time.time()
    best_cost, best_perm_matrix = solver.solve(max_iter=args.iter, lr_dual=args.lr_dual)
    end_t = time.time()
    elapsed = end_t - start_t

    # 4. 结果比对逻辑
    gap = 0.0
    is_optimal_cost = False
    perm_match = "N/A"
    
    if opt_val is not None:
        # (a) 计算 Cost 差距 (Gap %)
        # Gap = (MyCost - OptCost) / OptCost * 100%
        gap = (best_cost - opt_val) / abs(opt_val) * 100
        
        # 判断是否达到最优值 (允许微小浮点误差)
        if abs(best_cost - opt_val) < 1e-4:
            is_optimal_cost = True
            
        # (b) 比较排列 (Permutation)
        if opt_perm_vec is not None and len(opt_perm_vec) == n:
            # best_perm_matrix 是 tensor，转 numpy
            my_perm_mat = best_perm_matrix.cpu().numpy()
            # argmax 得到每行 1 所在的列索引
            my_perm_vec = np.argmax(my_perm_mat, axis=1)
            
            # 比较两个向量是否完全一致
            if np.array_equal(my_perm_vec, opt_perm_vec):
                perm_match = "YES"
            else:
                # 注意：QAP 经常有对称解，排列不同但 Cost 相同也是最优解
                perm_match = "NO" 
    
    # 打印单行结果
    status_str = "OPTIMAL" if is_optimal_cost else f"Gap {gap:.2f}%"
    print(f"-> Found: {best_cost:.2f} | Optimal: {opt_val if opt_val else 'Unknown'}")
    print(f"-> Status: {status_str} | Permutation Match: {perm_match}")

    return {
        "Instance": instance_name,
        "Size": n,
        "MyCost": best_cost,
        "OptCost": opt_val if opt_val else -1,
        "Gap(%)": gap if opt_val else -1,
        "PermMatch": perm_match,
        "Time(s)": elapsed
    }

def main():
    args = Config()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[System] Using device: {device}")

    if not os.path.exists(args.data_dir):
        print(f"[Error] Data directory '{args.data_dir}' not found.")
        return

    # 获取所有 .dat 文件
    all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.dat')]
    all_files.sort()
    target_files = all_files[:args.limit]
    
    results = []

    for idx, filename in enumerate(target_files):
        dat_path = os.path.join(args.data_dir, filename)
        res = run_single_instance(dat_path, args.soln_dir, args, device)
        if res:
            results.append(res)

    # 5. 输出汇总表格
    print("\n" + "="*85)
    print("FINAL BENCHMARK SUMMARY")
    print("="*85)
    header = f"{'Instance':<15} | {'Size':<5} | {'MyCost':<10} | {'OptCost':<10} | {'Gap(%)':<8} | {'SamePerm?':<10} | {'Time(s)':<6}"
    print(header)
    print("-" * 85)
    
    avg_gap = 0
    valid_count = 0
    
    for r in results:
        gap_str = f"{r['Gap(%)']:.2f}" if r['OptCost'] != -1 else "N/A"
        opt_str = f"{r['OptCost']:.0f}" if r['OptCost'] != -1 else "N/A"
        
        print(f"{r['Instance']:<15} | {r['Size']:<5} | {r['MyCost']:<10.2f} | {opt_str:<10} | {gap_str:<8} | {r['PermMatch']:<10} | {r['Time(s)']:<6.2f}")
        
        if r['OptCost'] != -1:
            avg_gap += abs(r['Gap(%)'])
            valid_count += 1
            
    if valid_count > 0:
        print("-" * 85)
        print(f"Average Optimality Gap: {avg_gap / valid_count :.2f}%")

if __name__ == "__main__":
    main()