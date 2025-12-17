import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import os
import pandas as pd  # 新增：用于处理表格数据
from parse import parse_qap_dat, parse_qap_solution

class GurobiQAPSolver:
    def __init__(self, n, A, B, time_limit=300, verbose=False):
        """
        Args:
            verbose: 是否打印 Gurobi 内部详细日志
        """
        self.n = n
        self.A = A
        self.B = B
        self.time_limit = time_limit
        self.verbose = verbose
        self.model = None
        self.x = None

    def build_model(self):
        self.model = gp.Model("QAP")
        
        # === 参数设置 ===
        self.model.Params.TimeLimit = self.time_limit
        self.model.Params.OutputFlag = 1 if self.verbose else 0  # 批量运行时通常关闭详细日志
        self.model.Params.NonConvex = 2   
        self.model.Params.MIPGap = 0.0    
        self.model.Params.Threads = 8     

        # === 变量与约束 ===
        self.x = self.model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")

        for i in range(self.n):
            self.model.addConstr(self.x.sum(i, '*') == 1)

        for j in range(self.n):
            self.model.addConstr(self.x.sum('*', j) == 1)

        # === 目标函数 ===
        obj_expr = gp.QuadExpr()
        # 利用稀疏性加速构建
        for i in range(self.n):
            for k in range(self.n):
                if self.A[i, k] == 0: continue
                for j in range(self.n):
                    for l in range(self.n):
                        if self.B[j, l] == 0: continue
                        coeff = self.A[i, k] * self.B[j, l]
                        obj_expr.add(self.x[i, j] * self.x[k, l], coeff)

        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def solve(self):
        if self.model is None:
            self.build_model()
        
        start_time = time.time()
        self.model.optimize()
        end_time = time.time()
        
        status = self.model.Status
        solve_time = end_time - start_time

        result = {
            "status": status,
            "obj_val": None,
            "perm": None,
            "mip_gap": None, # Gurobi 自身的上下界 Gap
            "time": solve_time
        }

        if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT:
            if self.model.SolCount > 0:
                result["obj_val"] = self.model.ObjVal
                result["mip_gap"] = self.model.MIPGap
                
                # 提取排列
                perm = np.zeros(self.n, dtype=int)
                for i in range(self.n):
                    for j in range(self.n):
                        if self.x[i, j].X > 0.5:
                            perm[i] = j
                            break
                result["perm"] = perm
        
        return result

def main():
    # === 配置路径 ===
    data_dir = "/home/opt/szy/QAP/QAPs/qaplibs/qapdata"
    soln_dir = "/home/opt/szy/QAP/QAPs/qaplibs/qapsoln"
    output_file = "Gurobi_QAP_Benchmark.xlsx"
    
    # === 设置 ===
    # 你可以在这里限制只跑前几个文件，例如 all_files[:5]
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    all_files.sort()
    
    # 示例：只跑文件名包含 'tai' 的前 3 个小规模算例用于测试
    # target_files = [f for f in all_files if 'tai' in f][:3]
    # 如果要跑所有文件：
    target_filename = 'nug12.dat'  # 要筛选的文件名
    if target_filename in all_files:
        target_files = [target_filename]  # 只保留这个文件
    results_list = []

    print(f"Found {len(target_files)} instances. Starting benchmark...")
    print("-" * 80)
    print(f"{'Instance':<15} | {'Size':<5} | {'Gurobi Cost':<12} | {'Opt Cost':<12} | {'Gap(%)':<10} | {'Time(s)':<8}")
    print("-" * 80)

    for filename in target_files:
        instance_name = os.path.splitext(filename)[0]
        dat_path = os.path.join(data_dir, filename)
        soln_path = os.path.join(soln_dir, f"{instance_name}.sln")

        # 1. 解析数据
        try:
            n, A, B = parse_qap_dat(dat_path)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue
        
        # 跳过过大的算例 (可选)
        if n > 20: 
            print(f"{instance_name:<15} | {n:<5} | {'SKIPPED (Too Large)':<40}")
            continue

        # 2. 获取已知最优解
        opt_val_known, opt_perm_known = parse_qap_solution(soln_path)

        # 3. Gurobi 求解
        solver = GurobiQAPSolver(n, A, B, time_limit=600, verbose=False)
        res = solver.solve()

        # 4. 计算指标
        my_cost = res['obj_val']
        gap_percent = None
        perm_match = False
        
        if my_cost is not None and opt_val_known is not None:
             if abs(opt_val_known) > 1e-9:
                 gap_percent = (my_cost - opt_val_known) / abs(opt_val_known) * 100
             else:
                 gap_percent = 0.0 if abs(my_cost) < 1e-9 else float('inf')
        
        if res['perm'] is not None and opt_perm_known is not None:
            if len(res['perm']) == len(opt_perm_known):
                perm_match = np.array_equal(res['perm'], opt_perm_known)

        # 5. 打印单行日志
        cost_str = f"{my_cost:.2f}" if my_cost else "N/A"
        opt_str = f"{opt_val_known:.2f}" if opt_val_known else "N/A"
        gap_str = f"{gap_percent:.2f}%" if gap_percent is not None else "N/A"
        
        print(f"{instance_name:<15} | {n:<5} | {cost_str:<12} | {opt_str:<12} | {gap_str:<10} | {res['time']:.2f}")

        # 6. 收集结果到列表
        results_list.append({
            "Instance": instance_name,
            "Size": n,
            "Gurobi_Obj": my_cost,
            "Known_Opt": opt_val_known,
            "Gap_to_Opt(%)": gap_percent,
            "Gurobi_MIP_Gap": res['mip_gap'],
            "Solve_Time(s)": res['time'],
            "Permutation_Found": str(res['perm'].tolist()) if res['perm'] is not None else None,
            "Permutation_Match": perm_match,
            "Status_Code": res['status']
        })

    # === 导出到 Excel ===
    if results_list:
        df = pd.DataFrame(results_list)
        
        # 调整列顺序，好看一点
        cols = ["Instance", "Size", "Gurobi_Obj", "Known_Opt", "Solve_Time(s)"]
        df = df[cols]
        
        try:
            df.to_excel(output_file, index=False, sheet_name="Gurobi_Results")
            print("-" * 80)
            print(f"Successfully saved results to: {os.path.abspath(output_file)}")
            
            # 计算平均 Gap
            avg_gap = df["Gap_to_Opt(%)"].mean()
            print(f"Average Gap: {avg_gap:.2f}%")
            
        except Exception as e:
            print(f"Error saving Excel file: {e}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()