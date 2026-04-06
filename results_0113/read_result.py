import numpy as np
import pandas as pd
import os

result_dir = "results/"
result_file = "result.txt"
gurobi_file = "result_gurobi.txt"

def read_results(filename):
    file_path = os.path.join(result_dir, filename)
    data = pd.read_csv(file_path, sep=" ", header=None, skiprows=2)
    data.columns = ["Instance", "time", "incumbent", "bks", "gap"]
    
    # Convert columns to appropriate data types
    data["Instance"] = data["Instance"].astype(str)
    data["time"] = data["time"].astype(float)
    data["incumbent"] = data["incumbent"].astype(int)
    data["bks"] = data["bks"].astype(int)
    data["gap"] = data["gap"].astype(float)
    data["gap"] = data["gap"] * 100
    data["gap"] = data["gap"].round(2)
    # data["gap"] = data["gap"].astype(str) + "\%"
    return data

def read_gurobi_results(filename):
    file_path = os.path.join(result_dir, filename)
    data = pd.read_csv(file_path, sep=" ", header=None, skiprows=2)
    data.columns = ["Instance", "time", "incumbent", "bks"]
    
    # Convert columns to appropriate data types
    data["Instance"] = data["Instance"].astype(str)
    data["time"] = data["time"].astype(float)
    data["incumbent"] = data["incumbent"].astype(int)
    data["bks"] = data["bks"].astype(int)
    data["gap"] = ((data["incumbent"] - data["bks"]) / data["bks"]).astype(float)
    data["gap"] = data["gap"] * 100
    data["gap"] = data["gap"].round(2)
    # data["gap"] = data["gap"].astype(str) + "\%"
    return data

# 3. 定义高亮逻辑 (通用函数，支持传入样式模板)
def apply_style_to_best(df_source, df_target, cols, style_template):
    """
    df_source: 原始数值 DataFrame (用于比较大小)
    df_target: 字符串 DataFrame (用于修改输出内容)
    cols: 需要比较的列组
    style_template: 样式模板，例如 "\\textbf{{{}}}" 或 "\\underline{{{}}}"
    """
    for idx, row in df_source.iterrows():
        vals = row[cols]
        
        # 将 0 替换为无穷大，避免 0 (失败) 被选为最小值
        # vals_clean = pd.to_numeric(vals, errors='coerce').replace(0, np.inf)
        vals_clean = vals        
        # 找到最小值
        min_val = vals_clean.min()
        
        # 如果全失败 (inf)，跳过
        if min_val == np.inf:
            continue

        # 找到最优列
        best_cols = vals_clean[vals_clean == min_val].index.tolist()
        
        # 应用样式
        for col in best_cols:
            current_str = df_target.at[idx, col]
            # 使用 format 将当前字符串填入模板中
            df_target.at[idx, col] = style_template.format(current_str)



data = read_results(result_file)
gurobi_data = read_gurobi_results(gurobi_file)

# data["gurobi_time"] = gurobi_data["time"]
# data["gurobi_incumbent"] = gurobi_data["incumbent"]
# data["gurobi_gap"] = gurobi_data["gap"]

# merge gurobi_data into data on "Instance"
data = pd.merge(data, gurobi_data[["Instance", "time", "incumbent", "gap"]], on="Instance", suffixes=("", "-gurobi"))

data_str = data.copy()
# 转换为字符串格式，便于后续处理
data_str["time"] = data_str["time"].map("{:.2f}".format)
data_str["incumbent"] = data_str["incumbent"].map("{}".format)
data_str["gap"] = data_str["gap"].map("{:.2f}\\%".format)
data_str["time-gurobi"] = data_str["time-gurobi"].map("{:.2f}".format)
data_str["incumbent-gurobi"] = data_str["incumbent-gurobi"].map("{}".format)
data_str["gap-gurobi"] = data_str["gap-gurobi"].map("{:.2f}\\%".format)
# 应用加粗样式到incumnent列
apply_style_to_best(data, data_str, ["incumbent", "incumbent-gurobi"], "\\textbf{{{}}}")
# 应用加粗样式到gap列
apply_style_to_best(data, data_str, ["gap", "gap-gurobi"], "\\textbf{{{}}}")

data_str.set_index("Instance", inplace=True)




latex_code = data_str.to_latex(
    # index=False,
    escape=False,          # 禁用转义，确保 latex 命令生效
    # column_format='ll' + 'r'*5,
    # caption="Experiment Results (Bold: Best Obj, Underline: Best Time)",
    label="tab:results"
)

# 增加 booktabs 风格
latex_code = latex_code.replace('\\begin{tabular}', '\\begin{tabular}')
latex_code = latex_code.replace('\\hline', '\\midrule')


# write to latex table
latex_file = os.path.join(result_dir, "result_table.tex")
with open(latex_file, "w") as f:
    f.write(latex_code)