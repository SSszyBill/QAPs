# data_parser.py
import numpy as np
import os

def parse_qap_dat(file_path):
    """
    解析 QAPLIB 标准 .dat 文件
    
    Args:
        file_path (str): .dat 文件路径
        
    Returns:
        n (int): 问题规模
        A (np.array): 流矩阵 (Flow)
        B (np.array): 距离矩阵 (Distance)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[Parser] Reading {os.path.basename(file_path)}...")
    
    with open(file_path, 'r') as f:
        # split() 会自动处理所有的空格、换行符，非常健壮
        tokens = f.read().split()
    
    if not tokens:
        raise ValueError("File is empty")

    iterator = iter(tokens)
    
    try:
        # 1. 读取维度 n
        n = int(next(iterator))
        
        # 2. 读取矩阵数据的辅助函数
        def read_matrix_np(size):
            data = []
            for _ in range(size * size):
                data.append(float(next(iterator)))
            return np.array(data).reshape(size, size)
            
        # 3. 提取核心矩阵
        A = read_matrix_np(n) # Flow Matrix
        B = read_matrix_np(n) # Distance Matrix
        
        # 忽略后续可能存在的 C 矩阵或 EOF
        
    except StopIteration:
        raise ValueError("File format incorrect (unexpected end of file)")

    print(f"[Parser] Done. Size n={n}")
    return n, A, B



# data_parser.py (追加内容)
def parse_qap_solution(file_path):
    """
    解析 QAPLIB 解文件 (.sin/.sln)
    格式通常为:
    n optimal_value
    p1 p2 ... pn (排列数组)
    """
    if not os.path.exists(file_path):
        return None, None

    with open(file_path, 'r') as f:
        tokens = f.read().split()
    
    if not tokens:
        return None, None

    try:
        iterator = iter(tokens)
        
        # 1. 读取维度 n 和 最优值
        n = int(next(iterator))
        opt_val = float(next(iterator))
        
        # 2. 读取排列 (Permutation Vector)
        # QAPLIB 通常使用 1-based indexing (1..n)，我们需要转为 0-based (0..n-1)
        opt_perm = []
        try:
            for _ in range(n):
                val = int(next(iterator))
                opt_perm.append(val - 1) # 转为 0-based
        except StopIteration:
            pass # 有些文件可能只给了 optimal value 没有给 permutation
            
        return opt_val, np.array(opt_perm)

    except Exception as e:
        print(f"[Parser Warning] Could not parse solution file {file_path}: {e}")
        return None, None