import os

# ==========================================
# 1. 路径与真值配置
# ==========================================
# 请确保你的 .dat 文件都在这个目录下
DATA_DIR = "/home/szy/QAPszy/qaplibs/qapdata/"
BKS_FILE = "/home/szy/QAPszy/experiment1/qaplib_labels.txt"

def load_bks(filepath):
    """读取 BKS 真值文件，返回字典 {instance_name: best_obj}"""
    bks_dict = {}
    if not os.path.exists(filepath):
        print(f"[Config] Warning: {filepath} not found. Gap will be 0.")
        return {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                name = parts[0]
                val = float(parts[1])
                bks_dict[name] = val
            except ValueError:
                continue
    return bks_dict

# 加载 BKS
BKS_DICT = load_bks(BKS_FILE)


TARGET_INSTANCES = [
    'nug12', 'had12',
    'nug20', 'nug30', 'kra30a', 'tho30',
    'tai40b', 'tai50b', 'tai60b', 'tai80b', 'tho150'
]

# 随机种子 (跑5次取平均)
SEEDS = [0, 1, 2, 3, 4]

# ==========================================
# 3. 算法参数与扩展性策略
# ==========================================

# Baseline PGD 参数
PGD_PARAMS = {
    'gamma': 0.05, 
    'beta': 0.05
}

# Ours Sinkhorn 参数
SINKHORN_PARAMS = {
    'lr': 0.02, 
    'dual_init': 1.0
}

def get_adaptive_settings(n):
    """
    根据问题规模 N 动态调整 Batch Size 和 Iterations。
    这是展示 Scalability 的关键：大图不能用太大的 Batch (显存限制)，但需要更多 Iter。
    """
    if n < 30:
        return {'batch_size': 2000, 'max_iter': 10}
    elif n < 60:
        return {'batch_size': 1000, 'max_iter': 15}
    elif n < 80:
        return {'batch_size': 500,  'max_iter': 20}
    else: # >= 80 (如 tai80b, tai100b)
        return {'batch_size': 200,  'max_iter': 25}