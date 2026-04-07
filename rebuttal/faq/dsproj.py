"""
dsproj.py — Frank–Wolfe 方向子问题（双随机投影）
对应 MATLAB dsproj.m

在当前点 x 处，求解线性子问题：
  w* = argmin_{w ∈ Birkhoff} g^T w
即对梯度的 Q 分量取负后求最大权匹配（LAP），得到顶点排列矩阵。
FW 方向 d = w* - x。
"""

import numpy as np
from .core import stack, unstack, perm2mat
from .lap import assign


def dsproj(x: np.ndarray, g: np.ndarray, m: int, n: int):
    """Frank–Wolfe 线性子问题，返回下降方向 d 和对应排列 q。

    参数
    ----
    x : 当前点（列向量）
    g : 当前（对称化后的）梯度
    m, n : A 的行数和列数（通常 m == n）

    返回
    ----
    d : FW 方向，d = w - x
    q : 0-indexed 排列向量
    """
    _, gQ = unstack(g, m, n)
    # assign(-gQ) 最大化 sum_i (-gQ)[q[i], i] = 最小化 sum_i gQ[q[i], i]
    # wQ = perm2mat(q).T：wQ[q[i], i] = 1（MATLAB 列→行约定）
    # FW 线性目标 g_half^T wQ_vec = sum_i gQ[q[i], i] 已最小 ✓
    q = assign(-gQ)
    wQ = perm2mat(q).T     # wQ[q[i], i] = 1，MATLAB 列→行约定，fun(wQ) = QAP 目标值
    wP = wQ                # 与 MATLAB 一致：wP = wQ
    w = stack(wP, wQ, m, n)
    d = w - x
    return d, q
