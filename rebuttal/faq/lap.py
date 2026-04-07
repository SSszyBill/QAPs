"""
lap.py — 线性指派问题包装
对应 MATLAB: assign.m（使用 lapjv(-A', 0.01)）

MATLAB lapjv(-A', 0.01) 约定：
  - lapjv(C) 返回 rowsol，其中 rowsol[i] = 分配给行 i 的列索引
  - lapjv(-A') 最小化 sum_i (-A')[i, rowsol[i]] = 最大化 sum_i A[rowsol[i], i]
  即：返回 col_ind 使 sum_i A[col_ind[i], i] 最大（列→行方向的最大权匹配）

Python 等价：linear_sum_assignment(-A.T)
  - 最小化 sum_i (-A.T)[i, col_ind[i]] = 最大化 sum_i A[col_ind[i], i] ✓

配合 dsproj 中 wQ = perm2mat(q).T（MATLAB 列→行约定），
fun(x, A, B) 才能正确计算 QAP 目标值 sum A * B(q, q)。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(A: np.ndarray) -> np.ndarray:
    """求矩阵 A 的最大权匹配，返回 0-indexed 排列 q。

    返回 col_ind 使 sum_i A[col_ind[i], i] 最大（列→行方向）。
    对应 MATLAB assign(A) = lapjv(-A', 0.01)。
    与 dsproj 中 wQ = perm2mat(q).T 配合，保证 fun(x,A,B) 正确等于 QAP 目标值。
    """
    _, col_ind = linear_sum_assignment(-A.T)
    return col_ind
