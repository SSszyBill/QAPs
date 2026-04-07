"""
lines.py — 精确二次线搜索
对应 MATLAB lines.m 中 type==2 分支（sfw.m 中 stype=2）

将目标函数视为关于步长 α 的二次多项式，直接求极小值：
  f(x + α·d) ≈ a·α² + b·α + c
  b = g^T d，c = f(x)，a = f(x+d) - b - c
  α* = clamp(-b/(2a), 0, 1)
再做三点比较保证不劣于端点。
"""

import numpy as np
from .core import fun


def lines(x: np.ndarray, d: np.ndarray, g: np.ndarray,
          A: np.ndarray, B: np.ndarray) -> float:
    """二次线搜索，返回步长 salpha ∈ [0, 1]。

    参数
    ----
    x : 当前点
    d : Frank–Wolfe 方向
    g : 当前梯度
    A, B : QAP 矩阵

    返回
    ----
    salpha : 最优步长
    """
    b = float(g @ d)
    c = fun(x, A, B)
    a = fun(x + d, A, B) - b - c

    if abs(a) < np.finfo(float).eps:
        salpha = 1.0
    else:
        salpha = min(1.0, max(-b / (2.0 * a), 0.0))

    fun_alpha = fun(x + salpha * d, A, B)

    if fun_alpha > c:
        salpha = 0.0
    elif fun_alpha > fun(x + d, A, B):
        salpha = 1.0

    return salpha
