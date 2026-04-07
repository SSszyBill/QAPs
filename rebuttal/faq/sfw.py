"""
sfw.py — Frank–Wolfe 主循环（Soft Frank-Wolfe for QAP）
对应 MATLAB sfw.m

最小化目标函数 f(p) = sum(A * B[p,:][:,p])。
调用方若需图匹配语义（最小化 -sum(...)），在传入时传 -B。

返回：f, myp, x, iter, fs, myps
"""

import numpy as np
from .core import stack, unstack, fun, fungrad, perm2mat, sink
from .lap import assign
from .lines import lines as _lines
from .dsproj import dsproj


def sfw(
    A: np.ndarray,
    B: np.ndarray,
    IMAX: float = 30,
    x0=None,
    rng: np.random.Generator = None,
):
    """Frank–Wolfe 近似 QAP 求解器。

    参数
    ----
    A, B : n×n 方阵
    IMAX : 最大迭代次数，默认 30。
           设为 0.5 时执行一次 FW 步（Priebe LAP 近似），步长强制为 1。
    x0   : 起点，可以是：
           None      → 平坦双随机矩阵 ones(n,n)/n
           -1        → 随机 Sinkhorn 起点
           长度 n 的 0-indexed 排列向量
           n×n 双随机矩阵
    rng  : numpy 随机数生成器，None 时使用默认生成器

    返回
    ----
    f    : 最终目标函数值
    myp  : 0-indexed 最优排列
    x    : FW 内点（双随机矩阵拼接的列向量）
    iters: 实际迭代次数
    fs   : 每步迭代的目标函数值列表
    myps : 每步迭代的排列列表
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = A.shape
    stype = 2  # 始终使用二次线搜索

    # ---- 起点初始化（对应 sfw.m 第 44–68 行）----
    if x0 is None:
        if IMAX == 0.5:
            t = np.eye(m)
            x = stack(t, t, m, n)
        else:
            x = np.concatenate([np.ones(m * m) / m, np.ones(n * n) / n])
    elif np.isscalar(x0) and x0 == -1:
        X = np.ones((m, m)) / m
        lam = 0.5
        X = (1 - lam) * X + lam * sink(rng.random((m, m)), 10)
        x = stack(X, X, m, n)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim == 1 and x0.size == m:
            # 0-indexed 排列向量
            M = perm2mat(x0.astype(int)).T
            x = stack(M, M, m, n)
        elif x0.size == m * n:
            # n×n 双随机矩阵
            DS = x0.reshape(m, n)
            x = stack(DS, DS, m, n)
        else:
            x = x0.ravel(order='F')

    stoptol = 1.0e-4
    myp = None
    iters = 0
    stop = 0
    salpha = 1.0  # 初始化，防止循环 0 次时 salpha 未定义

    fs = []       # 每步离散排列投影的 QAP 目标值（不保证单调）
    myps = []
    fun_vals = [] # 每步 FW 连续内点 x 的目标值（保证单调不升）

    while (iters < IMAX) and (stop == 0):
        # ---- 目标函数 + 梯度 ----
        f0, g = fungrad(x, A, B)

        # ---- 梯度对称化（对应 sfw.m 第 80 行）----
        # 两个块大小相同（m==n 时），各取均值后重复
        gP = g[:m * m]
        gQ = g[m * m:]
        g_half = (gP + gQ) / 2.0
        g = np.concatenate([g_half, g_half])

        # ---- FW 方向子问题 ----
        d, myp = dsproj(x, g, m, n)

        stopnorm = np.linalg.norm(d)

        # ---- 收敛检测 ----
        if stopnorm < stoptol:
            stop = 1

        # ---- 线搜索 ----
        if IMAX > 0.5:
            salpha = _lines(x, d, g, A, B)
        else:
            salpha = 1.0  # Priebe LAP 近似

        x = x + salpha * d
        iters += 1

        if salpha == 0:
            stop = 1

        # ---- 记录每步结果 ----
        fun_vals.append(fun(x, A, B))  # 连续目标值，保证单调不升
        P, Q = unstack(x, m, n)
        if salpha != 1.0:
            temp = assign(P)
        else:
            temp = myp
        myps.append(temp.copy())
        fs.append(float(np.sum(A * B[np.ix_(temp, temp)])))

    # ---- 最终排列投影 ----
    if salpha != 1.0:
        P, Q = unstack(x, m, n)
        myp = assign(P)

    f = float(np.sum(A * B[np.ix_(myp, myp)]))
    return f, myp, x, iters, fs, myps, fun_vals
