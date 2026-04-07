"""
core.py — FAQ 算法基础工具函数
对应 MATLAB: stack.m, unstack.m, fun.m, fungrad.m, perm2mat.m, stoch.m, sink.m

关键约定：
- 所有 reshape/ravel 使用 order='F'（MATLAB 列主序）
- 排列向量全程 0-indexed
"""

import numpy as np


# ---------------------------------------------------------------------------
# stack / unstack
# ---------------------------------------------------------------------------

def stack(P: np.ndarray, Q: np.ndarray, m: int = None, n: int = None) -> np.ndarray:
    """将 P (m×m) 和 Q (n×n) 拼接成列向量 x，对应 MATLAB stack.m。"""
    return np.concatenate([P.ravel(order='F'), Q.ravel(order='F')])


def unstack(x: np.ndarray, m: int, n: int):
    """从列向量 x 中还原 P (m×m) 和 Q (n×n)，对应 MATLAB unstack.m。"""
    P = x[:m * m].reshape(m, m, order='F')
    Q = x[m * m: m * m + n * n].reshape(n, n, order='F')
    return P, Q


# ---------------------------------------------------------------------------
# perm2mat
# ---------------------------------------------------------------------------

def perm2mat(p: np.ndarray) -> np.ndarray:
    """0-indexed 排列向量 -> 置换矩阵，对应 MATLAB perm2mat.m。

    MATLAB 版: P(i, p(i)) = 1（1-indexed）
    Python 版: np.eye(n)[p]（0-indexed）
    """
    n = len(p)
    P = np.zeros((n, n))
    P[np.arange(n), p] = 1.0
    return P


# ---------------------------------------------------------------------------
# fun / fungrad
# ---------------------------------------------------------------------------

def fun(x: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """目标函数 f = sum(P @ A @ Q^T * B)，对应 MATLAB fun.m。"""
    m, n = A.shape
    P, Q = unstack(x, m, n)
    return float(np.sum(P @ A @ Q.T * B))


def fungrad(x: np.ndarray, A: np.ndarray, B: np.ndarray):
    """目标函数值及梯度，对应 MATLAB fungrad.m。

    梯度：
      ∂f/∂P = B @ Q @ A^T   (按列主序 ravel 后拼接)
      ∂f/∂Q = B^T @ P @ A
    """
    m, n = A.shape
    P, Q = unstack(x, m, n)
    f0 = float(np.sum(P @ A @ Q.T * B))
    gP = (B @ Q @ A.T).ravel(order='F')
    gQ = (B.T @ P @ A).ravel(order='F')
    g = np.concatenate([gP, gQ])
    return f0, g


# ---------------------------------------------------------------------------
# stoch / sink
# ---------------------------------------------------------------------------

def stoch(A: np.ndarray, dim: int = 2) -> np.ndarray:
    """Sinkhorn 单步归一化，对应 MATLAB stoch.m。

    dim=1 → 列归一化；dim=2 → 行归一化（与 MATLAB 约定一致）。
    MATLAB stoch(A,1) 按列归一化，stoch(A,2) 按行归一化。
    """
    tiny = np.finfo(float).tiny
    if dim == 2:  # 行归一化
        s = A.sum(axis=1, keepdims=True)
        s = np.maximum(s, tiny)
        return A / s
    else:          # 列归一化
        s = A.sum(axis=0, keepdims=True)
        s = np.maximum(s, tiny)
        return A / s


def sink(A: np.ndarray, n_iter: int) -> np.ndarray:
    """n_iter 轮 Sinkhorn 平衡，对应 MATLAB sink.m。

    MATLAB sink 调用 stoch(A,1) 再 stoch(A,2)，即先列后行。
    """
    for _ in range(n_iter):
        A = stoch(A, dim=1)
        A = stoch(A, dim=2)
    return A
