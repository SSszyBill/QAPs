"""
qap_read.py — 读取 QAPLIB 格式的问题文件和解文件
对应 MATLAB: qap_read.m

兼容两类常见目录布局：
    1. <root>/qapprob/*.dat  + <root>/qapsoln/*.sln
    2. <root>/qapdata/*.dat  + <root>/qapsoln/*.sln

QAPLIB .dat 格式：
    n
    （空行）
    n×n 矩阵 A（流量矩阵）
    （空行）
    n×n 矩阵 B（距离矩阵）

QAPLIB .sln 格式：
    n  optimal_value
    p[0] p[1] ... p[n-1]   （1-indexed 排列）
"""

import numpy as np
from pathlib import Path


def _resolve_problem_dir(root: Path) -> Path:
    """兼容 qapprob/ 与 qapdata/ 两种目录名。"""
    for dirname in ("qapprob", "qapdata"):
        candidate = root / dirname
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find problem directory under {root}. "
        "Expected either qapprob/ or qapdata/."
    )


def qap_read(problem: str, directory: str):
    """读取 QAPLIB 问题及已知最优解。

    参数
    ----
    problem   : 问题名，如 'chr12a'
    directory : qaplib 根目录，下含 qapprob/ 和 qapsoln/

    返回
    ----
    A    : n×n 流量矩阵
    B    : n×n 距离矩阵
    p    : 0-indexed 最优排列（若无解文件则为 None）
    s    : 已知最优目标值（若无解文件则为 None）
    """
    root = Path(directory)
    problem_dir = _resolve_problem_dir(root)

    # 读问题文件
    dat_path = problem_dir / f"{problem}.dat"
    with open(dat_path, "r") as f:
        tokens = f.read().split()

    idx = 0
    n = int(tokens[idx]); idx += 1
    A = np.array([float(tokens[idx + i]) for i in range(n * n)]).reshape(n, n)
    idx += n * n
    B = np.array([float(tokens[idx + i]) for i in range(n * n)]).reshape(n, n)

    # 读解文件（可选）
    sln_path = root / "qapsoln" / f"{problem}.sln"
    if not sln_path.exists():
        return A, B, None, None

    with open(sln_path, "r") as f:
        tokens = f.read().split()

    idx = 0
    _n = int(tokens[idx]); idx += 1
    s = float(tokens[idx]); idx += 1
    # .sln 文件中排列是 1-indexed，转为 0-indexed
    p = np.array([int(tokens[idx + i]) - 1 for i in range(_n)])
    return A, B, p, s
