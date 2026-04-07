"""
tests/test_core.py — 覆盖 PLAN.md 中各步骤的验证点
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from faq.core import stack, unstack, fun, fungrad, perm2mat, stoch, sink
from faq.lap import assign
from faq.lines import lines
from faq.dsproj import dsproj
from faq.sfw import sfw


# ===========================================================================
# 步骤 2：core.py — perm2mat / stack / unstack / fun / fungrad
# ===========================================================================

class TestPerm2mat:
    def test_identity(self):
        p = np.array([0, 1, 2])
        P = perm2mat(p)
        np.testing.assert_array_equal(P, np.eye(3))

    def test_reversal(self):
        p = np.array([2, 1, 0])
        P = perm2mat(p)
        expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
        np.testing.assert_array_equal(P, expected)


class TestStackUnstack:
    def test_roundtrip(self):
        rng = np.random.default_rng(42)
        P = rng.random((3, 3))
        Q = rng.random((3, 3))
        x = stack(P, Q, 3, 3)
        P2, Q2 = unstack(x, 3, 3)
        np.testing.assert_allclose(P, P2, atol=1e-15)
        np.testing.assert_allclose(Q, Q2, atol=1e-15)

    def test_column_major_order(self):
        """确认 stack/unstack 使用列主序（F order），与 MATLAB 一致。"""
        P = np.arange(9, dtype=float).reshape(3, 3, order='F')
        x = stack(P, P, 3, 3)
        # 列主序下 ravel 后第一个元素应是 P[0,0]=0
        assert x[0] == P[0, 0]
        # 列主序下第二个元素是 P[1,0]
        assert x[1] == P[1, 0]


class TestFunFungrad:
    """固定 n=3 的 A, B, x0，验证 fun 与 fungrad 数值。"""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.n = 3
        self.A = rng.standard_normal((self.n, self.n))
        self.B = rng.standard_normal((self.n, self.n))
        flat = np.ones((self.n, self.n)) / self.n
        self.x = stack(flat, flat, self.n, self.n)

    def test_fun_value(self):
        f = fun(self.x, self.A, self.B)
        # 验证人工计算：P=Q=ones/n
        P = np.ones((self.n, self.n)) / self.n
        expected = float(np.sum(P @ self.A @ P.T * self.B))
        assert abs(f - expected) < 1e-10

    def test_fungrad_value_matches_fun(self):
        f0, g = fungrad(self.x, self.A, self.B)
        f1 = fun(self.x, self.A, self.B)
        assert abs(f0 - f1) < 1e-10

    def test_fungrad_gradient_finite_diff(self):
        """数值梯度验证（有限差分），容差 1e-6。"""
        _, g = fungrad(self.x, self.A, self.B)
        eps = 1e-6
        g_fd = np.zeros_like(self.x)
        for i in range(len(self.x)):
            xp = self.x.copy(); xp[i] += eps
            xm = self.x.copy(); xm[i] -= eps
            g_fd[i] = (fun(xp, self.A, self.B) - fun(xm, self.A, self.B)) / (2 * eps)
        np.testing.assert_allclose(g, g_fd, atol=1e-5)


# ===========================================================================
# 步骤 3：lap.py — assign 最大权匹配
# ===========================================================================

class TestAssign:
    def test_small_matrix_optimal(self):
        """验证 assign(C) 返回使 sum_i C[p[i], i] 最大的排列（列→行方向）。

        MATLAB lapjv(-A', ...) 语义：最大化 sum_i A[rowsol[i], i]。
        """
        rng = np.random.default_rng(7)
        n = 4
        C = rng.standard_normal((n, n))
        p = assign(C)
        best = float(C[p, np.arange(n)].sum())  # sum_i C[p[i], i]

        from itertools import permutations
        max_val = max(
            sum(C[perm[i], i] for i in range(n))
            for perm in permutations(range(n))
        )
        assert abs(best - max_val) < 1e-10

    def test_identity_cost(self):
        n = 5
        C = np.eye(n)
        p = assign(C)
        assert sorted(p.tolist()) == list(range(n))


# ===========================================================================
# 步骤 4：stoch / sink
# ===========================================================================

class TestStochSink:
    def test_stoch_row(self):
        rng = np.random.default_rng(1)
        A = rng.random((4, 4))
        A_row = stoch(A, dim=2)
        np.testing.assert_allclose(A_row.sum(axis=1), np.ones(4), atol=1e-15)

    def test_stoch_col(self):
        rng = np.random.default_rng(2)
        A = rng.random((4, 4))
        A_col = stoch(A, dim=1)
        np.testing.assert_allclose(A_col.sum(axis=0), np.ones(4), atol=1e-15)

    def test_sink_doubly_stochastic(self):
        """随机矩阵经 sink(A, 10) 后各行列和接近 1（容差 1e-12）。"""
        rng = np.random.default_rng(3)
        A = rng.random((6, 6))
        A_ds = sink(A, 10)
        np.testing.assert_allclose(A_ds.sum(axis=1), np.ones(6), atol=1e-12)
        np.testing.assert_allclose(A_ds.sum(axis=0), np.ones(6), atol=1e-12)


# ===========================================================================
# 步骤 5：lines.py — 线搜索
# ===========================================================================

class TestLines:
    def test_salpha_in_unit_interval(self):
        rng = np.random.default_rng(10)
        n = 3
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        flat = np.ones((n, n)) / n
        x = stack(flat, flat, n, n)
        _, g = fungrad(x, A, B)
        d, _ = dsproj(x, g, n, n)
        salpha = lines(x, d, g, A, B)
        assert 0.0 <= salpha <= 1.0

    def test_lines_non_increasing(self):
        """f(x + salpha*d) <= f(x)"""
        rng = np.random.default_rng(11)
        n = 4
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        flat = np.ones((n, n)) / n
        x = stack(flat, flat, n, n)
        _, g = fungrad(x, A, B)
        g_half = (g[:n*n] + g[n*n:]) / 2
        g_sym = np.concatenate([g_half, g_half])
        d, _ = dsproj(x, g_sym, n, n)
        salpha = lines(x, d, g_sym, A, B)
        assert fun(x + salpha * d, A, B) <= fun(x, A, B) + 1e-10


# ===========================================================================
# 步骤 6：dsproj.py — FW 下降方向
# ===========================================================================

class TestDsproj:
    def test_fw_descent_direction(self):
        """验证 g^T d < 0（Frank–Wolfe 下降方向）。"""
        rng = np.random.default_rng(20)
        n = 4
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        flat = np.ones((n, n)) / n
        x = stack(flat, flat, n, n)
        _, g = fungrad(x, A, B)
        g_half = (g[:n*n] + g[n*n:]) / 2
        g_sym = np.concatenate([g_half, g_half])
        d, q = dsproj(x, g_sym, n, n)
        dot = float(g_sym @ d)
        # d 为零向量（已在顶点）时为 0；否则应严格 < 0
        assert dot <= 1e-10, f"g^T d = {dot:.4e}，不是下降方向"


# ===========================================================================
# 步骤 7：sfw.py — 端到端测试
# ===========================================================================

class TestSfw:
    def test_isomorphic_n4(self):
        """n=4 同构图：已知置换 p，验证 sfw 找到全局最优解 f = -||A||_F^2。

        B = A[p,p] => 最小化 sum A*(-B)[q,q] 的全局最优为 q = p^{-1}，
        对应 f* = -sum_{i,j} A[i,j]^2 = -||A||_F^2。
        """
        rng = np.random.default_rng(42)
        n = 4
        A = rng.random((n, n))
        A = (A + A.T) / 2  # 对称化，保证同构图有唯一最优
        p = rng.permutation(n)
        B = A[np.ix_(p, p)]
        f, myp, x, iters, fs, myps, fun_vals = sfw(A, -B, IMAX=30)
        optimal_f = -float(np.sum(A * A))  # = -||A||_F^2
        assert abs(f - optimal_f) < 1e-4, (
            f"f={f:.6f} 与全局最优 {optimal_f:.6f} 不符（差 {abs(f-optimal_f):.2e}）"
        )

    def test_monotone_objective(self):
        """n=20 Erdős–Rényi，验证 FW 连续内点目标值 fun_vals 单调不升。

        fs（离散排列投影值）不保证单调；fun_vals（内点 x 处的连续目标值）
        由线搜索保证单调不升。
        """
        rng = np.random.default_rng(99)
        n = 20
        A = (rng.random((n, n)) < 0.3).astype(float)
        A = (A + A.T) / 2
        B = (rng.random((n, n)) < 0.3).astype(float)
        B = (B + B.T) / 2
        _, _, _, _, fs, _, fun_vals = sfw(A, B, IMAX=30)
        for i in range(1, len(fun_vals)):
            assert fun_vals[i] <= fun_vals[i - 1] + 1e-9, (
                f"第 {i} 步连续目标值上升：{fun_vals[i-1]:.6f} → {fun_vals[i]:.6f}"
            )
