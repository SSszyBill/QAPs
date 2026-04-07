"""
bench_matlab_compare.py — 对 QAPLIB 论文 16 个测试问题跑 FW1 / FW3 / FW10，
并将 Python 与 MATLAB 的结果写入独立 txt 文件。

输出格式：
    instance_id, method, obj_py, obj_matlab, time_py, time_matlab
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from faq import sfw
from faq.qap_read import qap_read


PROBLEMS = [
    "chr12c", "chr15a", "chr15c", "chr20b", "chr22b",
    "esc16b",
    "rou12", "rou15", "rou20",
    "tai10a", "tai15a", "tai17a", "tai20a", "tai30a", "tai35a", "tai40a",
]

QAPLIB_DIR = Path("/Users/shazhengyang/Code/FastApproximateQAP/data/qaplib")
MATLAB_RESULTS_PATH = Path("/Users/shazhengyang/Code/FastApproximateQAP/data/results/problems16_results.mat")
OUTPUT_PATH = Path("/Users/shazhengyang/Code/FastApproximateQAP/rebuttal/results/qaplib_bench_matlab_compare.txt")
IMAX = 30
RNG_SEED = 12345678


def load_matlab_bench():
    """读取 Matlab benchmark 的目标值和累计时间。"""
    mat = loadmat(MATLAB_RESULTS_PATH, squeeze_me=True, struct_as_record=False)

    matlab_problems = [str(x) for x in np.atleast_1d(mat["probs"]).tolist()]
    if matlab_problems != PROBLEMS:
        raise ValueError(
            f"MATLAB problems order mismatch:\nexpected={PROBLEMS}\nactual={matlab_problems}"
        )

    mn = np.atleast_1d(mat["mn"])
    s1 = np.atleast_1d(mat["s1"]).astype(float)
    s3 = np.atleast_1d(mat["s3"]).astype(float)
    s10 = np.atleast_1d(mat["s10"]).astype(float)

    per_trial_times = np.vstack([np.atleast_1d(rec.time).astype(float) for rec in mn[:10]])
    time_fw1 = per_trial_times[:1].sum(axis=0)
    time_fw3 = per_trial_times[:3].sum(axis=0)
    time_fw10 = per_trial_times[:10].sum(axis=0)

    return {
        "FW1": {"obj": s1, "time": time_fw1},
        "FW3": {"obj": s3, "time": time_fw3},
        "FW10": {"obj": s10, "time": time_fw10},
    }


def run_python_bench(A: np.ndarray, B: np.ndarray, rng: np.random.Generator):
    """一次性完成 1 次平坦起点 + 9 次随机起点，记录 FW1/FW3/FW10。"""
    starts = []

    t0 = time.perf_counter()
    f_flat, p_flat = sfw(A, B, IMAX=IMAX)[:2]
    starts.append((f_flat, p_flat, time.perf_counter() - t0))

    for _ in range(9):
        t0 = time.perf_counter()
        f_rand, p_rand = sfw(A, B, IMAX=IMAX, x0=-1, rng=rng)[:2]
        starts.append((f_rand, p_rand, time.perf_counter() - t0))

    results = {}
    cumulative_time = 0.0
    best_f = float("inf")
    best_p = None
    milestones = {1: "FW1", 3: "FW3", 10: "FW10"}

    for idx, (f_val, p_val, elapsed) in enumerate(starts, start=1):
        cumulative_time += elapsed
        if f_val < best_f:
            best_f = float(f_val)
            best_p = p_val
        if idx in milestones:
            method = milestones[idx]
            results[method] = {
                "obj": best_f,
                "perm": best_p.copy(),
                "time": cumulative_time,
            }

    return results


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def main():
    matlab = load_matlab_bench()
    rng = np.random.default_rng(RNG_SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    console_lines = []
    header = "instance_id, method, obj_py, obj_matlab, time_py, time_matlab"
    rows.append(header)

    for idx, prob in enumerate(PROBLEMS):
        A, B, _, _ = qap_read(prob, str(QAPLIB_DIR))
        py_results = run_python_bench(A, B, rng)

        for method in ("FW1", "FW3", "FW10"):
            obj_py = py_results[method]["obj"]
            time_py = py_results[method]["time"]
            obj_matlab = matlab[method]["obj"][idx]
            time_matlab = matlab[method]["time"][idx]

            rows.append(
                f"{prob}, {method}, {int(round(obj_py))}, {int(round(obj_matlab))}, "
                f"{format_float(time_py)}, {format_float(time_matlab)}"
            )

            delta = int(round(obj_py - obj_matlab))
            console_lines.append(
                f"{prob:>8} {method:>4}  "
                f"py={int(round(obj_py)):>8}  ml={int(round(obj_matlab)):>8}  "
                f"delta={delta:>+7}  "
                f"t_py={time_py:>8.3f}s  t_ml={time_matlab:>8.3f}s"
            )

    OUTPUT_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(header)
    for line in console_lines:
        print(line)


if __name__ == "__main__":
    main()
