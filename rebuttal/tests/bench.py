"""
bench.py — 对 QAPLIB 论文 16 个测试问题跑 Python 版 FW1 / FW100 / FW1000 / FW2000，
并将结果写入 txt 文件。

输出格式：
    instance_id, method, obj_py, gap, time_py

其中：
    gap = (obj_py - opt) / opt * 100

运行方式：
    cd /Users/shazhengyang/Code/FastApproximateQAP/rebuttal
    python bench.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from faq import sfw
from faq.qap_read import qap_read


PROBLEMS = [
    "chr12c", "chr15a", "chr15c", "chr20b", "chr22b",
    "esc16b",
    "rou12", "rou15", "rou20",
    "tai10a", "tai15a", "tai17a", "tai20a", "tai30a", "tai35a", "tai40a",
]

METHOD_TO_TOTAL_STARTS = {
    "FW1": 1,
    "FW100": 100,
    "FW1000": 1000,
    "FW2000": 2000,
}

OPTIMAL_VALUES = {
    "chr12c": 11156,
    "chr15a": 9896,
    "chr15c": 9504,
    "chr20b": 2298,
    "chr22b": 6194,
    "esc16b": 292,
    "rou12": 235528,
    "rou15": 354210,
    "rou20": 725522,
    "tai10a": 135028,
    "tai15a": 388214,
    "tai17a": 491812,
    "tai20a": 703482,
    "tai30a": 1818146,
    "tai35a": 2422002,
    "tai40a": 3139370,
}

QAPLIB_DIR = Path("/Users/shazhengyang/Code/FastApproximateQAP/data/qaplib")
OUTPUT_PATH = Path("/Users/shazhengyang/Code/FastApproximateQAP/rebuttal/results/qaplib_bench_results.txt")
IMAX = 30
RNG_SEED = 12345678


def run_python_multistart(A: np.ndarray, B: np.ndarray, rng: np.random.Generator):
    """一次性完成 1 次平坦起点 + 1999 次随机起点，记录各里程碑最优值与累计时间。"""
    max_total_starts = max(METHOD_TO_TOTAL_STARTS.values())
    milestones = {count: method for method, count in METHOD_TO_TOTAL_STARTS.items()}

    results = {}
    cumulative_time = 0.0
    best_f = float("inf")
    best_p = None

    t0 = time.perf_counter()
    f_flat, p_flat = sfw(A, B, IMAX=IMAX)[:2]
    elapsed = time.perf_counter() - t0
    cumulative_time += elapsed
    best_f = float(f_flat)
    best_p = p_flat
    results["FW1"] = {"obj": best_f, "perm": best_p.copy(), "time": cumulative_time}

    for start_idx in range(2, max_total_starts + 1):
        t0 = time.perf_counter()
        f_rand, p_rand = sfw(A, B, IMAX=IMAX, x0=-1, rng=rng)[:2]
        cumulative_time += time.perf_counter() - t0

        if f_rand < best_f:
            best_f = float(f_rand)
            best_p = p_rand

        if start_idx in milestones:
            method = milestones[start_idx]
            results[method] = {"obj": best_f, "perm": best_p.copy(), "time": cumulative_time}

    return results


def format_float(value: float) -> str:
    return f"{float(value):.6f}"


def main():
    rng = np.random.default_rng(RNG_SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    console_lines = []
    header = "instance_id, method, obj_py, gap, time_py"
    rows.append(header)

    for prob in PROBLEMS:
        A, B, _, _ = qap_read(prob, str(QAPLIB_DIR))
        s_opt = OPTIMAL_VALUES[prob]

        py_results = run_python_multistart(A, B, rng)

        for method in ("FW1", "FW100", "FW1000", "FW2000"):
            obj_py = py_results[method]["obj"]
            time_py = py_results[method]["time"]
            gap = (obj_py - s_opt) / s_opt * 100.0

            rows.append(
                f"{prob}, {method}, {int(round(obj_py))}, {format_float(gap)}, {format_float(time_py)}"
            )

            console_lines.append(
                f"{prob:>8} {method:>6}  obj={int(round(obj_py)):>8}  "
                f"gap={gap:>8.3f}%  t_py={time_py:>9.3f}s"
            )

    OUTPUT_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(header)
    for line in console_lines:
        print(line)


if __name__ == "__main__":
    main()
