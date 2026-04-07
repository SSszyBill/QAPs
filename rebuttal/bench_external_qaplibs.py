"""
bench_external_qaplibs.py — 使用外部 QAPs 仓库中的 qaplibs 数据跑 FAQ benchmark。

默认行为：
    - 从外部仓库的 qaplib.txt 读取实例列表
    - 使用 qaplibs 目录作为数据根目录
    - 输出格式：instance_id, method, obj_py, gap, time_py

可通过命令行参数控制：
    --dataset-root   数据根目录（包含 qapdata/qapprob 和 qapsoln）
    --instance-list  实例列表文件，每行一个实例名
    --limit          只跑前 N 个实例，便于本地 smoke test
    --methods        逗号分隔的方法名，默认 FW1,FW100,FW1000,FW2000
    --output         输出 txt 路径
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from faq import sfw
from faq.qap_read import qap_read


DEFAULT_REPO_ROOT = Path("/Users/shazhengyang/Code/FastApproximateQAP/rebuttal/external/QAPs_xjx")
DEFAULT_DATASET_ROOT = DEFAULT_REPO_ROOT / "qaplibs"
DEFAULT_INSTANCE_LIST = DEFAULT_REPO_ROOT / "qaplib.txt"
DEFAULT_OUTPUT = Path("/Users/shazhengyang/Code/FastApproximateQAP/rebuttal/results/qaplib_external_bench.txt")
IMAX = 30
RNG_SEED = 12345678
DEFAULT_METHOD_TO_TOTAL_STARTS = {
    "FW1": 1,
    "FW100": 100,
    "FW1000": 1000,
    "FW2000": 2000,
}


def load_problem_list(path: Path, limit: int | None = None) -> list[str]:
    problems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        problems = problems[:limit]
    return problems


def load_known_optimum(problem: str, dataset_root: Path):
    """优先从 qapsoln 读取最优值；若缺失则返回 None。"""
    _, _, _, s_opt = qap_read(problem, str(dataset_root))
    return s_opt


def run_python_multistart(
    A: np.ndarray,
    B: np.ndarray,
    rng: np.random.Generator,
    method_to_total_starts: dict[str, int],
):
    """一次性完成所需启动次数，并记录各里程碑最优值与累计时间。"""
    max_total_starts = max(method_to_total_starts.values())
    milestones = {count: method for method, count in method_to_total_starts.items()}

    results = {}
    cumulative_time = 0.0
    best_f = float("inf")
    best_p = None

    t0 = time.perf_counter()
    f_flat, p_flat = sfw(A, B, IMAX=IMAX)[:2]
    cumulative_time += time.perf_counter() - t0
    best_f = float(f_flat)
    best_p = p_flat

    if 1 in milestones:
        results[milestones[1]] = {"obj": best_f, "perm": best_p.copy(), "time": cumulative_time}

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


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--instance-list", type=Path, default=DEFAULT_INSTANCE_LIST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--methods", type=str, default="FW1,FW100,FW1000,FW2000")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    selected_methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    method_to_total_starts = {name: DEFAULT_METHOD_TO_TOTAL_STARTS[name] for name in selected_methods}
    problems = load_problem_list(args.instance_list, limit=args.limit)
    rng = np.random.default_rng(RNG_SEED)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    console_lines = []
    header = "instance_id, method, obj_py, gap, time_py"
    rows.append(header)

    for prob in problems:
        A, B, _, _ = qap_read(prob, str(args.dataset_root))
        s_opt = load_known_optimum(prob, args.dataset_root)
        py_results = run_python_multistart(A, B, rng, method_to_total_starts)

        for method in selected_methods:
            obj_py = py_results[method]["obj"]
            time_py = py_results[method]["time"]
            gap = None if s_opt is None else (obj_py - s_opt) / s_opt * 100.0

            rows.append(
                f"{prob}, {method}, {int(round(obj_py))}, {format_float(gap)}, {format_float(time_py)}"
            )

            gap_text = "" if gap is None else f"{gap:>8.3f}%"
            console_lines.append(
                f"{prob:>10} {method:>6}  obj={int(round(obj_py)):>10}  "
                f"gap={gap_text:>9}  t_py={time_py:>9.3f}s"
            )

    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(header)
    for line in console_lines:
        print(line)


if __name__ == "__main__":
    main()
