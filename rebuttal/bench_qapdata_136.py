"""
bench_qapdata_136.py — 对 qapdata 目录中的全部实例分别跑 FW1 / FW1000 / FW2000，
并输出 3 个独立 txt 文件。

每个输出文件格式：
    instance, obj, time

默认输入目录：
    /Users/shazhengyang/Code/FastApproximateQAP/rebuttal/external/qapdata

默认输出目录：
    /Users/shazhengyang/Code/FastApproximateQAP/rebuttal/results/qapdata_136

用法示例：
    python bench_qapdata_136.py
    python bench_qapdata_136.py --limit 5
    python bench_qapdata_136.py --data-dir /path/to/qapdata --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from faq import sfw
from faq.qap_read import qap_read


DEFAULT_DATA_DIR = Path("/Users/shazhengyang/Code/FastApproximateQAP/rebuttal/external/qapdata")
DEFAULT_OUTPUT_DIR = Path("/Users/shazhengyang/Code/FastApproximateQAP/rebuttal/results/qapdata_136")
METHOD_TO_TOTAL_STARTS = {
    "FW1": 1,
    "FW1000": 1000,
    "FW2000": 2000,
}
IMAX = 30
RNG_SEED = 12345678


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个实例，用于 smoke test")
    return parser.parse_args()


def list_instances(data_dir: Path, limit: int | None = None) -> list[str]:
    problems = sorted(path.stem for path in data_dir.glob("*.dat"))
    if limit is not None:
        problems = problems[:limit]
    return problems


def run_multistart(
    A: np.ndarray,
    B: np.ndarray,
    rng: np.random.Generator,
):
    """一次性完成 1 次平坦起点 + 1999 次随机起点，并记录里程碑结果。"""
    max_total_starts = max(METHOD_TO_TOTAL_STARTS.values())
    milestones = {count: method for method, count in METHOD_TO_TOTAL_STARTS.items()}

    results = {}
    cumulative_time = 0.0
    best_f = float("inf")

    t0 = time.perf_counter()
    f_flat, _ = sfw(A, B, IMAX=IMAX)[:2]
    cumulative_time += time.perf_counter() - t0
    best_f = float(f_flat)
    results[milestones[1]] = {"obj": best_f, "time": cumulative_time}

    for start_idx in range(2, max_total_starts + 1):
        t0 = time.perf_counter()
        f_rand, _ = sfw(A, B, IMAX=IMAX, x0=-1, rng=rng)[:2]
        cumulative_time += time.perf_counter() - t0

        if f_rand < best_f:
            best_f = float(f_rand)

        if start_idx in milestones:
            method = milestones[start_idx]
            results[method] = {"obj": best_f, "time": cumulative_time}

    return results


def write_method_file(output_dir: Path, method: str, rows: list[tuple[str, int, float]]):
    output_path = output_dir / f"{method.lower()}.txt"
    lines = ["instance, obj, time"]
    for instance, obj, elapsed in rows:
        lines.append(f"{instance}, {obj}, {elapsed:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problems = list_instances(args.data_dir, args.limit)
    rng = np.random.default_rng(RNG_SEED)

    rows_by_method: dict[str, list[tuple[str, int, float]]] = {
        method: [] for method in METHOD_TO_TOTAL_STARTS
    }

    for idx, problem in enumerate(problems, start=1):
        A, B, _, _ = qap_read(problem, str(args.data_dir.parent))
        results = run_multistart(A, B, rng)

        print(f"[{idx}/{len(problems)}] {problem}")
        for method in ("FW1", "FW1000", "FW2000"):
            obj = int(round(results[method]["obj"]))
            elapsed = results[method]["time"]
            rows_by_method[method].append((problem, obj, elapsed))
            print(f"  {method:>6}  obj={obj:>10}  time={elapsed:>9.3f}s")

    written = []
    for method in ("FW1", "FW1000", "FW2000"):
        output_path = write_method_file(args.output_dir, method, rows_by_method[method])
        written.append(output_path)

    print("\nWrote files:")
    for output_path in written:
        print(output_path)


if __name__ == "__main__":
    main()
