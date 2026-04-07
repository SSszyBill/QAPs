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

断点续跑：
    - 若输出目录下已存在 fw1.txt / fw1000.txt / fw2000.txt
    - 且同一个 instance 已同时出现在这 3 个文件中
    - 则脚本会自动跳过该 instance
    - 每跑完一个 instance，会立刻 append 到 3 个结果文件
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from faq import sfw
from faq.qap_read import qap_read


REBUTTAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REBUTTAL_DIR / "external" / "qapdata"
DEFAULT_OUTPUT_DIR = REBUTTAL_DIR / "results" / "qapdata_136"
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


def output_path(output_dir: Path, method: str) -> Path:
    return output_dir / f"{method.lower()}.txt"


def ensure_output_files(output_dir: Path):
    for method in ("FW1", "FW1000", "FW2000"):
        path = output_path(output_dir, method)
        if not path.exists():
            path.write_text("instance, obj, time\n", encoding="utf-8")


def load_completed_instances(output_dir: Path) -> set[str]:
    """只有当 instance 同时出现在 3 个结果文件中时，才认为它已完成。"""
    completed_sets = []
    for method in ("FW1", "FW1000", "FW2000"):
        path = output_path(output_dir, method)
        instances = set()
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines[1:]:
                if not line.strip():
                    continue
                instance = line.split(",", 1)[0].strip()
                if instance:
                    instances.add(instance)
        completed_sets.append(instances)
    if not completed_sets:
        return set()
    return set.intersection(*completed_sets)


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


def append_method_result(output_dir: Path, method: str, instance: str, obj: int, elapsed: float):
    path = output_path(output_dir, method)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{instance}, {obj}, {elapsed:.6f}\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_files(args.output_dir)

    problems = list_instances(args.data_dir, args.limit)
    rng = np.random.default_rng(RNG_SEED)
    completed = load_completed_instances(args.output_dir)
    skipped = 0

    for idx, problem in enumerate(problems, start=1):
        if problem in completed:
            skipped += 1
            print(f"[{idx}/{len(problems)}] {problem}  SKIP (already completed)")
            continue

        A, B, _, _ = qap_read(problem, str(args.data_dir.parent))
        results = run_multistart(A, B, rng)

        print(f"[{idx}/{len(problems)}] {problem}")
        for method in ("FW1", "FW1000", "FW2000"):
            obj = int(round(results[method]["obj"]))
            elapsed = results[method]["time"]
            append_method_result(args.output_dir, method, problem, obj, elapsed)
            print(f"  {method:>6}  obj={obj:>10}  time={elapsed:>9.3f}s")

    print("\nWrote files:")
    for method in ("FW1", "FW1000", "FW2000"):
        print(output_path(args.output_dir, method))
    if skipped:
        print(f"Skipped completed instances: {skipped}")


if __name__ == "__main__":
    main()
