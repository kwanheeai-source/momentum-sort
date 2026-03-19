# benchmarks/real.py
"""
Real-world benchmark — now matches synthetic.py output format exactly
"""

import numpy as np
import argparse
import sys
from pathlib import Path
import pandas as pd

from core.momentumsort import MomentumSort
from core.momentumsortoptimal import MomentumSortOptimal


def parse_args():
    parser = argparse.ArgumentParser(
        description="MomentumSort Real-World Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n", "-n", type=int, default=100_000)
    parser.add_argument("--trials", "-t", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--count", action="store_true", help="Enable exact comparison counting")
    parser.add_argument("--output", "-o", type=str, default=None)
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)

DATA_DIR = Path(__file__).parent.parent / "data"

datasets = {
    "Abalone (whole_weight)": {"file": DATA_DIR / "abalone.data", "type": "data", "column": 4, "name": "whole_weight"},
    "Wine Quality (alcohol)": {"file": DATA_DIR / "winequality-red.csv", "type": "csv", "column": "alcohol", "name": "alcohol"},
    "California Housing (MedInc)": {"file": DATA_DIR / "california_medinc.csv", "type": "csv", "column": "median_income", "name": "MedInc"},
    "NYC Taxi Fares (1M)": {"file": DATA_DIR / "nyc_taxi_fares_1000000.npy", "type": "npy", "name": "fare_amount"},
    "NYC Taxi Fares (2.96M)": {"file": DATA_DIR / "nyc_taxi_fares_2964624.npy", "type": "npy", "name": "fare_amount"},
}


def load_data(info):
    fp = info["file"]
    if not fp.exists():
        print(f"❌ Missing: {fp}")
        sys.exit(1)
    if info["type"] == "npy":
        return np.load(fp, allow_pickle=True).astype(np.float64).flatten()
    elif info["type"] == "csv":
        return pd.read_csv(fp)[info["column"]].values.astype(np.float64)
    else:
        return pd.read_csv(fp, header=None).iloc[:, info["column"]].values.astype(np.float64)


def run_benchmark():
    if args.original:
        sorter = MomentumSort(z=args.z)
        version = "MomentumSort (Original - First Paper)"
    else:
        sorter = MomentumSortOptimal(target_bucket_size=64, z=args.z, parallel=False, count_comparisons=args.count)
        version = "MomentumSortOptimal (Recursive - Second Paper)"

    print(f"🚀 {version}")
    print(f"   n = {args.n:,} | trials = {args.trials} | counting = {args.count} | SHUFFLED = True\n")

    print(f"{'Dataset':<35} {'Residual Comps':>14} {'Max Bucket':>11} {'Ratio':>8} {'Saved':>8}")
    print("-" * 92)

    for name, info in datasets.items():
        data_full = load_data(info)
        comp_list = []
        max_bucket_list = []

        for _ in range(args.trials):
            data = np.random.choice(data_full, min(args.n, len(data_full)), replace=False).copy()
            np.random.shuffle(data)
            _, comps = sorter.sort(data)
            comp_list.append(comps)

            ranks = (data - data.min()) / (data.max() - data.min() + 1e-12)
            m = max(2, int(np.sqrt(args.n))) if args.original else 64
            idx = np.minimum((ranks * (m - 1)).astype(np.int64), m - 1)
            bucket_sizes = np.bincount(idx, minlength=m)
            max_bucket_list.append(int(bucket_sizes.max()))

        avg_comps = int(np.mean(comp_list))
        avg_max = int(np.mean(max_bucket_list))
        classical = args.n * np.log2(args.n)
        ratio = avg_comps / classical if classical > 0 else 0
        saved = round(100 * (1 - ratio), 1)

        print(f"{name:<35} {avg_comps:14,} {avg_max:11,} {ratio:8.3f} {saved:7.1f}%")

    print(f"\n✅ {version} finished")


if __name__ == "__main__":
    run_benchmark()