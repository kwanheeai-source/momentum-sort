# benchmarks/real.py — COMPLETE & FIXED (real Python sorted counting as baseline)
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import sys

from core.momentumsort import MomentumSort
from core.momentumsortoptimal import MomentumSortOptimal


# CountedItem (local copy so benchmark works without import issues)
class CountedItem:
    def __init__(self, value: float, counter: list):
        self.value = value
        self.counter = counter

    def __lt__(self, other): self.counter[0] += 1; return self.value < other.value
    def __le__(self, other): self.counter[0] += 1; return self.value <= other.value
    def __gt__(self, other): self.counter[0] += 1; return self.value > other.value
    def __ge__(self, other): self.counter[0] += 1; return self.value >= other.value
    def __eq__(self, other): self.counter[0] += 1; return self.value == other.value


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
    else:  # abalone
        return pd.read_csv(fp, header=None).iloc[:, info["column"]].values.astype(np.float64)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n", "-n", type=int, default=100_000)
    parser.add_argument("--trials", "-t", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--count", action="store_true")
    return parser.parse_args()


args = parse_args()
np.random.seed(args.seed)


def count_python_comps(data):
    """Run Python's real sorted() and count every comparison on this exact data."""
    data = list(data)
    counter = [0]
    wrapped = [CountedItem(float(v), counter) for v in data]
    sorted(wrapped)
    return counter[0]


def run_benchmark():
    if args.original:
        sorter = MomentumSort(z=args.z)
        version = "MomentumSort (Original)"
    else:
        sorter = MomentumSortOptimal(target_leaf_size=args.k, z=args.z, count_comparisons=args.count)
        version = f"MomentumSortOptimal (leaf={args.k})"

    print(f"🚀 {version} — Real Datasets\n")
    print(f"{'Dataset':<35} {'Python Comps':>14} {'Residual':>12} {'Bucket':>10} {'Total':>12} {'Saved vs Python':>15}")
    print("-" * 110)

    for name, info in datasets.items():
        python_list = []
        residual_list = []
        bucket_list = []
        total_list = []

        for _ in range(args.trials):
            data = load_data(info)[:args.n].copy()
            np.random.shuffle(data)

            python_comps = count_python_comps(data)          # ← real counted Python baseline
            result = sorter.sort(data)

            if len(result) == 4:
                _, residual, bucket_cost, total = result
            else:
                _, residual = result
                bucket_cost = 0
                total = residual

            python_list.append(python_comps)
            residual_list.append(residual)
            bucket_list.append(bucket_cost)
            total_list.append(total)

        avg_python = int(np.mean(python_list))
        avg_residual = int(np.mean(residual_list))
        avg_bucket = int(np.mean(bucket_list))
        avg_total = int(np.mean(total_list))

        saved = round(100 * (avg_python - avg_total) / avg_python, 1)

        print(f"{name:<35} {avg_python:14,} {avg_residual:12,} {avg_bucket:10,} {avg_total:12,} {saved:14.1f}%")

    print(f"\n✅ {version} finished")


if __name__ == "__main__":
    run_benchmark()