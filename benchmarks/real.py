# benchmarks/real.py — Clean version (no CV, respects your data loading)
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import sys

from core.momentumsort import MomentumSort
from core.momentumsortoptimal import MomentumSortOptimal

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
    else:  # "data" type (abalone)
        return pd.read_csv(fp, header=None).iloc[:, info["column"]].values.astype(np.float64)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n", "-n", type=int, default=100_000)
    parser.add_argument("--trials", "-t", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--count", action="store_true")
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

def run_benchmark():
    if args.original:
        sorter = MomentumSort(z=args.z)
        version = "MomentumSort (Original)"
    else:
        sorter = MomentumSortOptimal(target_bucket_size=args.k, z=args.z, count_comparisons=args.count)
        version = f"MomentumSortOptimal (Recursive, k={args.k})"

    print(f"🚀 {version} — Real Datasets\n")

    print(f"{'Dataset':<35} {'Residual Comps':>14} {'Max Leaf':>9} {'Ratio':>8} {'Saved':>8}")
    print("-" * 95)

    for name, info in datasets.items():
        comp_list = []

        for _ in range(args.trials):
            data = load_data(info)[:args.n].copy()   # take first n elements
            np.random.shuffle(data)
            _, comps = sorter.sort(data)
            comp_list.append(comps)

        avg_comps = int(np.mean(comp_list))
        classical = args.n * np.log2(args.n)
        ratio = avg_comps / classical
        saved = round(100 * (1 - ratio), 1)

        print(f"{name:<35} {avg_comps:14,} {args.k:>9} {ratio:8.3f} {saved:7.1f}%")

    print(f"\n✅ {version} finished")

if __name__ == "__main__":
    run_benchmark()