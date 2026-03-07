# benchmarks/real.py
"""
Real-world benchmark for MomentumSort paper
Reproduces the exact real-world table shown in Section 6.2

Usage examples:
  python -m benchmarks.real                    # default n=100000
  python -m benchmarks.real --n 200000 --trials 10
  python -m benchmarks.real --z 1.96 --output real_results.csv
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from core.momentumsort import MomentumSort


# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MomentumSort real-world benchmark (paper Section 6.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n", "-n", type=int, default=100_000,
                        help="Number of elements per run (subsampled if needed)")
    parser.add_argument("--trials", "-t", type=int, default=5,
                        help="Number of trials for averaging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--z", type=float, default=2.0,
                        help="Statistical threshold (z-score) for curvature degeneracy test")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results as CSV")
    return parser.parse_args()


args = parse_args()

# ====================== SETUP ======================
np.random.seed(args.seed)
DATA_DIR = Path(__file__).parent.parent / "data"

# ====================== REAL DATASETS ======================
datasets = {
    "Abalone (whole_weight)": {
        "file": DATA_DIR / "abalone.data",
        "column": 4,  # whole_weight (index 4 in the raw file)
        "name": "whole_weight"
    },
    "Wine Quality (alcohol)": {
        "file": DATA_DIR / "winequality-red.csv",
        "column": "alcohol",
        "name": "alcohol"
    },
    "California Housing (MedInc)": {
        "file": DATA_DIR / "california_medinc.csv",
        "column": "median_income",
        "name": "MedInc"
    },
}


# ====================== LOAD DATA ======================
def load_column(file_path, column):
    if not file_path.exists():
        print(f"❌ Missing {file_path}")
        print("   Run: python -m data.download  first!")
        sys.exit(1)

    if file_path.suffix == ".data":  # Abalone
        df = pd.read_csv(file_path, header=None)
        data = df.iloc[:, column].values.astype(float)
    else:  # CSV with header
        df = pd.read_csv(file_path)
        data = df[column].values.astype(float)

    return data[~np.isnan(data)]  # drop any NaNs just in case


# ====================== RESIDUAL PROXY (same logic as synthetic) ======================
def compute_residual_proxy(data_full, n=args.n, trials=args.trials, z=args.z):
    ms = MomentumSort(z=z)
    c_res_list = []
    max_bucket_list = []
    proj_list = []

    for _ in range(trials):
        # Subsample without replacement
        data = np.random.choice(data_full, size=min(n, len(data_full)), replace=False)
        N = len(data)  # FIXED: Use actual size

        # Run the actual sort (unchanged)
        _ = ms.sort(data)

        # Recompute ranks for bucket analysis
        mu = data.mean()
        sigma = data.std(ddof=0)
        g1 = float(np.mean(((data - mu) / sigma) ** 3)) if sigma > 0 else 0.0
        se = np.sqrt(6.0 / N)  # FIXED: Use N

        if abs(g1) <= z * se:
            ranks = (data - data.min()) / (data.max() - data.min() + 1e-12)
            proj_type = "Linear"
        else:
            s = max(abs(g1), 1e-12)
            k = 4.0 / (s * s)
            theta = sigma / np.sqrt(k)
            loc = mu - k * theta
            if g1 >= 0:
                z_vals = (data - loc) / theta
                ranks = sp.gammainc(k, np.maximum(z_vals, 0.0))
            else:
                z_vals = (loc + 2.0 * k * theta - data) / theta
                ranks = 1 - sp.gammainc(k, np.maximum(z_vals, 0.0))
            proj_type = "Gamma"

        m = max(2, int(np.sqrt(N)))  # FIXED: Use N
        idx = np.minimum((ranks * (m - 1)).astype(np.int64), m - 1)
        bucket_sizes = np.bincount(idx, minlength=m)
        ni = bucket_sizes[bucket_sizes > 0].astype(float)

        c_res = np.sum(ni * np.log2(ni))
        c_res_list.append(c_res)
        max_bucket_list.append(int(bucket_sizes.max()))
        proj_list.append(proj_type)

    avg_c_res = np.mean(c_res_list)
    avg_max = np.mean(max_bucket_list)
    c_full = N * np.log2(N) if N > 1 else 0.0  # FIXED: Use N
    ratio = avg_c_res / c_full if c_full > 0 else 0.0

    return {
        "projection": proj_list[0],
        "avg_max_bucket": int(avg_max),
        "ratio": round(ratio, 3),
        "saved": round((1 - ratio) * 100, 1)
    }

# ====================== MAIN ======================
if __name__ == "__main__":
    print(f"🚀 MomentumSort Real-World Benchmark")
    print(f"   n      = {args.n:,}")
    print(f"   trials = {args.trials}")
    print(f"   seed   = {args.seed}")
    print(f"   z      = {args.z}\n")

    print(f"{'Dataset':<30} {'Column':<15} {'Projection':<10} {'Max Bucket':>10} {'Ratio':>8} {'Saved':>8}")
    print("-" * 88)

    results = {}
    import pandas as pd
    import scipy.special as sp  # needed for gamma

    for name, info in datasets.items():
        full_data = load_column(info["file"], info["column"])
        res = compute_residual_proxy(full_data)

        print(f"{name:<30} {info['name']:<15} {res['projection']:<10} "
              f"{res['avg_max_bucket']:10,} {res['ratio']:8.3f} {res['saved']:7.1f}%")

        results[name] = res

    # Optional CSV output
    if args.output:
        import pandas as pd

        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "Dataset"
        df.to_csv(args.output)
        print(f"\n✅ Results saved to {args.output}")

    print("\n✅ Real-world benchmark complete!")
    print("Copy the table above directly into Section 6.2 of the paper.")