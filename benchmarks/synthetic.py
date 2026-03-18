# benchmarks/synthetic.py
"""
Synthetic benchmark for MomentumSort paper
Reproduces the exact synthetic table shown in Section 6.1

Usage examples:
  python -m benchmarks.synthetic                    # default n=100000
  python -m benchmarks.synthetic --n 200000 --trials 10
  python -m benchmarks.synthetic --n 100000 --z 3.0 --seed 123
  python -m benchmarks.synthetic --n 500000 --output results.csv
"""

import numpy as np
import argparse
import scipy.special as sp
import sys
from pathlib import Path
from core.momentumsort import MomentumSort

# ====================== ARGUMENT PARSER ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MomentumSort synthetic benchmark (paper Section 6.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n", "-n", type=int, default=100_000,
                        help="Number of elements per run")
    parser.add_argument("--trials", "-t", type=int, default=10,
                        help="Number of trials for averaging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--z", type=float, default=2.0,
                        help="Statistical threshold (z-score) for curvature degeneracy test")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results as CSV (e.g. results.csv)")
    return parser.parse_args()

args = parse_args()

# ====================== SETUP ======================
np.random.seed(args.seed)

# ====================== DISTRIBUTIONS ======================
def uniform_gen(n):   return np.random.uniform(0, 1, n)
def normal_gen(n):    return np.random.normal(0, 1, n)
def exp_gen(n):       return np.random.exponential(1, n)
def gamma_gen(n):     return np.random.gamma(0.5, 1, n)
def pareto_gen(n):    return np.random.pareto(2.0, n) + 1

distributions = {
    "Uniform":          uniform_gen,
    "Normal":           normal_gen,
    "Exponential":      exp_gen,
    "Gamma (shape=0.5)": gamma_gen,
    "Pareto (a=2)":     pareto_gen,
}

# ====================== RESIDUAL PROXY (now uses --z) ======================
def compute_residual_proxy(gen_func, n=args.n, trials=args.trials, z=args.z):
    ms = MomentumSort(z=z)                    # pass the z value to the class
    c_res_list = []
    max_bucket_list = []
    proj_list = []

    for _ in range(trials):
        data = gen_func(n)

        # Run the actual sort (triggers projection)
        _ = ms.sort(data)

        # Recompute ranks exactly as the class does (for bucket analysis)
        mu = data.mean()
        sigma = data.std(ddof=0)
        g1 = float(np.mean(((data - mu) / sigma) ** 3)) if sigma > 0 else 0.0
        se = np.sqrt(6.0 / n)

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

        m = max(2, int(np.sqrt(n)))
        idx = np.minimum((ranks * (m - 1)).astype(np.int64), m - 1)
        bucket_sizes = np.bincount(idx, minlength=m)
        ni = bucket_sizes[bucket_sizes > 0].astype(float)

        c_res = np.sum(ni * np.log2(ni))
        c_res_list.append(c_res)
        max_bucket_list.append(int(bucket_sizes.max()))
        proj_list.append(proj_type)

    avg_c_res = np.mean(c_res_list)
    avg_max = np.mean(max_bucket_list)
    c_full = n * np.log2(n)
    ratio = avg_c_res / c_full

    return {
        "projection": proj_list[0],
        "avg_max_bucket": int(avg_max),
        "ratio": round(ratio, 3),
        "saved": round((1 - ratio) * 100, 1)
    }

# ====================== MAIN ======================
if __name__ == "__main__":
    print(f"🚀 MomentumSort Synthetic Benchmark")
    print(f"   n      = {args.n:,}")
    print(f"   trials = {args.trials}")
    print(f"   seed   = {args.seed}")
    print(f"   z      = {args.z}\n")

    print(f"{'Distribution':<25} {'Projection':<10} {'Max Bucket':>10} {'Ratio':>8} {'Saved':>8}")
    print("-" * 72)

    results = {}
    for name, gen in distributions.items():
        res = compute_residual_proxy(gen)
        print(f"{name:<25} {res['projection']:<10} {res['avg_max_bucket']:10,} "
              f"{res['ratio']:8.3f} {res['saved']:7.1f}%")
        results[name] = res

    # Optional CSV output
    if args.output:
        import pandas as pd
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "Distribution"
        df.to_csv(args.output)
        print(f"\n✅ Results saved to {args.output}")

    print("\n✅ Benchmark complete! Copy the table above into your paper.")