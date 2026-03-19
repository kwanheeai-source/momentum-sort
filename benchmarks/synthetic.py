# benchmarks/synthetic.py
"""
Synthetic benchmark — now matches real.py output format exactly (no Projection column)
"""

import numpy as np
import argparse

from core.momentumsort import MomentumSort
from core.momentumsortoptimal import MomentumSortOptimal


def parse_args():
    parser = argparse.ArgumentParser(
        description="MomentumSort Synthetic Benchmark",
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


def uniform_gen(n):   return np.random.uniform(0, 1, n)
def normal_gen(n):    return np.random.normal(0, 1, n)
def exp_gen(n):       return np.random.exponential(1, n)
def gamma_gen(n):     return np.random.gamma(0.5, 1, n)
def pareto_gen(n):    return np.random.pareto(2.0, n) + 1
def bimodal_gen(n):   return np.concatenate([np.random.normal(-3,1,n//2), np.random.normal(3,1,n//2)])
def trimodal_gen(n):  return np.concatenate([np.random.normal(-4,0.8,n//3), np.random.normal(0,1.2,n//3), np.random.normal(4,0.9,n//3)])
def overlapping_bimodal_gen(n): return np.concatenate([np.random.normal(-1,1.5,n//2), np.random.normal(1.5,1.8,n//2)])
def many_clusters_gen(n):
    return np.concatenate([np.random.normal(i*1.8, 0.6+i*0.2, n//6) for i in range(6)])

distributions = {
    "Uniform": uniform_gen, "Normal": normal_gen, "Exponential": exp_gen,
    "Gamma (shape=0.5)": gamma_gen, "Pareto (a=2)": pareto_gen,
    "Bimodal": bimodal_gen, "Trimodal": trimodal_gen,
    "Overlapping Bimodal": overlapping_bimodal_gen, "Many Small Clusters": many_clusters_gen,
}


def run_benchmark():
    if args.original:
        sorter = MomentumSort(z=args.z)
        version = "MomentumSort (Original - First Paper)"
    else:
        sorter = MomentumSortOptimal(target_bucket_size=64, z=args.z, parallel=False, count_comparisons=args.count)
        version = "MomentumSortOptimal (Recursive - Second Paper)"

    print(f"🚀 {version}")
    print(f"   n = {args.n:,} | trials = {args.trials} | counting = {args.count} | SHUFFLED = True\n")

    print(f"{'Distribution':<25} {'Residual Comps':>14} {'Max Bucket':>11} {'Ratio':>8} {'Saved':>8}")
    print("-" * 92)

    for name, gen in distributions.items():
        comp_list = []
        max_bucket_list = []

        for _ in range(args.trials):
            data = gen(args.n).copy()
            np.random.shuffle(data)               # consistent with real.py
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

        print(f"{name:<25} {avg_comps:14,} {avg_max:11,} {ratio:8.3f} {saved:7.1f}%")

    print(f"\n✅ {version} finished")


if __name__ == "__main__":
    run_benchmark()