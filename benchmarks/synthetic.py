# benchmarks/synthetic.py — COMPLETE & FIXED (real Python sorted counting)
import numpy as np
import argparse

from core.momentumsort import MomentumSort
from core.momentumsortoptimal import MomentumSortOptimal


# CountedItem must be here too for the baseline
class CountedItem:
    def __init__(self, value: float, counter: list):
        self.value = value
        self.counter = counter

    def __lt__(self, other): self.counter[0] += 1; return self.value < other.value
    def __le__(self, other): self.counter[0] += 1; return self.value <= other.value
    def __gt__(self, other): self.counter[0] += 1; return self.value > other.value
    def __ge__(self, other): self.counter[0] += 1; return self.value >= other.value
    def __eq__(self, other): self.counter[0] += 1; return self.value == other.value


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


def count_python_comps(data):
    """Run Python's real sorted() and count every single comparison."""
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

    print(f"🚀 {version}\n")
    print(f"{'Distribution':<25} {'Python Comps':>14} {'Residual':>12} {'Bucket':>10} {'Total':>12} {'Saved vs Python':>15}")
    print("-" * 110)

    for name, gen in distributions.items():
        python_list = []
        residual_list = []
        bucket_list = []
        total_list = []

        for _ in range(args.trials):
            data = gen(args.n).copy()
            np.random.shuffle(data)

            python_comps = count_python_comps(data)
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

        print(f"{name:<25} {avg_python:14,} {avg_residual:12,} {avg_bucket:10,} {avg_total:12,} {saved:14.1f}%")

    print(f"\n✅ {version} finished")


if __name__ == "__main__":
    run_benchmark()