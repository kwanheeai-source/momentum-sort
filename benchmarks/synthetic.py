# benchmarks/synthetic.py — Updated for Unified Geometric MomentumSort
import numpy as np
import argparse

from core.momentumsort import MomentumSort

class CountedItem:
    """Wrapper to count the actual number of comparisons performed during sorting."""
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
    parser.add_argument("--n", "-n", type=int, default=100_000, help="Number of elements to sort")
    parser.add_argument("--trials", "-t", type=int, default=5, help="Number of trials per distribution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

# Generative Sieve: Defining the structural regularities of input data
def uniform_gen(n):   return np.random.uniform(0, 1, n)
def normal_gen(n):    return np.random.normal(0, 1, n)
def exp_gen(n):       return np.random.exponential(1, n)
def gamma_gen(n):     return np.random.gamma(0.5, 1, n)
def pareto_gen(n):    return np.random.pareto(2.0, n) + 1
def bimodal_gen(n):   return np.concatenate([np.random.normal(-3,1,n//2), np.random.normal(3,1,n//2)])
def trimodal_gen(n):  return np.concatenate([np.random.normal(-4,0.8,n//3), np.random.normal(0,1.2,n//3), np.random.normal(4,0.9,n//3)])
def overlapping_bimodal_gen(n): return np.concatenate([np.random.normal(-1,1.5,n//2), np.random.normal(1.5,1.8,n//2)])
def bifurcated_gen(n):
    """Extreme bifurcation: mass at boundaries, empty middle manifold."""
    low_cluster = np.random.normal(0.05, 0.01, n // 2)
    high_cluster = np.random.normal(0.95, 0.01, n // 2)
    return np.clip(np.concatenate([low_cluster, high_cluster]), 0, 1)
def many_clusters_gen(n):
    return np.concatenate([np.random.normal(i*1.8, 0.6+i*0.2, n//6) for i in range(6)])

distributions = {
    "Uniform": uniform_gen,
    "Normal": normal_gen,
    "Exponential": exp_gen,
    "Gamma (shape=0.5)": gamma_gen,
    "Pareto (a=2)": pareto_gen,
    "Bimodal": bimodal_gen,
    "Trimodal": trimodal_gen,
    "Extreme Bifurcation": bifurcated_gen,
    "Overlapping Bimodal": overlapping_bimodal_gen,
    "Many Small Clusters": many_clusters_gen,
}

def count_python_comps(data):
    """Measure the Timsort baseline (maximum ignorance assumption)."""
    data = list(data)
    counter = [0]
    wrapped = [CountedItem(float(v), counter) for v in data]
    sorted(wrapped)
    return counter[0]

def run_benchmark():
    sorter = MomentumSort()
    version = "MomentumSort (Unified Geometric)"

    print(f"🚀 {version}\n")
    print(f"{'Distribution':<25} {'Python Comps':>14} {'Residual':>12} {'Total':>12} {'Saved vs Python':>15}")
    print("-" * 90)

    for name, gen in distributions.items():
        python_list = []
        residual_list = []

        for _ in range(args.trials):
            data = gen(args.n).copy()
            np.random.shuffle(data)

            python_comps = count_python_comps(data)
            # Unified sort returns (sorted_array, comparisons)
            _, residual = sorter.sort(data)

            python_list.append(python_comps)
            residual_list.append(residual)

        avg_python = int(np.mean(python_list))
        avg_residual = int(np.mean(residual_list))
        saved = round(100 * (avg_python - avg_residual) / avg_python, 1)

        print(f"{name:<25} {avg_python:14,} {avg_residual:12,} {avg_residual:12,} {saved:14.1f}%")

    print(f"\n✅ {version} finished")

if __name__ == "__main__":
    run_benchmark()