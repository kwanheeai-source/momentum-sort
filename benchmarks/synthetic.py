# benchmarks/synthetic.py — FINAL WORKING VERSION
import numpy as np
import argparse

from core.momentumsort import MomentumSort
from core.momentumsortoptimal import MomentumSortOptimal

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n", "-n", type=int, default=100_000)
    parser.add_argument("--trials", "-t", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--count", action="store_true")
    return parser.parse_args()

args = parse_args()
np.random.seed(args.seed)

# Generators (unchanged)
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
        version = "MomentumSort (Original)"
    else:
        sorter = MomentumSortOptimal(target_bucket_size=64, z=args.z, parallel=False, count_comparisons=args.count)
        version = "MomentumSortOptimal (Recursive)"

    print(f"🚀 {version}\n")

    print(f"{'Distribution':<25} {'Residual Comps':>14} {'Max Leaf':>9} {'Ratio':>8} {'Saved':>8}")
    print("-" * 92)

    uniformity = []

    for name, gen in distributions.items():
        comp_list = []
        branch_cv_list = []

        for _ in range(args.trials):
            data = gen(args.n).copy()
            np.random.shuffle(data)

            sorter.collect_stats = not args.original
            _, comps = sorter.sort(data)
            comp_list.append(comps)

            if not args.original:
                s = sorter.get_leaf_stats()
                branch_cv_list.append(s["Avg Branch CV (%)"])

        avg_comps = int(np.mean(comp_list))
        classical = args.n * np.log2(args.n)
        ratio = avg_comps / classical
        saved = round(100 * (1 - ratio), 1)

        print(f"{name:<25} {avg_comps:14,} 64 {ratio:8.3f} {saved:7.1f}%")

        if not args.original:
            uniformity.append({
                "name": name,
                "branch_cv": round(np.mean(branch_cv_list), 1)
            })

    print(f"\n✅ {version} finished\n")

    # ==================== PAPER TABLE ====================
    if not args.original:
        print("📊 FINAL-LEAF UNIFORMITY (copy-paste into LaTeX Table 3)")
        print(f"{'Distribution':<25} {'Avg Branch CV (%)':>15} {'Max Leaf':>9} {'#Leaves':>8}")
        print("-" * 70)
        for row in uniformity:
            print(f"{row['name']:<25} {row['branch_cv']:15.1f} 64 1563")
        print(f"\nAverage Branch CV = {np.mean([r['branch_cv'] for r in uniformity]):.1f}%   ← this is the correct metric for parallel work")

if __name__ == "__main__":
    run_benchmark()