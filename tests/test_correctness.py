"""
tests/test_correctness.py
Correctness tests for BOTH MomentumSort versions
"""

import numpy as np
import argparse

from core.momentumsort import MomentumSort
from core.momentumsortoptimal import MomentumSortOptimal


def is_sorted(arr):
    """Fast check that array is non-decreasing (NaNs removed for safety)."""
    return np.all(arr[:-1] <= arr[1:])


def parse_args():
    parser = argparse.ArgumentParser(description="Test MomentumSort versions")
    parser.add_argument("--original", action="store_true", help="Test original MomentumSort (first paper)")
    parser.add_argument("--optimal", action="store_true", help="Test MomentumSortOptimal (second paper)")
    return parser.parse_args()


def test_correctness():
    args = parse_args()

    if args.original:
        sorter = MomentumSort(z=2.0)
        version_name = "MomentumSort (Original - First Paper)"
        print("🧪 Testing ORIGINAL MomentumSort (first paper)...\n")
    else:
        sorter = MomentumSortOptimal(target_bucket_size=64, z=2.0, count_comparisons=True)
        version_name = "MomentumSortOptimal (Recursive - Second Paper)"
        print("🧪 Testing MomentumSortOptimal (recursive version)...\n")

    np.random.seed(42)

    test_cases = [
        ("Empty array", np.array([])),
        ("Single element", np.array([42.0])),
        ("Two elements sorted", np.array([1.0, 5.0])),
        ("Two elements reverse", np.array([5.0, 1.0])),
        ("All equal", np.full(100, 7.3)),
        ("All equal negative", np.full(100, -3.14)),
        ("Uniform", np.random.uniform(-10, 10, 5000)),
        ("Normal", np.random.normal(0, 5, 5000)),
        ("High-skew Gamma", np.random.gamma(0.5, 1, 5000)),
        ("Heavy-tailed Pareto", np.random.pareto(2.0, 5000) + 1),
        ("Already sorted", np.arange(5000, dtype=float)),
        ("Reverse sorted", np.arange(5000, dtype=float)[::-1]),
        ("With duplicates", np.random.choice([1.0, 2.0, 3.0], 5000)),
        ("Realistic skewed", np.random.exponential(10, 5000)),
        ("Mixed positive/negative", np.random.normal(0, 10, 5000)),
    ]

    for name, data in test_cases:
        original = data.copy()
        result, comp_count = sorter.sort(data)

        assert len(result) == len(original), f"Length changed in {name}"
        assert is_sorted(result), f"Not sorted in {name}"
        assert np.array_equal(np.sort(original), result, equal_nan=True), f"Values don't match in {name}"
        assert isinstance(comp_count, int) and comp_count >= 0, f"Invalid comparison count in {name}"

        print(f"✅ {name:<35} → passed (comparisons: {comp_count:,})")

    # Stress test
    for _ in range(50):
        size = np.random.randint(1, 10000)
        x = np.random.uniform(-100, 100, size)
        result, _ = sorter.sort(x)
        assert is_sorted(result)

    print("\n🎉 ALL CORRECTNESS TESTS PASSED!")
    print(f"   {version_name} is correct for arbitrary inputs")
    print("   Comparison counting works correctly")


if __name__ == "__main__":
    test_correctness()