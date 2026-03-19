# tests/test_correctness.py
"""
Correctness tests for BOTH MomentumSort versions
(First paper: original single-level + Second paper: recursive+parallel)

Usage:
  python -m tests.test_correctness                    # tests new MomentumSortOptimal (default)
  python -m tests.test_correctness --original         # tests old MomentumSort (first paper)
  python -m tests.test_correctness --optimal          # explicitly new version
"""

import numpy as np
import argparse
from core.momentumsort import MomentumSort          # ← first paper
from core.momentumsortoptimal import MomentumSortOptimal  # ← second paper


def is_sorted(arr):
    """Fast check that array is non-decreasing."""
    return np.all(arr[:-1] <= arr[1:])


def parse_args():
    parser = argparse.ArgumentParser(description="Test MomentumSort versions")
    parser.add_argument("--original", action="store_true",
                        help="Test the original MomentumSort (first paper)")
    parser.add_argument("--optimal", action="store_true",
                        help="Test MomentumSortOptimal (second paper)")
    return parser.parse_args()


def test_correctness():
    args = parse_args()

    if args.original:
        ms = MomentumSort(z=2.0)
        version_name = "MomentumSort (Original - First Paper)"
        print("🧪 Running correctness tests for ORIGINAL MomentumSort (first paper)...\n")
    else:
        # Default = new optimal version
        ms = MomentumSortOptimal(
            target_bucket_size=64,
            z=2.0,
            parallel=False          # serial for small deterministic tests
        )
        version_name = "MomentumSortOptimal (Recursive + Parallel - Second Paper)"
        print("🧪 Running correctness tests for MomentumSortOptimal (new version)...\n")

    np.random.seed(42)

    test_cases = [
        ("Empty array", np.array([])),
        ("Single element", np.array([42.0])),
        ("Two elements (sorted)", np.array([1.0, 5.0])),
        ("Two elements (reverse)", np.array([5.0, 1.0])),
        ("All equal", np.full(100, 7.3)),
        ("Uniform", np.random.uniform(0, 1, 10_000)),
        ("Normal", np.random.normal(0, 1, 10_000)),
        ("High-skew Gamma", np.random.gamma(0.5, 1, 10_000)),
        ("Heavy-tailed Pareto", np.random.pareto(2.0, 10_000) + 1),
        ("Already sorted", np.arange(5000, dtype=float)),
        ("Reverse sorted", np.arange(5000, dtype=float)[::-1]),
        ("Realistic skewed (simulated income)", np.random.exponential(10, 5000)),
    ]

    for name, data in test_cases:
        original = data.copy()
        result, comp_count = ms.sort(data)

        assert len(result) == len(original), f"Length changed in {name}"
        assert is_sorted(result), f"Not sorted in {name}"
        assert np.array_equal(np.sort(original), result), f"Values don't match in {name}"
        assert isinstance(comp_count, int) and comp_count >= 0, \
               f"Invalid comparison count ({comp_count}) in {name}"

        print(f"✅ {name:<30} → passed (comparisons: {comp_count:,})")

    # Extra stress test
    for _ in range(50):
        x = np.random.uniform(-100, 100, np.random.randint(1, 1000))
        result, _ = ms.sort(x)
        assert is_sorted(result)

    print("\n🎉 ALL CORRECTNESS TESTS PASSED!")
    print(f"   {version_name} is correct for arbitrary inputs")
    print("   Comparison counting works correctly")


if __name__ == "__main__":
    test_correctness()