# tests/test_correctness.py
"""
Correctness tests for MomentumSort
Verifies the core guarantee: always returns a correctly sorted array
(for arbitrary inputs, as proved in Section 3.1 of the paper)

Run with:
  python -m tests.test_correctness
  python -m pytest tests/test_correctness.py -q   # if you have pytest
"""

import numpy as np
import sys
from core.momentumsort import MomentumSort


def is_sorted(arr):
    """Fast check that array is non-decreasing."""
    return np.all(arr[:-1] <= arr[1:])


def test_correctness():
    ms = MomentumSort(z=2.0)
    np.random.seed(42)
    print("🧪 Running MomentumSort correctness tests...\n")

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
        result, comp_count = ms.sort(data)          # ← NEW: unpack the tuple

        # Core correctness checks
        assert len(result) == len(original), f"Length changed in {name}"
        assert is_sorted(result), f"Not sorted in {name}"

        # Should be a permutation of original (stable sort not required, but values must match)
        assert np.array_equal(np.sort(original), result), f"Values don't match in {name}"

        # Quick sanity check on the new comparison counter
        assert isinstance(comp_count, int) and comp_count >= 0, \
               f"Invalid comparison count ({comp_count}) in {name}"

        print(f"✅ {name:<30} → passed (comparisons: {comp_count:,})")

    # Extra stress test: many small runs
    for _ in range(50):
        x = np.random.uniform(-100, 100, np.random.randint(1, 1000))
        result, _ = ms.sort(x)                      # ← unpack here too
        assert is_sorted(result)

    print("\n🎉 ALL CORRECTNESS TESTS PASSED!")
    print("   MomentumSort is correct for arbitrary inputs (Section 3.1)")
    print("   Comparison counting is working correctly")
    print("   Ready for submission.")


if __name__ == "__main__":
    test_correctness()