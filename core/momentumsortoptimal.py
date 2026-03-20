"""
core/momentumsortoptimal.py — Clean version (no CV/stats)
"""

import numpy as np
import scipy.special as sp
from scipy.stats import skew as sample_skew
from typing import List
import sys

sys.setrecursionlimit(10000)


class CountedItem:
    def __init__(self, value: float, counter: List[int]):
        self.value = value
        self.counter = counter

    def __lt__(self, other): self.counter[0] += 1; return self.value < other.value
    def __le__(self, other): self.counter[0] += 1; return self.value <= other.value
    def __gt__(self, other): self.counter[0] += 1; return self.value > other.value
    def __ge__(self, other): self.counter[0] += 1; return self.value >= other.value
    def __eq__(self, other): self.counter[0] += 1; return self.value == other.value


class MomentumSortOptimal:
    def __init__(self,
                 target_bucket_size: int = 64,
                 z: float = 2.0,
                 count_comparisons: bool = False):
        self.target_bucket_size = int(target_bucket_size)
        self.z = float(z)
        self.count_comparisons = count_comparisons

    def sort(self, data):
        x = np.asarray(data, dtype=np.float64)
        if len(x) <= 1:
            return x.copy(), 0

        counter = [0]
        sorted_list = self._recursive_sort(x, counter)
        return np.array(sorted_list, dtype=np.float64), counter[0] if self.count_comparisons else 0

    def _recursive_sort(self, x: np.ndarray, counter: List[int]) -> List[float]:
        n = len(x)

        if n <= self.target_bucket_size:
            if self.count_comparisons:
                wrapped = [CountedItem(float(v), counter) for v in x]
                return [item.value for item in sorted(wrapped)]
            return sorted(x)

        # Projection + split
        mu = float(x.mean())
        sigma = float(x.std(ddof=0))
        if sigma == 0:
            return list(x)

        g1 = float(sample_skew(x))
        se_g1 = np.sqrt(6.0 / n)
        ranks = self._linear_ranks(x) if abs(g1) <= self.z * se_g1 else self._gamma_ranks(x, mu, sigma, g1)

        m = max(2, n // self.target_bucket_size)
        idx = np.minimum((ranks * (m - 1)).astype(np.int64), m - 1)

        buckets = [[] for _ in range(m)]
        for i, b_idx in enumerate(idx):
            buckets[b_idx].append(float(x[i]))

        if max(len(b) for b in buckets) == n:
            if self.count_comparisons:
                wrapped = [CountedItem(float(v), counter) for v in x]
                return [item.value for item in sorted(wrapped)]
            return sorted(x)

        result = []
        for b in buckets:
            if b:
                result.extend(self._recursive_sort(np.array(b), counter))

        return result

    @staticmethod
    def _linear_ranks(x):
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)

    @staticmethod
    def _gamma_ranks(x, mean, std, skewness):
        s = max(abs(skewness), 1e-12)
        k = 4.0 / (s * s)
        theta = std / np.sqrt(k)
        loc = mean - k * theta
        if skewness >= 0:
            z = (x - loc) / theta
            return sp.gammainc(k, np.maximum(z, 0.0))
        else:
            z = (loc + 2.0 * k * theta - x) / theta
            return 1.0 - sp.gammainc(k, np.maximum(z, 0.0))