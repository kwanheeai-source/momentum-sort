"""
core/momentumsortoptimal.py — To be improved for future recursive application
"""

import numpy as np
import scipy.special as sp
from scipy.stats import skew as sample_skew
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import sys

sys.setrecursionlimit(5000)


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
                 target_leaf_size: int = 256,
                 z: float = 2.0,
                 max_workers: int = 8,
                 count_comparisons: bool = False):
        self.target_leaf_size = int(target_leaf_size)
        self.z = float(z)
        self.max_workers = max_workers
        self.count_comparisons = count_comparisons
        self._leaf_sizes = []

    def sort(self, data):
        x = np.asarray(data, dtype=np.float64).copy()
        n = len(x)
        if n <= 1:
            return x, 0

        counter = [0]
        self._leaf_sizes.clear()

        # ==================== 1. PARALLEL MOMENTS ====================
        moments = self._parallel_moments(x)

        # ==================== 2. GAMMA PARAMETERS + BOUNDARIES ====================
        boundaries, from_top = self._compute_boundaries(moments, n)

        # ==================== 3. SPLIT INTO EQUAL CHUNKS + PARALLEL ASSIGNMENT ====================
        chunks = np.array_split(x, self.max_workers)
        buckets = [[] for _ in range(len(boundaries) + 1)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self._assign_chunk, chunk, boundaries, from_top, buckets))
            for f in futures:
                f.result()

        # Record leaf sizes for CV
        self._leaf_sizes = [len(b) for b in buckets if b]

        # ==================== 4. PARALLEL LEAF SORTING ====================
        if self.max_workers > 1 and n > 2000:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._leaf_sort, b, counter) for b in buckets if b]
                result = []
                for f in futures:
                    result.extend(f.result())
        else:
            result = []
            for b in buckets:
                if b:
                    result.extend(self._leaf_sort(b, counter))

        return np.array(result, dtype=np.float64), counter[0] if self.count_comparisons else 0

    def _parallel_moments(self, x: np.ndarray) -> Tuple[float, float, float]:
        """Parallel mean + std + skew"""
        chunks = np.array_split(x, min(3, self.max_workers))
        with ThreadPoolExecutor(max_workers=3) as ex:
            means = list(ex.map(np.mean, chunks))
            stds = list(ex.map(lambda c: np.std(c, ddof=0), chunks))
            skews = list(ex.map(sample_skew, chunks))

        mu = np.mean(means)
        sigma = np.sqrt(np.mean([s**2 for s in stds]))
        g1 = np.mean(skews)
        return mu, sigma, float(g1)

    def _compute_boundaries(self, moments: Tuple[float, float, float], n: int):
        mu, sigma, g1 = moments
        m = max(8, int(np.sqrt(n)))                     # number of buckets

        if abs(g1) <= self.z * np.sqrt(6.0 / n):
            # Linear case
            return np.linspace(0, 1, m + 1)[1:-1], False
        else:
            # Gamma case
            s = max(abs(g1), 1e-12)
            k = 4.0 / (s * s)
            theta = sigma / np.sqrt(k)
            loc = mu - k * theta
            boundaries = sp.gammaincinv(k, np.linspace(0.01, 0.99, m)) * theta + loc
            return boundaries, g1 < 0   # from_top = True if negative skew

    def _assign_chunk(self, chunk, boundaries, from_top, buckets):
        """Fully parallel assignment per chunk"""
        if from_top:
            # Negative skew → assign from high to low
            ranks = 1.0 - np.searchsorted(boundaries[::-1], chunk[::-1]).astype(float) / len(boundaries)
            ranks = 1.0 - ranks[::-1]
        else:
            ranks = np.searchsorted(boundaries, chunk).astype(float) / len(boundaries)

        idx = np.minimum((ranks * len(buckets)).astype(np.int64), len(buckets) - 1)
        for i, b_idx in enumerate(idx):
            buckets[b_idx].append(float(chunk[i]))

    def _leaf_sort(self, bucket: list, counter: List[int]):
        if len(bucket) <= 1:
            return bucket
        if self.count_comparisons:
            wrapped = [CountedItem(v, counter) for v in bucket]
            return [item.value for item in sorted(wrapped)]
        return sorted(bucket)

    def get_leaf_stats(self):
        if not self._leaf_sizes:
            return {"Avg Leaf CV (%)": 0.0, "Min": 0, "Max": 0, "Num": 0}
        sizes = np.array(self._leaf_sizes)
        cv = 100 * sizes.std(ddof=0) / sizes.mean() if sizes.mean() > 0 else 0.0
        return {
            "Avg Leaf CV (%)": round(cv, 2),
            "Min": int(sizes.min()),
            "Max": int(sizes.max()),
            "Num": len(sizes)
        }