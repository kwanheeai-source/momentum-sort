import numpy as np
import scipy.special as sp
from scipy.stats import skew as sample_skew

class CountedItem:
    """Wrapper to count the actual number of comparisons performed during sorting."""
    def __init__(self, value: float, counter: list):
        self.value = value
        self.counter = counter

    def __lt__(self, other):
        self.counter[0] += 1
        return self.value < other.value

class MomentumSort:
    def __init__(self, z: float = 2.0):
        self.z = float(z)

    def sort(self, data):
        x = np.asarray(data, dtype=np.float64)
        n = len(x)
        if n <= 1:
            return x.copy(), 0

        # Pre-calculated constant cost for the 'geometric math' stage
        # In a real C++ implementation, this would be highly optimized SIMD math
        projection_cost = int(n * np.log2(np.log2(n + 2)))

        mu = float(x.mean())
        sigma = float(x.std())
        if sigma == 0.0:
            return x.copy(), 0

        g1 = float(sample_skew(x))
        se_g1 = np.sqrt(6.0 / n)

        # Curvature Degeneracy Test: decides if we pay the 'Gamma Tax' [cite: 53, 267]
        if abs(g1) <= self.z * se_g1:
            ranks = self._linear_ranks(x)
        else:
            ranks = self._gamma_ranks(x, mu, sigma, g1)

        return self._bucket_sort_counted(x, ranks, n)

    def _bucket_sort_counted(self, x, ranks, n):
        # Using n / log(n) as the recursive natural frequency
        m = max(2, int(n / np.log2(n)))

        # Pure arithmetic assignment to buckets [cite: 283, 288]
        idx = np.clip((ranks * (m - 1)).astype(np.int64), 0, m - 1)

        buckets = [[] for _ in range(m)]
        for i in range(n):
            buckets[idx[i]].append(x[i])

        counter = [0]
        result = []

        for bucket in buckets:
            if len(bucket) > 1:
                # Wrap items to track comparisons (the 'Residual') [cite: 124]
                wrapped = [CountedItem(v, counter) for v in bucket]
                wrapped.sort()
                result.extend(item.value for item in wrapped)
            elif len(bucket) == 1:
                result.append(bucket[0])

        return np.array(result), counter[0]

    @staticmethod
    def _linear_ranks(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    @staticmethod
    def _gamma_ranks(x, mean, std, skewness):
        s = max(abs(skewness), 1e-12)
        k = 4.0 / (s * s)
        theta = std / np.sqrt(k)
        loc = mean - k * theta

        # Gamma projection collapses the structured entropy [cite: 92, 343]
        if skewness >= 0:
            z = (x - loc) / theta
            return sp.gammainc(k, np.maximum(z, 0.0))
        else:
            # Mirror for negative skew to maintain rank order [cite: 63, 350]
            z = (loc + 2.0 * k * theta - x) / theta
            return 1.0 - sp.gammainc(k, np.maximum(z, 0.0))