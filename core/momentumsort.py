import numpy as np
import scipy.special as sp
from scipy.stats import skew as sample_skew


class ComparisonCounter:
    """
    Clean class-based comparison counter.
    No globals, no risky side effects.
    """
    def __init__(self):
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class CountedItem:
    """
    Transparent wrapper that counts EVERY comparison.
    Only used inside the residual bucket sorting phase.
    """
    def __init__(self, value, counter: ComparisonCounter):
        self.value = value
        self.counter = counter

    def __lt__(self, other):
        self.counter.count += 1
        return self.value < other.value

    def __le__(self, other):
        self.counter.count += 1
        return self.value <= other.value

    def __gt__(self, other):
        self.counter.count += 1
        return self.value > other.value

    def __ge__(self, other):
        self.counter.count += 1
        return self.value >= other.value

    def __eq__(self, other):
        self.counter.count += 1
        return self.value == other.value


class MomentumSort:
    """
    MomentumSort: Distribution-Aware Sorting via Entropy Projection
    Reference implementation from the paper (Kwanhee Lee, 2026).

    Now returns (sorted_array, comparison_count)
    Only the local bucket sorts are counted (exactly the residual disorder part).
    Bucket assignment remains comparison-free.
    """
    def __init__(self, bucket_policy=None, z: float = 2.0):
        self.bucket_policy = bucket_policy or (lambda n: max(4, int(np.sqrt(n))))
        self.z = float(z)

    def sort(self, data):
        """Main sorting method (API updated only for the return value)."""
        x = np.asarray(data, dtype=np.float64)
        n = x.size
        if n <= 1:
            return x.copy(), 0

        mu = x.mean()
        sigma = x.std(ddof=0)
        if sigma == 0.0:
            return x.copy(), 0

        g1 = float(sample_skew(x))
        se_g1 = np.sqrt(6.0 / n)

        if abs(g1) <= self.z * se_g1:
            ranks = self._linear_ranks(x)
        else:
            ranks = self._gamma_ranks(x, mu, sigma, g1)

        return self._bucket_sort_counted(x, ranks, n)

    def _bucket_sort_counted(self, x, ranks, n):
        """
        Vectorised bucket assignment (0 comparisons) + counted local sorting.
        """
        m = int(self.bucket_policy(n))
        m = max(2, m)

        # Bucket assignment — still pure arithmetic (no data comparisons)
        idx = np.minimum((ranks * (m - 1)).astype(np.int64), m - 1)

        # Group into Python lists for counted sorting
        buckets: list[list[float]] = [[] for _ in range(m)]
        for i, b_idx in enumerate(idx):
            buckets[b_idx].append(float(x[i]))

        # Counted residual sorting
        counter = ComparisonCounter()
        result = []
        with counter:
            for bucket in buckets:
                if len(bucket) > 1:
                    wrapped = [CountedItem(v, counter) for v in bucket]
                    sorted_bucket = sorted(wrapped)
                    result.extend(item.value for item in sorted_bucket)
                elif bucket:
                    result.append(bucket[0])

        return np.array(result, dtype=np.float64), counter.count

    @staticmethod
    def _linear_ranks(x):
        """Original linear projection."""
        lo, hi = x.min(), x.max()
        if hi == lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    @staticmethod
    def _gamma_ranks(x, mean, std, skewness):
        """Original gamma projection (exactly as in the paper)."""
        s = max(abs(skewness), 1e-12)
        k = 4.0 / (s * s)
        theta = std / np.sqrt(k)
        loc = mean - k * theta

        if skewness >= 0:
            z = (x - loc) / theta
            return sp.gammainc(k, np.maximum(z, 0.0))
        else:
            z = (loc + 2.0 * k * theta - x) / theta
            return 1 - sp.gammainc(k, np.maximum(z, 0.0))


# For easy import / backward compatibility in benchmarks
__all__ = ["MomentumSort"]


