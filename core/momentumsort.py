import numpy as np
import scipy.special as sp
from scipy.stats import skew as sample_skew

class MomentumSort:
    """
    MomentumSort: Distribution-Aware Sorting via Entropy Projection
    Reference implementation from the paper (Kwanhee Lee, 2026).
    """
    def __init__(self, bucket_policy=None, z: float = 2.0):
        self.bucket_policy = bucket_policy or (lambda n: max(4, int(np.sqrt(n))))
        self.z = float(z)

    def sort(self, data):
        """Main sorting method (exactly as in the paper)."""
        x = np.asarray(data, dtype=np.float64)
        n = x.size
        if n <= 1:
            return x.copy()
        mu = x.mean()
        sigma = x.std(ddof=0)
        if sigma == 0.0:
            return x.copy()

        g1 = float(sample_skew(x))
        se_g1 = np.sqrt(6.0 / n)

        if abs(g1) <= self.z * se_g1:
            ranks = self._linear_ranks(x)
        else:
            ranks = self._gamma_ranks(x, mu, sigma, g1)

        return self._bucket_sort(x, ranks, n)

    def _bucket_sort(self, x, ranks, n):
        m = int(self.bucket_policy(n))
        m = max(2, m)
        idx = np.minimum((ranks * (m - 1)).astype(np.int64), m - 1)

        # Fast vectorised bucketing (much faster than original list.append)
        sort_idx = np.argsort(idx)
        sorted_x = x[sort_idx].copy()

        bucket_sizes = np.bincount(idx, minlength=m)
        start = 0
        for size in bucket_sizes:
            if size > 1:
                sorted_x[start:start + size].sort()
            start += size
        return sorted_x

    @staticmethod
    def _linear_ranks(x):
        lo, hi = x.min(), x.max()
        if hi == lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

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
            return 1 - sp.gammainc(k, np.maximum(z, 0.0))