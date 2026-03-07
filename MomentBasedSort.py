import numpy as np
import scipy.special as sp
from scipy.stats import skew as sample_skew

class MomentumSort:
    """
    Minimal momentum-based sort:
    1) estimate low-order structure (mean, std, skew)
    2) project to rank space via CDF (linear if curvature is degenerate; gamma otherwise)
    3) bucket by rank
    4) pay residual entropy locally with Timsort
    """

    def __init__(self, bucket_policy=None, z=2.0):
        # bucket_policy: n -> number of buckets (execution-layer choice)
        self.bucket_policy = bucket_policy or (lambda n: max(4, int(np.sqrt(n))))
        # z: statistical confidence multiplier for “skew is indistinguishable from 0”
        self.z = float(z)

    def sort(self, data):
        x = np.asarray(data, dtype=np.float64)
        n = x.size
        if n <= 1:
            return x

        mu = x.mean()
        sigma = x.std()
        if sigma == 0.0:
            return x.copy()

        g1 = float(sample_skew(x))  # sample skewness

        # Theoretical (sample-size aware) degeneracy test:
        # under mild conditions, SE(skew) ≈ sqrt(6/n). If |g1| <= z*SE, treat curvature as degenerate.
        se_g1 = np.sqrt(6.0 / n)
        if abs(g1) <= self.z * se_g1:
            return self._bucket_sort(x, self._linear_ranks(x), n)
        else:
            return self._bucket_sort(x, self._gamma_ranks(x, mu, sigma, g1), n)

    def _bucket_sort(self, x, ranks, n):
        m = int(self.bucket_policy(n))
        m = max(2, m)

        # ranks in [0,1] -> bucket index in [0, m-1]
        idx = np.minimum((ranks * (m - 1)).astype(np.int64), m - 1)

        buckets = [[] for _ in range(m)]
        for i, v in enumerate(x):
            buckets[idx[i]].append(v)

        out = []
        for b in buckets:
            if b:
                b.sort()           # local residual entropy payment
                out.extend(b)
        return np.asarray(out, dtype=np.float64)

    def _linear_ranks(self, x):
        lo = x.min()
        hi = x.max()
        if hi == lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _gamma_ranks(self, x, mean, std, skewness):
        s = max(abs(skewness), 1e-12)
        k = 4.0 / (s * s)
        theta = std / np.sqrt(k)
        loc = mean - k * theta

        if skewness >= 0:
            z = (x - loc) / theta
            ranks = sp.gammainc(k, np.maximum(z, 0.0))
        else:
            z = (loc + 2.0 * k * theta - x) / theta
            ranks = 1 - sp.gammainc(k, np.maximum(z, 0.0))  # Invert for order preservation

        return ranks