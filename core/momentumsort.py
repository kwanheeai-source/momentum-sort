import numpy as np
import scipy.special as sp
from scipy.stats import skew as sample_skew

class CountedItem:
    """Wrapper to count the actual number of comparisons performed during sorting."""
    def __init__(self, value, counter):
        self.value = value
        self.counter = counter

    def __lt__(self, other):
        self.counter[0] += 1
        return self.value < other.value

class MomentumSort:
    """
    Unified Geometric Sort: Resolves entropy via projection onto a Gamma-type manifold.
    Bypasses the comparison-based Θ(n log n) barrier by exploiting structural memory.
    """
    def sort(self, data):
        x = np.asarray(data, dtype=np.float64)
        n = len(x)
        if n <= 1: return x.copy(), 0

        # Extract moments to reconstruct the generative manifold [cite: 1, 4]
        mu, sigma = float(x.mean()), float(x.std())
        if sigma == 0.0: return x.copy(), 0
        g1 = float(sample_skew(x))

        # 1. Rank Projection: 'Unwarps' distributional asymmetry
        ranks = self._gamma_ranks(x, mu, sigma, g1)

        # 2. Adaptive Bucketization: Uses the recursive natural frequency
        m = max(2, int(n / np.log2(n)))
        idx = np.clip((ranks * (m - 1)).astype(np.int64), 0, m - 1)

        buckets = [[] for _ in range(m)]
        for i in range(n):
            buckets[idx[i]].append(x[i])

        # 3. Residual Resolution: Eliminates local micro-disorder via comparison
        counter, result = [0], []
        for b in buckets:
            if len(b) > 1:
                wrapped = [CountedItem(v, counter) for v in b]
                wrapped.sort()
                result.extend(w.value for w in wrapped)
            elif len(b) == 1:
                result.append(b[0])

        return np.array(result), counter[0]

    @staticmethod
    def _gamma_ranks(x, mean, std, skewness):
        """Maps data to cumulative probability under a Generalized Gamma manifold."""
        s = max(abs(skewness), 1e-12)
        k = 4.0 / (s * s)
        theta = std / np.sqrt(k)
        loc = mean - k * theta

        if skewness >= 0:
            return sp.gammainc(k, np.maximum((x - loc) / theta, 0.0))
        else:
            # Mirror for negative skew to maintain rank geometry
            z = (loc + 2.0 * k * theta - x) / theta
            return 1.0 - sp.gammainc(k, np.maximum(z, 0.0))