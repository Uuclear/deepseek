import numpy as np
rng = np.random.default_rng(42)
rolls = rng.integers(1, 7, size=10000)
counts = np.bincount(rolls)[1:]
probs = counts / counts.sum()
print("骰子 1-6 频率:", np.round(probs, 3))
print("理论 1/6 =", round(1/6, 3))
