import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

print("公平硬币 H =", entropy([0.5, 0.5]))
print("偏置 0.9/0.1 H =", round(entropy([0.9, 0.1]), 3))
