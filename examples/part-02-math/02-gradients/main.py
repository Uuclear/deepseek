import numpy as np

def f(x):
    return x[0]**2 + 3 * x[1]**2

def grad_f(x):
    return np.array([2 * x[0], 6 * x[1]])

x = np.array([3.0, 2.0])
lr = 0.1
for step in range(10):
    g = grad_f(x)
    x = x - lr * g
    print(f"step {step+1}: x={x}, f={f(x):.4f}")
