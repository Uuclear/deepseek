import numpy as np

y_true = np.array([1, 0, 1, 1])
y_prob = np.array([0.9, 0.2, 0.8, 0.4])

bce = -np.mean(y_true * np.log(y_prob + 1e-9) + (1 - y_true) * np.log(1 - y_prob + 1e-9))
mse = np.mean((y_true - y_prob) ** 2)
print("二元交叉熵:", round(bce, 4))
print("MSE:", round(mse, 4))
