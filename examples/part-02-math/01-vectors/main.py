import numpy as np
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
print("点积 u·v =", u @ v)
print("范数 ||u|| =", np.linalg.norm(u))
A = np.array([[1, 2], [3, 4]])
print("A 的转置:\n", A.T)
print("det(A) =", np.linalg.det(A))
