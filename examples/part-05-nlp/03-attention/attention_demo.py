import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

Q = np.array([[1., 0.], [0., 1.]])
K = np.array([[1., 0.], [0., 1.], [1., 1.]])
V = np.array([[1., 0., 0.], [0., 1., 0.], [0.5, 0.5, 1.]])
scores = Q @ K.T / np.sqrt(2)
weights = np.array([softmax(scores[0]), softmax(scores[1])])
out = weights @ V
print("attention weights row0:", weights[0].round(3))
print("output shape:", out.shape)
