import numpy as np
vocab = {"我": 0, "爱": 1, "AI": 2}
emb = np.random.randn(len(vocab), 4) * 0.1
sentence = ["我", "爱", "AI"]
vectors = [emb[vocab[w]] for w in sentence]
print("句子向量 shape:", np.stack(vectors).shape)
print("平均池化:", np.mean(vectors, axis=0).round(3))
