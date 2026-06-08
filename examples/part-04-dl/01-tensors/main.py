import torch
x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.ones(2, 2)
print("x+y:", x + y)
print("shape:", x.shape, "dtype:", x.dtype)
