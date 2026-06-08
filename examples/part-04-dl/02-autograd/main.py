import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print("dy/dx at x=2:", x.grad.item())  # 2*2+3=7
