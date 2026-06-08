import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
X = torch.randn(200, 1)
y = 3 * X.squeeze() + 0.5 * torch.randn(200)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

big = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
small = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))

def fit(model, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            opt.step()
    with torch.no_grad():
        mse = loss_fn(model(X).squeeze(), y).item()
    return mse

print("大模型 train MSE:", round(fit(big), 4))
print("小模型 train MSE:", round(fit(small), 4))
