import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

model = MLP()
x = torch.randn(4, 1, 28, 28)
print("logits shape:", model(x).shape)
