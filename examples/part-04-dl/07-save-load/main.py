import torch
import torch.nn as nn
from pathlib import Path

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

model = Tiny()
path = Path(__file__).parent / "tiny.pt"
torch.save({"state_dict": model.state_dict(), "arch": "Tiny"}, path)
ckpt = torch.load(path, weights_only=False)
model2 = Tiny()
model2.load_state_dict(ckpt["state_dict"])
print("loaded ok, sample output shape:", model2(torch.randn(1, 10)).shape)
