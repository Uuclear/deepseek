"""简化 Transformer 块：LayerNorm + FFN。"""
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return x

x = torch.randn(2, 8, 64)
print("out shape:", TransformerBlock()(x).shape)
