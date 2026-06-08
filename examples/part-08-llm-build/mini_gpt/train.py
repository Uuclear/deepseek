"""迷你 GPT 预训练：Next Token Prediction。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import MiniGPT

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "data" / "corpus_zh_en.txt"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int = 64):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()

    text = CORPUS.read_text(encoding="utf-8")
    ds = CharDataset(text, block_size=args.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniGPT(vocab_size=ds.vocab_size, max_seq_len=args.block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, opt, device)
        print(f"epoch {epoch}/{args.epochs} loss={loss:.4f} device={device}")

    ckpt = {
        "model_state": model.state_dict(),
        "stoi": ds.stoi,
        "itos": ds.itos,
        "block_size": args.block_size,
        "vocab_size": ds.vocab_size,
    }
    path = CKPT_DIR / "mini_gpt.pt"
    torch.save(ckpt, path)
    print("已保存:", path)


if __name__ == "__main__":
    main()
