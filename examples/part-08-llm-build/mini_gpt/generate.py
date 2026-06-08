"""从 checkpoint 生成文本。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import MiniGPT

CKPT = Path(__file__).resolve().parent / "checkpoints" / "mini_gpt.pt"


@torch.no_grad()
def generate(model, stoi, itos, prompt: str, max_new: int = 80, temperature: float = 0.8):
    device = next(model.parameters()).device
    ids = [stoi.get(c, 0) for c in prompt]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model(x[:, -model.max_seq_len :])
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
    return "".join(itos[i] for i in x[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="机器学习")
    parser.add_argument("--max-new", type=int, default=80)
    args = parser.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(f"请先运行 train.py，缺少 {CKPT}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(
        vocab_size=ckpt["vocab_size"], max_seq_len=ckpt["block_size"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out = generate(model, ckpt["stoi"], ckpt["itos"], args.prompt, args.max_new)
    print("Prompt:", args.prompt)
    print("Generated:", out)


if __name__ == "__main__":
    main()
