"""迷你 SFT 演示：在 Alpaca 格式数据上继续训练字符级模型。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SFT_DATA = ROOT / "data" / "sft_instructions.jsonl"
CKPT_IN = Path(__file__).resolve().parents[1] / "mini_gpt" / "checkpoints" / "mini_gpt.pt"


def load_sft_texts(limit: int = 30) -> str:
    lines = []
    with SFT_DATA.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            row = json.loads(line)
            inst = row.get("instruction", "")
            inp = row.get("input", "")
            out = row.get("output", "")
            lines.append(f"指令:{inst}\n输入:{inp}\n回答:{out}\n")
    return "\n".join(lines)


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_gpt"))
    from model import MiniGPT

    if not CKPT_IN.exists():
        raise FileNotFoundError("请先运行 mini_gpt/train.py 生成预训练权重")

    ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(vocab_size=ckpt["vocab_size"], max_seq_len=ckpt["block_size"]).to(
        device
    )
    model.load_state_dict(ckpt["model_state"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    text = load_sft_texts()
    ids = [stoi.get(c, 0) for c in text]
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)

    model.train()
    for step in range(50):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 10 == 0:
            print(f"sft step {step+1} loss={loss.item():.4f}")

    out = Path(__file__).resolve().parent / "sft_mini_gpt.pt"
    torch.save({**ckpt, "model_state": model.state_dict(), "sft": True}, out)
    print("SFT 完成，已保存:", out)


if __name__ == "__main__":
    main()
