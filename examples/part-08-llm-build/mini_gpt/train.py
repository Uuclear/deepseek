"""迷你 GPT 预训练：Next Token Prediction，支持 config.yaml 与训练曲线记录。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import MiniGPT

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = ROOT / "data" / "corpus_zh_en.txt"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def load_yaml_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("使用 --config 需安装 pyyaml: pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def save_training_plot(history: list[dict], plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示: pip install matplotlib 可生成训练曲线图")
        return
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o", color="#3b82f6")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("mini_gpt training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print("已保存曲线:", plot_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        cfg = load_yaml_config(Path(args.config))

    train_cfg = cfg.get("training", {})
    log_cfg = cfg.get("logging", {})
    path_cfg = cfg.get("paths", {})

    epochs = args.epochs or train_cfg.get("epochs", 3)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 3e-3)
    block_size = args.block_size or train_cfg.get("block_size", 64)

    corpus_path = ROOT / path_cfg.get("corpus", "data/corpus_zh_en.txt")
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path
    text = corpus_path.read_text(encoding="utf-8")

    log_dir = Path(__file__).resolve().parent / log_cfg.get("log_dir", "checkpoints")
    log_file = log_dir / log_cfg.get("log_file", "training_log.jsonl")
    plot_file = log_dir / log_cfg.get("plot_file", "training_curves.png")

    ds = CharDataset(text, block_size=block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniGPT(vocab_size=ds.vocab_size, max_seq_len=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    log_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, opt, device)
        record = {"epoch": epoch, "loss": round(loss, 6), "device": str(device)}
        history.append(record)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"epoch {epoch}/{epochs} loss={loss:.4f} device={device}")

    ckpt = {
        "model_state": model.state_dict(),
        "stoi": ds.stoi,
        "itos": ds.itos,
        "block_size": block_size,
        "vocab_size": ds.vocab_size,
        "history": history,
    }
    path = log_dir / "mini_gpt.pt"
    torch.save(ckpt, path)
    print("已保存:", path)

    save_training_plot(history, plot_file)


if __name__ == "__main__":
    main()
