"""从零训练 BPE 分词器（教学用，依赖 tokenizers 库）。"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus_zh_en.txt"
OUT_DIR = Path(__file__).resolve().parent / "output"
VOCAB_SIZE = 800


def main() -> None:
    if not CORPUS.exists():
        raise FileNotFoundError(f"语料不存在: {CORPUS}")

    text = CORPUS.read_text(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "corpus_tmp.txt"
    tmp.write_text(text, encoding="utf-8")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC(), normalizers.Lowercase()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<unk>", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )
    tokenizer.train([str(tmp)], trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    out_path = OUT_DIR / "bpe_tokenizer.json"
    tokenizer.save(str(out_path))

    sample = "机器学习 Machine learning 很棒！"
    encoded = tokenizer.encode(sample)
    print("样例文本:", sample)
    print("Token IDs:", encoded.ids)
    print("Tokens:", encoded.tokens)
    print("已保存:", out_path)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "corpus": str(CORPUS),
        "corpus_bytes": CORPUS.stat().st_size,
    }
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
